from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(
    title="Reg-Explain API",
    description="PPNR Stress-Testing Backend API",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Models ==============

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class ScenarioConfig(BaseModel):
    scenario_type: str = "baseline"  # baseline, adverse, severely_adverse, custom
    n_quarters: int = 9
    custom_params: Optional[Dict[str, float]] = None


class BankDataUpload(BaseModel):
    data: List[Dict[str, Any]]


class TrainingConfig(BaseModel):
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    xgb_estimators: int = 200
    xgb_depth: int = 6
    xgb_learning_rate: float = 0.05


class ForecastRequest(BaseModel):
    scenario_type: str = "baseline"
    n_quarters: int = 9
    bank_ids: Optional[List[str]] = None


class StressTestResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: str
    run_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    banks_count: int
    min_cet1: float
    breach_count: int
    total_losses: float


# ============== Endpoints ==============

@api_router.get("/")
async def root():
    return {
        "message": "Reg-Explain API",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/scenarios",
            "/api/banks",
            "/api/forecast",
            "/api/results"
        ]
    }


@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected"
    }


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks


# ============== Scenario Endpoints ==============

@api_router.get("/scenarios")
async def get_scenarios():
    """Get available stress test scenarios"""
    from data.synthetic_generator import get_fed_2026_scenarios
    return get_fed_2026_scenarios()


@api_router.post("/scenarios/project")
async def project_scenario(config: ScenarioConfig):
    """Project macro variables for a scenario"""
    from data.fred_integration import FREDDataFetcher
    
    fetcher = FREDDataFetcher()
    scenario_df = fetcher.project_scenario(
        config.scenario_type, 
        config.n_quarters
    )
    
    return scenario_df.to_dict(orient='records')


@api_router.get("/macro/current")
async def get_current_macro():
    """Get current macroeconomic conditions from FRED"""
    from data.fred_integration import FREDDataFetcher
    
    fetcher = FREDDataFetcher()
    conditions = fetcher.get_current_conditions()
    return conditions


# ============== Bank Data Endpoints ==============

@api_router.post("/banks/synthetic")
async def generate_synthetic_banks(n_banks: int = 10, n_quarters: int = 36):
    """Generate synthetic bank portfolio data"""
    from data.synthetic_generator import SyntheticBankDataGenerator
    
    generator = SyntheticBankDataGenerator()
    bank_df = generator.generate_bank_portfolio(n_banks, n_quarters)
    
    # Store in MongoDB
    records = bank_df.to_dict(orient='records')
    for record in records:
        record['quarter_date'] = record['quarter_date'].isoformat()
        record['created_at'] = datetime.now(timezone.utc).isoformat()
    
    await db.bank_portfolios.delete_many({})  # Clear existing
    await db.bank_portfolios.insert_many(records)
    
    return {
        "message": f"Generated {n_banks} banks with {n_quarters} quarters of data",
        "total_records": len(records),
        "banks": bank_df['bank_id'].unique().tolist()
    }


@api_router.get("/banks")
async def get_banks():
    """Get all bank portfolio data"""
    records = await db.bank_portfolios.find({}, {"_id": 0}).to_list(10000)
    return records


@api_router.get("/banks/{bank_id}")
async def get_bank(bank_id: str):
    """Get specific bank data"""
    records = await db.bank_portfolios.find(
        {"bank_id": bank_id}, 
        {"_id": 0}
    ).to_list(1000)
    
    if not records:
        raise HTTPException(status_code=404, detail="Bank not found")
    
    return records


@api_router.post("/banks/upload")
async def upload_bank_data(file: UploadFile = File(...)):
    """Upload custom bank data CSV"""
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    records = df.to_dict(orient='records')
    for record in records:
        record['created_at'] = datetime.now(timezone.utc).isoformat()
    
    await db.bank_portfolios.delete_many({})
    await db.bank_portfolios.insert_many(records)
    
    return {
        "message": "Bank data uploaded successfully",
        "total_records": len(records),
        "columns": list(df.columns)
    }


# ============== Model Training Endpoints ==============

@api_router.post("/model/train")
async def train_models(config: TrainingConfig):
    """Train LSTM-XGBoost ensemble model"""
    from data.synthetic_generator import SyntheticBankDataGenerator
    from models.lstm_model import PPNRLSTMModel
    from models.xgboost_model import PPNRXGBoostModel
    from models.ensemble import PPNREnsembleModel
    
    # Generate training data
    generator = SyntheticBankDataGenerator()
    X_temporal, X_static, y = generator.generate_training_data(n_samples=1000)
    
    # Train LSTM
    lstm_model = PPNRLSTMModel(sequence_length=9, n_features=3)
    lstm_model.build_model(
        lstm_units=config.lstm_units, 
        dropout_rate=config.lstm_dropout
    )
    lstm_history = lstm_model.fit(
        X_temporal, y, 
        epochs=config.lstm_epochs, 
        batch_size=32
    )
    
    # Train XGBoost
    xgb_model = PPNRXGBoostModel()
    xgb_model.build_model(
        n_estimators=config.xgb_estimators,
        max_depth=config.xgb_depth,
        learning_rate=config.xgb_learning_rate
    )
    
    feature_names = ['cre_exposure', 'residential_exposure', 'cet1_ratio', 
                    'total_assets_log', 'npl_ratio']
    xgb_model.fit(X_static, y, feature_names=feature_names)
    
    # Calibrate ensemble
    ensemble = PPNREnsembleModel(lstm_model, xgb_model)
    lstm_weight, xgb_weight = ensemble.calibrate_weights(
        X_temporal, X_static, y, method='optimize'
    )
    
    # Save models
    model_dir = ROOT_DIR / 'saved_models'
    model_dir.mkdir(exist_ok=True)
    
    lstm_model.save(str(model_dir / 'lstm'))
    xgb_model.save(str(model_dir / 'xgboost'))
    ensemble.save(str(model_dir / 'ensemble'))
    
    # Store training record
    training_record = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': config.model_dump(),
        'lstm_weight': lstm_weight,
        'xgb_weight': xgb_weight,
        'feature_importance': xgb_model.get_feature_importance()
    }
    
    await db.training_runs.insert_one(training_record)
    
    return {
        "message": "Models trained successfully",
        "lstm_weight": lstm_weight,
        "xgb_weight": xgb_weight,
        "feature_importance": xgb_model.get_feature_importance()
    }


@api_router.get("/model/status")
async def get_model_status():
    """Check if models are trained and available"""
    model_dir = ROOT_DIR / 'saved_models'
    
    lstm_exists = (model_dir / 'lstm_model.keras').exists()
    xgb_exists = (model_dir / 'xgboost_model.json').exists()
    ensemble_exists = (model_dir / 'ensemble_ensemble.pkl').exists()
    
    # Get latest training run
    latest = await db.training_runs.find_one(
        {},
        {"_id": 0},
        sort=[("timestamp", -1)]
    )
    
    return {
        "lstm_ready": lstm_exists,
        "xgboost_ready": xgb_exists,
        "ensemble_ready": ensemble_exists,
        "all_ready": lstm_exists and xgb_exists and ensemble_exists,
        "latest_training": latest
    }


# ============== Forecast Endpoints ==============

@api_router.post("/forecast/run")
async def run_forecast(request: ForecastRequest):
    """Run capital erosion forecast"""
    from data.fred_integration import FREDDataFetcher
    from data.synthetic_generator import SyntheticBankDataGenerator
    from models.lstm_model import PPNRLSTMModel
    from models.xgboost_model import PPNRXGBoostModel
    from models.ensemble import PPNREnsembleModel, QuarterlyPPNRForecaster
    from capital_engine import CapitalEngine
    
    model_dir = ROOT_DIR / 'saved_models'
    
    # Check if models exist, if not train with defaults
    if not (model_dir / 'lstm_model.keras').exists():
        # Quick train
        generator = SyntheticBankDataGenerator()
        X_temporal, X_static, y = generator.generate_training_data(n_samples=500)
        
        lstm_model = PPNRLSTMModel(sequence_length=9, n_features=3)
        lstm_model.build_model()
        lstm_model.fit(X_temporal, y, epochs=50)
        
        xgb_model = PPNRXGBoostModel()
        xgb_model.build_model()
        feature_names = ['cre_exposure', 'residential_exposure', 'cet1_ratio', 
                        'total_assets_log', 'npl_ratio']
        xgb_model.fit(X_static, y, feature_names=feature_names)
        
        ensemble = PPNREnsembleModel(lstm_model, xgb_model)
        ensemble.calibrate_weights(X_temporal, X_static, y)
        
        model_dir.mkdir(exist_ok=True)
        lstm_model.save(str(model_dir / 'lstm'))
        xgb_model.save(str(model_dir / 'xgboost'))
        ensemble.save(str(model_dir / 'ensemble'))
    else:
        # Load models
        lstm_model = PPNRLSTMModel()
        lstm_model.load(str(model_dir / 'lstm'))
        
        xgb_model = PPNRXGBoostModel()
        xgb_model.load(str(model_dir / 'xgboost'))
        
        ensemble = PPNREnsembleModel(lstm_model, xgb_model)
        ensemble.load(str(model_dir / 'ensemble'))
    
    # Get bank data
    bank_records = await db.bank_portfolios.find({}, {"_id": 0}).to_list(10000)
    
    if not bank_records:
        # Generate synthetic if none exists
        generator = SyntheticBankDataGenerator()
        bank_df = generator.generate_bank_portfolio(n_banks=10, n_quarters=36)
        bank_records = bank_df.to_dict(orient='records')
    
    bank_df = pd.DataFrame(bank_records)
    
    # Get latest quarter for each bank
    if 'quarter' in bank_df.columns:
        bank_df = bank_df.groupby('bank_id').last().reset_index()
    
    # Generate scenario
    fetcher = FREDDataFetcher()
    scenario_df = fetcher.project_scenario(request.scenario_type, request.n_quarters)
    
    # Run forecast
    forecaster = QuarterlyPPNRForecaster(ensemble)
    ppnr_forecasts = forecaster.forecast(bank_df, scenario_df, request.n_quarters)
    
    # Calculate capital trajectory
    engine = CapitalEngine()
    trajectory = engine.calculate_cet1_trajectory(
        bank_df, ppnr_forecasts, scenario_df, request.n_quarters
    )
    
    # Store results
    result_id = str(uuid.uuid4())
    result_record = {
        'id': result_id,
        'scenario': request.scenario_type,
        'run_date': datetime.now(timezone.utc).isoformat(),
        'n_quarters': request.n_quarters,
        'trajectory': trajectory.to_dict(orient='records'),
        'ppnr_forecasts': ppnr_forecasts.to_dict(orient='records'),
        'scenario_data': scenario_df.to_dict(orient='records')
    }
    
    await db.forecast_results.insert_one(result_record)
    
    # Get breach analysis
    breach_analysis = engine.get_breach_analysis(trajectory)
    
    return {
        "result_id": result_id,
        "scenario": request.scenario_type,
        "min_cet1": trajectory['cet1_ratio'].min(),
        "breach_count": breach_analysis['banks_with_breach'],
        "total_banks": breach_analysis['total_banks'],
        "first_breach_quarter": breach_analysis['first_breach_quarter'],
        "trajectory_preview": trajectory.head(20).to_dict(orient='records')
    }


@api_router.get("/forecast/results")
async def get_forecast_results(limit: int = 10):
    """Get historical forecast results"""
    results = await db.forecast_results.find(
        {},
        {"_id": 0, "trajectory": 0, "ppnr_forecasts": 0, "scenario_data": 0}
    ).sort("run_date", -1).to_list(limit)
    
    return results


@api_router.get("/forecast/results/{result_id}")
async def get_forecast_result(result_id: str):
    """Get specific forecast result with full trajectory"""
    result = await db.forecast_results.find_one(
        {"id": result_id},
        {"_id": 0}
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return result


# ============== SHAP Explanation Endpoints ==============

@api_router.post("/explain/global")
async def get_global_explanation(scenario_type: str = "severely_adverse"):
    """Get global SHAP feature importance"""
    from data.synthetic_generator import SyntheticBankDataGenerator
    from models.xgboost_model import PPNRXGBoostModel, create_static_features
    from shap_explainer import SHAPExplainer
    from data.fred_integration import FREDDataFetcher
    
    model_dir = ROOT_DIR / 'saved_models'
    
    if not (model_dir / 'xgboost_model.json').exists():
        raise HTTPException(status_code=400, detail="Models not trained. Please train first.")
    
    # Load model
    xgb_model = PPNRXGBoostModel()
    xgb_model.load(str(model_dir / 'xgboost'))
    
    # Get bank data
    bank_records = await db.bank_portfolios.find({}, {"_id": 0}).to_list(10000)
    
    if not bank_records:
        generator = SyntheticBankDataGenerator()
        bank_df = generator.generate_bank_portfolio(n_banks=10, n_quarters=36)
    else:
        bank_df = pd.DataFrame(bank_records)
        if 'quarter' in bank_df.columns:
            bank_df = bank_df.groupby('bank_id').last().reset_index()
    
    # Get scenario data
    fetcher = FREDDataFetcher()
    scenario_df = fetcher.project_scenario(scenario_type, n_quarters=9)
    
    # Create features
    X_static, feature_names = create_static_features(bank_df, scenario_df)
    
    # Ensure consistent feature dimensions (use only the 5 core features)
    core_features = ['cre_exposure', 'residential_exposure', 'cet1_ratio', 'total_assets_log', 'npl_ratio']
    if len(feature_names) > 5:
        # Find indices of core features
        core_indices = []
        for core_feat in core_features:
            if core_feat in feature_names:
                core_indices.append(feature_names.index(core_feat))
        
        if len(core_indices) == 5:
            X_static = X_static[:, core_indices]
            feature_names = core_features
    
    # SHAP analysis
    explainer = SHAPExplainer(xgb_model)
    explainer.initialize_explainer(X_static, feature_names)
    shap_results = explainer.explain_predictions(X_static)
    
    return {
        "scenario": scenario_type,
        "global_importance": shap_results['global_importance'],
        "expected_value": float(shap_results['expected_value']),
        "feature_names": shap_results['feature_names']
    }


@api_router.post("/explain/ai")
async def get_ai_explanation(scenario_type: str = "severely_adverse"):
    """Get AI-powered natural language explanation using Gemini"""
    from shap_explainer import GeminiExplainer
    
    # First get SHAP results
    shap_response = await get_global_explanation(scenario_type)
    
    # Generate AI explanation
    gemini = GeminiExplainer()
    
    import asyncio
    explanation = await gemini.generate_global_explanation(
        shap_response, 
        scenario_type
    )
    
    return {
        "scenario": scenario_type,
        "shap_importance": shap_response['global_importance'],
        "ai_explanation": explanation
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
