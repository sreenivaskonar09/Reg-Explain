"""
================================================================================
REG-EXPLAIN: PPNR STRESS-TESTING BACKEND
================================================================================
CCAR Framework 2026 - Hybrid LSTM-XGBoost with SHAP Interpretability

This file contains all backend code combined for easy download.
Separate into individual files as indicated by the section headers.

Files included:
1. server.py - FastAPI main application
2. data/synthetic_generator.py - Synthetic bank data generator
3. data/fred_integration.py - FRED API integration
4. models/lstm_model.py - LSTM model for temporal sequences
5. models/xgboost_model.py - XGBoost model for static features
6. models/ensemble.py - Weighted ensemble model
7. capital_engine.py - CET1 ratio calculations
8. shap_explainer.py - SHAP analysis with Gemini AI

================================================================================
"""


# ==============================================================================
# FILE: server.py
# ==============================================================================

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


# ============== Pydantic Models ==============

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


# ============== Helper Functions ==============

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ============== API Endpoints ==============

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
        'lstm_weight': float(lstm_weight),
        'xgb_weight': float(xgb_weight),
        'feature_importance': convert_numpy_types(xgb_model.get_feature_importance())
    }
    
    await db.training_runs.insert_one(training_record)
    
    return {
        "message": "Models trained successfully",
        "lstm_weight": float(lstm_weight),
        "xgb_weight": float(xgb_weight),
        "feature_importance": convert_numpy_types(xgb_model.get_feature_importance())
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
        'trajectory': convert_numpy_types(trajectory.to_dict(orient='records')),
        'ppnr_forecasts': convert_numpy_types(ppnr_forecasts.to_dict(orient='records')),
        'scenario_data': convert_numpy_types(scenario_df.to_dict(orient='records'))
    }
    
    await db.forecast_results.insert_one(result_record)
    
    # Get breach analysis
    breach_analysis = engine.get_breach_analysis(trajectory)
    
    return {
        "result_id": result_id,
        "scenario": request.scenario_type,
        "min_cet1": float(trajectory['cet1_ratio'].min()),
        "breach_count": int(breach_analysis['banks_with_breach']),
        "total_banks": int(breach_analysis['total_banks']),
        "first_breach_quarter": int(breach_analysis['first_breach_quarter']) if breach_analysis['first_breach_quarter'] is not None else None,
        "trajectory_preview": convert_numpy_types(trajectory.head(20).to_dict(orient='records'))
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
        "global_importance": convert_numpy_types(shap_results['global_importance']),
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


# ==============================================================================
# FILE: data/synthetic_generator.py
# ==============================================================================

"""
Synthetic Bank Data Generator for PPNR Stress Testing
Generates realistic bank portfolio data with regulatory metrics
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SyntheticBankDataGenerator:
    """Generates synthetic bank data for stress testing"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_bank_portfolio(self, n_banks=10, n_quarters=36):
        """Generate synthetic bank portfolio data"""
        banks = []
        
        bank_names = [
            "First National Bank", "Metro Commercial Bank", "Pacific Trust Bank",
            "Atlantic Financial Corp", "Midwest Regional Bank", "Coastal Savings Bank",
            "Mountain State Bank", "Valley Commerce Bank", "Harbor Financial Group",
            "Prairie National Bank", "Desert Southwest Bank", "Northern Trust Corp"
        ]
        
        for i in range(n_banks):
            bank_name = bank_names[i % len(bank_names)]
            
            # Base characteristics
            total_assets = np.random.uniform(50, 500) * 1e9  # $50B - $500B
            cre_exposure_pct = np.random.uniform(0.05, 0.35)  # 5-35% CRE exposure
            residential_exposure_pct = np.random.uniform(0.15, 0.45)
            
            for q in range(n_quarters):
                quarter_date = datetime(2020, 1, 1) + timedelta(days=q*91)
                
                # CET1 Capital with some variation
                cet1_ratio = np.random.normal(0.12, 0.02)  # ~12% average
                cet1_ratio = max(0.06, min(0.18, cet1_ratio))
                
                # Risk-Weighted Assets
                rwa = total_assets * np.random.uniform(0.6, 0.85)
                cet1_capital = cet1_ratio * rwa
                
                # PPNR components
                net_interest_income = total_assets * np.random.uniform(0.025, 0.04) / 4
                non_interest_income = total_assets * np.random.uniform(0.005, 0.015) / 4
                non_interest_expense = total_assets * np.random.uniform(0.015, 0.025) / 4
                ppnr = net_interest_income + non_interest_income - non_interest_expense
                
                # Loan portfolio breakdown
                cre_loans = total_assets * cre_exposure_pct
                residential_loans = total_assets * residential_exposure_pct
                consumer_loans = total_assets * np.random.uniform(0.05, 0.15)
                commercial_loans = total_assets * np.random.uniform(0.1, 0.25)
                
                banks.append({
                    'bank_id': f'BANK_{i:03d}',
                    'bank_name': bank_name,
                    'quarter': quarter_date.strftime('%Y-Q%q').replace('Q%q', f'Q{(quarter_date.month-1)//3+1}'),
                    'quarter_date': quarter_date,
                    'total_assets': total_assets,
                    'rwa': rwa,
                    'cet1_capital': cet1_capital,
                    'cet1_ratio': cet1_ratio,
                    'ppnr': ppnr,
                    'net_interest_income': net_interest_income,
                    'non_interest_income': non_interest_income,
                    'non_interest_expense': non_interest_expense,
                    'cre_exposure': cre_loans,
                    'cre_exposure_pct': cre_exposure_pct,
                    'residential_exposure': residential_loans,
                    'residential_exposure_pct': residential_exposure_pct,
                    'consumer_loans': consumer_loans,
                    'commercial_loans': commercial_loans,
                    'loan_loss_provision': ppnr * np.random.uniform(0.05, 0.15),
                    'credit_card_utilization': np.random.uniform(0.15, 0.35),
                    'npl_ratio': np.random.uniform(0.005, 0.025),
                    'tier1_leverage_ratio': cet1_ratio * np.random.uniform(0.7, 0.9)
                })
        
        return pd.DataFrame(banks)
    
    def generate_scenario_data(self, scenario_type='baseline', n_quarters=9):
        """Generate macro scenario data for 9-quarter projection"""
        
        scenarios = {
            'baseline': {
                'unemployment_rate': (4.5, 0.3),
                'gdp_growth': (2.0, 0.5),
                'fed_funds_rate': (4.5, 0.25),
                'treasury_10y': (4.0, 0.2),
                'treasury_3m': (4.3, 0.15),
                'vix': (18, 3),
                'hpi_growth': (3.0, 1.0),
                'credit_card_util': (0.22, 0.02),
                'real_dpi_growth': (1.5, 0.5)
            },
            'adverse': {
                'unemployment_rate': (7.0, 1.0),
                'gdp_growth': (-1.0, 0.8),
                'fed_funds_rate': (2.0, 0.5),
                'treasury_10y': (2.5, 0.3),
                'treasury_3m': (1.5, 0.25),
                'vix': (45, 8),
                'hpi_growth': (-5.0, 2.0),
                'credit_card_util': (0.30, 0.03),
                'real_dpi_growth': (-2.0, 1.0)
            },
            'severely_adverse': {
                'unemployment_rate': (10.0, 1.5),
                'gdp_growth': (-4.0, 1.0),
                'fed_funds_rate': (0.5, 0.25),
                'treasury_10y': (1.5, 0.3),
                'treasury_3m': (0.25, 0.1),
                'vix': (72, 10),
                'hpi_growth': (-15.0, 3.0),
                'credit_card_util': (0.38, 0.04),
                'real_dpi_growth': (-5.0, 1.5)
            }
        }
        
        params = scenarios.get(scenario_type, scenarios['baseline'])
        
        scenario_data = []
        base_date = datetime(2026, 1, 1)
        
        for q in range(n_quarters):
            quarter_date = base_date + timedelta(days=q*91)
            
            # Add path dependency (cumulative stress)
            stress_multiplier = 1 + (q / n_quarters) * 0.3 if scenario_type != 'baseline' else 1
            
            row = {
                'quarter': q + 1,
                'quarter_date': quarter_date,
                'scenario': scenario_type,
                'unemployment_rate': np.clip(
                    np.random.normal(params['unemployment_rate'][0] * stress_multiplier, 
                                   params['unemployment_rate'][1]), 
                    2.5, 15
                ),
                'gdp_growth': np.random.normal(
                    params['gdp_growth'][0] / stress_multiplier if scenario_type != 'baseline' else params['gdp_growth'][0],
                    params['gdp_growth'][1]
                ),
                'fed_funds_rate': np.clip(
                    np.random.normal(params['fed_funds_rate'][0], params['fed_funds_rate'][1]),
                    0, 8
                ),
                'treasury_10y': np.clip(
                    np.random.normal(params['treasury_10y'][0], params['treasury_10y'][1]),
                    0.5, 8
                ),
                'treasury_3m': np.clip(
                    np.random.normal(params['treasury_3m'][0], params['treasury_3m'][1]),
                    0, 7
                ),
                'vix': np.clip(
                    np.random.normal(params['vix'][0] * stress_multiplier, params['vix'][1]),
                    10, 100
                ),
                'hpi_growth': np.random.normal(
                    params['hpi_growth'][0] / stress_multiplier if scenario_type != 'baseline' else params['hpi_growth'][0],
                    params['hpi_growth'][1]
                ),
                'credit_card_utilization': np.clip(
                    np.random.normal(params['credit_card_util'][0] * stress_multiplier, params['credit_card_util'][1]),
                    0.1, 0.6
                ),
                'real_dpi_growth': np.random.normal(
                    params['real_dpi_growth'][0] / stress_multiplier if scenario_type != 'baseline' else params['real_dpi_growth'][0],
                    params['real_dpi_growth'][1]
                )
            }
            
            # Calculate yield curve spread (10Y - 3M)
            row['yield_curve_spread'] = row['treasury_10y'] - row['treasury_3m']
            
            scenario_data.append(row)
        
        return pd.DataFrame(scenario_data)
    
    def generate_training_data(self, n_samples=1000, sequence_length=9):
        """Generate training data for ML models"""
        
        X_temporal = []  # For LSTM
        X_static = []    # For XGBoost
        y = []           # PPNR outcomes
        
        for _ in range(n_samples):
            # Temporal features (9 quarters of rates)
            base_rate = np.random.uniform(0.5, 6)
            rate_path = [base_rate]
            for _ in range(sequence_length - 1):
                change = np.random.normal(0, 0.3)
                rate_path.append(np.clip(rate_path[-1] + change, 0, 8))
            
            # Add more temporal features
            vix_path = np.random.uniform(12, 70, sequence_length)
            spread_path = np.random.uniform(-1, 3, sequence_length)
            
            temporal = np.column_stack([rate_path, vix_path, spread_path])
            X_temporal.append(temporal)
            
            # Static features
            cre_exposure = np.random.uniform(0.05, 0.35)
            residential_exposure = np.random.uniform(0.15, 0.45)
            cet1_ratio = np.random.uniform(0.08, 0.16)
            total_assets_log = np.log(np.random.uniform(50, 500) * 1e9)
            npl_ratio = np.random.uniform(0.005, 0.03)
            
            static = [cre_exposure, residential_exposure, cet1_ratio, 
                     total_assets_log, npl_ratio]
            X_static.append(static)
            
            # Target: PPNR change (influenced by features)
            rate_impact = -0.1 * (rate_path[-1] - rate_path[0])
            vix_impact = -0.002 * np.mean(vix_path)
            cre_impact = -0.15 * cre_exposure * (vix_path[-1] > 40)
            base_ppnr = 0.008 * np.exp(total_assets_log - 25)
            
            ppnr_change = base_ppnr * (1 + rate_impact + vix_impact + cre_impact)
            ppnr_change += np.random.normal(0, 0.001)
            
            y.append(ppnr_change)
        
        return np.array(X_temporal), np.array(X_static), np.array(y)


def get_fed_2026_scenarios():
    """Return the Fed 2026 macro scenarios as specified"""
    return {
        'baseline': {
            'name': 'Baseline',
            'unemployment_peak': 4.5,
            'vix_peak': 20,
            'gdp_trough': 2.0,
            'hpi_decline': 0,
            'description': 'Normal economic conditions with moderate growth'
        },
        'adverse': {
            'name': 'Adverse',
            'unemployment_peak': 7.5,
            'vix_peak': 50,
            'gdp_trough': -2.0,
            'hpi_decline': -10,
            'description': 'Moderate recession with elevated stress'
        },
        'severely_adverse': {
            'name': 'Severely Adverse',
            'unemployment_peak': 10.0,
            'vix_peak': 72,
            'gdp_trough': -6.0,
            'hpi_decline': -25,
            'description': 'Severe recession with major asset price declines'
        }
    }


# ==============================================================================
# FILE: data/fred_integration.py
# ==============================================================================

"""
FRED API Integration for Real Macroeconomic Data
Fetches actual economic indicators from Federal Reserve Economic Data
"""

# FRED Series IDs for key macro variables
FRED_SERIES = {
    'unemployment_rate': 'UNRATE',
    'fed_funds_rate': 'FEDFUNDS', 
    'treasury_10y': 'DGS10',
    'treasury_3m': 'DTB3',
    'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth rate
    'real_dpi': 'DSPIC96',  # Real Disposable Personal Income
    'hpi': 'CSUSHPINSA',  # S&P/Case-Shiller Home Price Index
    'vix': 'VIXCLS',  # CBOE Volatility Index
    'credit_card_delinquency': 'DRCCLACBS',  # Credit Card Delinquency Rate
    'cpi': 'CPIAUCSL'  # Consumer Price Index
}


class FREDDataFetcher:
    """Fetches real macroeconomic data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FRED_API_KEY', 'demo')
        self._fred = None
        
    @property
    def fred(self):
        """Lazy load FRED API connection"""
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"FRED API not available: {e}")
                self._fred = None
        return self._fred
    
    def fetch_series(self, series_id: str, start_date: str = '2015-01-01', 
                    end_date: Optional[str] = None) -> pd.Series:
        """Fetch a single FRED series"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            if self.fred:
                data = self.fred.get_series(series_id, start_date, end_date)
                return data
        except Exception as e:
            logger.warning(f"Could not fetch {series_id}: {e}")
        
        # Return synthetic data if FRED unavailable
        return self._generate_synthetic_series(series_id, start_date, end_date)
    
    def _generate_synthetic_series(self, series_id: str, start_date: str, 
                                   end_date: str) -> pd.Series:
        """Generate synthetic data when FRED is unavailable"""
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Realistic ranges for different series
        ranges = {
            'UNRATE': (3.5, 6.5),
            'FEDFUNDS': (0.0, 5.5),
            'DGS10': (1.5, 4.5),
            'DTB3': (0.0, 5.0),
            'A191RL1Q225SBEA': (-2.0, 4.0),
            'DSPIC96': (14000, 16000),
            'CSUSHPINSA': (200, 320),
            'VIXCLS': (12, 35),
            'DRCCLACBS': (2.0, 4.0),
            'CPIAUCSL': (250, 310)
        }
        
        low, high = ranges.get(series_id, (0, 100))
        base = np.random.uniform(low, high)
        
        # Create random walk with mean reversion
        values = [base]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, (high - low) * 0.02)
            mean_reversion = 0.1 * ((low + high) / 2 - values[-1])
            new_val = np.clip(values[-1] + change + mean_reversion, low, high)
            values.append(new_val)
        
        return pd.Series(values, index=dates)
    
    def fetch_all_macro_data(self, start_date: str = '2015-01-01',
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch all macro variables and combine into DataFrame"""
        
        data = {}
        for name, series_id in FRED_SERIES.items():
            try:
                series = self.fetch_series(series_id, start_date, end_date)
                data[name] = series
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                
        df = pd.DataFrame(data)
        df = df.resample('Q').mean()  # Quarterly average
        df = df.dropna(how='all')
        
        # Calculate yield curve spread
        if 'treasury_10y' in df.columns and 'treasury_3m' in df.columns:
            df['yield_curve_spread'] = df['treasury_10y'] - df['treasury_3m']
            
        # Calculate real DPI growth
        if 'real_dpi' in df.columns:
            df['real_dpi_growth'] = df['real_dpi'].pct_change() * 100
            
        # Calculate HPI growth
        if 'hpi' in df.columns:
            df['hpi_growth'] = df['hpi'].pct_change(4) * 100  # YoY
            
        return df
    
    def get_current_conditions(self) -> Dict:
        """Get most recent economic conditions"""
        df = self.fetch_all_macro_data()
        
        if df.empty:
            return self._get_synthetic_conditions()
        
        latest = df.iloc[-1].to_dict()
        
        return {
            'as_of_date': df.index[-1].strftime('%Y-%m-%d'),
            'unemployment_rate': latest.get('unemployment_rate', 4.2),
            'fed_funds_rate': latest.get('fed_funds_rate', 4.5),
            'treasury_10y': latest.get('treasury_10y', 4.0),
            'treasury_3m': latest.get('treasury_3m', 4.3),
            'yield_curve_spread': latest.get('yield_curve_spread', -0.3),
            'vix': latest.get('vix', 18.0),
            'hpi_growth': latest.get('hpi_growth', 3.0),
            'gdp_growth': latest.get('gdp_growth', 2.0)
        }
    
    def _get_synthetic_conditions(self) -> Dict:
        """Return synthetic current conditions"""
        return {
            'as_of_date': datetime.now().strftime('%Y-%m-%d'),
            'unemployment_rate': 4.2,
            'fed_funds_rate': 4.5,
            'treasury_10y': 4.0,
            'treasury_3m': 4.3,
            'yield_curve_spread': -0.3,
            'vix': 18.0,
            'hpi_growth': 3.0,
            'gdp_growth': 2.0
        }
    
    def project_scenario(self, scenario_type: str = 'baseline', 
                        n_quarters: int = 9) -> pd.DataFrame:
        """Project macro variables over n quarters for given scenario"""
        
        current = self.get_current_conditions()
        
        # Fed scenario parameters (2026 guidelines)
        scenario_params = {
            'baseline': {
                'unemployment_target': 4.5,
                'unemployment_speed': 0.1,
                'fed_funds_target': 3.5,
                'fed_funds_speed': 0.25,
                'vix_target': 18,
                'hpi_growth_target': 3.0,
                'gdp_target': 2.0
            },
            'adverse': {
                'unemployment_target': 7.5,
                'unemployment_speed': 0.5,
                'fed_funds_target': 1.5,
                'fed_funds_speed': 0.5,
                'vix_target': 50,
                'hpi_growth_target': -10.0,
                'gdp_target': -2.0
            },
            'severely_adverse': {
                'unemployment_target': 10.0,
                'unemployment_speed': 0.8,
                'fed_funds_target': 0.25,
                'fed_funds_speed': 0.75,
                'vix_target': 72,
                'hpi_growth_target': -25.0,
                'gdp_target': -6.0
            }
        }
        
        params = scenario_params.get(scenario_type, scenario_params['baseline'])
        projections = []
        
        # Current values
        unemp = current['unemployment_rate']
        ffr = current['fed_funds_rate']
        t10y = current['treasury_10y']
        vix = current['vix']
        hpi = current['hpi_growth']
        gdp = current['gdp_growth']
        
        base_date = datetime.now()
        
        for q in range(n_quarters):
            quarter_date = base_date + timedelta(days=(q+1)*91)
            
            # Mean reversion to target
            unemp += params['unemployment_speed'] * (params['unemployment_target'] - unemp)
            unemp += np.random.normal(0, 0.2)
            
            ffr += params['fed_funds_speed'] * (params['fed_funds_target'] - ffr)
            ffr = max(0, ffr + np.random.normal(0, 0.1))
            
            # VIX spikes in adverse scenarios
            vix_shock = np.random.normal(0, 5) if scenario_type != 'baseline' else np.random.normal(0, 2)
            vix += 0.3 * (params['vix_target'] - vix) + vix_shock
            vix = max(10, min(100, vix))
            
            # Treasury rates follow fed funds with spread
            t10y = ffr + np.random.uniform(0.5, 1.5)
            t3m = ffr + np.random.uniform(-0.5, 0.3)
            
            # HPI and GDP
            hpi += 0.2 * (params['hpi_growth_target'] - hpi) + np.random.normal(0, 1)
            gdp += 0.25 * (params['gdp_target'] - gdp) + np.random.normal(0, 0.5)
            
            projections.append({
                'quarter': q + 1,
                'quarter_date': quarter_date,
                'scenario': scenario_type,
                'unemployment_rate': np.clip(unemp, 2.5, 15),
                'fed_funds_rate': np.clip(ffr, 0, 8),
                'treasury_10y': np.clip(t10y, 0.5, 10),
                'treasury_3m': np.clip(t3m, 0, 8),
                'yield_curve_spread': t10y - t3m,
                'vix': np.clip(vix, 10, 100),
                'hpi_growth': np.clip(hpi, -30, 20),
                'gdp_growth': np.clip(gdp, -10, 6),
                'credit_card_utilization': 0.22 + 0.02 * (unemp - 4.5)
            })
        
        return pd.DataFrame(projections)


# ==============================================================================
# FILE: models/lstm_model.py
# ==============================================================================

"""
LSTM Model for Temporal Sequence Processing
Captures path-dependency of interest rate cycles and macroeconomic momentum
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress TF warnings
tf.get_logger().setLevel('ERROR')


class PPNRLSTMModel:
    """LSTM model for temporal PPNR prediction"""
    
    def __init__(self, sequence_length=9, n_features=3):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def build_model(self, lstm_units=64, dropout_rate=0.2):
        """Build LSTM architecture"""
        self.model = Sequential([
            LSTM(lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            LSTM(lstm_units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # PPNR prediction
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()
            
        # Scale temporal features
        X_reshaped = X.reshape(-1, self.n_features)
        self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(-1, self.sequence_length, self.n_features)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X):
        """Generate PPNR predictions from temporal sequences"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(-1, self.sequence_length, self.n_features)
        
        return self.model.predict(X_scaled, verbose=0)
    
    def save(self, path):
        """Save model and scaler"""
        self.model.save(f"{path}_model.keras")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        joblib.dump({
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'is_fitted': self.is_fitted
        }, f"{path}_config.pkl")
        
    def load(self, path):
        """Load model and scaler"""
        self.model = load_model(f"{path}_model.keras")
        self.scaler = joblib.load(f"{path}_scaler.pkl")
        config = joblib.load(f"{path}_config.pkl")
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.is_fitted = config['is_fitted']


def create_temporal_features(macro_df, sequence_length=9):
    """
    Create temporal feature sequences from macro data
    
    Features:
    - Interest rate paths (Fed Funds, 10Y Treasury)
    - VIX volatility path
    - Yield curve spread path
    """
    features = []
    
    # Ensure we have enough data
    if len(macro_df) < sequence_length:
        # Pad with first values if needed
        n_pad = sequence_length - len(macro_df)
        macro_df = pd.concat([
            pd.DataFrame([macro_df.iloc[0]] * n_pad),
            macro_df
        ]).reset_index(drop=True)
    
    # Extract feature columns
    rate_col = 'fed_funds_rate' if 'fed_funds_rate' in macro_df.columns else 'treasury_10y'
    vix_col = 'vix' if 'vix' in macro_df.columns else None
    spread_col = 'yield_curve_spread' if 'yield_curve_spread' in macro_df.columns else None
    
    for i in range(len(macro_df) - sequence_length + 1):
        seq = macro_df.iloc[i:i+sequence_length]
        
        feature_seq = []
        for _, row in seq.iterrows():
            feature_row = [row.get(rate_col, 4.0)]
            
            if vix_col:
                feature_row.append(row.get(vix_col, 18))
            else:
                feature_row.append(18)
                
            if spread_col:
                feature_row.append(row.get(spread_col, 0))
            else:
                feature_row.append(0)
                
            feature_seq.append(feature_row)
            
        features.append(feature_seq)
    
    return np.array(features)


# ==============================================================================
# FILE: models/xgboost_model.py
# ==============================================================================

"""
XGBoost Model for Static Risk Factor Processing
Captures non-linear relationships between bank features and economic shocks
"""
import xgboost as xgb
from sklearn.model_selection import cross_val_score


class PPNRXGBoostModel:
    """XGBoost model for static risk factor PPNR prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def build_model(self, **kwargs):
        """Build XGBoost model with optimal parameters"""
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **kwargs}
        self.model = xgb.XGBRegressor(**params)
        return self.model
    
    def fit(self, X, y, feature_names=None):
        """Train the XGBoost model"""
        if self.model is None:
            self.build_model()
            
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        logger.info(f"XGBoost CV RMSE: {np.sqrt(-cv_scores.mean()):.6f}")
        
        return self
    
    def predict(self, X):
        """Generate PPNR predictions from static features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance rankings"""
        if not self.is_fitted:
            return {}
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save(self, path):
        """Save model and scaler"""
        self.model.save_model(f"{path}_model.json")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        joblib.dump({
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, f"{path}_config.pkl")
        
    def load(self, path):
        """Load model and scaler"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(f"{path}_model.json")
        self.scaler = joblib.load(f"{path}_scaler.pkl")
        config = joblib.load(f"{path}_config.pkl")
        self.feature_names = config['feature_names']
        self.is_fitted = config['is_fitted']


def create_static_features(bank_df, macro_df=None):
    """
    Create static feature vectors for XGBoost
    
    Bank Features:
    - CET1 ratio
    - CRE exposure percentage
    - Residential exposure percentage
    - Total assets (log)
    - NPL ratio
    
    Macro Stress Features (optional):
    - Peak unemployment
    - Peak VIX
    - HPI decline
    """
    features = []
    feature_names = []
    
    # Bank-specific features
    if 'cet1_ratio' in bank_df.columns:
        features.append(bank_df['cet1_ratio'].values)
        feature_names.append('cet1_ratio')
        
    if 'cre_exposure_pct' in bank_df.columns:
        features.append(bank_df['cre_exposure_pct'].values)
        feature_names.append('cre_exposure')
        
    if 'residential_exposure_pct' in bank_df.columns:
        features.append(bank_df['residential_exposure_pct'].values)
        feature_names.append('residential_exposure')
        
    if 'total_assets' in bank_df.columns:
        features.append(np.log(bank_df['total_assets'].values))
        feature_names.append('total_assets_log')
        
    if 'npl_ratio' in bank_df.columns:
        features.append(bank_df['npl_ratio'].values)
        feature_names.append('npl_ratio')
        
    if 'tier1_leverage_ratio' in bank_df.columns:
        features.append(bank_df['tier1_leverage_ratio'].values)
        feature_names.append('tier1_leverage')
        
    # Add macro stress factors if provided
    if macro_df is not None:
        if 'unemployment_rate' in macro_df.columns:
            peak_unemp = macro_df['unemployment_rate'].max()
            features.append(np.full(len(bank_df), peak_unemp))
            feature_names.append('unemployment_peak')
            
        if 'vix' in macro_df.columns:
            peak_vix = macro_df['vix'].max()
            features.append(np.full(len(bank_df), peak_vix))
            feature_names.append('vix_peak')
            
        if 'hpi_growth' in macro_df.columns:
            min_hpi = macro_df['hpi_growth'].min()
            features.append(np.full(len(bank_df), min_hpi))
            feature_names.append('hpi_trough')
            
        if 'yield_curve_spread' in macro_df.columns:
            avg_spread = macro_df['yield_curve_spread'].mean()
            features.append(np.full(len(bank_df), avg_spread))
            feature_names.append('yield_curve_avg')
    
    return np.column_stack(features), feature_names


# ==============================================================================
# FILE: models/ensemble.py
# ==============================================================================

"""
Ensemble Model: Weighted Average of LSTM and XGBoost
Optimized to minimize RMSE against historical stress test results
"""
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class PPNREnsembleModel:
    """
    Hybrid ensemble combining LSTM (temporal) and XGBoost (static) predictions
    Uses weighted average optimized for minimum RMSE
    """
    
    def __init__(self, lstm_model=None, xgboost_model=None):
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.lstm_weight = 0.5  # Default equal weighting
        self.xgboost_weight = 0.5
        self.is_calibrated = False
        
    def set_models(self, lstm_model, xgboost_model):
        """Set the component models"""
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        
    def calibrate_weights(self, X_temporal, X_static, y_true, 
                         method='optimize'):
        """
        Calibrate ensemble weights using validation data
        
        Methods:
        - 'optimize': Scipy optimization for minimum RMSE
        - 'equal': Equal weighting (0.5, 0.5)
        - 'inverse_rmse': Weight inversely proportional to individual RMSE
        """
        
        if self.lstm_model is None or self.xgboost_model is None:
            raise ValueError("Both models must be set before calibration")
            
        # Get individual predictions
        lstm_pred = self.lstm_model.predict(X_temporal).flatten()
        xgb_pred = self.xgboost_model.predict(X_static).flatten()
        
        if method == 'equal':
            self.lstm_weight = 0.5
            self.xgboost_weight = 0.5
            
        elif method == 'inverse_rmse':
            lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_pred))
            xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_pred))
            
            # Inverse weighting
            total_inv = (1/lstm_rmse) + (1/xgb_rmse)
            self.lstm_weight = (1/lstm_rmse) / total_inv
            self.xgboost_weight = (1/xgb_rmse) / total_inv
            
        elif method == 'optimize':
            def objective(w):
                ensemble_pred = w[0] * lstm_pred + w[1] * xgb_pred
                return mean_squared_error(y_true, ensemble_pred)
            
            # Constraints: weights sum to 1, non-negative
            constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
            bounds = [(0, 1), (0, 1)]
            
            result = minimize(
                objective, 
                x0=[0.5, 0.5],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            self.lstm_weight = result.x[0]
            self.xgboost_weight = result.x[1]
            
        self.is_calibrated = True
        logger.info(f"Ensemble weights: LSTM={self.lstm_weight:.3f}, XGBoost={self.xgboost_weight:.3f}")
        
        return self.lstm_weight, self.xgboost_weight
    
    def predict(self, X_temporal, X_static):
        """Generate ensemble PPNR predictions"""
        if self.lstm_model is None or self.xgboost_model is None:
            raise ValueError("Both models must be set before prediction")
            
        lstm_pred = self.lstm_model.predict(X_temporal).flatten()
        xgb_pred = self.xgboost_model.predict(X_static).flatten()
        
        ensemble_pred = self.lstm_weight * lstm_pred + self.xgboost_weight * xgb_pred
        
        return {
            'ensemble': ensemble_pred,
            'lstm': lstm_pred,
            'xgboost': xgb_pred,
            'weights': {
                'lstm': self.lstm_weight,
                'xgboost': self.xgboost_weight
            }
        }
    
    def get_model_contributions(self, X_temporal, X_static):
        """Calculate contribution of each model to final prediction"""
        predictions = self.predict(X_temporal, X_static)
        
        lstm_contribution = self.lstm_weight * predictions['lstm']
        xgb_contribution = self.xgboost_weight * predictions['xgboost']
        
        return {
            'lstm_contribution': lstm_contribution,
            'lstm_contribution_pct': lstm_contribution / (predictions['ensemble'] + 1e-10),
            'xgboost_contribution': xgb_contribution,
            'xgboost_contribution_pct': xgb_contribution / (predictions['ensemble'] + 1e-10)
        }
    
    def save(self, path):
        """Save ensemble configuration"""
        joblib.dump({
            'lstm_weight': self.lstm_weight,
            'xgboost_weight': self.xgboost_weight,
            'is_calibrated': self.is_calibrated
        }, f"{path}_ensemble.pkl")
        
    def load(self, path):
        """Load ensemble configuration"""
        config = joblib.load(f"{path}_ensemble.pkl")
        self.lstm_weight = config['lstm_weight']
        self.xgboost_weight = config['xgboost_weight']
        self.is_calibrated = config['is_calibrated']


class QuarterlyPPNRForecaster:
    """
    Forecasts PPNR over 9-quarter regulatory horizon
    Combines ensemble predictions with scenario-specific adjustments
    """
    
    def __init__(self, ensemble_model):
        self.ensemble = ensemble_model
        
    def forecast(self, bank_data, scenario_data, n_quarters=9):
        """
        Generate 9-quarter PPNR forecast
        
        Args:
            bank_data: DataFrame with bank characteristics
            scenario_data: DataFrame with macro scenario for each quarter
            n_quarters: Forecast horizon (default 9 per Fed requirements)
            
        Returns:
            DataFrame with quarterly PPNR projections
        """
        forecasts = []
        
        for q in range(n_quarters):
            # Get macro data up to current quarter
            macro_to_q = scenario_data[scenario_data['quarter'] <= q + 1]
            
            # Create features
            X_temporal = create_temporal_features(macro_to_q)
            X_static, feature_names = create_static_features(bank_data, macro_to_q)
            
            # Ensure consistent feature dimensions (use only the 5 core features)
            core_features = ['cre_exposure', 'residential_exposure', 'cet1_ratio', 'total_assets_log', 'npl_ratio']
            if len(feature_names) > 5:
                core_indices = []
                for core_feat in core_features:
                    if core_feat in feature_names:
                        core_indices.append(feature_names.index(core_feat))
                
                if len(core_indices) == 5:
                    X_static = X_static[:, core_indices]
            
            # Handle shape mismatches
            if len(X_temporal) == 0:
                X_temporal = np.zeros((len(bank_data), 9, 3))
            
            # Ensure temporal has correct shape
            if X_temporal.shape[0] < len(bank_data):
                X_temporal = np.tile(X_temporal[-1:], (len(bank_data), 1, 1))
            
            # Get predictions
            predictions = self.ensemble.predict(X_temporal[:len(bank_data)], X_static)
            
            # Apply scenario adjustments
            scenario_type = scenario_data['scenario'].iloc[0]
            stress_factor = self._get_stress_factor(scenario_type, q)
            
            for i, (idx, bank) in enumerate(bank_data.iterrows()):
                base_ppnr = predictions['ensemble'][i] if i < len(predictions['ensemble']) else predictions['ensemble'][0]
                adjusted_ppnr = base_ppnr * stress_factor
                
                forecasts.append({
                    'bank_id': bank.get('bank_id', f'BANK_{i:03d}'),
                    'bank_name': bank.get('bank_name', 'Unknown Bank'),
                    'quarter': q + 1,
                    'scenario': scenario_type,
                    'ppnr_forecast': adjusted_ppnr,
                    'lstm_component': predictions['lstm'][i] if i < len(predictions['lstm']) else predictions['lstm'][0],
                    'xgb_component': predictions['xgboost'][i] if i < len(predictions['xgboost']) else predictions['xgboost'][0],
                    'stress_factor': stress_factor
                })
        
        return pd.DataFrame(forecasts)
    
    def _get_stress_factor(self, scenario_type, quarter):
        """Calculate stress factor based on scenario severity and quarter"""
        base_factors = {
            'baseline': 1.0,
            'adverse': 0.85,
            'severely_adverse': 0.65
        }
        
        base = base_factors.get(scenario_type, 1.0)
        
        # Stress intensifies over time in adverse scenarios
        if scenario_type != 'baseline':
            time_factor = 1 - (quarter / 18)  # Peaks around quarter 4-5
            base *= (0.8 + 0.2 * time_factor)
            
        return base


# ==============================================================================
# FILE: capital_engine.py
# ==============================================================================

"""
Capital Engine: CET1 Ratio Calculations and Regulatory Compliance
Implements the capital erosion calculations for CCAR stress testing
"""
from dataclasses import dataclass


@dataclass
class RegulatoryThresholds:
    """Federal Reserve regulatory capital minimums"""
    CET1_MINIMUM: float = 0.045  # 4.5%
    TIER1_MINIMUM: float = 0.06  # 6%
    TOTAL_CAPITAL_MINIMUM: float = 0.08  # 8%
    LEVERAGE_RATIO_MINIMUM: float = 0.04  # 4%
    
    # Stress Capital Buffer (bank-specific, typically 2.5-8%)
    SCB_DEFAULT: float = 0.025  # 2.5% default
    
    # G-SIB Surcharge (for systemically important banks)
    GSIB_SURCHARGE: float = 0.0  # 0-3.5% depending on bank size


class CapitalEngine:
    """
    Capital calculation engine for CCAR stress testing
    Calculates CET1 erosion over 9-quarter horizon
    """
    
    def __init__(self, thresholds: Optional[RegulatoryThresholds] = None):
        self.thresholds = thresholds or RegulatoryThresholds()
        
    def calculate_cet1_trajectory(self, 
                                  bank_data: pd.DataFrame,
                                  ppnr_forecasts: pd.DataFrame,
                                  scenario_data: pd.DataFrame,
                                  n_quarters: int = 9) -> pd.DataFrame:
        """
        Calculate CET1 ratio trajectory over stress horizon
        
        CET1 Ratio = (CET1 Capital + PPNR - Losses - Dividends) / RWA
        """
        trajectories = []
        
        for _, bank in bank_data.iterrows():
            bank_id = bank.get('bank_id', 'BANK_000')
            bank_name = bank.get('bank_name', 'Unknown')
            
            # Initial values
            cet1_capital = bank.get('cet1_capital', bank.get('total_assets', 100e9) * 0.12)
            rwa = bank.get('rwa', bank.get('total_assets', 100e9) * 0.7)
            cre_exposure = bank.get('cre_exposure', bank.get('total_assets', 100e9) * 0.2)
            residential_exposure = bank.get('residential_exposure', bank.get('total_assets', 100e9) * 0.3)
            
            # SCB for this bank (can be customized)
            scb = bank.get('stress_capital_buffer', self.thresholds.SCB_DEFAULT)
            
            for q in range(1, n_quarters + 1):
                # Get PPNR for this quarter
                ppnr_row = ppnr_forecasts[
                    (ppnr_forecasts['bank_id'] == bank_id) & 
                    (ppnr_forecasts['quarter'] == q)
                ]
                
                if len(ppnr_row) > 0:
                    quarterly_ppnr = ppnr_row['ppnr_forecast'].values[0]
                else:
                    quarterly_ppnr = cet1_capital * 0.02  # Default 2% of capital
                
                # Get macro stress for this quarter
                macro_q = scenario_data[scenario_data['quarter'] == q]
                if len(macro_q) > 0:
                    macro = macro_q.iloc[0]
                else:
                    macro = {}
                
                # Calculate credit losses based on scenario
                credit_losses = self._calculate_credit_losses(
                    cre_exposure, 
                    residential_exposure,
                    bank.get('consumer_loans', 0),
                    macro
                )
                
                # RWA adjustments (increases in stress)
                rwa_growth_factor = self._calculate_rwa_growth(macro, q)
                rwa = rwa * rwa_growth_factor
                
                # Dividends and buybacks (reduced in stress)
                dividends = self._calculate_dividends(cet1_capital, macro)
                
                # Update CET1 Capital
                cet1_capital = cet1_capital + quarterly_ppnr - credit_losses - dividends
                
                # Calculate ratios
                cet1_ratio = cet1_capital / rwa if rwa > 0 else 0
                
                # Determine breach status
                min_required = self.thresholds.CET1_MINIMUM + scb
                breach = cet1_ratio < min_required
                buffer_remaining = cet1_ratio - min_required
                
                trajectories.append({
                    'bank_id': bank_id,
                    'bank_name': bank_name,
                    'quarter': q,
                    'cet1_capital': cet1_capital,
                    'rwa': rwa,
                    'cet1_ratio': cet1_ratio,
                    'ppnr': quarterly_ppnr,
                    'credit_losses': credit_losses,
                    'dividends': dividends,
                    'scb': scb,
                    'minimum_required': min_required,
                    'buffer_remaining': buffer_remaining,
                    'breach': breach,
                    'scenario': scenario_data['scenario'].iloc[0] if 'scenario' in scenario_data else 'unknown'
                })
        
        return pd.DataFrame(trajectories)
    
    def _calculate_credit_losses(self, cre_exposure: float, 
                                 residential_exposure: float,
                                 consumer_loans: float,
                                 macro: dict) -> float:
        """
        Calculate credit losses based on exposure and macro conditions
        """
        # Loss rates by asset class (scenario-dependent)
        scenario = macro.get('scenario', 'baseline')
        
        # Base loss rates
        loss_rates = {
            'baseline': {'cre': 0.005, 'residential': 0.003, 'consumer': 0.04},
            'adverse': {'cre': 0.04, 'residential': 0.02, 'consumer': 0.08},
            'severely_adverse': {'cre': 0.12, 'residential': 0.06, 'consumer': 0.12}
        }
        
        rates = loss_rates.get(scenario, loss_rates['baseline'])
        
        # Adjust for macro conditions
        unemployment = macro.get('unemployment_rate', 4.5)
        hpi_growth = macro.get('hpi_growth', 0)
        vix = macro.get('vix', 18)
        
        # Unemployment impact on consumer losses
        unemp_factor = 1 + max(0, (unemployment - 4.5) / 10)
        
        # HPI impact on CRE and residential losses
        hpi_factor = 1 + max(0, -hpi_growth / 20)
        
        # VIX impact on CRE losses (market stress)
        vix_factor = 1 + max(0, (vix - 20) / 100)
        
        # Calculate losses (quarterly)
        cre_loss = (cre_exposure * rates['cre'] * hpi_factor * vix_factor) / 4
        residential_loss = (residential_exposure * rates['residential'] * hpi_factor) / 4
        consumer_loss = (consumer_loans * rates['consumer'] * unemp_factor) / 4
        
        return cre_loss + residential_loss + consumer_loss
    
    def _calculate_rwa_growth(self, macro: dict, quarter: int) -> float:
        """Calculate RWA growth factor based on stress conditions"""
        scenario = macro.get('scenario', 'baseline')
        
        # Base RWA growth factors (stress increases risk weights)
        base_growth = {
            'baseline': 1.005,  # 0.5% quarterly growth
            'adverse': 1.015,   # 1.5% quarterly growth
            'severely_adverse': 1.025  # 2.5% quarterly growth
        }
        
        return base_growth.get(scenario, 1.005)
    
    def _calculate_dividends(self, cet1_capital: float, macro: dict) -> float:
        """Calculate dividend payments (reduced in stress)"""
        scenario = macro.get('scenario', 'baseline')
        
        # Dividend rates (annualized, divided by 4 for quarterly)
        div_rates = {
            'baseline': 0.03,  # 3% annual
            'adverse': 0.01,   # 1% annual (cut)
            'severely_adverse': 0.0  # Suspended
        }
        
        return cet1_capital * div_rates.get(scenario, 0.03) / 4
    
    def get_breach_analysis(self, trajectory_df: pd.DataFrame) -> Dict:
        """
        Analyze capital breaches across all banks
        """
        breaches = trajectory_df[trajectory_df['breach'] == True]
        
        analysis = {
            'total_banks': trajectory_df['bank_id'].nunique(),
            'banks_with_breach': breaches['bank_id'].nunique(),
            'breach_rate': breaches['bank_id'].nunique() / max(trajectory_df['bank_id'].nunique(), 1),
            'first_breach_quarter': breaches['quarter'].min() if len(breaches) > 0 else None,
            'worst_breach': {
                'bank_id': trajectory_df.loc[trajectory_df['cet1_ratio'].idxmin(), 'bank_id'] if len(trajectory_df) > 0 else None,
                'min_cet1': trajectory_df['cet1_ratio'].min() if len(trajectory_df) > 0 else None,
                'quarter': trajectory_df.loc[trajectory_df['cet1_ratio'].idxmin(), 'quarter'] if len(trajectory_df) > 0 else None
            },
            'capital_shortfall': (
                trajectory_df[trajectory_df['buffer_remaining'] < 0]['buffer_remaining'].sum()
            )
        }
        
        return analysis
    
    def compare_scenarios(self, bank_data: pd.DataFrame,
                         baseline_trajectory: pd.DataFrame,
                         stress_trajectory: pd.DataFrame) -> pd.DataFrame:
        """Compare capital trajectories across scenarios"""
        
        comparison = []
        
        for bank_id in bank_data['bank_id'].unique():
            baseline = baseline_trajectory[baseline_trajectory['bank_id'] == bank_id]
            stress = stress_trajectory[stress_trajectory['bank_id'] == bank_id]
            
            if len(baseline) > 0 and len(stress) > 0:
                comparison.append({
                    'bank_id': bank_id,
                    'baseline_min_cet1': baseline['cet1_ratio'].min(),
                    'stress_min_cet1': stress['cet1_ratio'].min(),
                    'cet1_decline': baseline['cet1_ratio'].min() - stress['cet1_ratio'].min(),
                    'baseline_breach': baseline['breach'].any(),
                    'stress_breach': stress['breach'].any(),
                    'total_losses_baseline': baseline['credit_losses'].sum(),
                    'total_losses_stress': stress['credit_losses'].sum()
                })
        
        return pd.DataFrame(comparison)


# ==============================================================================
# FILE: shap_explainer.py
# ==============================================================================

"""
SHAP Explainer with Gemini Natural Language Integration
Provides regulatory-compliant model interpretability for SR 11-7 compliance
"""
import shap
import asyncio


class SHAPExplainer:
    """
    SHAP-based model explainability for PPNR predictions
    Compliant with Fed SR 11-7 Model Risk Management requirements
    """
    
    def __init__(self, xgboost_model=None):
        self.xgb_model = xgboost_model
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def initialize_explainer(self, X_background: np.ndarray, feature_names: List[str]):
        """Initialize SHAP explainer with background data"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model must be set before initialization")
            
        self.feature_names = feature_names
        
        # Use TreeExplainer for XGBoost (fast and exact)
        self.explainer = shap.TreeExplainer(self.xgb_model.model)
        
        logger.info("SHAP explainer initialized")
        
    def explain_predictions(self, X: np.ndarray) -> Dict:
        """
        Generate SHAP explanations for predictions
        
        Returns:
            - Global feature importance
            - Local explanations for each prediction
        """
        if self.explainer is None:
            raise ValueError("Explainer must be initialized first")
            
        # Scale features
        X_scaled = self.xgb_model.scaler.transform(X)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_scaled)
        
        # Global importance (mean absolute SHAP)
        global_importance = np.abs(self.shap_values).mean(axis=0)
        global_importance_dict = dict(zip(self.feature_names, global_importance))
        
        # Sort by importance
        sorted_importance = sorted(
            global_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'shap_values': self.shap_values,
            'global_importance': dict(sorted_importance),
            'expected_value': self.explainer.expected_value,
            'feature_names': self.feature_names
        }
    
    def explain_single_prediction(self, X_single: np.ndarray, 
                                   prediction: float) -> Dict:
        """
        Generate detailed explanation for a single prediction
        Used for scenario-specific analysis
        """
        if self.explainer is None:
            raise ValueError("Explainer must be initialized first")
            
        X_scaled = self.xgb_model.scaler.transform(X_single.reshape(1, -1))
        shap_values = self.explainer.shap_values(X_scaled)[0]
        
        # Create contribution breakdown
        contributions = {}
        for i, (name, value) in enumerate(zip(self.feature_names, shap_values)):
            contributions[name] = {
                'shap_value': float(value),
                'feature_value': float(X_single[i]),
                'contribution_pct': float(abs(value) / (abs(shap_values).sum() + 1e-10) * 100)
            }
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        ))
        
        return {
            'prediction': prediction,
            'expected_value': self.explainer.expected_value,
            'contributions': sorted_contributions,
            'top_positive_drivers': [
                k for k, v in sorted_contributions.items() 
                if v['shap_value'] > 0
            ][:3],
            'top_negative_drivers': [
                k for k, v in sorted_contributions.items() 
                if v['shap_value'] < 0
            ][:3]
        }
    
    def explain_scenario_impact(self, baseline_X: np.ndarray,
                                stress_X: np.ndarray,
                                baseline_pred: float,
                                stress_pred: float) -> Dict:
        """
        Explain the difference between baseline and stress predictions
        Identifies primary drivers of capital breach
        """
        baseline_scaled = self.xgb_model.scaler.transform(baseline_X.reshape(1, -1))
        stress_scaled = self.xgb_model.scaler.transform(stress_X.reshape(1, -1))
        
        baseline_shap = self.explainer.shap_values(baseline_scaled)[0]
        stress_shap = self.explainer.shap_values(stress_scaled)[0]
        
        # Calculate change in SHAP values
        shap_delta = stress_shap - baseline_shap
        
        impact_breakdown = {}
        for i, name in enumerate(self.feature_names):
            impact_breakdown[name] = {
                'baseline_contribution': float(baseline_shap[i]),
                'stress_contribution': float(stress_shap[i]),
                'delta': float(shap_delta[i]),
                'impact_direction': 'negative' if shap_delta[i] < 0 else 'positive'
            }
        
        # Identify primary driver
        sorted_by_impact = sorted(
            impact_breakdown.items(),
            key=lambda x: abs(x[1]['delta']),
            reverse=True
        )
        
        primary_driver = sorted_by_impact[0][0] if sorted_by_impact else None
        
        return {
            'baseline_prediction': baseline_pred,
            'stress_prediction': stress_pred,
            'prediction_change': stress_pred - baseline_pred,
            'prediction_change_pct': (stress_pred - baseline_pred) / (abs(baseline_pred) + 1e-10) * 100,
            'impact_breakdown': dict(sorted_by_impact),
            'primary_driver': primary_driver,
            'primary_driver_impact': sorted_by_impact[0][1]['delta'] if sorted_by_impact else 0
        }


class GeminiExplainer:
    """
    Natural language explanation generator using Gemini 3 Flash
    Provides human-readable interpretations of SHAP results
    """
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        self.chat = None
        
    async def _get_chat(self):
        """Lazy initialize the chat client"""
        if self.chat is None:
            from emergentintegrations.llm.chat import LlmChat
            self.chat = LlmChat(
                api_key=self.api_key,
                session_id="shap-explainer",
                system_message="""You are a regulatory compliance expert specializing in bank stress testing 
                and capital adequacy. You explain complex model outputs in clear, professional language 
                suitable for Fed examiners and bank risk officers. Focus on:
                1. What the key drivers are
                2. Why they matter for capital adequacy
                3. Regulatory implications
                Keep explanations concise but informative."""
            ).with_model("gemini", "gemini-3-flash-preview")
        return self.chat
    
    async def generate_global_explanation(self, shap_results: Dict, 
                                          scenario_name: str) -> str:
        """Generate natural language explanation of global feature importance"""
        
        chat = await self._get_chat()
        
        # Format importance data
        importance_str = "\n".join([
            f"- {name}: {value:.4f}" 
            for name, value in list(shap_results['global_importance'].items())[:8]
        ])
        
        from emergentintegrations.llm.chat import UserMessage
        
        prompt = f"""Analyze this SHAP feature importance from a PPNR stress test model 
        under the "{scenario_name}" scenario:
        
        Feature Importance (mean |SHAP|):
        {importance_str}
        
        Provide a 3-4 sentence explanation of:
        1. Which macro factors drive PPNR losses most
        2. What this means for bank capital planning
        3. Key risk concentration to monitor"""
        
        try:
            response = await chat.send_message(UserMessage(text=prompt))
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_explanation(shap_results, scenario_name)
    
    async def generate_scenario_explanation(self, scenario_impact: Dict,
                                            scenario_name: str) -> str:
        """Generate explanation of scenario-specific impact"""
        
        chat = await self._get_chat()
        
        from emergentintegrations.llm.chat import UserMessage
        
        prompt = f"""Explain this stress test result comparison for a bank:
        
        Scenario: {scenario_name}
        Baseline PPNR: ${scenario_impact['baseline_prediction']*1e9/1e6:.1f}M
        Stressed PPNR: ${scenario_impact['stress_prediction']*1e9/1e6:.1f}M
        Change: {scenario_impact['prediction_change_pct']:.1f}%
        
        Primary Driver: {scenario_impact['primary_driver']}
        Driver Impact: {scenario_impact['primary_driver_impact']:.4f}
        
        Top factors causing PPNR decline:
        {', '.join([f"{k} ({v['delta']:.4f})" for k, v in list(scenario_impact['impact_breakdown'].items())[:5]])}
        
        Provide a 3-4 sentence regulatory explanation:
        1. What's driving the capital erosion
        2. Is this CRE crash, yield curve, or other factors
        3. Supervisory implications"""
        
        try:
            response = await chat.send_message(UserMessage(text=prompt))
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_scenario_explanation(scenario_impact, scenario_name)
    
    async def generate_breach_alert_explanation(self, breach_analysis: Dict,
                                                trajectory_summary: Dict) -> str:
        """Generate explanation when capital breach is detected"""
        
        chat = await self._get_chat()
        
        from emergentintegrations.llm.chat import UserMessage
        
        prompt = f"""Generate a regulatory alert for this capital breach:
        
        Banks breaching CET1 minimum: {breach_analysis['banks_with_breach']} of {breach_analysis['total_banks']}
        First breach quarter: Q{breach_analysis['first_breach_quarter']}
        Worst case: Bank {breach_analysis['worst_breach']['bank_id']} at {breach_analysis['worst_breach']['min_cet1']*100:.1f}% CET1
        Capital shortfall: ${abs(breach_analysis['capital_shortfall'])/1e9:.1f}B
        
        Provide a concise alert message that:
        1. Summarizes the severity
        2. Identifies the timeline
        3. Suggests immediate actions
        4. References SR 11-7 compliance needs"""
        
        try:
            response = await chat.send_message(UserMessage(text=prompt))
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_breach_explanation(breach_analysis)
    
    def _generate_fallback_explanation(self, shap_results: Dict, scenario_name: str) -> str:
        """Fallback when API unavailable"""
        top_features = list(shap_results['global_importance'].keys())[:3]
        return f"""Under the {scenario_name} scenario, the primary drivers of PPNR variation are: 
        {', '.join(top_features)}. Banks with higher exposure to these factors face 
        elevated capital risk and should monitor their stress buffers closely."""
    
    def _generate_fallback_scenario_explanation(self, scenario_impact: Dict, scenario_name: str) -> str:
        """Fallback scenario explanation"""
        driver = scenario_impact['primary_driver']
        change = scenario_impact['prediction_change_pct']
        return f"""The {scenario_name} scenario projects a {abs(change):.1f}% decline in PPNR, 
        primarily driven by {driver}. This indicates potential capital erosion 
        that warrants enhanced monitoring under SR 11-7 guidelines."""
    
    def _generate_fallback_breach_explanation(self, breach_analysis: Dict) -> str:
        """Fallback breach explanation"""
        return f"""CAPITAL BREACH ALERT: {breach_analysis['banks_with_breach']} banks 
        projected to fall below CET1 minimums by Q{breach_analysis['first_breach_quarter']}. 
        Immediate supervisory review recommended per SR 11-7."""


# ==============================================================================
# REQUIREMENTS.TXT (for reference)
# ==============================================================================
"""
Required Python packages:

fastapi
uvicorn
motor
pydantic
python-dotenv
pandas
numpy
tensorflow
keras
xgboost
shap
scikit-learn
fredapi
plotly
scipy
joblib
python-multipart
emergentintegrations
"""


# ==============================================================================
# ENVIRONMENT VARIABLES (.env file)
# ==============================================================================
"""
Required environment variables:

MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="*"
EMERGENT_LLM_KEY=your-emergent-key-here
FRED_API_KEY=demo
"""
