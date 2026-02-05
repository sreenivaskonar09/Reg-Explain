"""
Ensemble Model: Weighted Average of LSTM and XGBoost
Optimized to minimize RMSE against historical stress test results
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import joblib
import logging

logger = logging.getLogger(__name__)


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
    
        return {                                    # ← FUNCTION RETURNS HERE
            'ensemble': ensemble_pred,
            'lstm': lstm_pred,
            'xgboost': xgb_pred,
            'weights': {
                'lstm': self.lstm_weight,
                'xgboost': self.xgboost_weight
            }}
    # ↓ THIS CODE NEVER RUNS (unreachable after return)
        print(f"DEBUG [ensemble.py]: X_static shape before XGBoost: {X_static.shape}")
        if hasattr(self.xgboost_model, 'scaler') and self.xgboost_model.scaler is not None:
            print(f"DEBUG [ensemble.py]: XGBoost expects {self.xgboost_model.scaler.n_features_in_} features")

        xgb_pred = self.xgboost_model.predict(X_static).flatten()  # ← DUPLICATE LINE
        
    
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
        from .lstm_model import create_temporal_features
        from .xgboost_model import create_static_features
        
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
                # Find indices of core features
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
