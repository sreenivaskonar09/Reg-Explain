"""
XGBoost Model for Static Risk Factor Processing
Captures non-linear relationships between bank features and economic shocks
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)


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
