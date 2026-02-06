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
            'n_estimators': 100,
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
    
        logger.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features...")
    
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
    
        # Train model with progress indicator
        self.model.fit(X_scaled, y, verbose=1)
        self.is_fitted = True
    
        # SIMPLIFIED: Just do 3-fold CV instead of 5
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
        logger.info(f"XGBoost CV RMSE: {np.sqrt(-cv_scores.mean()):.6f}")
    
        return self

    def preprocess(self, X):
        """
        Expose the internal scaler for SHAP explanations.
        Standardizes features using the fitted scaler.
        """
        if not hasattr(self, 'scaler') or self.scaler is None:
            # If no scaler exists, return data as-is (fallback)
            return X
            
        # Transform the data using the already fitted scaler
        return self.scaler.transform(X)
    
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



def create_static_features(bank_df, macro_df=None, include_macro=True, verbose=False):
    """
    Create static feature vectors for XGBoost with RIGID SCHEMA.
    
    Args:
        bank_df: DataFrame with bank-level features
        macro_df: DataFrame or dict with macro scenario data
        include_macro: If False, skip macro features entirely
        verbose: If True, print debug information
    """
    feature_map = [
        ('cet1_ratio', 0.0),
        ('cre_exposure_pct', 0.0),
        ('residential_exposure_pct', 0.0),
        ('total_assets', 0.0),
        ('npl_ratio', 0.0),
        ('tier1_leverage_ratio', 0.0)
    ]
    
    macro_map = [
        ('unemployment_rate', 'max'),
        ('vix', 'max'),
        ('hpi_growth', 'min'),
        ('yield_curve_spread', 'mean')
    ]
    
    n_samples = len(bank_df)
    features = []
    feature_names = []
    
    # Build Bank Features
    for col, default_val in feature_map:
        if col == 'total_assets':
            if col in bank_df.columns:
                val = bank_df[col].values
                val = np.where(val > 0, np.log(val), np.log(1000000))
            else:
                val = np.full(n_samples, np.log(1000000))
            name = 'total_assets_log'
        else:
            if col in bank_df.columns:
                val = bank_df[col].fillna(default_val).values
            else:
                val = np.full(n_samples, default_val)
            name = col
        
        features.append(val)
        feature_names.append(name)
    
    # Build Macro Features - ONLY if include_macro=True
    if include_macro:
        if isinstance(macro_df, dict):
            macro_df = pd.DataFrame([macro_df])
        
        for col, agg_func in macro_map:
            if macro_df is not None and col in macro_df.columns:
                if agg_func == 'max':
                    val = macro_df[col].max()
                elif agg_func == 'min':
                    val = macro_df[col].min()
                else:
                    val = macro_df[col].mean()
                
                val = 0.0 if pd.isna(val) else val
            else:
                val = 0.0
            
            features.append(np.full(n_samples, val))
            feature_names.append(f"{col}_{agg_func}")
    
    X = np.column_stack(features)
    
    # Validate shape
    expected_features = len(feature_map) + (len(macro_map) if include_macro else 0)
    assert X.shape[1] == expected_features, \
        f"Feature count mismatch! Expected {expected_features}, got {X.shape[1]}"
    
    if verbose:
        logger.info(f"Created feature matrix: {X.shape}, features: {feature_names}")
    
    return X, feature_names
