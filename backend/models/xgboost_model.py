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



def create_static_features(bank_df, macro_df=None):
    """
    Create static feature vectors for XGBoost with RIGID SCHEMA.
    Ensures the feature count never varies between Train and Test.
    """
    # 1. Define the exact order of features the model expects
    # This list MUST match what you used during training
    feature_map = [
        ('cet1_ratio', 0.0),
        ('cre_exposure_pct', 0.0),
        ('residential_exposure_pct', 0.0),
        ('total_assets', 0.0), # Will be log-transformed
        ('npl_ratio', 0.0),
        ('tier1_leverage_ratio', 0.0)
    ]
    
    # Macro features
    macro_map = [
        ('unemployment_rate', 'max'),
        ('vix', 'max'),
        ('hpi_growth', 'min'),
        ('yield_curve_spread', 'mean')
    ]
    
    features = []
    feature_names = []
    
    # 2. Build Bank Features (Force 0 if missing)
    for col, default_val in feature_map:
        if col == 'total_assets':
            # Handle log transform specifically
            val = np.log(bank_df.get(col, 1000000).values) # Default to avoid log(0)
            name = 'total_assets_log'
        else:
            # .get() ensures we never crash or skip a column
            val = bank_df.get(col, default_val).values
            name = col
            
        features.append(val)
        feature_names.append(name)

    # 3. Build Macro Features (Force 0 if missing)
    # Even if macro_df is None, we append 0s to maintain shape
    if macro_df is not None:
        for col, agg_func in macro_map:
            if col in macro_df.columns:
                if agg_func == 'max': val = macro_df[col].max()
                elif agg_func == 'min': val = macro_df[col].min()
                else: val = macro_df[col].mean()
            else:
                val = 0.0
            
            features.append(np.full(len(bank_df), val))
            feature_names.append(f"{col}_{agg_func if agg_func != 'mean' else 'avg'}")
    else:
        # If no macro data, fill with zeros to match training shape
        for col, agg_func in macro_map:
            features.append(np.full(len(bank_df), 0.0))
            feature_names.append(f"{col}_{agg_func if agg_func != 'mean' else 'avg'}")

    return np.column_stack(features), feature_names
