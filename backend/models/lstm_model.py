"""
LSTM Model for Temporal Sequence Processing
Captures path-dependency of interest rate cycles and macroeconomic momentum
"""
import numpy as np
import pandas as pd  # ← MOVED HERE
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)

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
# ← REMOVED: import pandas as pd from here
