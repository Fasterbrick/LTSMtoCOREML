import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import coremltools as ct
import matplotlib.pyplot as plt
import tempfile
import os
import shutil
import pickle
import json
import warnings

class FinancialLSTMPredictor:
    def __init__(self, sequence_length=10, prediction_features=['open', 'high', 'low', 'close', 'tick_volume'], max_training_rows=500):
        self.sequence_length = sequence_length
        self.prediction_features = prediction_features
        self.max_training_rows = max_training_rows  # New parameter to limit training data
        self.scalers = {}
        self.model = None
        self.coreml_model = None
        self.feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range',
                               'candle_type_numeric', 'price_change', 'volatility', 'volume_price_trend']
        
    def load_data_from_db(self, db_path, table_name):
        """Load financial data from SQLite database"""
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            if df.empty:
                raise ValueError("Database query returned no data")
            return df
        except Exception as e:
            raise ConnectionError(f"Error connecting to database: {str(e)}")
    
    def limit_training_data(self, df):
        """Limit the data to the last N rows for training"""
        if self.max_training_rows is None:
            return df
            
        total_rows = len(df)
        
        if total_rows <= self.max_training_rows:
            print(f" Using all {total_rows} rows (requested {self.max_training_rows})")
            return df
        else:
            # Take the last N rows (most recent data)
            limited_df = df.tail(self.max_training_rows).reset_index(drop=True)
            print(f" Limited training data: {self.max_training_rows} rows (from {total_rows} total)")
            
            if 'time' in limited_df.columns:
                print(f"   • Date range: {limited_df['time'].min()} to {limited_df['time'].max()}")
            
            return limited_df
    
    def preprocess_data(self, df):
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        numeric_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Convert candle_type to numeric (0 for bearish, 1 for bullish)
        if 'candle_type' in df.columns:
            df['candle_type_numeric'] = (df['candle_type'] == 'bullish').astype(int)
        
        # Sort by time (oldest first for proper sequence)
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)
        
        # Limit training data AFTER sorting
        df = self.limit_training_data(df)
        
        # Add technical indicators
        if all(col in df.columns for col in ['open', 'close']):
            df['price_change'] = df['close'] - df['open']
        
        if all(col in df.columns for col in ['high', 'low', 'open']):
            df['volatility'] = (df['high'] - df['low']) / df['open']
        
        if all(col in df.columns for col in ['tick_volume', 'close']):
            df['volume_price_trend'] = df['tick_volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
            df['volume_price_trend'] = df['volume_price_trend'].fillna(0)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0  # Add missing columns with default value
                print(f"Warning: Added missing column {col} with default values")
        
        return df
    
    def create_sequences(self, df):
        """Create sequences for LSTM training - Fixed to eliminate warnings"""
        # Check if we have enough data
        if len(df) <= self.sequence_length:
            raise ValueError(f"Need more data! Got {len(df)} rows but require > {self.sequence_length}")
        
        # Predefined scaling parameters
        predefined_params = {
            "open": {"data_min": 10000.0, "data_max": 200000.0},
            "high": {"data_min": 10000.0, "data_max": 201000.0},
            "low": {"data_min": 9900.0, "data_max": 199000.0},
            "close": {"data_min": 10000.0, "data_max": 200000.0},
            "tick_volume": {"data_min": 0.0, "data_max": 100000.0},
            "range": {"data_min": 0.0, "data_max": 5000.0},
            "candle_type_numeric": {"data_min": 0.0, "data_max": 1.0},
            "price_change": {"data_min": -1000.0, "data_max": 1000.0},
            "volatility": {"data_min": 0.0, "data_max": 0.1},
            "volume_price_trend": {"data_min": -100000.0, "data_max": 100000.0}
        }
        
        # Create and fit scalers properly to avoid warnings
        scaled_data = {}
        
        for col in self.feature_columns:
            scaler = MinMaxScaler()
            
            # Extract column data as numpy array (no feature names)
            column_data = df[col].values.reshape(-1, 1)
            
            if col in predefined_params:
                # Set predefined parameters
                params = predefined_params[col]
                scaler.data_min_ = np.array([params["data_min"]])
                scaler.data_max_ = np.array([params["data_max"]])
                scaler.data_range_ = scaler.data_max_ - scaler.data_min_
                scaler.scale_ = 1.0 / scaler.data_range_
                scaler.min_ = -scaler.data_min_ * scaler.scale_
                scaler.feature_range = (0, 1)
                scaler.n_features_in_ = 1
                scaler.n_samples_seen_ = len(df)
                
                # Transform using numpy array (no warnings)
                scaled_data[col] = scaler.transform(column_data).flatten()
            else:
                # Standard fit_transform using numpy array
                scaled_data[col] = scaler.fit_transform(column_data).flatten()
            
            # Store the scaler
            self.scalers[col] = scaler
        
        # Create feature matrix
        feature_matrix = np.column_stack([scaled_data[col] for col in self.feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(feature_matrix)):
            X.append(feature_matrix[i-self.sequence_length:i])
            # Predict the main features for next candle
            target_indices = [self.feature_columns.index(feat) for feat in self.prediction_features]
            y.append(feature_matrix[i][target_indices])
        
        return np.array(X), np.array(y), self.feature_columns
    
    def build_model(self, input_shape, output_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_1'),
            Dropout(0.2, name='dropout_1'),
            LSTM(32, return_sequences=True, name='lstm_2'),
            Dropout(0.2, name='dropout_2'),
            LSTM(16, return_sequences=False, name='lstm_3'),
            Dropout(0.2, name='dropout_3'),
            Dense(8, activation='relu', name='dense_1'),
            Dense(output_shape, activation='linear', name='predictions')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df, validation_split=0.2, epochs=100, batch_size=32):
        X, y, feature_columns = self.create_sequences(df)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Effective training samples: {len(X)} (from {len(df)} rows)")
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        self.model = self.build_model(
            input_shape=(X.shape[1], X.shape[2]),
            output_shape=y.shape[1]
        )
        
        print("Model architecture:")
        self.model.summary()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=10,
                monitor='val_loss',
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        
        return history, feature_columns
    
    def predict_next_candle(self, df, last_n_candles=None):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if last_n_candles is None:
            last_n_candles = self.sequence_length
        recent_data = df.tail(last_n_candles).copy()
        recent_data = self._add_technical_indicators(recent_data)
        scaled_sequence = []
        for col in self.feature_columns:
            if col in self.scalers and col in recent_data.columns:
                column_data = recent_data[col].values.reshape(-1, 1)
                scaled_values = self.scalers[col].transform(column_data).flatten()
                scaled_sequence.append(scaled_values)
            else:
                scaled_sequence.append(np.zeros(len(recent_data)))
        input_sequence = np.column_stack(scaled_sequence)
        input_sequence = input_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        prediction_scaled = self.model.predict(input_sequence, verbose=0)[0]
        predictions = {}
        for i, feature in enumerate(self.prediction_features):
            if feature in self.scalers:
                pred_array = np.array([[prediction_scaled[i]]])
                pred_value = self.scalers[feature].inverse_transform(pred_array)[0][0]
                if feature == 'tick_volume':
                    pred_value = int(max(0, pred_value))  # Ensure positive integer
                
                predictions[feature] = pred_value
        
        return predictions
    
    def _add_technical_indicators(self, df):
        df = df.copy()
        if 'candle_type_numeric' not in df.columns and 'candle_type' in df.columns:
            df['candle_type_numeric'] = (df['candle_type'] == 'bullish').astype(int)
        elif 'candle_type_numeric' not in df.columns:
            df['candle_type_numeric'] = 0
        if 'price_change' not in df.columns and all(col in df.columns for col in ['close', 'open']):
            df['price_change'] = df['close'] - df['open']
        elif 'price_change' not in df.columns:
            df['price_change'] = 0
        if 'volatility' not in df.columns and all(col in df.columns for col in ['high', 'low', 'open']):
            df['volatility'] = (df['high'] - df['low']) / df['open']
        elif 'volatility' not in df.columns:
            df['volatility'] = 0
            
        if 'volume_price_trend' not in df.columns and all(col in df.columns for col in ['tick_volume', 'close']):
            df['volume_price_trend'] = df['tick_volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
            df['volume_price_trend'] = df['volume_price_trend'].fillna(0)
        elif 'volume_price_trend' not in df.columns:
            df['volume_price_trend'] = 0
            
        return df
    
    def convert_to_coreml(self, model_name="FinancialLSTM"):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        input_shape = self.model.input_shape
        coreml_input_shape = (1, input_shape[1], input_shape[2])
        input_name = self.model.input.name.split(':')[0]
        try:
            print(f"Converting with input name: {input_name}")
            print(f"Input shape: {coreml_input_shape}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.coreml_model = ct.convert(
                    self.model,
                    inputs=[ct.TensorType(shape=coreml_input_shape, name=input_name)],
                    minimum_deployment_target=ct.target.iOS14,
                    convert_to="neuralnetwork"
                )
            print("CoreML conversion successful!")
        except Exception as e:
            print(f" CoreML conversion failed: {e}")
            raise Exception(f"Could not convert model to CoreML format: {e}")
        self._set_coreml_metadata(model_name)
        
        return self.coreml_model
    
    def _set_coreml_metadata(self, model_name):
        if self.coreml_model is not None:
            print("Setting CoreML model metadata...")
            try:
                spec = self.coreml_model.get_spec()
                spec.description.metadata.author = "Financial LSTM Predictor"
                spec.description.metadata.shortDescription = f"{model_name} - LSTM model for financial prediction (trained on {self.max_training_rows} rows)"
                spec.description.metadata.versionString = "1.0"
                for input_spec in spec.description.input:
                    input_spec.shortDescription = f"Financial time series sequence ({self.sequence_length} timesteps, 10 features)"
                for output_spec in spec.description.output:
                    output_spec.shortDescription = f"Predicted values for: {', '.join(self.prediction_features)}"
                
                # Update the model with the new spec
                self.coreml_model = ct.models.MLModel(spec)
                
            except Exception as e:
                print(f"Warning: Could not set metadata: {e}")
    
    def save_coreml_model(self, filename="FinancialLSTM.mlmodel"):
        """Save CoreML model to file"""
        if self.coreml_model is None:
            self.convert_to_coreml()
        
        self.coreml_model.save(filename)
        print(f"CoreML model saved as {filename}")
        
        # Print detailed model information
        self._print_model_info()
    
    def _print_model_info(self):
        """Helper function to print model information"""
        try:
            spec = self.coreml_model.get_spec()
            print(f"\n INPUTS:")
            for input_spec in spec.description.input:
                print(f"  • Name: '{input_spec.name}'")
                print(f"    Description: {input_spec.shortDescription}")
                if hasattr(input_spec.type, 'multiArrayType'):
                    shape = list(input_spec.type.multiArrayType.shape)
                    print(f"    Shape: {shape}")
            print(f"\n OUTPUTS:")
            for output_spec in spec.description.output:
                print(f"  • Name: '{output_spec.name}'")
                print(f"    Description: {output_spec.shortDescription}")
                if hasattr(output_spec.type, 'multiArrayType'):
                    shape = list(output_spec.type.multiArrayType.shape)
                    print(f"    Shape: {shape}")
                    
        except Exception as e:
            print(f"Could not print detailed model info: {e}")
    def save_scaler_params_json(self, filename="scaler_params.json"):
        scaler_params = {}
        for feature_name, scaler in self.scalers.items():
            scaler_params[feature_name] = {
                "data_min": float(scaler.data_min_[0]) if hasattr(scaler, 'data_min_') else 0.0,
                "data_max": float(scaler.data_max_[0]) if hasattr(scaler, 'data_max_') else 1.0,
                "scale": float(scaler.scale_[0]) if hasattr(scaler, 'scale_') else 1.0,
                "min": float(scaler.min_[0]) if hasattr(scaler, 'min_') else 0.0
            }
        scaler_params["_training_info"] = {
            "max_training_rows": self.max_training_rows,
            "sequence_length": self.sequence_length,
            "prediction_features": self.prediction_features,
            "created_at": datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print(f"Scaler parameters saved as {filename}")
        return scaler_params
    def load_scalers(self, filename="scalers.pkl"):
        """Load scalers"""
        with open(filename, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"Scalers loaded from {filename}")
def main():
    predictor = FinancialLSTMPredictor(
        sequence_length=5,
        max_training_rows=100  # Limit to last 500 rows for faster training
    )
    
    try:
        # Database configuration
        db_path = "/Users/swift/Desktop/BTCUSDdaily.db"
        table_name = "BTCUSDdaily"
        
        # Load and preprocess data
        print(f"Loading data from: {db_path}")
        raw_df = predictor.load_data_from_db(db_path, table_name)
        print(f"Total rows in database: {len(raw_df)}")
        print("Preprocessing data...")
        df = predictor.preprocess_data(raw_df)  # This will apply the row limit
        print(f"Data processed successfully!")
        print(f"Final training rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        if 'time' in df.columns:
            print(f"   • Date range: {df['time'].min()} to {df['time'].max()}")
        if len(df) < predictor.sequence_length + 1:
            raise ValueError(f"Insufficient data: {len(df)} rows. Need at least {predictor.sequence_length + 1} rows")
        history, feature_columns = predictor.train_model(
            df,
            epochs=50,
            validation_split=0.1,
            batch_size=32
        )
        print(f"\n🔮 Generating prediction for next candle...")
        prediction = predictor.predict_next_candle(df)
        print("📈 Predicted next candle values:")
        for feature, value in prediction.items():
            if feature == 'tick_volume':
                print(f"   • {feature.capitalize().replace('_', ' ')}: {value:,} ticks")
            else:
                print(f"   • {feature.capitalize()}: ${value:,.2f}")
        # Convert to CoreML
        print(f"\n Converting to CoreML...")
        coreml_model = predictor.convert_to_coreml("FinancialLSTM")
        predictor.save_coreml_model("FinancialLSTM.mlmodel")
        scaler_params = predictor.save_scaler_params_json("scaler_params.json")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\n Troubleshooting tips:")
        print("   • Check database path and table name")
        print("   • Ensure sufficient data (>5 rows)")
        print("   • Verify data format and columns")
        print("   • Try adjusting max_training_rows parameter")
        import traceback
        traceback.print_exc()

# Example usage with different row limits
def train_with_custom_limit(max_rows=500, sequence_length=5):
    """Train model with custom row limit"""
    predictor = FinancialLSTMPredictor(
        sequence_length=sequence_length,
        max_training_rows=max_rows
    )
    
    # Your training code here...
    return predictor

if __name__ == "__main__":
    main()

# Additional utility functions for iOS integration
def prepare_input_for_ios(recent_candles, scaler_params, sequence_length=5):
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range',
                      'candle_type_numeric', 'price_change', 'volatility', 'volume_price_trend']
    processed_candles = []
    for i, candle in enumerate(recent_candles[-sequence_length:]):
        processed_candle = []
        # Calculate technical indicators
        prev_candle = recent_candles[i-1] if i > 0 else candle
        indicators = {
            'candle_type_numeric': 1.0 if candle.get('candle_type') == 'bullish' else 0.0,
            'price_change': candle.get('close', 0) - candle.get('open', 0),
            'volatility': (candle.get('high', 0) - candle.get('low', 0)) / max(candle.get('open', 1), 1),
            'volume_price_trend': candle.get('tick_volume', 0) * (candle.get('close', 0) - prev_candle.get('close', 0)) / max(prev_candle.get('close', 1), 1)
        }
        
        # Scale each feature
        for feature in feature_columns:
            if feature in indicators:
                value = indicators[feature]
            else:
                value = candle.get(feature, 0)
            scaled_value = scale_value_with_json(value, feature, scaler_params)
            processed_candle.append(scaled_value)
        processed_candles.append(processed_candle)
    return processed_candles

def scale_value_with_json(value, feature_name, scaler_params):
    if feature_name not in scaler_params:
        return value
    params = scaler_params[feature_name]
    data_min = params["data_min"]
    data_max = params["data_max"]
    if data_max - data_min == 0:
        return 0.0
    scaled = (value - data_min) / (data_max - data_min)
    return max(0.0, min(1.0, scaled))  # Clamp to [0,1]
def unscale_prediction_with_json(scaled_value, feature_name, scaler_params):
    if feature_name not in scaler_params:
        return scaled_value
    params = scaler_params[feature_name]
    data_min = params["data_min"]
    data_max = params["data_max"]
    unscaled = scaled_value * (data_max - data_min) + data_min
    if feature_name == 'tick_volume':
        return int(max(0, unscaled))  # Ensure positive integer
    return unscaled
