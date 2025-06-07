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

class FinancialLSTMPredictor:
    def __init__(self, sequence_length=10, prediction_features=['open', 'high', 'low', 'close', 'tick_volume']):
        self.sequence_length = sequence_length
        self.prediction_features = prediction_features
        self.scalers = {}
        self.model = None
        self.coreml_model = None
        
    def load_and_preprocess_data(self, csv_data):
        """Load and preprocess the financial data"""
        # Parse the CSV data
        lines = csv_data.strip().split('\n')
        header = lines[0].split(',')
        
        data = []
        for line in lines[1:]:
            row = line.split(',')
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
        
        # Convert data types
        df['time'] = pd.to_datetime(df['time'])
        numeric_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert candle_type to numeric (0 for bearish, 1 for bullish)
        df['candle_type_numeric'] = (df['candle_type'] == 'bullish').astype(int)
        
        # Sort by time (oldest first for proper sequence)
        df = df.sort_values('time').reset_index(drop=True)
        
        # Add technical indicators
        df['price_change'] = df['close'] - df['open']
        df['volatility'] = (df['high'] - df['low']) / df['open']
        df['volume_price_trend'] = df['tick_volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['volume_price_trend'] = df['volume_price_trend'].fillna(0)
        
        return df
    
    def create_sequences(self, df):
        """Create sequences for LSTM training"""
        # Check if we have enough data
        if len(df) <= self.sequence_length:
            raise ValueError(f"Need more data! Got {len(df)} rows but require > {self.sequence_length}")
        
        # Select features for training
        feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range',
                          'candle_type_numeric', 'price_change', 'volatility', 'volume_price_trend']
        
        # Scale features
        scaled_data = {}
        for col in feature_columns:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(df[[col]]).flatten()
            self.scalers[col] = scaler
        
        # Create feature matrix
        feature_matrix = np.column_stack([scaled_data[col] for col in feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(feature_matrix)):
            X.append(feature_matrix[i-self.sequence_length:i])
            # Predict the main features for next candle
            target_indices = [feature_columns.index(feat) for feat in self.prediction_features]
            y.append(feature_matrix[i][target_indices])
        
        return np.array(X), np.array(y), feature_columns
    
    def build_model(self, input_shape, output_shape):
        """Build LSTM model with simplified input layer"""
        model = Sequential([
            # Simplified input handling - let first layer define input shape
            LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_1'),
            Dropout(0.2, name='dropout_1'),
            LSTM(32, return_sequences=True, name='lstm_2'),
            Dropout(0.2, name='dropout_2'),
            LSTM(16, return_sequences=False, name='lstm_3'),
            Dropout(0.2, name='dropout_3'),
            Dense(8, activation='relu', name='dense_1'),
            Dense(output_shape, activation='linear', name='predictions')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def train_model(self, df, validation_split=0.2, epochs=100, batch_size=32):
        """Train the LSTM model"""
        # Create sequences
        X, y, feature_columns = self.create_sequences(df)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model(
            input_shape=(X.shape[1], X.shape[2]),
            output_shape=y.shape[1]
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
            ]
        )
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"Training Loss: {train_loss[0]:.6f}, Training MAE: {train_loss[1]:.6f}")
        print(f"Validation Loss: {val_loss[0]:.6f}, Validation MAE: {val_loss[1]:.6f}")
        
        return history, feature_columns
    
    def predict_next_candle(self, df, last_n_candles=None):
        """Predict the next candle"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Use last sequence_length candles for prediction
        if last_n_candles is None:
            last_n_candles = self.sequence_length
        
        # Get the last sequence
        recent_data = df.tail(last_n_candles).copy()
        
        # Preprocess recent data same way as training
        feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range',
                          'candle_type_numeric', 'price_change', 'volatility', 'volume_price_trend']
        
        # Add missing technical indicators if needed
        if 'candle_type_numeric' not in recent_data.columns:
            recent_data['candle_type_numeric'] = (recent_data['candle_type'] == 'bullish').astype(int)
        if 'price_change' not in recent_data.columns:
            recent_data['price_change'] = recent_data['close'] - recent_data['open']
        if 'volatility' not in recent_data.columns:
            recent_data['volatility'] = (recent_data['high'] - recent_data['low']) / recent_data['open']
        if 'volume_price_trend' not in recent_data.columns:
            recent_data['volume_price_trend'] = recent_data['tick_volume'] * (recent_data['close'] - recent_data['close'].shift(1)) / recent_data['close'].shift(1)
            recent_data['volume_price_trend'] = recent_data['volume_price_trend'].fillna(0)
        
        # Scale the data using trained scalers
        scaled_sequence = []
        for col in feature_columns:
            if col in self.scalers:
                scaled_values = self.scalers[col].transform(recent_data[[col]]).flatten()
                scaled_sequence.append(scaled_values)
        
        # Create input sequence
        input_sequence = np.column_stack(scaled_sequence)
        input_sequence = input_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction_scaled = self.model.predict(input_sequence, verbose=0)[0]
        
        # Unscale predictions
        predictions = {}
        for i, feature in enumerate(self.prediction_features):
            if feature in self.scalers:
                pred_value = self.scalers[feature].inverse_transform([[prediction_scaled[i]]])[0][0]
                predictions[feature] = pred_value
        
        return predictions
    
    def convert_to_coreml(self, model_name="FinancialLSTM"):
        """Convert trained model to CoreML format with correct input name"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print("Converting to CoreML...")
        
        # Get input shape from model
        input_shape = self.model.input_shape
        coreml_input_shape = (1, input_shape[1], input_shape[2])
        
        # Dynamically get the input name from the model
        input_name = self.model.input.name.split(':')[0]  # e.g., 'lstm_1_input'
        
        try:
            print(f"Attempting conversion with input name: {input_name}")
            
            self.coreml_model = ct.convert(
                self.model,
                inputs=[ct.TensorType(shape=coreml_input_shape, name=input_name)],
                minimum_deployment_target=ct.target.iOS14,
                convert_to="neuralnetwork"
            )
            print("CoreML conversion successful!")
            
        except Exception as e:
            print(f"CoreML conversion failed: {e}")
            raise Exception("Could not convert model to CoreML format")
        
        # Set metadata
        if self.coreml_model is not None:
            print("Setting CoreML model metadata...")
            try:
                spec = self.coreml_model.get_spec()
                
                # Set general metadata
                spec.description.metadata.author = "Financial LSTM Predictor"
                spec.description.metadata.shortDescription = f"{model_name} - LSTM model for financial prediction"
                spec.description.metadata.versionString = "1.0"
                
                # Set input descriptions
                for input_spec in spec.description.input:
                    input_spec.shortDescription = f"Financial time series sequence ({self.sequence_length} timesteps, 10 features)"
                
                # Set output descriptions
                for output_spec in spec.description.output:
                    output_spec.shortDescription = f"Predicted values for: {', '.join(self.prediction_features)}"
                
                # Update the model with the new spec
                self.coreml_model = ct.models.MLModel(spec)
                
            except Exception as e:
                print(f"Warning: Could not set metadata: {e}")
        
        return self.coreml_model
    
    def save_coreml_model(self, filename="FinancialLSTM.mlmodel"):
        """Save CoreML model to file"""
        if self.coreml_model is None:
            self.convert_to_coreml()
        
        self.coreml_model.save(filename)
        print(f"CoreML model saved as {filename}")
        
        # Print model info
        try:
            spec = self.coreml_model.get_spec()
            
            print("\n=== CoreML Model Information ===")
            print(f"Model version: {spec.description.metadata.versionString}")
            print(f"Author: {spec.description.metadata.author}")
            print(f"Description: {spec.description.metadata.shortDescription}")
            
            print("\nInputs:")
            for input_spec in spec.description.input:
                print(f"  - Name: '{input_spec.name}'")
                print(f"    Description: {input_spec.shortDescription}")
                if hasattr(input_spec.type, 'multiArrayType'):
                    shape = input_spec.type.multiArrayType.shape
                    print(f"    Shape: {list(shape)}")
            
            print("\nOutputs:")
            for output_spec in spec.description.output:
                print(f"  - Name: '{output_spec.name}'")
                print(f"    Description: {output_spec.shortDescription}")
                if hasattr(output_spec.type, 'multiArrayType'):
                    shape = output_spec.type.multiArrayType.shape
                    print(f"    Shape: {list(shape)}")
                    
        except Exception as e:
            print(f"Could not print detailed model info: {e}")
    
    def save_scalers(self, filename="scalers.pkl"):
        """Save scalers for later use"""
        with open(filename, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"Scalers saved as {filename}")
    
    def load_scalers(self, filename="scalers.pkl"):
        """Load scalers"""
        with open(filename, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"Scalers loaded from {filename}")

# Example usage with improved error handling
def main():
    # Your sample data
    sample_data = """time,open,high,low,close,tick_volume,candle_type,range
2025-06-05 22:00:00,100942.41,105267.07,100406.11,104497.31,41323,bullish,4860.960000000006
2025-06-04 22:00:00,104477.31,105861.17,100602.51,100609.51,43946,bearish,5258.6600000000035
2025-06-03 22:00:00,105636.01,105948.87,104100.47,104619.11,28029,bearish,1848.3999999999942
2025-06-02 22:00:00,104915.61,106737.77,104507.67,105735.11,32119,bullish,2230.100000000006
2025-06-01 22:00:00,104882.41,105880.07,103601.37,104646.31,36529,bearish,2278.7000000000116
2025-05-31 22:00:00,104755.01,105262.51,103685.88,104954.21,21004,bullish,1576.62999999999
2025-05-30 22:00:00,104612.41,104832.41,103014.61,104699.41,31992,bullish,1817.800000000003
2025-05-29 22:00:00,106189.61,106399.07,103634.7,104597.81,52559,bearish,2764.37000000001
2025-05-28 22:00:00,107181.71,108844.07,105564.17,106058.11,45835,bearish,3279.9000000000087
2025-05-27 22:00:00,109382.21,109615.51,106712.41,107159.11,38316,bearish,2903.0999999999913
2025-05-26 22:00:00,109341.41,110662.47,107434.17,109560.21,46600,bullish,3228.300000000003
2025-05-25 22:00:00,107646.41,110375.17,107102.97,109418.81,40421,bullish,3272.199999999997"""

    # Initialize predictor with shorter sequence for limited data
    predictor = FinancialLSTMPredictor(sequence_length=5)
    
    try:
        # Load and preprocess data
        df = predictor.load_and_preprocess_data(sample_data)
        print("Data loaded and preprocessed:")
        print(df.head())
        print(f"Data shape: {df.shape}")
        
        # Train model
        print("\nTraining model...")
        history, feature_columns = predictor.train_model(df, epochs=50, validation_split=0.1)
        
        # Make prediction for next candle
        print("\nPredicting next candle...")
        prediction = predictor.predict_next_candle(df)
        print("Predicted next candle values:")
        for feature, value in prediction.items():
            print(f"  {feature}: {value:.2f}")
        
        # Convert to CoreML
        print("\nConverting to CoreML...")
        coreml_model = predictor.convert_to_coreml("FinancialLSTM")
        
        # Save CoreML model
        predictor.save_coreml_model("FinancialLSTM.mlmodel")
        
        # Save scalers for use in iOS app
        predictor.save_scalers("scalers.pkl")
        
        print("\n✅ Process completed successfully!")
        print("Files created:")
        print("  - FinancialLSTM.mlmodel (CoreML model)")
        print("  - scalers.pkl (feature scalers)")
        print("\n🚀 CoreML model ready for iOS/macOS applications!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check your data and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Utility functions for iOS integration
def prepare_input_for_coreml(recent_candles_df, scalers, sequence_length=5):
    """
    Prepare input data for CoreML model prediction
    Use this function in your iOS app (Swift/Objective-C equivalent)
    
    Args:
        recent_candles_df: DataFrame with recent candle data
        scalers: Dictionary of fitted MinMaxScaler objects
        sequence_length: Number of timesteps for the sequence
    
    Returns:
        numpy array ready for CoreML prediction
    """
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'range',
                      'candle_type_numeric', 'price_change', 'volatility', 'volume_price_trend']
    
    # Add technical indicators
    df = recent_candles_df.copy()
    df['candle_type_numeric'] = (df['candle_type'] == 'bullish').astype(int)
    df['price_change'] = df['close'] - df['open']
    df['volatility'] = (df['high'] - df['low']) / df['open']
    df['volume_price_trend'] = df['tick_volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volume_price_trend'] = df['volume_price_trend'].fillna(0)
    
    # Scale data
    scaled_sequence = []
    for col in feature_columns:
        if col in scalers:
            scaled_values = scalers[col].transform(df[[col]]).flatten()
            scaled_sequence.append(scaled_values)
    
    # Create input array for CoreML
    input_array = np.column_stack(scaled_sequence)
    input_array = input_array[-sequence_length:].reshape(1, sequence_length, -1)
    
    return input_array.astype(np.float32)  # CoreML expects float32

def unscale_predictions(predictions, scalers, prediction_features=['open', 'high', 'low', 'close', 'tick_volume']):
    """
    Unscale predictions from CoreML model
    
    Args:
        predictions: Raw predictions from CoreML model
        scalers: Dictionary of fitted MinMaxScaler objects
        prediction_features: List of features being predicted
    
    Returns:
        Dictionary of unscaled predictions
    """
    unscaled = {}
    for i, feature in enumerate(prediction_features):
        if feature in scalers and i < len(predictions):
            unscaled_value = scalers[feature].inverse_transform([[predictions[i]]])[0][0]
            unscaled[feature] = unscaled_value
    
    return unscaled
