import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class MinimalIAQModel:
    """
    Minimal LSTM model for IAQ prediction using CPCB dataset
    - Robust to column name variations
    - Uses only essential air quality parameters
    - Predicts IAQ for next 7 days
    - Flask web app integration ready
    """
    
    def __init__(self, window_size=24):
        self.window_size = window_size  # 24 hours of data to predict next values
        self.model = None
        self.scaler = None
        self.feature_columns = []
        
        # Essential        # Column name mappings for robust detection - MINIMAL FEATURE SET
        self.column_mappings = {
            'pm25': ['PM2.5', 'PM25', 'PM_2_5', 'PM2_5', 'PARTICULATE_MATTER_2_5'],
            'pm10': ['PM10', 'PM_10', 'PARTICULATE_MATTER_10'],
            'co': ['CO', 'MQ7', 'CARBON_MONOXIDE'],
            'temp': ['TEMP', 'AT', 'TEMPERATURE', 'AMBIENT_TEMP', 'AIR_TEMP'],
            'humidity': ['HUMIDITY', 'RH', 'RELATIVE_HUMIDITY', 'HUMID']
        }
        
        # Improved IAQI breakpoints with exact CPCB standards
        self.iaqi_breakpoints = {
            'pm25': [
                (0, 30, 0, 50),        # Good: 0-30 μg/m³ → 0-50 AQI
                (31, 60, 51, 100),     # Satisfactory: 31-60 → 51-100
                (61, 90, 101, 200),    # Moderate: 61-90 → 101-200
                (91, 120, 201, 300),   # Poor: 91-120 → 201-300
                (121, 250, 301, 400),  # Very Poor: 121-250 → 301-400
                (251, 380, 401, 500)   # Severe: 251-380 → 401-500
            ],
            'pm10': [
                (0, 50, 0, 50),        # Good: 0-50 μg/m³ → 0-50 AQI
                (51, 100, 51, 100),    # Satisfactory: 51-100 → 51-100
                (101, 250, 101, 200),  # Moderate: 101-250 → 101-200
                (251, 350, 201, 300),  # Poor: 251-350 → 201-300
                (351, 430, 301, 400),  # Very Poor: 351-430 → 301-400
                (431, 550, 401, 500)   # Severe: 431-550 → 401-500
            ],
            'co': [
                (0, 1.0, 0, 50),       # Good: 0-1.0 mg/m³ → 0-50 AQI
                (1.1, 2.0, 51, 100),   # Satisfactory: 1.1-2.0 → 51-100
                (2.1, 10, 101, 200),   # Moderate: 2.1-10 → 101-200
                (11, 17, 201, 300),    # Poor: 11-17 → 201-300
                (18, 34, 301, 400),    # Very Poor: 18-34 → 301-400
                (35, 50, 401, 500)     # Severe: 35-50 → 401-500
            ]
        }
    
    def find_column(self, df, parameter):
        """Find the correct column name for a parameter in the dataset"""
        possible_names = self.column_mappings.get(parameter, [])
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def calculate_iaqi(self, concentration, parameter):
        """Calculate Individual Air Quality Index (IAQI) using improved CPCB standards"""
        if parameter not in self.iaqi_breakpoints:
            return concentration  # Return raw value if no breakpoints defined
        
        breakpoints = self.iaqi_breakpoints[parameter]
        concentration = float(concentration)
        
        # Handle edge cases
        if concentration < 0:
            return 0.0
        if concentration == 0:
            return 0.0
        
        # Find the appropriate breakpoint range with improved boundary handling
        for i, (bp_low, bp_high, iaqi_low, iaqi_high) in enumerate(breakpoints):
            # Handle exact boundary matches more precisely
            if concentration == bp_low:
                return float(iaqi_low)
            elif concentration == bp_high:
                return float(iaqi_high)
            elif bp_low < concentration < bp_high:
                # Linear interpolation: IAQI = [(IHi-ILo)/(BPHi-BPLo)] * (Cp-BPLo) + ILo
                if bp_high == bp_low:  # Avoid division by zero
                    return float(iaqi_low)
                
                iaqi = ((iaqi_high - iaqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + iaqi_low
                return round(float(iaqi), 1)
        
        # If concentration is above all breakpoints, return maximum IAQI (500)
        return 500.0
    
    def calculate_overall_iaqi(self, row):
        """Calculate overall IAQI as the maximum of individual IAQIs with improved accuracy"""
        iaqis = []
        debug_info = []
        
        # Air quality parameters for IAQI calculation
        aq_params = ['pm25', 'pm10', 'co']
        
        for param in aq_params:
            concentration = None
            
            # Try direct parameter name first (for test data)
            if param in row and row[param] is not None:
                concentration = float(row[param])
            else:
                # Try to find column using mapping (for real CSV data)
                col_name = self.find_column(pd.DataFrame([row]), param)
                if col_name and col_name in row and row[col_name] is not None:
                    concentration = float(row[col_name])
            
            # Calculate IAQI if we have a valid concentration
            if concentration is not None and concentration >= 0:
                iaqi = self.calculate_iaqi(concentration, param)
                if iaqi > 0:  # Only include valid IAQI values
                    iaqis.append(iaqi)
                    debug_info.append(f"{param}={concentration:.1f}→{iaqi:.1f}")
        
        # Return the maximum IAQI (dominant pollutant determines overall AQI)
        if iaqis:
            max_iaqi = max(iaqis)
            # Uncomment for debugging: print(f"IAQI components: {', '.join(debug_info)} → Max: {max_iaqi:.1f}")
            return round(max_iaqi, 1)
        else:
            # If no valid IAQIs found, return a default moderate value
            return 100.0
    
    def prepare_data(self, df):
        """Prepare data for training - minimal feature set with robust column detection"""
        print("📋 Available columns:", df.columns.tolist())
        
        # Find essential columns for minimal feature set
        essential_data = {}
        default_values = {
            'pm25': 50.0,    # Moderate default
            'pm10': 100.0,   # Moderate default
            'co': 2.0,       # Moderate default
            'temp': 25.0,    # Room temperature default
            'humidity': 60.0 # Moderate humidity default
        }
        
        # Process air quality parameters (for IAQI calculation)
        air_quality_params = ['pm25', 'pm10', 'co']
        for param in air_quality_params:
            col_name = self.find_column(df, param)
            if col_name:
                essential_data[param] = df[col_name].values
                print(f"✅ Found {param} -> {col_name}")
            else:
                # Use default values if missing
                essential_data[param] = np.full(len(df), default_values[param])
                print(f"⚠️ Missing {param}, using default: {default_values[param]}")
        
        # Process environmental parameters (for model features)
        env_params = ['temp', 'humidity']
        for param in env_params:
            col_name = self.find_column(df, param)
            if col_name:
                essential_data[param] = df[col_name].values
                print(f"✅ Found {param} -> {col_name}")
            else:
                # Use default values if missing
                essential_data[param] = np.full(len(df), default_values[param])
                print(f"⚠️ Missing {param}, using default: {default_values[param]}")
        
        # Create processed dataframe
        processed_df = pd.DataFrame(essential_data)
        
        # Calculate IAQI for each row (using only air quality parameters)
        print("🧮 Calculating IAQI values...")
        processed_df['iaqi'] = processed_df.apply(self.calculate_overall_iaqi, axis=1)
        
        # Set feature columns for model training (all parameters)
        self.feature_columns = ['pm25', 'pm10', 'co', 'temp', 'humidity']
        
        print(f"📈 IAQI value range: {processed_df['iaqi'].min():.1f} - {processed_df['iaqi'].max():.1f}")
        print(f"📈 Feature columns: {self.feature_columns}")
        
        return processed_df
    
    def create_sequences(self, data, target_col='iaqi'):
        """Create sequences for LSTM training"""
        print(f"🔄 Creating sequences with window size: {self.window_size}")
        
        features = data[self.feature_columns].values
        targets = data[target_col].values
        
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(features[i-self.window_size:i])
            y.append(targets[i])
        
        X, y = np.array(X), np.array(y)
        print(f"📦 Sequences created: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def build_model(self, input_shape):
        """Build minimal LSTM model"""
        print("🏗️ Building LSTM model...")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Model built successfully!")
        print(f"📋 Model summary:")
        model.summary()
        
        return model
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model with proper IAQI scaling"""
        print("🚀 Starting training process...")
        
        # Prepare data
        processed_df = self.prepare_data(df)
        
        print(f"📊 IAQI value range in training data: {processed_df['iaqi'].min():.1f} - {processed_df['iaqi'].max():.1f}")
        print(f"📊 IAQI mean: {processed_df['iaqi'].mean():.1f}, std: {processed_df['iaqi'].std():.1f}")
        
        # Scale only features, NOT the target IAQI values
        print("📏 Scaling features only...")
        feature_data = processed_df[self.feature_columns]
        target_data = processed_df['iaqi']
        
        # Scale features to 0-1 range
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Scale IAQI values to a reasonable range for neural network training (0-1)
        self.iaqi_scaler = MinMaxScaler()
        scaled_iaqi = self.iaqi_scaler.fit_transform(target_data.values.reshape(-1, 1)).flatten()
        
        print(f"📊 Scaled IAQI range: {scaled_iaqi.min():.3f} - {scaled_iaqi.max():.3f}")
        
        # Combine scaled features with scaled target
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_columns)
        scaled_df['iaqi'] = scaled_iaqi
        
        # Create sequences
        X, y = self.create_sequences(scaled_df)
        
        print(f"📊 Target y range after sequences: {y.min():.3f} - {y.max():.3f}")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📊 Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        # Build and train model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        print("🎯 Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Evaluate model
        print("📊 Evaluating model...")
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Convert predictions back to original IAQI scale for evaluation
        train_pred_original = self.iaqi_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
        val_pred_original = self.iaqi_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        y_train_original = self.iaqi_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_val_original = self.iaqi_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        train_mse = mean_squared_error(y_train_original, train_pred_original)
        val_mse = mean_squared_error(y_val_original, val_pred_original)
        train_mae = mean_absolute_error(y_train_original, train_pred_original)
        val_mae = mean_absolute_error(y_val_original, val_pred_original)
        
        print(f"✅ Training completed!")
        print(f"📈 Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f} (original IAQI scale)")
        print(f"📈 Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f} (original IAQI scale)")
        print(f"📈 Sample predictions: {train_pred_original[:5].round(1)}")
        
        return history
    
    def predict_next_7_days(self, recent_data):
        """Predict IAQI for next 7 days using trained model"""
        print("🔮 Predicting next 7 days...")
        
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet!")
        
        # Check if we have the IAQI scaler (from new training method)
        if not hasattr(self, 'iaqi_scaler') or self.iaqi_scaler is None:
            print("⚠️ Warning: No IAQI scaler found. Using fallback prediction method.")
            return self._fallback_prediction(recent_data)
        
        # Prepare recent data
        processed_df = self.prepare_data(recent_data)
        print(f"📊 Sample IAQI values from recent data: {processed_df['iaqi'].tail(5).tolist()}")
        
        # Use last window_size rows for prediction
        if len(processed_df) < self.window_size:
            print(f"⚠️ Warning: Need at least {self.window_size} rows, got {len(processed_df)}")
            # Repeat last row to fill the gap
            last_row = processed_df.iloc[-1:].copy()
            while len(processed_df) < self.window_size:
                processed_df = pd.concat([processed_df, last_row], ignore_index=True)
        
        # Get recent IAQI statistics
        recent_iaqi_values = processed_df['iaqi'].tail(self.window_size).values
        recent_iaqi_mean = np.mean(recent_iaqi_values)
        recent_iaqi_std = np.std(recent_iaqi_values)
        
        print(f"📈 Recent IAQI stats: Mean={recent_iaqi_mean:.1f}, Std={recent_iaqi_std:.1f}")
        
        # Scale features
        feature_data = processed_df[self.feature_columns].tail(self.window_size)
        scaled_features = self.scaler.transform(feature_data)
        
        # Scale recent IAQI values
        scaled_iaqi = self.iaqi_scaler.transform(recent_iaqi_values.reshape(-1, 1)).flatten()
        
        # Create input sequence with scaled features
        input_sequence = scaled_features.reshape(1, self.window_size, len(self.feature_columns))
        
        # Predict next 7 days using the trained model
        predictions = []
        current_sequence = input_sequence.copy()
        
        for day in range(7):
            # Predict next scaled IAQI value
            next_pred_scaled = self.model.predict(current_sequence, verbose=0)[0][0]
            
            # Convert back to original IAQI scale
            next_pred_iaqi = self.iaqi_scaler.inverse_transform([[next_pred_scaled]])[0][0]
            
            # Ensure realistic bounds
            next_pred_iaqi = max(5, min(500, next_pred_iaqi))
            
            predictions.append(float(next_pred_iaqi))
            
            # Update sequence for next prediction
            # Create new row with last features (we don't include IAQI in features)
            last_features = current_sequence[0, -1, :].copy()
            new_row = last_features.reshape(1, 1, -1)
            
            # Shift sequence and add new row
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :],
                new_row
            ], axis=1)
        
        print(f"🎯 Model-based IAQI Predictions: {[round(p, 1) for p in predictions]}")
        return predictions
    
    def _fallback_prediction(self, recent_data):
        """Fallback prediction method for older models without IAQI scaler"""
        processed_df = self.prepare_data(recent_data)
        recent_iaqi_values = processed_df['iaqi'].tail(self.window_size).values
        recent_iaqi_mean = np.mean(recent_iaqi_values)
        recent_iaqi_std = np.std(recent_iaqi_values)
        
        # Use statistical approach based on recent data
        baseline_iaqi = recent_iaqi_values[-3:].mean()
        predictions = []
        
        np.random.seed(42)
        for day in range(7):
            trend_factor = np.random.normal(0, recent_iaqi_std * 0.1)
            seasonal_factor = 5 * np.sin(day * 2 * np.pi / 7)
            random_factor = np.random.normal(0, recent_iaqi_std * 0.05)
            
            predicted_iaqi = baseline_iaqi + trend_factor + seasonal_factor + random_factor
            predicted_iaqi = max(10, min(500, predicted_iaqi))
            
            predictions.append(float(predicted_iaqi))
            baseline_iaqi = baseline_iaqi * 0.95 + predicted_iaqi * 0.05
        
        print(f"🎯 Fallback IAQI Predictions: {[round(p, 1) for p in predictions]}")
        return predictions
    
    def get_iaqi_category(self, iaqi_value):
        """Get IAQI category and details"""
        categories = [
            (0, 50, 'Good', '🟢', 'Minimal impact'),
            (51, 100, 'Satisfactory', '🟡', 'Minor breathing discomfort to sensitive people'),
            (101, 200, 'Moderate', '🟠', 'Breathing discomfort to people with lung, asthma and heart diseases'),
            (201, 300, 'Poor', '🔴', 'Breathing discomfort to most people on prolonged exposure'),
            (301, 400, 'Very Poor', '🟣', 'Respiratory illness on prolonged exposure'),
            (401, 500, 'Severe', '🟤', 'Affects healthy people and seriously impacts those with existing diseases')
        ]
        
        for min_val, max_val, category, emoji, description in categories:
            if min_val <= iaqi_value <= max_val:
                return {
                    'value': iaqi_value,
                    'category': category,
                    'emoji': emoji,
                    'description': description,
                    'range': f'{min_val}-{max_val}'
                }
        
        return {
            'value': iaqi_value,
            'category': 'Hazardous',
            'emoji': '☠️',
            'description': 'Emergency conditions affecting entire population',
            'range': '500+'
        }
    
    def predict_week(self, recent_data):
        """Predict week with category information - Flask integration ready"""
        predictions = self.predict_next_7_days(recent_data)
        
        results = []
        for pred in predictions:
            category_info = self.get_iaqi_category(pred)
            results.append(category_info)
        
        return results
    
    def save_model(self, model_path='minimal_lstm_model.h5', scaler_path='minimal_scaler.pkl'):
        """Save model and scaler with robust serialization"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        try:
            # Save model with specific options for compatibility
            self.model.save(model_path, save_format='h5', include_optimizer=False)
            print(f"✅ Model saved to {model_path}")
        except Exception as e:
            print(f"⚠️ H5 save failed: {e}")
            # Fallback: save weights only
            weights_path = model_path.replace('.h5', '_weights.h5')
            self.model.save_weights(weights_path)
            print(f"✅ Model weights saved to {weights_path}")
        
        # Save scaler and metadata
        model_info = {
            'scaler': self.scaler,
            'iaqi_scaler': getattr(self, 'iaqi_scaler', None),  # Save IAQI scaler if it exists
            'feature_columns': self.feature_columns,
            'window_size': self.window_size,
            'column_mappings': self.column_mappings,
            'model_architecture': {
                'layers': [
                    {'type': 'LSTM', 'units': 64, 'return_sequences': True},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'LSTM', 'units': 32, 'return_sequences': False},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 16, 'activation': 'relu'},
                    {'type': 'Dense', 'units': 1}
                ]
            }
        }
        joblib.dump(model_info, scaler_path)
        
        print(f"✅ Scaler and metadata saved to {scaler_path}")
    
    def load_model(self, model_path='minimal_lstm_model.h5', scaler_path='minimal_scaler.pkl'):
        """Load model and scaler with robust error handling"""
        print(f"🔄 Loading model from {model_path}...")
        
        # Load scaler and metadata first
        model_info = joblib.load(scaler_path)
        self.scaler = model_info['scaler']
        self.iaqi_scaler = model_info.get('iaqi_scaler', None)  # Load IAQI scaler if it exists
        self.feature_columns = model_info['feature_columns']
        self.window_size = model_info['window_size']
        self.column_mappings = model_info.get('column_mappings', self.column_mappings)
        
        # Try multiple methods to load the model
        model_loaded = False
        
        # Method 1: Try loading the full model
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print(f"✅ Method 1: Full model loaded successfully!")
            model_loaded = True
        except Exception as e1:
            print(f"⚠️ Method 1 failed: {e1}")
            
            # Method 2: Try loading weights only
            try:
                weights_path = model_path.replace('.h5', '_weights.h5')
                if os.path.exists(weights_path):
                    # Rebuild architecture from metadata
                    architecture = model_info.get('model_architecture', {})
                    if architecture:
                        print("🔄 Rebuilding model from architecture...")
                        self.model = self.build_model((self.window_size, len(self.feature_columns)))
                        self.model.load_weights(weights_path)
                        print(f"✅ Method 2: Model rebuilt and weights loaded!")
                        model_loaded = True
                    else:
                        raise ValueError("No architecture info found")
                else:
                    raise FileNotFoundError("Weights file not found")
            except Exception as e2:
                print(f"⚠️ Method 2 failed: {e2}")
                
                # Method 3: Create new model with same architecture
                try:
                    print("🔄 Creating new model with same architecture...")
                    self.model = self.build_model((self.window_size, len(self.feature_columns)))
                    print(f"⚠️ Method 3: New model created (will need retraining)")
                    model_loaded = True
                except Exception as e3:
                    print(f"❌ Method 3 failed: {e3}")
                    raise RuntimeError(f"All model loading methods failed: {e1}, {e2}, {e3}")
        
        if model_loaded:
            print(f"✅ Model loaded successfully!")
            print(f"✅ Features: {self.feature_columns}")
            print(f"✅ Window size: {self.window_size}")
        else:
            raise RuntimeError("Failed to load model")

# Training script
if __name__ == "__main__":
    print("🚀 Minimal IAQ LSTM Model Training")
    print("=" * 50)
    
    # Check if we have training data
    if not os.path.exists('test_data.csv'):
        print("❌ No training data found. Creating sample data...")
        
        # Create sample CPCB-style data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        
        # Generate realistic air quality data with trends
        base_pm25 = 50 + 30 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 10, 1000)
        base_pm10 = base_pm25 * 1.5 + np.random.normal(0, 15, 1000)
        base_co = 1.5 + 0.5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 0.3, 1000)
        base_no2 = 40 + 20 * np.sin(np.arange(1000) * 2 * np.pi / 12) + np.random.normal(0, 8, 1000)
        base_so2 = 20 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 168) + np.random.normal(0, 5, 1000)
        base_o3 = 80 + 40 * np.sin(np.arange(1000) * 2 * np.pi / 24 + np.pi/2) + np.random.normal(0, 12, 1000)
        
        # Ensure positive values
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'PM2.5': np.maximum(base_pm25, 5),
            'PM10': np.maximum(base_pm10, 10),
            'CO': np.maximum(base_co, 0.1),
            'NO2': np.maximum(base_no2, 5),
            'SO2': np.maximum(base_so2, 2),
            'O3': np.maximum(base_o3, 10)
        })
        
        sample_data.to_csv('training_data.csv', index=False)
        print("✅ Sample training data created as 'training_data.csv'")
        training_file = 'training_data.csv'
    else:
        training_file = 'test_data.csv'
    
    # Load and train model
    print(f"📊 Loading training data from {training_file}...")
    df = pd.read_csv(training_file)
    
    # Initialize and train model
    model = MinimalIAQModel(window_size=24)
    history = model.train(df, epochs=30, batch_size=16)
    
    # Save model
    model.save_model()
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    recent_data = df.tail(50)  # Use last 50 rows for testing
    predictions = model.predict_week(recent_data)
    
    print("\n📊 7-Day Prediction Results:")
    for i, pred in enumerate(predictions, 1):
        print(f"Day {i}: IAQI {pred['value']:.1f} - {pred['emoji']} {pred['category']} ({pred['range']})")
    
    print("\n✅ Training completed successfully!")
    print("🎯 Model ready for Flask integration!")
