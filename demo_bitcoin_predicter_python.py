# btc_predictor.py
# Bitcoin Price Predictor - Simple Interface for Real Trading Decisions

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from datetime import datetime, timedelta
import time

class BTCPricePredictor:
    """
    Bitcoin Price Predictor for 1-hour ahead forecasting.
    
    Usage:
        predictor = BTCPricePredictor()
        prediction = predictor.predict_next_hour()
        print(f"Predicted BTC price in 1 hour: ${prediction:,.2f}")
    """
    
    def __init__(self, model_path='gru_btc_final.keras', scaler_path='btc_scaler.pkl'):
        """
        Initialize the predictor with the trained model and scaler.
        
        Args:
            model_path: Path to the saved Keras model
            scaler_path: Path to the saved MinMaxScaler
        """
        # Load the trained model and scaler
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Constants (must match training parameters)
        self.WINDOW_SIZE = 1440  # 24 hours of 1-minute data
        self.HORIZON = 60  # Predict 1 hour ahead
        self.N_FEATURES = 6  # Close, Volume_BTC_log, hour_sin, hour_cos, dow_sin, dow_cos
        
        # Try to get data from API; fallback to manual input
        self.api_available = self._check_api_availability()
        
        print("✅ BTC Price Predictor initialized!")
        print(f"   Model: {model_path}")
        print(f"   Window: {self.WINDOW_SIZE} minutes (24 hours)")
        print(f"   Horizon: {self.HORIZON} minutes (1 hour)")
        if self.api_available:
            print("   Data source: Real-time API")
        else:
            print("   Data source: Manual entry mode")
    
    def _check_api_availability(self):
        """Check if we can fetch real-time Bitcoin price data."""
        try:
            import requests
            # Test connection to a free Bitcoin price API
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _fetch_recent_prices(self):
        """
        Fetch the last 24 hours of 1-minute Bitcoin price and volume data.
        
        Returns:
            DataFrame with columns: timestamp, price, volume
        """
        try:
            import requests
            
            # Use CoinGecko API (free, no API key required)
            # Get historical data for the last 24 hours
            end_time = int(time.time())
            start_time = end_time - 24 * 3600  # 24 hours ago
            
            # CoinGecko API for historical data (limited but free)
            # Note: For minute-level data, we need to aggregate from multiple sources
            
            # Alternative: Use Binance public API (more reliable for crypto data)
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    'symbol': 'BTCUSDT',
                    'interval': '1m',
                    'limit': self.WINDOW_SIZE + self.HORIZON + 10
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse the data
                prices = []
                volumes = []
                timestamps = []
                
                for candle in data:
                    timestamp = int(candle[0]) // 1000  # Convert to seconds
                    price = float(candle[4])  # Close price
                    volume = float(candle[5])  # Volume
                    
                    timestamps.append(timestamp)
                    prices.append(price)
                    volumes.append(volume)
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'price': prices,
                    'volume': volumes
                })
                
                return df
            
            else:
                print(f"⚠️ API returned status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def _get_price_data_manual(self):
        """
        Get price data manually from user input.
        """
        print("\n📊 Manual Data Entry Mode")
        print("=" * 50)
        print("Please enter the last 5-minute Bitcoin prices and volumes")
        print("(You can get this data from any exchange like Binance, Coinbase, etc.)")
        print("-" * 50)
        
        prices = []
        volumes = []
        
        for i in range(5):
            print(f"\nMinute {i+1}:")
            price = float(input(f"  BTC Price (USD): $"))
            volume = float(input(f"  BTC Volume: "))
            prices.append(price)
            volumes.append(volume)
        
        # Ask if user wants to enter more data
        more = input("\nEnter more data? (y/n): ").lower()
        if more == 'y':
            additional = int(input("How many more minutes? "))
            for i in range(additional):
                print(f"\nMinute {len(prices)+1}:")
                price = float(input(f"  BTC Price (USD): $"))
                volume = float(input(f"  BTC Volume: "))
                prices.append(price)
                volumes.append(volume)
        
        return prices, volumes
    
    def _create_features(self, prices, volumes, timestamps):
        """
        Create the feature set required by the model.
        
        Returns:
            Scaled feature array ready for prediction
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Close': prices,
            'Volume_BTC': volumes,
            'timestamp': timestamps
        })
        
        # Apply log1p to volume
        df['Volume_BTC_log'] = np.log1p(df['Volume_BTC'])
        
        # Create time features
        dt_idx = pd.to_datetime(df['timestamp'], unit='s')
        df['hour_sin'] = np.sin(2 * np.pi * dt_idx.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt_idx.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dt_idx.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dt_idx.dayofweek / 7)
        
        # Select features in the correct order
        features = ['Close', 'Volume_BTC_log', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        feature_data = df[features].values
        
        # Scale using the pre-fitted scaler
        scaled_data = self.scaler.transform(feature_data)
        
        return scaled_data
    
    def predict_next_hour(self, custom_prices=None, custom_volumes=None):
        """
        Predict the Bitcoin price 1 hour from now.
        
        Args:
            custom_prices: Optional list of recent prices (if API unavailable)
            custom_volumes: Optional list of recent volumes
            
        Returns:
            Predicted price in USD
        """
        print("\n🔮 Predicting BTC price for 1 hour from now...")
        print("-" * 50)
        
        # Get the data
        if custom_prices is not None and custom_volumes is not None:
            # Use custom data
            if len(custom_prices) < self.WINDOW_SIZE:
                print(f"⚠️ Warning: Only {len(custom_prices)} minutes of data provided.")
                print(f"   The model needs {self.WINDOW_SIZE} minutes for accurate prediction.")
                print("   Prediction may be less accurate.")
            
            prices = custom_prices[-self.WINDOW_SIZE:]
            volumes = custom_volumes[-self.WINDOW_SIZE:]
            timestamps = list(range(len(prices)))
            
        elif self.api_available:
            # Try to fetch from API
            df = self._fetch_recent_prices()
            
            if df is not None and len(df) >= self.WINDOW_SIZE:
                prices = df['price'].values[-self.WINDOW_SIZE:].tolist()
                volumes = df['volume'].values[-self.WINDOW_SIZE:].tolist()
                timestamps = df['timestamp'].values[-self.WINDOW_SIZE:].tolist()
            else:
                print("⚠️ Could not fetch enough data from API")
                print("   Switching to manual entry mode...")
                prices, volumes = self._get_price_data_manual()
                timestamps = list(range(len(prices)))
        else:
            # Manual entry mode
            prices, volumes = self._get_price_data_manual()
            timestamps = list(range(len(prices)))
        
        # Check if we have enough data
        if len(prices) < self.WINDOW_SIZE:
            print(f"\n❌ Error: Need {self.WINDOW_SIZE} minutes of data, but only have {len(prices)}")
            print("Please provide more data points.")
            return None
        
        # Create features and scale
        print("📊 Processing data...")
        scaled_data = self._create_features(prices, volumes, timestamps)
        
        # Get the last window of data (most recent 24 hours)
        window_data = scaled_data[-self.WINDOW_SIZE:].reshape(1, self.WINDOW_SIZE, self.N_FEATURES)
        
        # Make prediction
        print("🤖 Running model prediction...")
        scaled_prediction = self.model.predict(window_data, verbose=0)[0][0]
        
        # Inverse transform to get actual price
        prediction_usd = self._inverse_transform_close(scaled_prediction)
        
        # Get current price
        current_price = prices[-1]
        
        # Calculate change
        price_change = prediction_usd - current_price
        percent_change = (price_change / current_price) * 100
        
        # Display results
        print("\n" + "=" * 50)
        print("📈 PREDICTION RESULTS")
        print("=" * 50)
        print(f"Current BTC Price:      ${current_price:,.2f}")
        print(f"Predicted in 1 hour:    ${prediction_usd:,.2f}")
        print("-" * 50)
        
        if price_change > 0:
            print(f"📈 Expected increase:    +${price_change:,.2f} (+{percent_change:.2f}%)")
            print("💡 Recommendation: Consider BUYING")
        else:
            print(f"📉 Expected decrease:    ${price_change:,.2f} ({percent_change:.2f}%)")
            print("💡 Recommendation: Consider WAITING or SELLING")
        print("=" * 50)
        print("\n⚠️ DISCLAIMER: This is a prediction, not financial advice.")
        print("   Always do your own research before trading.")
        
        return prediction_usd
    
    def _inverse_transform_close(self, scaled_value):
        """
        Convert scaled prediction back to actual USD price.
        """
        dummy = np.zeros((1, self.scaler.n_features_in_))
        dummy[0, 0] = scaled_value
        return self.scaler.inverse_transform(dummy)[0, 0]
    
    def predict_multiple_hours(self, hours=4):
        """
        Predict prices for multiple hours ahead (recursive prediction).
        
        Args:
            hours: Number of hours to predict
            
        Returns:
            List of predicted prices for each hour
        """
        print(f"\n🔮 Predicting BTC prices for the next {hours} hours...")
        print("-" * 50)
        
        predictions = []
        
        for i in range(hours):
            print(f"\nHour {i+1}:")
            pred = self.predict_next_hour()
            if pred is not None:
                predictions.append(pred)
            time.sleep(1)  # Small delay between predictions
        
        return predictions
    
    def get_trading_signal(self):
        """
        Get a simple trading signal (BUY/SELL/HOLD) based on the prediction.
        
        Returns:
            Dictionary with signal, confidence, and details
        """
        prediction = self.predict_next_hour()
        
        if prediction is None:
            return {'signal': 'ERROR', 'confidence': 0, 'message': 'Could not make prediction'}
        
        # Get current price (use the last price from the API or last manual entry)
        if self.api_available:
            df = self._fetch_recent_prices()
            if df is not None:
                current_price = df['price'].values[-1]
            else:
                current_price = None
        else:
            current_price = None
        
        if current_price is None:
            # Just return the prediction without price comparison
            return {
                'signal': 'NEUTRAL',
                'predicted_price': prediction,
                'confidence': 0.7,
                'message': f'Model predicts ${prediction:,.2f} in 1 hour'
            }
        
        # Calculate expected change percentage
        change_pct = ((prediction - current_price) / current_price) * 100
        
        # Determine signal
        if change_pct > 1.5:
            signal = 'STRONG BUY'
            confidence = min(0.9, 0.5 + change_pct / 10)
        elif change_pct > 0.5:
            signal = 'BUY'
            confidence = 0.6 + change_pct / 10
        elif change_pct < -1.5:
            signal = 'STRONG SELL'
            confidence = min(0.9, 0.5 - change_pct / 10)
        elif change_pct < -0.5:
            signal = 'SELL'
            confidence = 0.6 - change_pct / 10
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        return {
            'signal': signal,
            'current_price': current_price,
            'predicted_price': prediction,
            'expected_change': change_pct,
            'confidence': confidence,
            'message': f'Predicted {change_pct:+.2f}% change in 1 hour'
        }


# ============================================================================
# Interactive CLI Interface
# ============================================================================

def main():
    """Main interactive interface for the Bitcoin predictor."""
    print("\n" + "=" * 60)
    print("   BITCOIN PRICE PREDICTOR - 1 Hour Forecast")
    print("=" * 60)
    print("\nWelcome! This tool predicts Bitcoin price 1 hour from now")
    print("using a GRU neural network trained on 2017-2019 data.\n")
    
    # Initialize predictor
    try:
        predictor = BTCPricePredictor(
            model_path='/kaggle/working/gru_btc_final.keras',
            scaler_path='/kaggle/working/btc_scaler.pkl'
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Please ensure the model files are in the correct location.")
        return
    
    print("\n" + "-" * 60)
    print("OPTIONS:")
    print("  1. Get single prediction (1 hour ahead)")
    print("  2. Get trading signal (BUY/SELL/HOLD)")
    print("  3. Predict multiple hours (2-24 hours)")
    print("  4. Manual data entry (if no API)")
    print("  5. Exit")
    print("-" * 60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                predictor.predict_next_hour()
                
            elif choice == '2':
                signal = predictor.get_trading_signal()
                print("\n" + "=" * 50)
                print("📊 TRADING SIGNAL")
                print("=" * 50)
                for key, value in signal.items():
                    print(f"{key}: {value}")
                print("=" * 50)
                
            elif choice == '3':
                hours = int(input("How many hours to predict? (2-24): "))
                hours = min(max(hours, 2), 24)
                predictions = predictor.predict_multiple_hours(hours)
                
            elif choice == '4':
                print("\n📝 Manual Data Entry")
                print("Enter the last 10 minutes of price and volume data:")
                prices = []
                volumes = []
                for i in range(10):
                    price = float(input(f"Minute {i+1} price (USD): $"))
                    volume = float(input(f"Minute {i+1} volume (BTC): "))
                    prices.append(price)
                    volumes.append(volume)
                predictor.predict_next_hour(custom_prices=prices, custom_volumes=volumes)
                
            elif choice == '5':
                print("\n👋 Goodbye! Happy trading!")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()