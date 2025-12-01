import pandas as pd
import requests
import json
import lightgbm as lgb
import numpy as np
import websocket
import threading
import time
import io
import os
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8120"
WS_URL = "ws://localhost:8120/ws/trade_groups"
AUTH_TOKEN = os.getenv("AUTH_BEARER_TOKEN", "4kEZu6v_W5y2oNf8oZg2L1b3bO8hJ8mX")

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

class MarketPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = [
            'delta', 'delta_pct', 'total_volume', 'trade_count', 
            'buy_volume', 'sell_volume', 'vwap', 'price_range',
            'large_trade_dominance', 'ratio_buy_sell_trades',
            'rsi', 'volatility', 'return'
        ]
        self.running = False
        self.latest_data = []
        self.lock = threading.Lock()
        
        # Telegram notification tracking
        self.previous_probs = {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
        self.last_hourly_message_time = None
        self.last_prediction_data = None

    def send_telegram_message(self, message):
        """Send a message via Telegram Bot API"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("Telegram credentials not configured. Skipping notification.")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print("Telegram message sent successfully.")
            return True
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")
            return False

    def check_and_send_notifications(self, timestamp, price, delta, direction, confidence, probs):
        """Check if notifications should be sent based on threshold crossing or hourly schedule"""
        current_time = datetime.now()
        
        # Store the latest prediction data
        self.last_prediction_data = {
            'timestamp': timestamp,
            'price': price,
            'delta': delta,
            'direction': direction,
            'confidence': confidence,
            'probs': probs
        }
        
        # Check for 60% threshold crossing from below for each category
        threshold = 0.60
        current_probs = {
            'NEUTRAL': probs[0],
            'PUMP': probs[1],
            'DUMP': probs[2]
        }
        
        for category in ['NEUTRAL', 'PUMP', 'DUMP']:
            prev_prob = self.previous_probs[category]
            curr_prob = current_probs[category]
            
            # Check if crossed 60% threshold from below
            if prev_prob < threshold <= curr_prob:
                message = f"🚨 <b>ALERT: {category} crossed 60% threshold!</b>\n\n"
                message += f"Current Price: ${price:.2f}\n"
                message += f"Order Flow Imbalance (OFI): {delta:.2f}\n\n"
                message += f"Prediction: <b>{direction}</b> (Confidence: {confidence:.2%})\n\n"
                message += f"Probability Distribution:\n"
                message += f"  - NEUTRAL: {probs[0]:.2%}\n"
                message += f"  - PUMP:    {probs[1]:.2%}\n"
                message += f"  - DUMP:    {probs[2]:.2%}"
                
                self.send_telegram_message(message)
        
        # Update previous probabilities
        self.previous_probs = current_probs.copy()
        
        # Check for hourly update
        if self.last_hourly_message_time is None:
            self.last_hourly_message_time = current_time
        
        time_since_last_hourly = (current_time - self.last_hourly_message_time).total_seconds()
        if time_since_last_hourly >= 3600:  # 3600 seconds = 1 hour
            message = f"📊 <b>Hourly Update</b>\n\n"
            message += f"Current Price: ${price:.2f}\n"
            message += f"Order Flow Imbalance (OFI): {delta:.2f}\n\n"
            message += f"Prediction: <b>{direction}</b> (Confidence: {confidence:.2%})\n\n"
            message += f"Probability Distribution:\n"
            message += f"  - NEUTRAL: {probs[0]:.2%}\n"
            message += f"  - PUMP:    {probs[1]:.2%}\n"
            message += f"  - DUMP:    {probs[2]:.2%}"
            
            self.send_telegram_message(message)
            self.last_hourly_message_time = current_time

    def fetch_historical_data(self, days=3):
        print(f"Fetching data for the last {days} days in 3 requests...")
        all_data = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Calculate the time interval for each of the 3 requests
        total_duration = end_time - start_time
        chunk_duration = total_duration / 3
        
        # Always make exactly 3 requests
        for i in range(3):
            # Calculate time range for this chunk
            chunk_end = end_time - (chunk_duration * i)
            chunk_start = chunk_end - chunk_duration
            
            params = {
                'start': chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'end': chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            try:
                print(f"Request {i+1}/3: Downloading data from {params['start']} to {params['end']}...")
                headers = {}
                if AUTH_TOKEN:
                    headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
                
                # Explicitly request gzip to avoid zstd issues if any
                headers["Accept-Encoding"] = "gzip, deflate"

                response = requests.get(f"{BASE_URL}/download/trade_groups", params=params, headers=headers, stream=True)
                response.raise_for_status()
                
                # Handle NDJSON
                for line in response.iter_lines():
                    if line:
                        try:
                            # Decode bytes to string and parse JSON
                            decoded_line = line.decode('utf-8')
                            item = json.loads(decoded_line)
                            all_data.append(item)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            continue
                        
            except Exception as e:
                print(f"Error fetching data: {e}")
            
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['start_ts'] = pd.to_datetime(df['start_ts'], format='ISO8601')
            df = df.sort_values('start_ts').reset_index(drop=True)
            print(f"Total records fetched: {len(df)}")
        else:
            print("No data fetched.")
            
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_features(self, df):
        if df.empty:
            return df
            
        df = df.copy()
        
        # Basic features
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=20).std()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Fill NaNs
        df = df.fillna(0)
        
        # Target: Max price change in next 20 bars
        # We want to predict if price will go UP or DOWN significantly
        # Let's define "significant" as > 0.1% move
        
        future_window = 20
        threshold = 0.001 # 0.1%
        
        # Calculate future return
        # We look at the max high and min low in the next window to catch volatility
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=future_window)
        df['future_high'] = df['high'].rolling(window=indexer).max()
        df['future_low'] = df['low'].rolling(window=indexer).min()
        
        df['target'] = 0
        
        # If max high > current close * (1 + threshold) -> Buy Signal (1)
        # If min low < current close * (1 - threshold) -> Sell Signal (-1)
        # If both (high volatility), maybe 0 or prioritize one? Let's prioritize the first one that happens.
        # For simplicity, let's just use close-to-close return for now to avoid lookahead bias complexity in labeling
        
        df['future_close'] = df['close'].shift(-future_window)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        df.loc[df['future_return'] > threshold, 'target'] = 1
        df.loc[df['future_return'] < -threshold, 'target'] = 2 # LightGBM likes 0, 1, 2...
        # 0 = Neutral, 1 = Pump, 2 = Dump
        
        return df

    def train_model(self, df):
        print("Training model...")
        df = self.prepare_features(df)
        
        # Drop last rows with NaN targets
        df = df.dropna(subset=['target', 'future_return'] + self.feature_cols)
        
        if len(df) < 10:
            print(f"Not enough data after feature engineering to train model. Rows: {len(df)}")
            return None

        X = df[self.feature_cols]
        y = df['target']
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        except ValueError as e:
            print(f"Error splitting data: {e}")
            return None
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_class = [np.argmax(x) for x in y_pred]
        print(classification_report(y_test, y_pred_class))
        
        return self.model

    def on_message(self, ws, message):
        # Skip empty or non-string messages
        if not message or not isinstance(message, (str, bytes)):
            return
        
        # Convert bytes to string if needed
        if isinstance(message, bytes):
            try:
                message = message.decode('utf-8')
            except UnicodeDecodeError:
                return
        
        # Skip empty strings after decoding
        message = message.strip()
        if not message:
            return
        
        # Try to parse JSON
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            # Skip non-JSON messages (ping/pong, connection frames, etc.)
            return
        
        with self.lock:
            self.latest_data.append(data)
            if len(self.latest_data) > 100: # Keep last 100 bars for feature calc
                self.latest_data.pop(0)
                
            if len(self.latest_data) >= 20: # Need enough data for rolling features
                df = pd.DataFrame(self.latest_data)
                df = self.prepare_features(df) # This adds features to all rows
                
                # Get last row
                last_row = df.iloc[-1]
                
                # Predict
                features = last_row[self.feature_cols].values.reshape(1, -1)
                prediction = self.model.predict(features)
                pred_class = np.argmax(prediction[0])
                probs = prediction[0]
                
                timestamp = last_row['start_ts']
                price = last_row['close']
                
                # Map prediction class to direction
                direction_map = {0: "NEUTRAL", 1: "PUMP", 2: "DUMP"}
                direction = direction_map[pred_class]
                confidence = probs[pred_class]
                
                # Display all prediction results with full probability distribution
                print(f"\n{'='*80}")
                print(f"[{timestamp}] LIVE PREDICTION")
                print(f"{'='*80}")
                print(f"Current Price: ${price:.2f}")
                print(f"Order Flow Imbalance (OFI): {last_row['delta']:.2f}")
                print(f"\nPrediction: {direction} (Confidence: {confidence:.2%})")
                print(f"\nProbability Distribution:")
                print(f"  - NEUTRAL: {probs[0]:.2%}")
                print(f"  - PUMP:    {probs[1]:.2%}")
                print(f"  - DUMP:    {probs[2]:.2%}")
                print(f"{'='*80}\n")
                
                # Send Telegram notifications if conditions are met
                self.check_and_send_notifications(
                    timestamp=timestamp,
                    price=price,
                    delta=last_row['delta'],
                    direction=direction,
                    confidence=confidence,
                    probs=probs
                )

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket Closed. Code: {close_status_code}, Msg: {close_msg}")

    def on_open(self, ws):
        print("WebSocket Connection Opened")

    def start_live_prediction(self):
        if self.model is None:
            print("Model not trained! Cannot start live prediction.")
            return

        print("Starting live prediction...")
        # Pre-fill buffer with some historical data so we don't wait too long for features
        # (Optional, but good practice. For now we just wait for 20 bars from WS)
        
        websocket.enableTrace(False)
        ws_headers = {}
        if AUTH_TOKEN:
            ws_headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

        ws = websocket.WebSocketApp(WS_URL,
                                  on_open=self.on_open,
                                  on_message=self.on_message,
                                  on_error=self.on_error,
                                  on_close=self.on_close,
                                  header=ws_headers)

        ws.run_forever()

if __name__ == "__main__":
    predictor = MarketPredictor()
    
    # 1. Fetch Data
    df = predictor.fetch_historical_data(days=5)
    
    if not df.empty and len(df) > 10:
        # 2. Train Model
        predictor.train_model(df)
        
        # 3. Live Predict
        try:
            predictor.start_live_prediction()
        except KeyboardInterrupt:
            print("Stopping...")
    else:
        print(f"Not enough data to train. Fetched {len(df)} records. Need > 10.")
