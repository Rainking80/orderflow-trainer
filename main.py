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
import sys
import signal
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8120"
WS_URL = "ws://localhost:8120/ws/trade_groups"
AUTH_TOKEN = os.getenv("AUTH_BEARER_TOKEN", "4kEZu6v_W5y2oNf8oZg2L1b3bO8hJ8mX")

# Symbols configuration
SYMBOLS = os.getenv("SYMBOLS", "btcusdt").split(",")
SYMBOLS = [s.strip().lower() for s in SYMBOLS]  # Normalize symbols

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

class MarketPredictor:
    def __init__(self, symbols):
        """
        Initialize predictor for multiple symbols
        Args:
            symbols: List of symbol strings (e.g., ['btcusdt', 'ethusdt', 'solusdt'])
        """
        self.symbols = symbols
        self.models = {}  # symbol -> model
        self.feature_cols = [
            'delta', 'delta_pct', 'total_volume', 'trade_count', 
            'buy_volume', 'sell_volume', 'vwap', 'price_range',
            'large_trade_dominance', 'ratio_buy_sell_trades',
            'rsi', 'volatility', 'return'
        ]
        self.running = False
        self.latest_data = {}  # symbol -> list of data
        self.lock = threading.Lock()
        
        # Telegram notification tracking per symbol
        self.previous_probs = {}  # symbol -> {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
        self.last_hourly_message_time = {}  # symbol -> datetime
        self.last_prediction_data = {}  # symbol -> prediction dict
        self.active_events = {}  # symbol -> {'NEUTRAL': False, 'PUMP': False, 'DUMP': False}
        
        # Initialize per-symbol data structures
        for symbol in self.symbols:
            self.latest_data[symbol] = []
            self.previous_probs[symbol] = {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
            self.last_hourly_message_time[symbol] = None
            self.last_prediction_data[symbol] = None
            self.active_events[symbol] = {'NEUTRAL': False, 'PUMP': False, 'DUMP': False}
        
        self.ws = None

    def handle_exit(self, signum, frame):
        print(f"\nReceived signal {signum}. Exiting gracefully...")
        if self.ws:
            self.ws.close()
        sys.exit(0)

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

    def send_telegram_photo(self, photo_bytes, caption=""):
        """Send a photo via Telegram Bot API"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("Telegram credentials not configured. Skipping photo.")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {
                'photo': ('chart.png', photo_bytes, 'image/png')
            }
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
            print("Telegram photo sent successfully.")
            return True
        except Exception as e:
            print(f"Failed to send Telegram photo: {e}")
            return False

    def generate_candle_chart(self, df_20min, symbol):
        """Generate a candlestick chart from the last 20 minutes of data"""
        try:
            # Prepare data for mplfinance
            # mplfinance expects DatetimeIndex and columns: Open, High, Low, Close, Volume
            chart_df = df_20min[['start_ts', 'open', 'high', 'low', 'close', 'total_volume']].copy()
            chart_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            chart_df.set_index('Date', inplace=True)
            
            # Create the chart
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style='charles',
                title=f'{symbol.upper()} Price Chart - Last {len(chart_df)} Candles (1m)',
                ylabel='Price (USD)',
                volume=True,
                ylabel_lower='Volume',
                figsize=(12, 8),
                returnfig=True
            )
            
            # Save to bytes buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            return buf.getvalue()
        except Exception as e:
            print(f"Error generating candle chart: {e}")
            return None

    def send_followup(self, symbol, category, trigger_price, trigger_time, label):
        """Send follow-up notification with chart"""
        try:
            # 1. Get Data (Locked)
            with self.lock:
                if symbol not in self.latest_data or not self.latest_data[symbol]:
                    return
                # Copy data to avoid holding lock during processing
                data_copy = list(self.latest_data[symbol])
            
            # 2. Process Data (Unlocked)
            df_all = pd.DataFrame(data_copy)
            if 'start_ts' not in df_all.columns:
                return
            
            # Handle mixed types
            df_all['start_ts'] = pd.to_datetime(df_all['start_ts'])
            df_all.set_index('start_ts', inplace=True)
            
            # Resample to 1m
            df_1m = df_all.resample('1min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'total_volume': 'sum'
            }).dropna()
            
            if df_1m.empty:
                return

            current_price = df_1m.iloc[-1]['close']
            
            # Determine chart window
            # Start from trigger_time - offset (60 mins before trigger to match initial chart)
            offset = timedelta(minutes=60)
            start_chart = trigger_time - offset
            
            # Filter data
            df_chart = df_1m[df_1m.index >= start_chart].copy()
            
            # Generate chart
            chart_bytes = None
            if len(df_chart) >= 5:
                # Reset index for plotting function
                df_chart_reset = df_chart.reset_index()
                # Rename columns to match what generate_candle_chart expects if needed
                # But generate_candle_chart expects 'start_ts' which we have after reset_index
                
                print(f"Generating {label} follow-up chart for {category}...")
                chart_bytes = self.generate_candle_chart(df_chart_reset, symbol)
            
            # Calculate stats
            pct_change = (current_price - trigger_price) / trigger_price
            
            emoji = "➖"
            if category == "PUMP":
                emoji = "✅" if pct_change > 0 else "❌"
            elif category == "DUMP":
                emoji = "✅" if pct_change < 0 else "❌"
            
            message = f"⏱️ <b>{label} Follow-up: {category} ({symbol.upper()})</b>\n\n"
            message += f"Trigger Price: ${trigger_price:.2f}\n"
            message += f"Current Price: ${current_price:.2f}\n"
            message += f"Change: {pct_change:.2%} {emoji}\n"
            
            if chart_bytes:
                self.send_telegram_photo(chart_bytes, message)
            else:
                self.send_telegram_message(message)
                
        except Exception as e:
            print(f"Error in follow-up for {symbol}: {e}")
        finally:
            # Reset flag if 2h
            if label == "2h":
                with self.lock:
                    self.active_events[symbol][category] = False
                    print(f"Reset active event for {category} on {symbol}")

    def check_and_send_notifications(self, symbol, timestamp, price, delta, direction, confidence, probs):
        """Check if notifications should be sent based on threshold crossing or hourly schedule"""
        current_time = datetime.now()
        
        # Store the latest prediction data for this symbol
        self.last_prediction_data[symbol] = {
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
            prev_prob = self.previous_probs[symbol][category]
            curr_prob = current_probs[category]
            
            # Check if crossed 60% threshold from below
            if prev_prob < threshold <= curr_prob:
                # Check if event is already active
                if self.active_events[symbol][category]:
                    print(f"Skipping {category} alert for {symbol} - Timer already running.")
                    continue
                
                # Set active
                self.active_events[symbol][category] = True
                
                # Schedule timers
                # 30 min timer
                t30 = threading.Timer(30 * 60, self.send_followup, args=[symbol, category, price, timestamp, "30min"])
                t30.start()
                
                # 2 hour timer
                t120 = threading.Timer(2 * 60 * 60, self.send_followup, args=[symbol, category, price, timestamp, "2h"])
                t120.start()
                
                message = f"🚨 <b>ALERT: {symbol.upper()} {category} crossed 60% threshold!</b>\n\n"
                message += f"Current Price: ${price:.2f}\n"
                message += f"Order Flow Imbalance (OFI): {delta:.2f}\n\n"
                message += f"Prediction: <b>{direction}</b> (Confidence: {confidence:.2%})\n\n"
                message += f"Probability Distribution:\n"
                message += f"  - NEUTRAL: {probs[0]:.2%}\n"
                message += f"  - PUMP:    {probs[1]:.2%}\n"
                message += f"  - DUMP:    {probs[2]:.2%}"
                
                self.send_telegram_message(message)
                
                # Collect data for chart
                # Note: self.lock is already held by the caller (on_message method)
                if len(self.latest_data[symbol]) > 0:
                    # Convert to DataFrame
                    df_all = pd.DataFrame(self.latest_data[symbol])
                    
                    # Ensure start_ts is datetime
                    if 'start_ts' in df_all.columns:
                        # Handle mixed types (strings from WS, datetimes from history)
                        df_all['start_ts'] = pd.to_datetime(df_all['start_ts'])
                        
                        # Set index for resampling
                        df_all.set_index('start_ts', inplace=True)
                        
                        # Resample to 1-minute candles
                        # Aggregation rules:
                        # Open: first
                        # High: max
                        # Low: min
                        # Close: last
                        # Volume: sum
                        df_1m = df_all.resample('1min').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'total_volume': 'sum'
                        }).dropna()
                        
                        # Take up to the last 60 candles (minimum 5)
                        if len(df_1m) >= 5:
                            df_chart = df_1m.tail(60).copy()
                            
                            # Reset index to make start_ts a column again for generate_candle_chart
                            df_chart = df_chart.reset_index()
                            
                            print(f"Generating candle chart for {symbol} with {len(df_chart)} 1m candles...")
                            chart_bytes = self.generate_candle_chart(df_chart, symbol)
                            
                            if chart_bytes:
                                caption = f"📈 <b>{symbol.upper()} Price Chart - Last {len(df_chart)} Candles (1m)</b>\n{category} crossed 60% threshold"
                                self.send_telegram_photo(chart_bytes, caption)
                                                                
                        else:
                            print(f"Not enough data for 1m chart for {symbol}. Have {len(df_1m)} candles, need 5.")
        
        # Update previous probabilities for this symbol
        self.previous_probs[symbol] = current_probs.copy()
        
        # Check for hourly update
        if self.last_hourly_message_time[symbol] is None:
            self.last_hourly_message_time[symbol] = current_time
        
        time_since_last_hourly = (current_time - self.last_hourly_message_time[symbol]).total_seconds()
        if time_since_last_hourly >= 3600:  # 3600 seconds = 1 hour
            message = f"📊 <b>Hourly Update ({symbol.upper()})</b>\n\n"
            message += f"Current Price: ${price:.2f}\n"
            message += f"Order Flow Imbalance (OFI): {delta:.2f}\n\n"
            message += f"Prediction: <b>{direction}</b> (Confidence: {confidence:.2%})\n\n"
            message += f"Probability Distribution:\n"
            message += f"  - NEUTRAL: {probs[0]:.2%}\n"
            message += f"  - PUMP:    {probs[1]:.2%}\n"
            message += f"  - DUMP:    {probs[2]:.2%}"
            
            self.send_telegram_message(message)
            self.last_hourly_message_time[symbol] = current_time

    def fetch_historical_data(self, symbols, days=3):
        print(f"Fetching data for symbols {symbols} for the last {days} days in 3 requests...")
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
                'end': chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'symbols': ','.join(symbols)  # Add symbols parameter
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

    def get_model_file(self, symbol):
        """Get model filename for a specific symbol"""
        return f"lgb_model_{symbol}.txt"
    
    def load_model_if_valid(self, symbol, max_age_days=7):
        """Load model for a symbol if it exists and is younger than max_age_days"""
        model_file = self.get_model_file(symbol)
        if not os.path.exists(model_file):
            return False
            
        try:
            # Check file age
            file_time = os.path.getmtime(model_file)
            file_age = datetime.now() - datetime.fromtimestamp(file_time)
            
            if file_age.days >= max_age_days:
                print(f"Model for {symbol} found but too old ({file_age.days} days). Retraining...")
                return False
                
            print(f"Loading saved model for {symbol} (Age: {file_age})...")
            self.models[symbol] = lgb.Booster(model_file=model_file)
            return True
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            return False

    def train_model(self, symbol, df):
        print(f"Training model for {symbol}...")
        df = self.prepare_features(df)
        
        # Drop last rows with NaN targets
        df = df.dropna(subset=['target', 'future_return'] + self.feature_cols)
        
        if len(df) < 10:
            print(f"Not enough data after feature engineering to train model for {symbol}. Rows: {len(df)}")
            return None

        X = df[self.feature_cols]
        y = df['target']
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        except ValueError as e:
            print(f"Error splitting data for {symbol}: {e}")
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
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_class = [np.argmax(x) for x in y_pred]
        print(classification_report(y_test, y_pred_class))
        
        # Save model
        model_file = self.get_model_file(symbol)
        try:
            model.save_model(model_file)
            print(f"Model for {symbol} saved to {model_file}")
        except Exception as e:
            print(f"Failed to save model for {symbol}: {e}")
        
        # Store in models dict
        self.models[symbol] = model
        return model

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
        
        # Extract symbol from data
        symbol = data.get('symbol', '').lower()
        if not symbol or symbol not in self.symbols:
            return  # Skip data for unknown symbols
        
        with self.lock:
            # Append to symbol-specific data
            self.latest_data[symbol].append(data)
            if len(self.latest_data[symbol]) > 20000: # Keep last 20000 bars for feature calc and 15m resampling
                self.latest_data[symbol].pop(0)
                
            if len(self.latest_data[symbol]) >= 20: # Need enough data for rolling features
                df = pd.DataFrame(self.latest_data[symbol])
                df = self.prepare_features(df) # This adds features to all rows
                
                # Get last row
                last_row = df.iloc[-1]
                
                # Check if model exists for this symbol
                if symbol not in self.models or self.models[symbol] is None:
                    return
                
                # Predict
                features = last_row[self.feature_cols].values.reshape(1, -1)
                prediction = self.models[symbol].predict(features)
                pred_class = np.argmax(prediction[0])
                probs = prediction[0]
                
                timestamp = last_row['start_ts']
                price = last_row['close']
                
                # Map prediction class to direction
                direction_map = {0: "NEUTRAL", 1: "PUMP", 2: "DUMP"}
                direction = direction_map[pred_class]
                confidence = probs[pred_class]
                
                # Display all prediction results with full probability distribution
                # print(f"\n{'='*80}")
                # print(f"[{timestamp}] LIVE PREDICTION for {symbol.upper()}")
                # print(f"{'='*80}")
                # print(f"Current Price: ${price:.2f}")
                # print(f"Order Flow Imbalance (OFI): {last_row['delta']:.2f}")
                # print(f"\nPrediction: {direction} (Confidence: {confidence:.2%})")
                # print(f"\nProbability Distribution:")
                # print(f"  - NEUTRAL: {probs[0]:.2%}")
                # print(f"  - PUMP:    {probs[1]:.2%}")
                # print(f"  - DUMP:    {probs[2]:.2%}")
                # print(f"{'='*80}\n")
                
                # Send Telegram notifications if conditions are met
                self.check_and_send_notifications(
                    symbol=symbol,
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

    def start_live_prediction(self, historical_data=None):
        # Check if at least one model is trained
        if not self.models or all(model is None for model in self.models.values()):
            print("No models trained! Cannot start live prediction.")
            return

        print("Starting live prediction...")
        
        # Pre-fill buffer with historical data per symbol
        if historical_data is not None and not historical_data.empty:
            # Group by symbol and pre-fill each buffer
            for symbol in self.symbols:
                symbol_data = historical_data[historical_data['symbol'] == symbol.upper()]
                if not symbol_data.empty:
                    # Convert to list of dicts and take last 20000
                    # Note: historical_data has datetime objects for start_ts, WS has strings.
                    # We'll handle this mixed type in check_and_send_notifications.
                    records = symbol_data.to_dict('records')
                    # Ensure we don't exceed the buffer size we want to maintain
                    self.latest_data[symbol.upper()] = records[-20000:]
                    print(f"Pre-filled buffer for {symbol.upper()} with {len(self.latest_data[symbol.upper()])} records from history.")
        
        websocket.enableTrace(False)
        ws_headers = {}
        if AUTH_TOKEN:
            ws_headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

        # Register signal handlers for systemd/OS signals
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

        # Add symbols parameter to WebSocket URL
        ws_url_with_symbols = f"{WS_URL}?symbols={','.join(self.symbols)}"
        
        self.ws = websocket.WebSocketApp(ws_url_with_symbols,
                                  on_open=self.on_open,
                                  on_message=self.on_message,
                                  on_error=self.on_error,
                                  on_close=self.on_close,
                                  header=ws_headers)

        self.ws.run_forever()

if __name__ == "__main__":
    print(f"Initializing Market Predictor for symbols: {SYMBOLS}")
    predictor = MarketPredictor(SYMBOLS)
    
    # 1. Check for existing models and load them
    models_loaded = {}
    for symbol in SYMBOLS:
        models_loaded[symbol] = predictor.load_model_if_valid(symbol, max_age_days=50)
        if models_loaded[symbol]:
            print(f"✓ Model loaded for {symbol}")
        else:
            print(f"✗ Model not found or too old for {symbol}")
    
    # 2. Fetch Data
    # If all models are loaded, we just need enough data for the buffer (e.g. 2 days)
    # If we need to train any model, we fetch more (e.g. 5 days)
    all_models_loaded = all(models_loaded.values())
    days_to_fetch = 2 if all_models_loaded else 5
    
    print(f"\nFetching {days_to_fetch} days of historical data...")
    df = predictor.fetch_historical_data(SYMBOLS, days=days_to_fetch)
    
    if not df.empty and len(df) > 10:
        # 3. Train Models if needed (per symbol)
        for symbol in SYMBOLS:
            if not models_loaded[symbol]:
                print(f"\nTraining model for {symbol.upper()}...")
                # Filter data for this symbol
                symbol_df = df[df['symbol'] == symbol.upper()].copy()
                if len(symbol_df) > 10:
                    predictor.train_model(symbol.upper(), symbol_df)
                    print(f"✓ Model trained for {symbol.upper()}")
                else:
                    print(f"✗ Not enough data for {symbol.upper()}. Skipping training.")
        
        # 4. Live Predict
        try:
            predictor.start_live_prediction(df)
        except KeyboardInterrupt:
            print("Stopping...")
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1) # Exit with error code so systemd knows it failed
    else:
        print(f"Not enough data. Fetched {len(df)} records.")
