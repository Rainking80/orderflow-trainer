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
import logging
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf

# Configure logging for systemd
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
        self.lock = threading.RLock()

        self.send_hourly_message = False
        
        # Telegram notification tracking per symbol
        self.previous_probs = {}  # symbol -> {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
        self.last_hourly_message_time = None  # Global timestamp for hourly updates
        self.last_prediction_data = {}  # symbol -> prediction dict
        self.active_events = {}  # symbol -> {'NEUTRAL': False, 'PUMP': False, 'DUMP': False}
        self.last_nightly_chart_time = None  # Track last nightly chart generation
        
        # Batching for threshold crossing alerts
        self.pending_alerts = []  # List of (symbol, category) tuples
        self.alert_batch_timer = None  # Timer for sending batched alerts
        self.alert_batch_delay = 2.0  # Wait 2 seconds to batch alerts from multiple symbols
        
        # Initialize per-symbol data structures
        for symbol in self.symbols:
            self.latest_data[symbol] = []
            self.previous_probs[symbol] = {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
            self.last_prediction_data[symbol] = None
            self.active_events[symbol] = {'NEUTRAL': False, 'PUMP': False, 'DUMP': False}
        
        self.initial_message_sent = False
        self.ws = None

    def handle_exit(self, signum, frame):
        logger.info(f"Received signal {signum}. Exiting gracefully...")
        if self.ws:
            self.ws.close()
        sys.exit(0)

    def send_telegram_message(self, message):
        """Send a message via Telegram Bot API"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured. Skipping notification.")
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
            logger.info("Telegram message sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_telegram_photo(self, photo_bytes, caption=""):
        """Send a photo via Telegram Bot API"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured. Skipping photo.")
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
            logger.info("Telegram photo sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
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
            logger.error(f"Error generating candle chart: {e}")
            return None

    def generate_latest_chart(self, symbol):
        """Generate 1m candle chart for the last 60 minutes for a symbol"""
        data_copy = []
        with self.lock:
            if symbol in self.latest_data:
                data_copy = list(self.latest_data[symbol])
        
        if not data_copy:
            return None
            
        try:
            df_all = pd.DataFrame(data_copy)
            if 'symbol' in df_all.columns:
                df_all = df_all[df_all['symbol'].str.lower() == symbol.lower()].copy()
            
            if 'start_ts' in df_all.columns:
                df_all['start_ts'] = pd.to_datetime(df_all['start_ts'])
                df_all.set_index('start_ts', inplace=True)
                
                df_1m = df_all.resample('1min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'total_volume': 'sum'
                }).dropna()
                
                if len(df_1m) >= 5:
                    df_chart = df_1m.tail(60).copy()
                    df_chart = df_chart.reset_index()
                    return self.generate_candle_chart(df_chart, symbol)
        except Exception as e:
            logger.error(f"Error generating latest chart for {symbol}: {e}")
            
        return None

    def send_status_message(self, title, alerts=None):
        """Send a status message with optional alerts highlighted"""
        if alerts is None:
            alerts = []
            
        # Build aggregated message
        message = f"{title}\n\n"
        
        # Iterate over ALL symbols to build the message
        for sym in self.symbols:
            pred_data = self.last_prediction_data.get(sym)
            if pred_data:
                # Get emoji for prediction direction
                direction_emoji = ""
                if pred_data['direction'] == "PUMP":
                    direction_emoji = "🚀"
                elif pred_data['direction'] == "DUMP":
                    direction_emoji = "📉"
                elif pred_data['direction'] == "NEUTRAL":
                    direction_emoji = "➖"
                
                # Check if this symbol has an alert
                crossed = [cat for s, cat in alerts if s == sym]
                
                # Add header with alert indicator
                if crossed:
                    message += f"⚠️ <b>{sym.upper()} (ALERT: {', '.join(crossed)})</b>\n"
                else:
                    message += f"<b>{sym.upper()}</b>\n"
                
                message += f"Price: ${pred_data['price']:.2f}\n"
                message += f"OFI: {pred_data['delta']:.2f}\n"
                message += f"Prediction: <b>{direction_emoji} {pred_data['direction']}</b> ({pred_data['confidence']:.2%})\n"
                message += f"Probabilities:\n"
                message += f"NEUTRAL: {pred_data['probs']['NEUTRAL']:.2%}\n"
                message += f"PUMP: {pred_data['probs']['PUMP']:.2%}\n"
                message += f"DUMP: {pred_data['probs']['DUMP']:.2%}\n"
                message += "\n"
        
        # Generate chart
        chart_bytes = None
        target_symbol = None
        
        if alerts:
            # If alerts exist, chart the first alerted symbol
            # alerts is list of (symbol, category)
            target_symbol = alerts[0][0]
        elif self.symbols:
            # If no alerts (e.g. initial message), chart the first configured symbol
            target_symbol = self.symbols[0]
            
        if target_symbol:
            chart_bytes = self.generate_latest_chart(target_symbol)
            
        if chart_bytes:
            self.send_telegram_photo(chart_bytes, message)
        else:
            self.send_telegram_message(message)

    def send_batched_alerts(self):
        """Send a single aggregated message for all pending threshold crossing alerts"""
        alerts_to_send = []
        with self.lock:
            if not self.pending_alerts:
                return
            
            # Copy and clear pending alerts
            alerts_to_send = list(self.pending_alerts)
            self.pending_alerts = []
            self.alert_batch_timer = None
        
        self.send_status_message("🚨 <b>ALERT: Threshold Crossed!</b>", alerts_to_send)
    
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
            
            # Filter to ensure only this symbol's data (in case of mixed data)
            if 'symbol' in df_all.columns:
                df_all = df_all[df_all['symbol'].str.lower() == symbol.lower()].copy()
            
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
                
                logger.info(f"Generating {label} follow-up chart for {category}...")
                chart_bytes = self.generate_candle_chart(df_chart_reset, symbol)
            
            # Calculate stats
            pct_change = (current_price - trigger_price) / trigger_price
            
            emoji = "➖"
            if category == "PUMP":
                emoji = "✅" if pct_change > 0 else "❌"
            elif category == "DUMP":
                emoji = "✅" if pct_change < 0 else "❌"
            
            # Get emoji for category
            category_emoji = "🚀" if category == "PUMP" else "📉" if category == "DUMP" else "➖"
            message = f"⏱️ <b>{label} Follow-up: {category_emoji} {category} ({symbol.upper()})</b>\n\n"
            message += f"Trigger Price: ${trigger_price:.2f}\n"
            message += f"Current Price: ${current_price:.2f}\n"
            message += f"Change: {pct_change:.2%} {emoji}\n"
            
            if chart_bytes:
                self.send_telegram_photo(chart_bytes, message)
            else:
                self.send_telegram_message(message)
                
        except Exception as e:
            logger.error(f"Error in follow-up for {symbol}: {e}")
        finally:
            # Reset flag if 2h
            if label == "2h":
                with self.lock:
                    self.active_events[symbol][category] = False
                    logger.info(f"Reset active event for {category} on {symbol}")

    def check_and_send_notifications(self, symbol, timestamp, price, delta, direction, confidence, probs):
        """Check if notifications should be sent based on threshold crossing or hourly schedule"""
        current_time = datetime.now()
        
        # Note: Prediction data is now stored in on_message method before calling this function
        
        # Check for 60% threshold crossing from below for each category
        threshold = 0.60
        current_probs = {
            'NEUTRAL': probs[0],
            'PUMP': probs[1],
            'DUMP': probs[2]
        }
        
        # Track which symbols crossed thresholds in this call
        symbols_with_alerts = []
        
        for category in ['NEUTRAL', 'PUMP', 'DUMP']:
            prev_prob = self.previous_probs[symbol][category]
            curr_prob = current_probs[category]
            
            # Check if crossed 60% threshold from below
            if prev_prob < threshold <= curr_prob:
                # Check if event is already active
                if self.active_events[symbol][category]:
                    logger.info(f"Skipping {category} alert for {symbol} - Timer already running.")
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
                
                # Track this alert
                symbols_with_alerts.append((symbol, category))
        
        # Batch alerts instead of sending immediately
        if symbols_with_alerts:
            with self.lock:
                # Add alerts to pending list
                self.pending_alerts.extend(symbols_with_alerts)
                
                # Cancel existing timer if any
                if self.alert_batch_timer is not None:
                    self.alert_batch_timer.cancel()
                
                # Start new timer to send batched alerts after delay
                self.alert_batch_timer = threading.Timer(self.alert_batch_delay, self.send_batched_alerts)
                self.alert_batch_timer.start()
        
        # Update previous probabilities for this symbol
        self.previous_probs[symbol] = current_probs.copy()
        
        # Check for hourly update
        if self.last_hourly_message_time is None:
            self.last_hourly_message_time = current_time
        
        time_since_last_hourly = (current_time - self.last_hourly_message_time).total_seconds()
        if time_since_last_hourly >= 3600 and self.send_hourly_message:  # 3600 seconds = 1 hour
            self.send_status_message("📊 <b>Hourly Update - All Symbols</b>")
            self.last_hourly_message_time = current_time
        
        # Check for nightly chart generation (once per day at night)
        # Trigger if: 
        # 1. Never sent before (last_nightly_chart_time is None), OR
        # 2. More than 24 hours have passed since last nightly chart
        # 3. Current time is between midnight and 1 AM (0:00 - 1:00)
        if self.last_nightly_chart_time is None:
            # First time - check if it's nighttime (between 0:00 and 1:00)
            if 0 <= current_time.hour < 1:
                logger.info("First nightly chart - triggering generation...")
                threading.Thread(target=self.generate_nightly_charts).start()
        else:
            time_since_last_nightly = (current_time - self.last_nightly_chart_time).total_seconds()
            # If more than 24 hours and it's nighttime (0:00 - 1:00)
            if time_since_last_nightly >= 86400 and 0 <= current_time.hour < 1:
                logger.info("24 hours passed and it's nighttime - triggering nightly chart generation...")
                threading.Thread(target=self.generate_nightly_charts).start()

    def generate_nightly_charts(self):
        """Generate and send 1-hour timeframe charts for all symbols covering the last 24 hours"""
        try:
            logger.info("Generating nightly charts for all symbols...")
            
            for symbol in self.symbols:
                with self.lock:
                    if symbol not in self.latest_data or not self.latest_data[symbol]:
                        logger.warning(f"No data available for {symbol} to generate nightly chart.")
                        continue
                    # Copy data to avoid holding lock during processing
                    data_copy = list(self.latest_data[symbol])
                
                # Process data (unlocked)
                df_all = pd.DataFrame(data_copy)
                if 'start_ts' not in df_all.columns:
                    logger.warning(f"Missing start_ts column for {symbol}.")
                    continue
                
                # Filter to ensure only this symbol's data
                if 'symbol' in df_all.columns:
                    df_all = df_all[df_all['symbol'].str.lower() == symbol.lower()].copy()
                
                # Handle mixed types (strings from WS, datetimes from history)
                df_all['start_ts'] = pd.to_datetime(df_all['start_ts'])
                df_all.set_index('start_ts', inplace=True)
                
                # Resample to 1-hour candles
                df_1h = df_all.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'total_volume': 'sum'
                }).dropna()
                
                if df_1h.empty:
                    logger.warning(f"No 1h candle data available for {symbol}.")
                    continue
                
                # Get last 24 hours of data (24 candles)
                df_chart = df_1h.tail(24).copy()
                
                if len(df_chart) < 5:
                    logger.warning(f"Not enough 1h candles for {symbol}. Have {len(df_chart)}, need at least 5.")
                    continue
                
                # Reset index for chart generation
                df_chart_reset = df_chart.reset_index()
                
                # Generate chart with modified title for 1h timeframe
                logger.info(f"Generating nightly 1h chart for {symbol} with {len(df_chart)} candles...")
                
                # Prepare data for mplfinance
                chart_df = df_chart_reset[['start_ts', 'open', 'high', 'low', 'close', 'total_volume']].copy()
                chart_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                chart_df.set_index('Date', inplace=True)
                
                # Create the chart with 1h title
                try:
                    fig, axes = mpf.plot(
                        chart_df,
                        type='candle',
                        style='charles',
                        title=f'{symbol.upper()} Price Chart - Last 24 Hours (1h candles)',
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
                    
                    chart_bytes = buf.getvalue()
                    
                    if chart_bytes:
                        caption = f"🌙 <b>Nightly Chart: {symbol.upper()}</b>\n📊 1-hour timeframe - Last 24 hours ({len(df_chart)} candles)"
                        self.send_telegram_photo(chart_bytes, caption)
                        logger.info(f"Nightly chart sent for {symbol}.")
                    
                except Exception as e:
                    logger.error(f"Error generating nightly chart for {symbol}: {e}")
                    
            # Update last nightly chart time
            self.last_nightly_chart_time = datetime.now()
            logger.info("Nightly chart generation completed.")
            
        except Exception as e:
            logger.error(f"Error in generate_nightly_charts: {e}")

    def fetch_historical_data(self, symbols, days=3):
        logger.info(f"Fetching data for symbols {symbols} for the last {days} days in 3 requests...")
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
                logger.info(f"Request {i+1}/3: Downloading data from {params['start']} to {params['end']}...")
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
                            logger.error(f"JSON decode error: {e}")
                            continue
                        
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
            
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['start_ts'] = pd.to_datetime(df['start_ts'], format='ISO8601')
            df = df.sort_values('start_ts').reset_index(drop=True)
            logger.info(f"Total records fetched: {len(df)}")
        else:
            logger.warning("No data fetched.")
            
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
                logger.info(f"Model for {symbol} found but too old ({file_age.days} days). Retraining...")
                return False
                
            logger.info(f"Loading saved model for {symbol} (Age: {file_age})...")
            self.models[symbol] = lgb.Booster(model_file=model_file)
            return True
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return False

    def train_model(self, symbol, df):
        logger.info(f"Training model for {symbol}...")
        df = self.prepare_features(df)
        
        # Drop last rows with NaN targets
        df = df.dropna(subset=['target', 'future_return'] + self.feature_cols)
        
        if len(df) < 10:
            logger.warning(f"Not enough data after feature engineering to train model for {symbol}. Rows: {len(df)}")
            return None

        X = df[self.feature_cols]
        y = df['target']
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        except ValueError as e:
            logger.error(f"Error splitting data for {symbol}: {e}")
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
        logger.info(f"Model evaluation:\n{classification_report(y_test, y_pred_class, zero_division=0)}")
        
        # Save model
        model_file = self.get_model_file(symbol)
        try:
            model.save_model(model_file)
            logger.info(f"Model for {symbol} saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
        
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
                
                # Store the current prediction values for this symbol
                self.last_prediction_data[symbol] = {
                    'timestamp': timestamp,
                    'price': price,
                    'delta': last_row['delta'],
                    'direction': direction,
                    'confidence': confidence,
                    'probs': {
                        'NEUTRAL': float(probs[0]),
                        'PUMP': float(probs[1]),
                        'DUMP': float(probs[2])
                    },
                    'pred_class': int(pred_class)
                }

                # Display all prediction results with full probability distribution
                logger.info(
                    f"{symbol.upper()} | Price: ${price:.2f} | OFI: {last_row['delta']:.2f} | Pred: {direction} ({confidence:.2%}) | NEUTRAL: {probs[0]:.2%} | PUMP: {probs[1]:.2%} | DUMP: {probs[2]:.2%}")

                # Check if we should send initial status message
                if not self.initial_message_sent:
                    # Check if we have data for all symbols
                    if all(self.last_prediction_data.get(s) for s in self.symbols):
                        logger.info("Sending initial status message...")
                        self.send_status_message("🚀 <b>Bot Started - Initial Status</b>")
                        self.initial_message_sent = True

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
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket Closed. Code: {close_status_code}, Msg: {close_msg}")

    def on_open(self, ws):
        logger.info("WebSocket Connection Opened")

    def start_live_prediction(self, historical_data=None):
        # Check if at least one model is trained
        if not self.models or all(model is None for model in self.models.values()):
            logger.error("No models trained! Cannot start live prediction.")
            return

        logger.info("Starting live prediction...")
        
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
                    self.latest_data[symbol] = records[-20000:]
                    logger.info(f"Pre-filled buffer for {symbol} with {len(self.latest_data[symbol])} records from history.")
                    
                    # Run initial prediction on historical data to populate status
                    if len(self.latest_data[symbol]) >= 20 and symbol in self.models and self.models[symbol]:
                        try:
                            df = pd.DataFrame(self.latest_data[symbol])
                            df = self.prepare_features(df)
                            last_row = df.iloc[-1]
                            
                            features = last_row[self.feature_cols].values.reshape(1, -1)
                            prediction = self.models[symbol].predict(features)
                            pred_class = np.argmax(prediction[0])
                            probs = prediction[0]
                            
                            direction_map = {0: "NEUTRAL", 1: "PUMP", 2: "DUMP"}
                            
                            self.last_prediction_data[symbol] = {
                                'timestamp': last_row['start_ts'],
                                'price': last_row['close'],
                                'delta': last_row['delta'],
                                'direction': direction_map[pred_class],
                                'confidence': probs[pred_class],
                                'probs': {
                                    'NEUTRAL': float(probs[0]),
                                    'PUMP': float(probs[1]),
                                    'DUMP': float(probs[2])
                                },
                                'pred_class': int(pred_class)
                            }
                        except Exception as e:
                            logger.error(f"Error running initial prediction for {symbol}: {e}")

        # Check if we can send initial status message immediately
        if not self.initial_message_sent:
            if all(self.last_prediction_data.get(s) for s in self.symbols):
                logger.info("Sending initial status message from historical data...")
                self.send_status_message("🚀 <b>Bot Started - Initial Status</b>")
                self.initial_message_sent = True
        
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
    logger.info(f"Initializing Market Predictor for symbols: {SYMBOLS}")
    predictor = MarketPredictor(SYMBOLS)
    
    # 1. Check for existing models and load them
    models_loaded = {}
    for symbol in SYMBOLS:
        models_loaded[symbol] = predictor.load_model_if_valid(symbol, max_age_days=2)
        if models_loaded[symbol]:
            logger.info(f"✓ Model loaded for {symbol}")
        else:
            logger.info(f"✗ Model not found or too old for {symbol}")
    
    # 2. Fetch Data
    # If all models are loaded, we just need enough data for the buffer (e.g. 2 days)
    # If we need to train any model, we fetch more (e.g. 5 days)
    all_models_loaded = all(models_loaded.values())
    days_to_fetch = 2 if all_models_loaded else 5
    
    logger.info(f"Fetching {days_to_fetch} days of historical data...")
    df = predictor.fetch_historical_data(SYMBOLS, days=days_to_fetch)
    
    if not df.empty and len(df) > 10:
        # 3. Train Models if needed (per symbol)
        for symbol in SYMBOLS:
            if not models_loaded[symbol]:
                logger.info(f"Training model for {symbol.upper()}...")
                # Filter data for this symbol
                symbol_df = df[df['symbol'] == symbol.upper()].copy()
                if len(symbol_df) > 10:
                    predictor.train_model(symbol.upper(), symbol_df)
                    logger.info(f"✓ Model trained for {symbol.upper()}")
                else:
                    logger.warning(f"✗ Not enough data for {symbol.upper()}. Skipping training.")
        
        # 4. Live Predict
        try:
            predictor.start_live_prediction(df)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1) # Exit with error code so systemd knows it failed
    else:
        logger.warning(f"Not enough data. Fetched {len(df)} records.")
