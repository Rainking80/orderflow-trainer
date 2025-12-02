import pandas as pd
import requests
import json
import lightgbm as lgb
import numpy as np
import websocket
import threading
import asyncio
import websockets
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
import tensorflow as tf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

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
        self.lstm_models = {} # symbol -> lstm model
        self.lstm_scalers = {} # symbol -> scaler
        self.lstm_prediction_counters = {} # symbol -> int
        self.last_lstm_predictions = {} # symbol -> dict
        self.feature_cols = [
            'delta', 'delta_pct', 'total_volume', 'trade_count', 
            'buy_volume', 'sell_volume', 'vwap', 'price_range',
            'large_trade_dominance', 'ratio_buy_sell_trades',
            'rsi', 'volatility', 'return',
            'ofi_sum_10s', 'ofi_sum_60s', 'ofi_sum_300s',
            'ofi_lag_1', 'ofi_lag_2', 'ofi_lag_3', 'ofi_lag_5'
        ]
        self.running = False
        self.latest_data = {}  # symbol -> list of data
        self.lock = threading.RLock()

        self.send_hourly_message = True

        self.PROB_THRESHOLD = 40.0  # If Pump/Dump > 40%, pay attention (even if Neutral is 50%)
        self.OFI_THRESHOLD = 0.5  # Minimum OFI strength to confirm the move
        self.CONSECUTIVE_SIGNAL_THRESHOLD = 3  # Number of consecutive signals required to trigger
        
        # Telegram notification tracking per symbol
        self.previous_probs = {}  # symbol -> {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
        self.last_hourly_message_time = None  # Global timestamp for hourly updates
        self.last_prediction_data = {}  # symbol -> prediction dict
        self.active_events = {}  # symbol -> {'NEUTRAL': False, 'PUMP': False, 'DUMP': False}
        self.signal_counters = {}  # symbol -> {'type': 'WAIT', 'count': 0}
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
            self.signal_counters[symbol] = {'type': 'WAIT', 'count': 0}
            self.lstm_prediction_counters[symbol] = 0
            self.last_lstm_predictions[symbol] = {'direction': 'N/A', 'probs': [0.0, 0.0, 0.0], 'pred_class': -1}
        
        self.market_regime = {} # Stores global market state (season, etc.)
        self.prediction_history = {s: [] for s in self.symbols} # Stores last N predictions for averaging
        self.initial_message_sent = False
        self.ENABLE_LSTM = True # Set to True to enable LSTM training
        self.ws = None
        
        # Hedge Fund WebSocket Server
        self.server_clients = set()
        self.server_loop = None
        self.server_thread = None

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
            
        # Send title as separate message
        self.send_telegram_message(f"<b>{title}</b>")
        
        # Iterate over ALL symbols to build the message
        for sym in self.symbols:
            pred_data = self.last_prediction_data.get(sym)
            if pred_data:
                sym_msg = ""
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
                    sym_msg += f"⚠️ <b><u>{sym.upper()}</u> (ALERT: {', '.join(crossed)})</b>\n"
                else:
                    sym_msg += f"<b><u>{sym.upper()}</u></b>\n"

                sym_msg += f"Price: <code>${pred_data['price']:.2f}</code>\n"
                sym_msg += f"OFI: <code>{pred_data['delta']:.2f}</code>\n"
                sym_msg += f"<b>LGBM:</b> {direction_emoji} <b>{pred_data['direction']}</b> ({pred_data['confidence']:.2%})\n"
                sym_msg += f"• NEUTRAL: <code>{pred_data['probs']['NEUTRAL']:.2%}</code>\n"
                sym_msg += f"• PUMP: <code>{pred_data['probs']['PUMP']:.2%}</code>\n"
                sym_msg += f"• DUMP: <code>{pred_data['probs']['DUMP']:.2%}</code>\n"

                if pred_data.get('lstm_direction') and pred_data['lstm_direction'] != "N/A":
                    l_emoji = ""
                    if pred_data['lstm_direction'] == "PUMP":
                        l_emoji = "🚀"
                    elif pred_data['lstm_direction'] == "DUMP":
                        l_emoji = "📉"
                    elif pred_data['lstm_direction'] == "NEUTRAL":
                        l_emoji = "➖"
                    sym_msg += f"\n<b>LSTM:</b> {l_emoji} <b>{pred_data['lstm_direction']}</b> ({pred_data['lstm_confidence']:.2%})\n"

                    # Add LSTM probabilities if available
                    if 'lstm_probs' in pred_data:
                        lp = pred_data['lstm_probs']
                        sym_msg += f"• NEUTRAL: <code>{lp['NEUTRAL']:.2%}</code>\n"
                        sym_msg += f"• PUMP: <code>{lp['PUMP']:.2%}</code>\n"
                        sym_msg += f"• DUMP: <code>{lp['DUMP']:.2%}</code>\n"
                
                # Generate chart for THIS symbol
                chart_bytes = self.generate_latest_chart(sym)
                
                if chart_bytes:
                    self.send_telegram_photo(chart_bytes, sym_msg)
                else:
                    self.send_telegram_message(sym_msg)

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
        
        current_probs = {
            'NEUTRAL': probs[0],
            'PUMP': probs[1],
            'DUMP': probs[2]
        }
        
        # Track which symbols crossed thresholds in this call
        symbols_with_alerts = []
        
        # Check for triggers using new logic
        # Convert probs to percentage for threshold check
        prob_pump_pct = probs[1] * 100
        prob_dump_pct = probs[2] * 100
        
        trigger = self.check_for_trigger(symbol, prob_pump_pct, prob_dump_pct, delta)
        
        category = None
        if trigger == "LONG_ENTRY":
            category = "PUMP"
        elif trigger == "SHORT_ENTRY":
            category = "DUMP"
            
        if category:
            # Check if event is already active
            if self.active_events[symbol][category]:
                logger.info(f"Skipping {category} alert for {symbol} - Timer already running.")
            else:
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
        
        # Ensure start_ts is datetime for time-based rolling
        if 'start_ts' in df.columns:
            df['start_ts'] = pd.to_datetime(df['start_ts'], format='mixed')
            
            # Set index to start_ts for time-based rolling, but keep it as column
            # We sort just in case, though data should be sorted
            df = df.sort_values('start_ts')
            temp_index = pd.DatetimeIndex(df['start_ts'])
            
            # A. Rolling Sum (Accumulation)
            # Sum of OFI (delta) over last N seconds
            # We use the temp_index for rolling
            df['ofi_sum_10s'] = df.set_index(temp_index)['delta'].rolling('10s').sum().values
            df['ofi_sum_60s'] = df.set_index(temp_index)['delta'].rolling('60s').sum().values
            df['ofi_sum_300s'] = df.set_index(temp_index)['delta'].rolling('300s').sum().values
            
            # B. Lag Features (The "Shape" of the move)
            # Previous values of OFI
            df['ofi_lag_1'] = df['delta'].shift(1)
            df['ofi_lag_2'] = df['delta'].shift(2)
            df['ofi_lag_3'] = df['delta'].shift(3)
            df['ofi_lag_5'] = df['delta'].shift(5)
        else:
            # Fallback if no start_ts (should not happen in main flow)
            for col in ['ofi_sum_10s', 'ofi_sum_60s', 'ofi_sum_300s', 'ofi_lag_1', 'ofi_lag_2', 'ofi_lag_3', 'ofi_lag_5']:
                df[col] = 0

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
        return f"lgb_model_{symbol}_v2.txt"
    
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

    def get_lstm_model_file(self, symbol):
        return f"lstm_model_{symbol}.keras"

    def get_lstm_scaler_file(self, symbol):
        return f"lstm_scaler_{symbol}.pkl"

    def load_lstm_model_if_valid(self, symbol, max_age_days=3):
        """Load LSTM model and scaler if valid"""
        if not self.ENABLE_LSTM:
            return False
            
        model_file = self.get_lstm_model_file(symbol)
        scaler_file = self.get_lstm_scaler_file(symbol)
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            return False
            
        try:
            # Check file age
            file_time = os.path.getmtime(model_file)
            file_age = datetime.now() - datetime.fromtimestamp(file_time)
            
            if file_age.days >= max_age_days:
                logger.info(f"LSTM Model for {symbol} found but too old ({file_age.days} days). Retraining...")
                return False
                
            logger.info(f"Loading saved LSTM model for {symbol} (Age: {file_age})...")
            self.lstm_models[symbol] = tf.keras.models.load_model(model_file)
            self.lstm_scalers[symbol] = joblib.load(scaler_file)
            return True
        except Exception as e:
            logger.error(f"Error loading LSTM model for {symbol}: {e}")
            return False

    def train_lstm_model(self, symbol, df):
        """Train an LSTM model for the given symbol"""
        if not self.ENABLE_LSTM:
            return

        logger.info(f"Training LSTM model for {symbol}...")
        
        try:
            # Prepare data
            df = self.prepare_features(df)
            df = df.dropna(subset=['target'] + self.feature_cols)
            
            if len(df) < 200:
                logger.warning(f"Not enough data for LSTM training for {symbol}. Need at least 200 rows.")
                return

            # Scale features
            scaler = MinMaxScaler()
            feature_data = df[self.feature_cols].values
            scaled_data = scaler.fit_transform(feature_data)
            
            # Save scaler
            scaler_path = self.get_lstm_scaler_file(symbol)
            joblib.dump(scaler, scaler_path)
            self.lstm_scalers[symbol] = scaler
            
            # Prepare sequences
            SEQ_LEN = 200
            X = []
            y = []
            
            targets = df['target'].values
            
            for i in range(SEQ_LEN, len(scaled_data)):
                X.append(scaled_data[i-SEQ_LEN:i])
                y.append(targets[i])
                
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Train
            model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test), verbose=1)
            
            # Save
            model_path = self.get_lstm_model_file(symbol)
            model.save(model_path)
            self.lstm_models[symbol] = model
            logger.info(f"LSTM model for {symbol} saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training LSTM for {symbol}: {e}")

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
                
            if len(self.latest_data[symbol]) >= 200: # Need enough data for rolling features and LSTM
                # Optimize: Only take last 300 rows for feature calculation to avoid processing 20k rows every ms
                # We need at least 200 for LSTM + some buffer for rolling window calculations
                recent_data = self.latest_data[symbol][-300:]
                df = pd.DataFrame(recent_data)
                df = self.prepare_features(df) # This adds features to all rows
                
                # Get last row
                last_row = df.iloc[-1]
                
                # Check if model exists for this symbol
                if symbol not in self.models or self.models[symbol] is None:
                    return
                
                # Predict (LightGBM)
                features = last_row[self.feature_cols].values.reshape(1, -1)
                prediction = self.models[symbol].predict(features)
                pred_class = np.argmax(prediction[0])
                probs = prediction[0]
                
                # Predict (LSTM) if available
                lstm_pred_class = -1
                lstm_probs = [0.0, 0.0, 0.0]
                lstm_direction = "N/A"
                
                # Throttle LSTM: Run only every 10th message
                self.lstm_prediction_counters[symbol] += 1
                should_run_lstm = (self.lstm_prediction_counters[symbol] % 10 == 0)
                
                if self.ENABLE_LSTM and symbol in self.lstm_models and symbol in self.lstm_scalers:
                    if should_run_lstm:
                        try:
                            # Need last 200 rows for LSTM
                            if len(df) >= 200:
                                lstm_input_df = df.iloc[-200:]
                                lstm_features = lstm_input_df[self.feature_cols].values
                                # Scale
                                lstm_scaled = self.lstm_scalers[symbol].transform(lstm_features)
                                # Reshape to (1, 200, n_features)
                                lstm_input = lstm_scaled.reshape(1, 200, lstm_scaled.shape[1])
                                
                                lstm_prediction = self.lstm_models[symbol].predict(lstm_input, verbose=0)
                                lstm_pred_class = np.argmax(lstm_prediction[0])
                                lstm_probs = lstm_prediction[0]
                                
                                direction_map = {0: "NEUTRAL", 1: "PUMP", 2: "DUMP"}
                                lstm_direction = direction_map[lstm_pred_class]
                                
                                # Update cache
                                self.last_lstm_predictions[symbol] = {
                                    'direction': lstm_direction,
                                    'probs': lstm_probs,
                                    'pred_class': lstm_pred_class
                                }
                        except Exception as e:
                            logger.error(f"LSTM prediction error for {symbol}: {e}")
                    
                    # Use cached/latest values
                    cached = self.last_lstm_predictions[symbol]
                    lstm_direction = cached['direction']
                    lstm_probs = cached['probs']
                    lstm_pred_class = cached['pred_class']
                
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
                    'pred_class': int(pred_class),
                    'lstm_direction': lstm_direction,
                    'lstm_confidence': float(lstm_probs[lstm_pred_class]) if lstm_direction != "N/A" else 0.0,
                    'lstm_probs': {
                        'NEUTRAL': float(lstm_probs[0]),
                        'PUMP': float(lstm_probs[1]),
                        'DUMP': float(lstm_probs[2])
                        } if lstm_direction != "N/A" else {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
                }

                # Add to history for averaging (keep last 20)
                self.prediction_history[symbol].append(self.last_prediction_data[symbol])
                if len(self.prediction_history[symbol]) > 20:
                    self.prediction_history[symbol].pop(0)

                # Display all prediction results with full probability distribution
                lstm_info = ""
                if lstm_direction != "N/A":
                    lstm_info = f" | LSTM: {lstm_direction} ({lstm_probs[lstm_pred_class]:.2%}) [N:{lstm_probs[0]:.0%} P:{lstm_probs[1]:.0%} D:{lstm_probs[2]:.0%}]"
                    
                logger.info(
                    f"{symbol.upper()} | Price: ${price:.2f} | OFI: {last_row['delta']:.2f} | LGBM: {direction} ({confidence:.2%}){lstm_info} | NEUTRAL: {probs[0]:.2%} | PUMP: {probs[1]:.2%} | DUMP: {probs[2]:.2%}")

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
                
                # Update Market Regime Analysis
                self.analyze_market_regime()

    def analyze_market_regime(self):
        """
        Analyze the current market regime (BTC Season vs Alt Season) based on all symbols.
        Stores the result in self.market_regime.
        """
        # Need data for all symbols to make a global decision
        if not all(self.last_prediction_data.get(s) for s in self.symbols):
            return

        btc_data = self.last_prediction_data.get('btcusdt')
        eth_data = self.last_prediction_data.get('ethusdt')
        sol_data = self.last_prediction_data.get('solusdt')
        
        # If we don't have specific symbols, try to infer from what we have
        # Assuming 'btcusdt' is always present as the anchor
        if not btc_data:
            return

        alts_data = [self.last_prediction_data[s] for s in self.symbols if s != 'btcusdt']
        
        if not alts_data:
            self.market_regime = {"season": "BTC_ONLY", "confidence": 0.0, "details": "Only BTC data available"}
            return

        # Metrics
        avg_alt_ofi = np.mean([d['delta'] for d in alts_data])
        
        # --- NEW LOGIC: Use Average of Last 20 Predictions ---
        # Instead of just current snapshot, let's calculate the average probabilities over the last 20 ticks
        # This smooths out noise and gives a more robust "regime" signal.
        
        # Helper to get average metrics for a symbol from history
        def get_avg_metrics(sym):
            history = self.prediction_history.get(sym, [])
            if not history:
                return None
            
            # Calculate means
            mean_pump = np.mean([item['probs']['PUMP'] for item in history])
            mean_dump = np.mean([item['probs']['DUMP'] for item in history])
            mean_ofi = np.mean([item['delta'] for item in history])
            
            # Calculate LSTM means (if available)
            lstm_pumps = [item.get('lstm_probs', {}).get('PUMP', 0.0) for item in history if item.get('lstm_direction', 'N/A') != 'N/A']
            lstm_dumps = [item.get('lstm_probs', {}).get('DUMP', 0.0) for item in history if item.get('lstm_direction', 'N/A') != 'N/A']
            
            mean_lstm_pump = np.mean(lstm_pumps) if lstm_pumps else 0.0
            mean_lstm_dump = np.mean(lstm_dumps) if lstm_dumps else 0.0
            
            return {
                'pump': mean_pump,
                'dump': mean_dump,
                'ofi': mean_ofi,
                'lstm_pump': mean_lstm_pump,
                'lstm_dump': mean_lstm_dump
            }

        # Initialize variables
        btc_pump_prob = 0.0
        btc_dump_prob = 0.0
        btc_ofi = 0.0
        
        avg_alt_pump_prob = 0.0
        avg_alt_dump_prob = 0.0
        avg_alt_ofi = 0.0

        # Calculate averages for BTC
        btc_avgs = get_avg_metrics('btcusdt')
        if btc_avgs:
            btc_pump_prob = btc_avgs['pump']
            btc_dump_prob = btc_avgs['dump']
            btc_ofi = btc_avgs['ofi']
        
        # Calculate averages for Alts (aggregate of all alt histories)
        alt_pump_probs = []
        alt_dump_probs = []
        alt_ofis = []
        
        for s in self.symbols:
            if s != 'btcusdt':
                avgs = get_avg_metrics(s)
                if avgs:
                    alt_pump_probs.append(avgs['pump'])
                    alt_dump_probs.append(avgs['dump'])
                    alt_ofis.append(avgs['ofi'])
        
        if alt_pump_probs:
            avg_alt_pump_prob = np.mean(alt_pump_probs)
            avg_alt_dump_prob = np.mean(alt_dump_probs)
            avg_alt_ofi = np.mean(alt_ofis)
        
        # --- BTC DOMINANCE PROXY ---
        # Calculate relative strength of BTC vs Alts
        # Positive = BTC Stronger (Dominance Rising)
        # Negative = Alts Stronger (Dominance Falling)
        btc_dominance_proxy = (btc_pump_prob - btc_dump_prob) - (avg_alt_pump_prob - avg_alt_dump_prob)
        
        # Determine Season
        season = "NEUTRAL"
        confidence = 0.0
        details = ""
        
        # 1. BEAR MARKET: Everything is dumping
        if btc_dump_prob > 0.5 and avg_alt_dump_prob > 0.5:
            season = "BEAR_MARKET"
            confidence = (btc_dump_prob + avg_alt_dump_prob) / 2
            details = "Global sell-off detected."
            
        # 2. BULL MARKET: Everything is pumping
        elif btc_pump_prob > 0.5 and avg_alt_pump_prob > 0.5:
            season = "BULL_MARKET"
            confidence = (btc_pump_prob + avg_alt_pump_prob) / 2
            details = "Global rally detected."
            
        # 3. BTC SEASON (Dominance Rising)
        elif btc_dominance_proxy > 0.15:
            season = "BTC_SEASON"
            confidence = abs(btc_dominance_proxy)
            details = f"Money flowing into BTC (Dom Proxy: {btc_dominance_proxy:.2f})"
            
        # 4. ALT SEASON (Dominance Falling)
        elif btc_dominance_proxy < -0.15:
            season = "ALT_SEASON"
            confidence = abs(btc_dominance_proxy)
            details = f"Money flowing into Alts (Dom Proxy: {btc_dominance_proxy:.2f})"
            
        # 5. DIVERGENCE: BTC Dumping, Alts Pumping (Rare/Risky)
        elif btc_dump_prob > 0.6 and avg_alt_pump_prob > 0.5:
            season = "DIVERGENCE_RISK"
            confidence = (btc_dump_prob + avg_alt_pump_prob) / 2
            details = "BTC dumping while Alts pump. High risk."
            
        # --- HEDGE FUND DATA BROADCAST ---
        # Calculate scores for the hedge bot
        
        # 1. Market Regime Score (-1.0 to 1.0)
        # Bullish = +1, Bearish = -1
        # We use the aggregate pump - dump probability
        # BTC weight 50%, Alts weight 50%
        btc_net_score = btc_pump_prob - btc_dump_prob
        alt_net_score = avg_alt_pump_prob - avg_alt_dump_prob
        market_regime_score = (btc_net_score + alt_net_score) / 2
        
        # 2. Spread Matrix (Relative Strength)
        # We need individual strengths
        # If we have specific alts, calculate their specific scores
        eth_avgs = get_avg_metrics('ethusdt')
        sol_avgs = get_avg_metrics('solusdt')
        
        # Helper to calculate composite score (LGBM + LSTM)
        def calc_composite_score(avgs):
            if not avgs: return 0.0
            lgbm_score = avgs['pump'] - avgs['dump']
            lstm_score = avgs['lstm_pump'] - avgs['lstm_dump']
            # Weight: 70% LGBM, 30% LSTM (since LSTM is throttled/slower)
            if lstm_score != 0:
                return (lgbm_score * 0.7) + (lstm_score * 0.3)
            return lgbm_score
            
        def get_direction_label(score):
            if score > 0.05: return "PUMP"
            if score < -0.05: return "DUMP"
            return "NEUTRAL"
        
        btc_comp_score = calc_composite_score(btc_avgs)
        eth_comp_score = calc_composite_score(eth_avgs)
        sol_comp_score = calc_composite_score(sol_avgs)
        
        # Calculate raw sub-scores for labeling
        btc_lgbm = btc_pump_prob - btc_dump_prob
        btc_lstm = (btc_avgs['lstm_pump'] - btc_avgs['lstm_dump']) if btc_avgs else 0.0
        
        eth_lgbm = (eth_avgs['pump'] - eth_avgs['dump']) if eth_avgs else 0.0
        eth_lstm = (eth_avgs['lstm_pump'] - eth_avgs['lstm_dump']) if eth_avgs else 0.0
        
        sol_lgbm = (sol_avgs['pump'] - sol_avgs['dump']) if sol_avgs else 0.0
        sol_lstm = (sol_avgs['lstm_pump'] - sol_avgs['lstm_dump']) if sol_avgs else 0.0
        
        spread_matrix = {
            "btc_vs_eth": btc_comp_score - eth_comp_score, # >0 means BTC stronger
            "btc_vs_sol": btc_comp_score - sol_comp_score,
            "eth_vs_sol": eth_comp_score - sol_comp_score
        }
        
        hedge_payload = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": {
                "season": season,
                "score": float(market_regime_score), # -1.0 (Bear) to 1.0 (Bull)
                "confidence": float(confidence),
                "details": details
            },
            "btc_dominance": {
                "score": float(btc_dominance_proxy), # >0 Rising, <0 Falling
                "trend": "RISING" if btc_dominance_proxy > 0.05 else "FALLING" if btc_dominance_proxy < -0.05 else "NEUTRAL"
            },
            "spread_matrix": spread_matrix,
            "assets": {
                "btc": {
                    "score": float(btc_comp_score),
                    "direction": get_direction_label(btc_comp_score),
                    "lgbm_score": float(btc_lgbm),
                    "lgbm_direction": get_direction_label(btc_lgbm),
                    "lstm_score": float(btc_lstm),
                    "lstm_direction": get_direction_label(btc_lstm),
                    "ofi": float(btc_ofi)
                },
                "eth": {
                    "score": float(eth_comp_score),
                    "direction": get_direction_label(eth_comp_score),
                    "lgbm_score": float(eth_lgbm),
                    "lgbm_direction": get_direction_label(eth_lgbm),
                    "lstm_score": float(eth_lstm),
                    "lstm_direction": get_direction_label(eth_lstm),
                    "ofi": float(eth_avgs['ofi'])
                } if eth_avgs else {},
                "sol": {
                    "score": float(sol_comp_score),
                    "direction": get_direction_label(sol_comp_score),
                    "lgbm_score": float(sol_lgbm),
                    "lgbm_direction": get_direction_label(sol_lgbm),
                    "lstm_score": float(sol_lstm),
                    "lstm_direction": get_direction_label(sol_lstm),
                    "ofi": float(sol_avgs['ofi'])
                } if sol_avgs else {}
            }
        }
        
        # Broadcast to WebSocket clients
        self.broadcast_hedge_data(hedge_payload)
            
        # Check for change
        previous_season = self.market_regime.get("season", "NEUTRAL")
            
        # Check for change
        previous_season = self.market_regime.get("season", "NEUTRAL")
        
        # Store state
        self.market_regime = {
            "season": season,
            "confidence": confidence,
            "details": details,
            "metrics": {
                "btc_pump": btc_pump_prob,
                "alt_pump": avg_alt_pump_prob,
                "btc_ofi": btc_ofi,
                "alt_ofi": avg_alt_ofi
            },
            "timestamp": datetime.now()
        }
        
        # Log significant changes
        if season != previous_season:
            logger.info(f"Market Regime Change: {previous_season} -> {season} ({confidence:.2%}) - {details}")
            # Send Telegram notification for season change
            msg = f"🔄 <b>Market Regime Change</b>\n\n"
            msg += f"Old: {previous_season}\n"
            msg += f"New: <b>{season}</b>\n"
            msg += f"Confidence: {confidence:.2%}\n"
            msg += f"Details: {details}\n"
            self.send_telegram_message(msg)
        
        # Check for midnight report (once per day)
        current_time = datetime.now()
        last_midnight = self.market_regime.get("last_midnight_report")
        
        # If it's between 00:00 and 00:05 and we haven't sent it today
        if 0 <= current_time.hour < 1 and current_time.minute < 5:
            # Check if we already sent it today
            if last_midnight is None or last_midnight.day != current_time.day:
                logger.info("Sending midnight market regime report...")
                msg = f"🌙 <b>Daily Market Regime Report</b>\n\n"
                msg += f"Current Season: <b>{season}</b>\n"
                msg += f"Confidence: {confidence:.2%}\n"
                msg += f"Details: {details}\n"
                self.send_telegram_message(msg)
                
                # Update last report time in the dictionary (preserving other keys)
                self.market_regime["last_midnight_report"] = current_time

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
        
        # Start Hedge Bot Server
        self.server_thread = threading.Thread(target=self.start_websocket_server, daemon=True)
        self.server_thread.start()
        
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
                            
                            # Predict (LSTM) if available
                            lstm_pred_class = -1
                            lstm_probs = [0.0, 0.0, 0.0]
                            lstm_direction = "N/A"
                            
                            if self.ENABLE_LSTM and symbol in self.lstm_models and symbol in self.lstm_scalers:
                                try:
                                    # Need last 200 rows for LSTM
                                    if len(df) >= 200:
                                        lstm_input_df = df.iloc[-200:]
                                        lstm_features = lstm_input_df[self.feature_cols].values
                                        # Scale
                                        lstm_scaled = self.lstm_scalers[symbol].transform(lstm_features)
                                        # Reshape to (1, 200, n_features)
                                        lstm_input = lstm_scaled.reshape(1, 200, lstm_scaled.shape[1])
                                        
                                        lstm_prediction = self.lstm_models[symbol].predict(lstm_input, verbose=0)
                                        lstm_pred_class = np.argmax(lstm_prediction[0])
                                        lstm_probs = lstm_prediction[0]
                                        
                                        lstm_direction = direction_map[lstm_pred_class]
                                        
                                        # Update cache
                                        self.last_lstm_predictions[symbol] = {
                                            'direction': lstm_direction,
                                            'probs': lstm_probs,
                                            'pred_class': lstm_pred_class
                                        }
                                except Exception as e:
                                    logger.error(f"Initial LSTM prediction error for {symbol}: {e}")
                            
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
                                'pred_class': int(pred_class),
                                'lstm_direction': lstm_direction,
                                'lstm_confidence': float(lstm_probs[lstm_pred_class]) if lstm_direction != "N/A" else 0.0,
                                'lstm_probs': {
                                    'NEUTRAL': float(lstm_probs[0]),
                                    'PUMP': float(lstm_probs[1]),
                                    'DUMP': float(lstm_probs[2])
                                } if lstm_direction != "N/A" else {'NEUTRAL': 0.0, 'PUMP': 0.0, 'DUMP': 0.0}
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


    def check_for_trigger(self, symbol, prob_pump, prob_dump, ofi):
        # Determine current raw signal
        current_signal = "WAIT"

        # --- UP TREND TRIGGER (PUMP) ---
        # Logic: Model thinks Pump is likely (>40%) AND Order Flow confirms buying (>0.5)
        if prob_pump > self.PROB_THRESHOLD and ofi > self.OFI_THRESHOLD:
            current_signal = "LONG_ENTRY"

        # --- DOWN TREND TRIGGER (DUMP) ---
        # Logic: Model thinks Dump is likely (>40%) AND Order Flow confirms selling (<-0.5)
        elif prob_dump > self.PROB_THRESHOLD and ofi < -self.OFI_THRESHOLD:
            current_signal = "SHORT_ENTRY"

        # Get previous state
        if symbol not in self.signal_counters:
            self.signal_counters[symbol] = {'type': 'WAIT', 'count': 0}
            
        state = self.signal_counters[symbol]
        
        # Check for consecutive signals
        if current_signal == state['type'] and current_signal != "WAIT":
            state['count'] += 1
            # Cap the counter to prevent infinite growth since we only care about the threshold
            if state['count'] > self.CONSECUTIVE_SIGNAL_THRESHOLD + 10:
                state['count'] = self.CONSECUTIVE_SIGNAL_THRESHOLD + 1
        else:
            # Reset or start new count
            state['type'] = current_signal
            state['count'] = 1 if current_signal != "WAIT" else 0
            
        # Trigger ONLY on the exact Nth occurrence
        # This prevents endless messages when the trigger remains True for a long time
        if state['count'] == self.CONSECUTIVE_SIGNAL_THRESHOLD:
            logger.info(f"Trigger confirmed for {symbol}: {current_signal} (3 consecutive signals)")
            return current_signal
            
        return "WAIT"

    # --- WEBSOCKET SERVER FOR HEDGE BOT ---
    
    async def websocket_handler(self, websocket):
        """Handle incoming WebSocket connections from Hedge Bots"""
        self.server_clients.add(websocket)
        logger.info(f"Hedge Bot connected. Total clients: {len(self.server_clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.server_clients.remove(websocket)
            logger.info(f"Hedge Bot disconnected. Total clients: {len(self.server_clients)}")

    def start_websocket_server(self):
        """Start the asyncio WebSocket server in a separate thread"""
        logger.info("Starting Hedge Bot WebSocket Server on port 8765...")
        
        # Create a new event loop for this thread
        self.server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.server_loop)
        
        async def run_server():
            # websockets.serve returns a Server object which is an async context manager
            async with websockets.serve(self.websocket_handler, "0.0.0.0", 8765):
                logger.info("WebSocket Server running...")
                # Keep the server running indefinitely
                await asyncio.Future() 

        try:
            self.server_loop.run_until_complete(run_server())
        except Exception as e:
            logger.error(f"WebSocket Server crashed: {e}")
        finally:
            self.server_loop.close()

    def broadcast_hedge_data(self, data):
        """Broadcast data to all connected WebSocket clients (Thread-Safe)"""
        if not self.server_clients or not self.server_loop:
            return

        async def _send():
            if not self.server_clients:
                return
            message = json.dumps(data)
            # websockets.broadcast is available in newer versions, but loop iteration is safer for older ones
            # or simply iterate
            tasks = [asyncio.create_task(client.send(message)) for client in self.server_clients]
            if tasks:
                await asyncio.wait(tasks)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self.server_loop)
        except Exception as e:
            logger.error(f"Error broadcasting hedge data: {e}")

if __name__ == "__main__":
    logger.info(f"Initializing Market Predictor for symbols: {SYMBOLS}")
    predictor = MarketPredictor(SYMBOLS)
    
    # 1. Check for existing models and load them
    models_loaded = {}
    lstm_models_loaded = {}
    
    for symbol in SYMBOLS:
        models_loaded[symbol] = predictor.load_model_if_valid(symbol, max_age_days=2)
        if models_loaded[symbol]:
            logger.info(f"✓ LGBM Model loaded for {symbol}")
        else:
            logger.info(f"✗ LGBM Model not found or too old for {symbol}")
            
        # Check LSTM
        lstm_models_loaded[symbol] = predictor.load_lstm_model_if_valid(symbol, max_age_days=3)
        if lstm_models_loaded[symbol]:
            logger.info(f"✓ LSTM Model loaded for {symbol}")
        else:
            logger.info(f"✗ LSTM Model not found or too old for {symbol}")
    
    # 2. Fetch Data
    # If all models are loaded, we just need enough data for the buffer (e.g. 2 days)
    # If we need to train any model, we fetch more (e.g. 5 days)
    all_lgbm_loaded = all(models_loaded.values())
    all_lstm_loaded = all(lstm_models_loaded.values()) if predictor.ENABLE_LSTM else True
    
    days_to_fetch = 2 if (all_lgbm_loaded and all_lstm_loaded) else 5
    
    logger.info(f"Fetching {days_to_fetch} days of historical data...")
    df = predictor.fetch_historical_data(SYMBOLS, days=days_to_fetch)
    
    if not df.empty and len(df) > 10:
        # 3. Train Models if needed (per symbol)
        for symbol in SYMBOLS:
            # Filter data for this symbol
            symbol_df = df[df['symbol'] == symbol.upper()].copy()
            
            if not models_loaded[symbol]:
                logger.info(f"Training LGBM model for {symbol.upper()}...")
                if len(symbol_df) > 10:
                    predictor.train_model(symbol.upper(), symbol_df)
                    logger.info(f"✓ LGBM Model trained for {symbol.upper()}")
                else:
                    logger.warning(f"✗ Not enough data for {symbol.upper()}. Skipping LGBM training.")
            
            # Train LSTM if enabled and not loaded
            if predictor.ENABLE_LSTM and not lstm_models_loaded[symbol]:
                predictor.train_lstm_model(symbol.upper(), symbol_df)
        
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
