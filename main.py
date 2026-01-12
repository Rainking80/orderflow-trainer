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

from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from paper_trader import PaperTradeManager
from orderflow_strategy import OrderFlowStrategy

from dataset_labeler import DatasetLabeler

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_URL = "http://localhost:8120"
WS_URL = "ws://localhost:8120/ws/trade_groups"
AUTH_TOKEN = os.getenv("AUTH_BEARER_TOKEN", "")
AUTH_BEARER_TOKEN_OUTGOING = os.getenv("AUTH_BEARER_TOKEN_OUTGOING", "")

# Configuration
SYMBOLS = [s.strip().lower() for s in os.getenv("SYMBOLS", "btcusdt").split(",")]
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Meta-Model Configuration
META_CONFIDENCE_THRESHOLD = 0.56  # Only trade if Meta-Model is 65% sure
PRIMARY_Z_SCORE_THRESHOLD = 2.0  # Trigger primary signal at 1.5 Std Dev
PAPER_TRADING_FEE = 0.001

class MarketPredictor:
    def __init__(self, symbols):
        self.symbols = symbols
        self.meta_models = {}  # symbol -> lgb.Booster

        # Features MUST match dataset_labeler.py exactly
        self.feature_cols = [
            'volatility_20', 'volume_ratio', 'rsi', 'flow_z_score',
            'momentum_5', 'momentum_10', 'momentum_20',
            'price_to_ma20', 'price_to_ma50', 'ma_20_50_diff',
            'log_duration', 'trades_per_second', 'duration_z_score',
            'delta_ratio', 'delta_pct', 'min_max_delta_ratio',
            'cumulative_delta_divergence', 'consecutive_delta', 'cvd_momentum',
            'whale_dominance', 'large_trade_ratio',
            'efficiency_ratio', 'close_location',
        ]

        # Buffer for live data
        self.latest_data = {s: [] for s in self.symbols}
        self.buffers_ready = False  # Flag to indicate buffers are populated with historical data
        self.lock = threading.RLock()
        self.ws = None
        self.running = True

        self.retrain_thread = None

        # Shadow Ledger for tracking trades locally
        self.active_trades = {}

        self.paper_trader = PaperTradeManager(
            initial_balance=1000.0,
            stop_loss_pct=0.004,  # 0.4% Hard Stop
            trailing_stop_pct=0.0015,  # 0.15% Trailing
            fee_pct=PAPER_TRADING_FEE,  # 0.1% Fees
            leverage=10,  # 10x Leverage
            entry_threshold=META_CONFIDENCE_THRESHOLD,  # Normaler Einstieg
            reversal_threshold=0.65,  # Harte Flip-Regel
            cooldown_seconds=60,  # Fast turnover (60s)
            on_trade_open_callback=self.send_telegram_message,  # Just send message on open
            on_trade_close_callback=self.on_trade_closed  # Send message + check charts on close
        )

        # Server for Hedge Bot
        self.server_clients = set()
        self.server_loop = None
        self.server_thread = None

        # Order Flow Strategy for signal detection
        self.orderflow_strategy = OrderFlowStrategy(
            divergence_lookback=20,
            absorption_volume_std=2.0,
            absorption_range_factor=0.5,
            momentum_std=1.5
        )
        self.last_orderflow_signal = {s: None for s in self.symbols}  # Track last signal per symbol
        self.orderflow_signal_cooldown = {s: None for s in self.symbols}  # Cooldown tracking
        
        # Graceful Exit
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, signum, frame):
        logger.info("Received exit signal. Shutting down...")
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        sys.exit(0)

    # ------------------------------------------------------------------------
    # TELEGRAM UTILS
    # ------------------------------------------------------------------------
    def on_trade_closed(self, message):
        """Callback for when a trade closes - sends notification and cleans shadow ledger"""
        # Send the trade notification message
        self.send_telegram_message(message)
        
        # Parse symbol from message and remove from shadow ledger
        # Message format: "Symbol: BTCUSDT\n..."
        try:
            for line in message.split('\n'):
                if line.startswith('Symbol:'):
                    symbol = line.split(':')[1].strip().lower()
                    if symbol in self.active_trades:
                        del self.active_trades[symbol]
                        logger.info(f"🗑️ Removed {symbol.upper()} from shadow ledger")
                    break
        except Exception as e:
            logger.error(f"Error cleaning shadow ledger: {e}")

    def send_telegram_message(self, message):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=5)
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    def send_telegram_photo(self, photo_bytes, caption=""):
        """Send a photo to Telegram"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': ('chart.png', photo_bytes, 'image/png')}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
            requests.post(url, files=files, data=data, timeout=10)
        except Exception as e:
            logger.error(f"Telegram Photo Error: {e}")

    def generate_chart_for_symbol(self, symbol):
        """Generate a 1-minute candle chart for the last 1 hour with orderflow data"""
        with self.lock:
            # Don't generate charts until buffers are populated with historical data
            if not self.buffers_ready:
                logger.warning(f"Buffers not ready yet - skipping chart generation for {symbol}")
                return None
            
            if symbol not in self.latest_data or len(self.latest_data[symbol]) == 0:
                return None
            
            # Get last 60 candles (1 hour of 1-minute candles)
            data = self.latest_data[symbol][-60:]
            if len(data) < 10:  # Need at least some data
                return None
            
            df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_cols = ['close', 'open', 'high', 'low']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Also convert orderflow columns if available
            orderflow_cols = ['total_volume', 'taker_buy_volume', 'buy_volume', 'sell_volume']
            has_orderflow = False
            for col in orderflow_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    has_orderflow = True
            
            # Drop any rows with NaN values in price columns
            df = df.dropna(subset=required_cols)
            if len(df) == 0:
                return None
            
            # Calculate buy/sell volume from orderflow data
            if has_orderflow:
                # If we have taker_buy_volume, use it; otherwise try buy_volume/sell_volume
                if 'taker_buy_volume' in df.columns and 'total_volume' in df.columns:
                    df['buy_vol'] = df['taker_buy_volume'].fillna(0)
                    df['sell_vol'] = df['total_volume'].fillna(0) - df['buy_vol']
                elif 'buy_volume' in df.columns and 'sell_volume' in df.columns:
                    df['buy_vol'] = df['buy_volume'].fillna(0)
                    df['sell_vol'] = df['sell_volume'].fillna(0)
                else:
                    df['buy_vol'] = 0
                    df['sell_vol'] = 0
                
                # Calculate buy pressure ratio for candle coloring
                df['total_vol'] = df['buy_vol'] + df['sell_vol']
                df['buy_pressure'] = df.apply(
                    lambda row: row['buy_vol'] / row['total_vol'] if row['total_vol'] > 0 else 0.5,
                    axis=1
                )
            
            # Create chart with 2 subplots: price and volume
            if has_orderflow:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                              gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot candlesticks with orderflow-based coloring
            for idx, row in df.iterrows():
                if has_orderflow and 'buy_pressure' in row:
                    # Color based on buy pressure
                    # Strong buying pressure (>60%) = bright green
                    # Strong selling pressure (<40%) = bright red
                    # Neutral (40-60%) = gray/yellow
                    if row['buy_pressure'] > 0.6:
                        color = '#00ff00' if row['close'] >= row['open'] else '#90ee90'
                    elif row['buy_pressure'] < 0.4:
                        color = '#ff0000' if row['close'] < row['open'] else '#ff6b6b'
                    else:
                        color = '#ffa500' if row['close'] >= row['open'] else '#ffcc00'
                else:
                    # Fallback to simple coloring
                    color = 'green' if row['close'] >= row['open'] else 'red'
                
                # Draw high-low line
                ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
                # Draw open-close box
                height = abs(row['close'] - row['open'])
                bottom = min(row['open'], row['close'])
                ax1.add_patch(plt.Rectangle((idx - 0.4, bottom), 0.8, height, 
                                          facecolor=color, edgecolor=color, alpha=0.8))
            
            ax1.set_ylabel('Price (USD)', fontsize=10)
            ax1.set_title(f'{symbol.upper()} - Last Hour (1-minute candles with Orderflow)', 
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Plot orderflow volume if available
            if has_orderflow:
                # Plot buy and sell volume as stacked bars
                indices = df.index.tolist()
                buy_vols = df['buy_vol'].tolist()
                sell_vols = df['sell_vol'].tolist()
                
                ax2.bar(indices, buy_vols, width=0.8, color='green', alpha=0.7, label='Buy Volume')
                ax2.bar(indices, sell_vols, width=0.8, color='red', alpha=0.7, 
                       bottom=buy_vols, label='Sell Volume')
                
                ax2.set_xlabel('Candle Index (1min each)', fontsize=10)
                ax2.set_ylabel('Volume', fontsize=10)
                ax2.legend(loc='upper left', fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax1.set_xlabel('Candle Index (1min each)', fontsize=10)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            # Close all figures to prevent accumulation of markers/plots
            plt.close('all')
            
            return buf.read()

    # ------------------------------------------------------------------------
    # HISTORICAL DATA FETCHING (From Old Code - Vital for Cold Start)
    # ------------------------------------------------------------------------
    def _parse_dates(self, df):
        """Helper to parse dates robustly"""
        if 'start_ts' in df.columns:
            if df['start_ts'].dtype == object:
                # Normalize timestamp format
                df['start_ts'] = df['start_ts'].astype(str).str.replace('Z', '+00:00', regex=False)
                df['start_ts'] = df['start_ts'].astype(str).str.replace('T', ' ', regex=False)
                df['start_ts'] = pd.to_datetime(df['start_ts'], errors='coerce')

                if df['start_ts'].dt.tz is None:
                    df['start_ts'] = df['start_ts'].dt.tz_localize('UTC')
                else:
                    df['start_ts'] = df['start_ts'].dt.tz_convert('UTC')
        return df

    def fetch_historical_data(self, days=2):
        """Fetches recent history to pre-fill buffers so model works immediately"""
        logger.info(f"Fetching last {days} days of data to warm up buffers...")

        headers = {}
        if AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        headers["Accept-Encoding"] = "gzip, deflate"

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        all_data = []

        params = {
            'start': start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'end': end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'symbols': ','.join(self.symbols)
        }

        try:
            r = requests.get(f"{BASE_URL}/download/trade_groups", params=params, headers=headers, stream=True)
            r.raise_for_status()

            for line in r.iter_lines():
                if line:
                    try:
                        item = json.loads(line.decode('utf-8'))
                        all_data.append(item)
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = self._parse_dates(df)
            df = df.sort_values('start_ts').reset_index(drop=True)
            logger.info(f"Fetched {len(df)} historical records.")
        return df

    # ------------------------------------------------------------------------
    # AUTO-RETRAINING: LABELING LOGIC (The "Teacher")
    # ------------------------------------------------------------------------
    def apply_triple_barrier_labels(self, df):
        """
        Generates 'meta_label' (1=Win, 0=Loss) for fresh historical data.
        Uses the Trailing Stop logic we defined earlier.
        """
        logger.info("Applying Triple Barrier Labels to fresh data...")
        df = df.copy()
        labels = []

        # Parameters (Match your strategy)
        initial_stop_loss_pct = 0.01  # 1%
        trailing_pct = 0.015  # 1.5% Trailing
        min_profit_pct = 0.015  # Minimum 1.5% profit to be a "Win"

        # We need future data to label the past
        # Iterate until the end minus a buffer (e.g., last 100 rows can't be labeled yet)
        lookahead_buffer = 500

        for i in range(len(df) - lookahead_buffer):
            entry_price = df['close'].iloc[i]

            # --- LOGIC: SIMULATE A LONG TRADE ---
            highest_price = entry_price
            stop_price = entry_price * (1 - initial_stop_loss_pct)
            outcome = 0  # Default: Loss/Neutral

            # Look forward up to 500 bars (e.g. ~5-10 hours depending on volume)
            for j in range(1, 500):
                current_price = df['close'].iloc[i + j]

                # 1. Check Trailing Stop Update
                if current_price > highest_price:
                    highest_price = current_price
                    new_stop = highest_price * (1 - trailing_pct)
                    if new_stop > stop_price:
                        stop_price = new_stop

                # 2. Check Exit (Hit Stop)
                if current_price < stop_price:
                    # Trade ended. Calculate PnL.
                    pnl = (stop_price - entry_price) / entry_price

                    if pnl > min_profit_pct:
                        outcome = 1  # WIN
                    break  # Exit simulation loop

            labels.append(outcome)

        # Fill the remaining buffer with 0 (cannot label yet)
        labels.extend([0] * lookahead_buffer)

        df['meta_label'] = labels

        # Only keep rows that actually had a Primary Signal (Z-Score Trigger)
        # Because we only want to train the Meta-Model on situations where it gets asked!
        # Re-calculate Z-Score if not present, but it should be there from feature eng.
        df['primary_signal'] = 0
        df.loc[df['flow_z_score'] > 1.5, 'primary_signal'] = 1  # Long triggers
        df.loc[
            df['flow_z_score'] < -1.5, 'primary_signal'] = 1  # Short triggers (mapped to same label logic for now)

        return df

    # ------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------------
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def add_market_features(self, df):
        """Adds all features required by the Meta-Model"""
        df = df.copy()

        # 1. Delta
        if 'taker_buy_volume' in df.columns and 'total_volume' in df.columns:
            df['delta'] = 2 * df['taker_buy_volume'] - df['total_volume']
        else:
            direction_proxy = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
            df['delta'] = df['total_volume'] * direction_proxy

        # 2. Flow Momentum & Z-Score
        df['flow_momentum'] = df['delta'].ewm(span=5, adjust=False).mean()
        rolling_window = 100
        df['flow_std'] = df['flow_momentum'].rolling(window=rolling_window).std().replace(0, 0.0001)
        df['flow_mean'] = df['flow_momentum'].rolling(window=rolling_window).mean()
        df['flow_z_score'] = (df['flow_momentum'] - df['flow_mean']) / df['flow_std']

        # 3. Basic Market Features
        df['returns'] = df['close'].pct_change()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['log_ret'].rolling(window=20).std()

        if 'delta' in df.columns:
            df['delta_ma_10'] = df['delta'].rolling(window=10).mean()
            df['cum_delta'] = df['delta'].cumsum()
        else:
            df['delta_ma_10'] = 0
            df['cum_delta'] = 0

        df['volume_ma_20'] = df['total_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['total_volume'] / df['volume_ma_20']

        # 4. Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)

        # 5. MA Relations
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['price_to_ma20'] = df['close'] / df['ma_20'] - 1
        df['price_to_ma50'] = df['close'] / df['ma_50'] - 1
        df['ma_20_50_diff'] = (df['ma_20'] / df['ma_50'] - 1)

        df['rsi'] = self._calculate_rsi(df['close'], 14)

        # 6. Duration Features
        if 'bar_duration_ms' in df.columns:
            df['log_duration'] = np.log1p(df['bar_duration_ms'])

            duration_seconds = df['bar_duration_ms'] / 1000.0
            duration_seconds = duration_seconds.replace(0, 0.001)
            if 'trade_count' in df.columns:
                df['trades_per_second'] = df['trade_count'] / duration_seconds
            else:
                df['trades_per_second'] = 0

            df['duration_mean'] = df['bar_duration_ms'].rolling(window=50).mean()
            df['duration_std'] = df['bar_duration_ms'].rolling(window=50).std().replace(0, 1)
            df['duration_z_score'] = (df['bar_duration_ms'] - df['duration_mean']) / df['duration_std']
        else:
            df['log_duration'] = 0
            df['trades_per_second'] = 0
            df['duration_z_score'] = 0

        # 7. Delta Features
        if 'delta' in df.columns and 'total_volume' in df.columns:
            total_vol_safe = df['total_volume'].replace(0, 0.0001)
            df['delta_ratio'] = (df['delta'] / total_vol_safe).clip(-1, 1)
            df['delta_pct'] = (df['delta'] / total_vol_safe) * 100
        else:
            df['delta_ratio'] = 0
            df['delta_pct'] = 0

        # Min/Max Delta
        if 'delta' in df.columns:
            df['rolling_max_delta'] = df['delta'].rolling(window=10).max()
            df['rolling_min_delta'] = df['delta'].rolling(window=10).min()
            df['min_max_delta_ratio'] = df['rolling_max_delta'] / df['rolling_min_delta'].replace(0, 0.0001).abs()
        else:
            df['min_max_delta_ratio'] = 0

        # Divergence
        if 'cum_delta' in df.columns:
            price_change = df['close'].diff(10)
            cvd_change = df['cum_delta'].diff(10)
            df['cumulative_delta_divergence'] = np.where(
                (price_change > 0) & (cvd_change < 0), 1,
                np.where((price_change < 0) & (cvd_change > 0), -1, 0)
            )
            df['cvd_momentum'] = df['cum_delta'].diff(5)
        else:
            df['cumulative_delta_divergence'] = 0
            df['cvd_momentum'] = 0

        # Consecutive Delta
        if 'delta' in df.columns:
            delta_sign = np.sign(df['delta'])
            df['consecutive_delta'] = (delta_sign.groupby(
                (delta_sign != delta_sign.shift()).cumsum()).cumcount() + 1) * delta_sign
        else:
            df['consecutive_delta'] = 0

        # 8. Trade Structure
        total_vol_safe = df['total_volume'].replace(0, 0.0001)

        if 'largest_trade_qty' in df.columns:
            df['whale_dominance'] = (df['largest_trade_qty'] / total_vol_safe) * 100
        else:
            df['whale_dominance'] = 0

        if 'large_trade_dominance' in df.columns:
            df['large_trade_ratio'] = df['large_trade_dominance'] / 100.0
        else:
            df['large_trade_ratio'] = 0

        if 'high' in df.columns and 'low' in df.columns:
            price_range = df['high'] - df['low']
            df['efficiency_ratio'] = price_range / total_vol_safe
            df['close_location'] = (df['close'] - df['low']) / price_range.replace(0, 0.0001)
        else:
            df['efficiency_ratio'] = 0
            df['close_location'] = 0.5

        # Fill NaNs
        df = df.fillna(0)
        return df

    # ------------------------------------------------------------------------
    # MODEL LOADING & TRAINING
    # ------------------------------------------------------------------------
    def get_meta_model_file(self, symbol):
        return f"meta_model_{symbol.upper()}.txt"

    def clean_up_training_data(self, df):
        """Cleans up training data, removing toxic/absolute columns"""
        # 1. Non-normalized / Absolute columns to drop
        non_normalized_cols = [
            'open', 'high', 'low', 'close', 'vwap',
            'total_volume', 'buy_volume', 'sell_volume',
            'delta', 'delta_ma_10', 'volume_ma_20', 'avg_trade_size',
            'duration_mean', 'duration_std', 'rolling_max_delta', 'rolling_min_delta'
        ]
        # 2. Target columns (Leaks)
        targets = ['tbm_label', 'meta_label', 'returns', 'log_ret']
        # 3. Metadata
        metadata = ['symbol', 'start_ts', 'primary_signal', 'primary_direction', 'largest_trade_side']

        drop_list = non_normalized_cols + targets + metadata
        X = df.drop(columns=drop_list, errors='ignore')
        return X

    def train_meta_model(self, symbol, csv_path):
        logger.info(f"Training Meta-Model for {symbol}...")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path)
        # Filter for primary signals only
        df = df[df['primary_signal'] == 1].copy()

        # Save Target before cleaning
        if 'meta_label' not in df.columns:
            logger.error("meta_label missing")
            return None
        y = df['meta_label']

        # Clean features
        X = self.clean_up_training_data(df)

        # Ensure correct features exist
        missing = [f for f in self.feature_cols if f not in X.columns]
        if missing:
            logger.error(f"Missing features: {missing}")
            return None

        X = X[self.feature_cols]

        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # Dynamic Threshold Optimization
        y_pred = model.predict(X_test)
        best_thresh = 0.5
        best_precision = 0.0

        for thresh in np.arange(0.3, 0.7, 0.01):
            preds = (y_pred >= thresh).astype(int)
            if preds.sum() < 10: continue

            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)

            if rec >= 0.10 and prec > best_precision:
                best_precision = prec
                best_thresh = thresh

        logger.info(f"{symbol} Best Threshold: {best_thresh:.2f} (Precision: {best_precision:.2%})")

        model.save_model(self.get_meta_model_file(symbol))
        self.meta_models[symbol] = model
        return model

    def load_meta_models(self):

        for symbol in self.symbols:
            f = self.get_meta_model_file(symbol)
            dataset_path = f"meta_labeling_dataset_{symbol}.csv"

            # Retrain if dataset exists, else load
            if os.path.exists(dataset_path):
                logger.info(f"Found dataset for {symbol}. Training...")
                self.train_meta_model(symbol, dataset_path)
            elif os.path.exists(f):
                logger.info(f"Loading model for {symbol}...")
                self.meta_models[symbol] = lgb.Booster(model_file=f)
            else:
                logger.warning(f"No model or dataset found for {symbol}. Generating new dataset and training model...")
                
                # Initialize DatasetLabeler to generate new dataset
                labeler = DatasetLabeler()
                
                # Use prepare_training_dataset to generate labeled dataset
                df = labeler.prepare_training_dataset(
                    symbol=symbol,
                    days=300,
                    primary_strategy='volume_imbalance',
                    lookahead=int(os.getenv("LOOKAHEAD_PERIODS", "100")),
                    tp_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.0005")),
                    sl_pct=float(os.getenv("STOP_LOSS_PCT", "0.00025"))
                )
                
                if df.empty or len(df) < 100:
                    logger.error(f"❌ Not enough data generated for {symbol}. Cannot train model.")
                    continue
                
                # Save dataset for training (keep only rows with primary signals)
                training_data = df[df['primary_signal'] == 1].copy()
                training_data.to_csv(dataset_path, index=False)
                logger.info(f"✅ Saved new dataset to {dataset_path} ({len(training_data)} samples)")
                
                # Train model using the new dataset
                self.train_meta_model(symbol, dataset_path)


    # ------------------------------------------------------------------------
    # ORDER FLOW STRATEGY INTEGRATION
    # ------------------------------------------------------------------------
    def _check_orderflow_signal(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Check for order flow signals using the OrderFlowStrategy.
        
        Args:
            symbol: Trading symbol (e.g., 'btcusdt')
            df: DataFrame with market features already calculated
            
        Returns:
            Signal dictionary with 'signal', 'confidence', 'reason' keys
        """
        try:
            # Prepare DataFrame for OrderFlowStrategy
            # It requires: timestamp, open, high, low, close, total_volume, taker_buy_volume
            required_cols = ['open', 'high', 'low', 'close', 'total_volume', 'taker_buy_volume']
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.debug(f"Order Flow: Missing columns for {symbol}: {missing_cols}")
                return {'signal': None, 'confidence': 0.0, 'reason': 'Missing required columns'}
            
            # Create OrderFlow-compatible DataFrame
            of_df = df[['open', 'high', 'low', 'close', 'total_volume', 'taker_buy_volume']].copy()
            
            # Add timestamp column (use index if not present)
            if 'start_ts' in df.columns:
                of_df['timestamp'] = df['start_ts']
            else:
                of_df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(of_df), freq='1min')
            
            # Need at least 40 candles for proper analysis
            if len(of_df) < 40:
                return {'signal': None, 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Get order flow signal
            signal_result = self.orderflow_strategy.get_signal(of_df)
            
            # Apply cooldown to avoid signal spam
            if signal_result['signal']:
                now = datetime.now()
                cooldown_seconds = 300  # 5 minute cooldown between same signals
                
                # Check if we're in cooldown for this signal type
                last_signal = self.last_orderflow_signal.get(symbol)
                last_time = self.orderflow_signal_cooldown.get(symbol)
                
                if last_signal == signal_result['signal'] and last_time:
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < cooldown_seconds:
                        logger.debug(f"Order Flow: {symbol} signal in cooldown ({elapsed:.0f}s / {cooldown_seconds}s)")
                        return {'signal': None, 'confidence': 0.0, 'reason': 'Signal in cooldown'}
                
                # Update tracking
                self.last_orderflow_signal[symbol] = signal_result['signal']
                self.orderflow_signal_cooldown[symbol] = now
                
                # Log the detected signal
                logger.info(
                    f"📈 {symbol.upper()} | ORDER FLOW SIGNAL | "
                    f"{signal_result['signal']} | Conf: {signal_result['confidence']:.1%} | "
                    f"{signal_result['reason']}"
                )
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Order Flow error for {symbol}: {e}")
            return {'signal': None, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def get_orderflow_analysis(self, symbol: str) -> dict:
        """
        Get comprehensive order flow analysis for a symbol.
        Useful for debugging and monitoring.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Full analysis dictionary from OrderFlowStrategy
        """
        with self.lock:
            if symbol not in self.latest_data or len(self.latest_data[symbol]) < 40:
                return {'error': 'Insufficient data'}
            
            df = pd.DataFrame(self.latest_data[symbol][-200:])
            
            # Ensure numeric types
            cols = ['close', 'open', 'high', 'low', 'total_volume', 'taker_buy_volume']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Create OrderFlow-compatible DataFrame
            of_df = df[['open', 'high', 'low', 'close', 'total_volume', 'taker_buy_volume']].copy()
            if 'start_ts' in df.columns:
                of_df['timestamp'] = df['start_ts']
            else:
                of_df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(of_df), freq='1min')
            
            return self.orderflow_strategy.get_analysis_summary(of_df)
    
    def execute_orderflow_trade(self, symbol: str, signal_result: dict, current_price: float) -> bool:
        """
        Execute a paper trade based on Order Flow signal with custom stop loss.
        
        This creates trades with pattern-specific stop losses:
        - DIVERGENCE: Stop above/below recent high/low
        - ABSORPTION: Stop just beyond absorption level  
        - MOMENTUM: Trailing stop based on ATR
        
        Args:
            symbol: Trading symbol
            signal_result: Signal dictionary from OrderFlowStrategy with trade_setup
            current_price: Current price for entry
            
        Returns:
            True if trade was opened, False otherwise
        """
        if not signal_result or not signal_result.get('signal'):
            return False
        
        trade_setup = signal_result.get('trade_setup')
        if not trade_setup:
            logger.warning(f"Order Flow signal without trade setup for {symbol}")
            return False
        
        signal_side = signal_result['signal']
        confidence = signal_result['confidence']
        pattern_type = trade_setup.get('pattern_type', 'UNKNOWN')
        stop_loss = trade_setup.get('stop_loss', 0)
        stop_loss_pct = trade_setup.get('stop_loss_pct', 0.01)  # Default 1%
        entry_trigger = trade_setup.get('entry_trigger', '')
        
        # Check if we already have an open position
        if self.paper_trader.has_open_position(symbol):
            # Check if this is a reversal signal
            active_trade = self.paper_trader.active_trades.get(symbol)
            if active_trade and active_trade['side'] != signal_side:
                logger.info(f"📊 {symbol.upper()} | Order Flow REVERSAL detected | {active_trade['side']} -> {signal_side}")
                # Close existing trade first - calculate PnL
                if active_trade['side'] == 'LONG':
                    pnl = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                else:
                    pnl = (active_trade['entry_price'] - current_price) / active_trade['entry_price']
                self.paper_trader.close_trade(symbol, current_price, f"OF_{pattern_type}_REVERSAL", pnl, datetime.now())
            else:
                logger.debug(f"Order Flow: {symbol} already has position in same direction")
                return False
        
        # Calculate custom stop loss percentage
        # Limit stop loss to reasonable bounds
        stop_loss_pct = max(0.002, min(stop_loss_pct, 0.05))  # Between 0.2% and 5%
        
        # Temporarily modify paper trader stop loss for this trade
        original_sl_pct = self.paper_trader.SL_PCT
        original_tsl_pct = self.paper_trader.TSL_PCT
        
        # For MOMENTUM trades, use tighter trailing stops
        if pattern_type == 'MOMENTUM':
            self.paper_trader.SL_PCT = stop_loss_pct
            self.paper_trader.TSL_PCT = stop_loss_pct * 0.75  # Tighter trailing
        else:
            # For ABSORPTION and DIVERGENCE, use the calculated stop
            self.paper_trader.SL_PCT = stop_loss_pct
            self.paper_trader.TSL_PCT = stop_loss_pct * 1.5  # Wider trailing for reversals
        
        try:
            # Open the trade
            self.paper_trader.open_trade(
                symbol=symbol,
                side=signal_side,
                price=current_price,
                timestamp=datetime.now()
            )
            
            # Populate Shadow Ledger with Order Flow specific info
            self.active_trades[symbol] = {
                'entry_price': current_price,
                'start_time': datetime.now(),
                'side': signal_side,
                'source': 'ORDERFLOW',
                'pattern_type': pattern_type,
                'stop_loss': stop_loss,
                'entry_trigger': entry_trigger
            }
            
            # Send detailed Telegram notification
            emoji = "🔄" if pattern_type == "DIVERGENCE" else ("🧱" if pattern_type == "ABSORPTION" else "🚀")
            msg = f"{emoji} <b>[OF] {pattern_type} TRADE OPENED</b>\n"
            msg += f"Symbol: {symbol.upper()}\n"
            msg += f"Side: <b>{signal_side}</b>\n"
            msg += f"Pattern: {pattern_type}\n"
            msg += f"Confidence: {confidence:.1%}\n"
            msg += f"Entry: ${current_price:.2f}\n"
            msg += f"Stop Loss: ${stop_loss:.2f} ({stop_loss_pct:.2%})\n"
            msg += f"Trigger: {entry_trigger}"
            
            self.send_telegram_message(msg)
            
            logger.info(
                f"📊 {symbol.upper()} | [OF] {pattern_type} TRADE OPENED | "
                f"{signal_side} @ ${current_price:.2f} | SL: ${stop_loss:.2f} ({stop_loss_pct:.2%})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing Order Flow trade for {symbol}: {e}")
            return False
        
        finally:
            # Restore original stop loss settings
            self.paper_trader.SL_PCT = original_sl_pct
            self.paper_trader.TSL_PCT = original_tsl_pct


    # ------------------------------------------------------------------------
    # LIVE PREDICTION LOOP (Client Consumation)
    # ------------------------------------------------------------------------
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
        except:
            return

        symbol = data.get('symbol', '').lower()
        if symbol not in self.symbols:
            return

        logger.info(f"📨 Incoming message for {symbol.upper()} | Price: ${data.get('close', 'N/A')}")

        with self.lock:
            # Append new data
            self.latest_data[symbol].append(data)

            # Maintain buffer size (keep last 500)
            if len(self.latest_data[symbol]) > 500:
                self.latest_data[symbol] = self.latest_data[symbol][-500:]

            # Need minimum 200 rows for rolling features
            if len(self.latest_data[symbol]) < 200:
                logger.info(f"⏳ {symbol.upper()} | Buffer size: {len(self.latest_data[symbol])}/200 - waiting for more data")
                return

            # Preis extrahieren für den Paper Trader
            try:
                current_price = float(data.get('close'))
            except:
                return  # Ohne Preis kein Trading

            # =========================================================
            # FEED DATA TO PAPER TRADER FOR CHART GENERATION
            # =========================================================
            # Add bucketed trade data to paper trader so it can generate charts on exit
            try:
                bucket_data = {
                    'timestamp': data.get('timestamp', datetime.now().timestamp()),
                    'price': current_price,
                    'volume': float(data.get('total_volume', 0)),
                    'buy_volume': float(data.get('taker_buy_volume', 0)),
                    'sell_volume': float(data.get('total_volume', 0)) - float(data.get('taker_buy_volume', 0))
                }
                self.paper_trader.add_bucketed_trade(symbol, bucket_data)
            except Exception as e:
                logger.warning(f"Failed to add bucketed trade data for {symbol}: {e}")

            # =========================================================
            # 1. PAPER TRADER UPDATE (Check Exits FIRST)
            # =========================================================
            # Prüft bei JEDEM Tick, ob der TSL oder SL getroffen wurde.
            # Wenn ja, wird der Trade geschlossen und geloggt.
            self.paper_trader.update_trade(symbol, current_price, datetime.now())

            # =========================================================
            # 2. ROTTING SIGNAL LOGIC (Shadow Ledger Check)
            # =========================================================
            # Check if symbol is in active_trades (Shadow Ledger)
            if symbol in self.active_trades:
                trade_info = self.active_trades[symbol]
                shadow_entry_price = trade_info['entry_price']
                start_time = trade_info['start_time']
                
                # Calculate trade duration
                now = datetime.now()
                duration = (now - start_time).total_seconds()
                
                # Calculate estimated PnL based on side
                if trade_info['side'] == 'LONG':
                    estimated_pnl = (current_price - shadow_entry_price) / shadow_entry_price
                else:  # SHORT
                    estimated_pnl = (shadow_entry_price - current_price) / shadow_entry_price
                
                # TIME-BASED EXIT: If 15 minutes passed and profit < 0.2%
                if duration > 900 and estimated_pnl < 0.002:
                    logger.info(f"⏳ {symbol.upper()} | Time Decay exit | Duration: {duration:.0f}s | PnL: {estimated_pnl:.2%}")
                    self.paper_trader.close_trade(symbol, current_price, "TIME_DECAY", estimated_pnl, datetime.now())
                    # Remove from shadow ledger
                    del self.active_trades[symbol]

            # 2. Feature Engineering
            # Convert list of dicts to DataFrame (last 200 rows)
            # Note: The websocket data keys must match what add_market_features expects
            df = pd.DataFrame(self.latest_data[symbol][-200:])

            # Ensure numeric types
            cols = ['close', 'open', 'high', 'low', 'total_volume', 'taker_buy_volume', 'bar_duration_ms',
                    'trade_count']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c])

            df = self.add_market_features(df)

            current_row = df.iloc[-1]
            z_score = current_row['flow_z_score']

            logger.info(f"✅ {symbol.upper()} | Features calculated | Z-Score: {z_score:.2f}")

            # =========================================================
            # 2.5. ORDER FLOW STRATEGY SIGNAL CHECK
            # =========================================================
            orderflow_signal = self._check_orderflow_signal(symbol, df)
            
            # Execute Order Flow trade if signal is strong enough
            # Order Flow trades run independently with their own stop loss logic
            orderflow_trade_executed = False
            if orderflow_signal and orderflow_signal.get('signal') and orderflow_signal.get('confidence', 0) >= 0.6:
                trade_setup = orderflow_signal.get('trade_setup')
                if trade_setup:
                    pattern_type = trade_setup.get('pattern_type', 'UNKNOWN')
                    logger.info(
                        f"📈 {symbol.upper()} | ORDER FLOW {pattern_type} | "
                        f"{orderflow_signal['signal']} | Conf: {orderflow_signal['confidence']:.1%}"
                    )
                    
                    # Execute the Order Flow trade with custom stops
                    orderflow_trade_executed = self.execute_orderflow_trade(
                        symbol=symbol,
                        signal_result=orderflow_signal,
                        current_price=current_price
                    )
                    
                    if orderflow_trade_executed:
                        # Broadcast to Hedge Bot with Order Flow marker
                        self.broadcast_hedge_signal(
                            symbol=symbol,
                            side=orderflow_signal['signal'],
                            confidence=orderflow_signal['confidence'],
                            price=current_price
                        )
                        # Return early - Order Flow trade was executed
                        return
            
            # 3. CHECK PRIMARY SIGNAL (Stage 1)
            is_long_signal = z_score > PRIMARY_Z_SCORE_THRESHOLD
            is_short_signal = z_score < -PRIMARY_Z_SCORE_THRESHOLD
            
            # Order Flow can still influence primary signal direction if trade wasn't executed
            if not orderflow_trade_executed and orderflow_signal and orderflow_signal['signal']:
                of_signal = orderflow_signal['signal']
                of_confidence = orderflow_signal['confidence']
                of_reason = orderflow_signal['reason']
                
                # If Order Flow has signal but didn't execute (low confidence), still consider direction
                if of_confidence >= 0.5:
                    if of_signal == 'LONG':
                        is_long_signal = True
                        is_short_signal = False
                    elif of_signal == 'SHORT':
                        is_short_signal = True
                        is_long_signal = False
                    logger.info(f"📊 {symbol.upper()} | Order Flow INFLUENCE | {of_signal} | Conf: {of_confidence:.1%} | {of_reason}")

            if not (is_long_signal or is_short_signal):
                logger.info(f"⚠️ {symbol.upper()} | No primary signal | Z-Score: {z_score:.2f} (threshold: ±{PRIMARY_Z_SCORE_THRESHOLD})")
                return


            signal_direction = "LONG" if is_long_signal else "SHORT"
            logger.info(f"🎯 {symbol.upper()} | Primary signal detected: {signal_direction} | Z-Score: {z_score:.2f}")

            # 4. CHECK META-MODEL (Stage 2)
            if symbol not in self.meta_models:
                logger.warning(f"❌ {symbol.upper()} | No meta-model available for prediction")
                return

            features = current_row[self.feature_cols].values.reshape(1, -1)
            success_prob = self.meta_models[symbol].predict(features)[0]

            logger.info(f"🤖 {symbol.upper()} | Meta-model prediction | Confidence: {success_prob:.1%} (threshold: {META_CONFIDENCE_THRESHOLD:.1%})")

            # =========================================================
            # 5. PAPER TRADER ENTRY
            # =========================================================
            if success_prob > META_CONFIDENCE_THRESHOLD:
                signal_type = "LONG" if is_long_signal else "SHORT"

                # Versuch Trade zu öffnen.
                # Die Klasse ignoriert es automatisch, wenn schon einer offen ist.
                self.paper_trader.execute_signal(
                    symbol=symbol,
                    signal_side=signal_type,
                    price=current_price,
                    timestamp=datetime.now(),
                    confidence=success_prob,
                    z_score=z_score
                )
                
                # Populate Shadow Ledger immediately after execute_signal
                self.active_trades[symbol] = {
                    'entry_price': current_price,
                    'start_time': datetime.now(),
                    'side': signal_type
                }


            # 6. EXECUTION
            if success_prob > META_CONFIDENCE_THRESHOLD:
                signal_type = "LONG" if is_long_signal else "SHORT"
                emoji = "🚀" if is_long_signal else "📉"

                logger.info(
                    f"🔥 VALID SIGNAL {symbol.upper()} | {signal_type} | Z: {z_score:.2f} | Conf: {success_prob:.1%}")

                msg = f"{emoji} <b>SIGNAL: {symbol.upper()}</b>\n"
                msg += f"Type: <b>{signal_type}</b>\n"
                msg += f"AI Conf: <code>{success_prob:.1%}</code>\n"
                msg += f"Price: <code>${current_row['close']:.2f}</code>"

                # self.send_telegram_message(msg)

                # Broadcast to Hedge Bot
                self.broadcast_hedge_signal(symbol, signal_type, success_prob, current_row['close'])
            else:
                logger.info(f"🚫 {symbol.upper()} | Signal REJECTED | Confidence too low: {success_prob:.1%}")

    # ------------------------------------------------------------------------
    # WEBSOCKET SERVER (Hedge Bot Communication)
    # ------------------------------------------------------------------------
    def broadcast_hedge_signal(self, symbol, side, confidence, price):
        if not self.server_clients:
            return
        payload = json.dumps({
            "type": "SIGNAL",
            "symbol": symbol,
            "side": side,
            "confidence": float(confidence),
            "price": float(price),
            "timestamp": datetime.now().isoformat()
        })
        # Thread-safe broadcast
        asyncio.run_coroutine_threadsafe(self._send_all(payload), self.server_loop)

    async def _send_all(self, msg):
        if not self.server_clients:
            return
        tasks = [asyncio.create_task(client.send(msg)) for client in self.server_clients]
        await asyncio.wait(tasks)

    async def websocket_handler(self, websocket):
        """Auth and Connection Handler"""
        try:
            auth_header = websocket.request_headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                await websocket.close(code=1008)
                return
            token = auth_header.split(" ")[1]
            if token != AUTH_BEARER_TOKEN_OUTGOING:
                await websocket.close(code=1008)
                return

            logger.info(f"Hedge Bot Connected: {websocket.remote_address}")
            self.server_clients.add(websocket)
            await websocket.wait_closed()
        except Exception as e:
            logger.error(f"WS Handler Error: {e}")
        finally:
            self.server_clients.discard(websocket)

    def start_websocket_server(self):
        """Starts AsyncIO Server in Thread"""
        self.server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.server_loop)

        async def run_server():
            async with websockets.serve(self.websocket_handler, "0.0.0.0", 8765):
                logger.info("Hedge Bot Signal Server running on port 8765")
                await asyncio.Future()  # Run forever

        try:
            self.server_loop.run_until_complete(run_server())
        except Exception as e:
            logger.error(f"Server crash: {e}")
        finally:
            self.server_loop.close()

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket Closed. Code: {close_status_code}, Msg: {close_msg}")

    def on_open(self, ws):
        logger.info("WebSocket Connection Opened")

    def run_scheduled_retraining(self):
        """Full pipeline: Download fresh data -> Label -> Train -> Hot Swap"""
        logger.info("🔄 STARTING SCHEDULED RETRAINING PIPELINE...")

        # Initialize DatasetLabeler to reuse training preparation functions
        labeler = DatasetLabeler()

        for symbol in self.symbols:
            try:
                logger.info(f"Processing {symbol} for retraining...")

                # Use prepare_training_dataset from dataset_labeler.py instead of re-implementing
                # This handles: data fetching, primary signals, TBM labels, meta labels, and features
                df = labeler.prepare_training_dataset(
                    symbol=symbol,
                    days=30,
                    primary_strategy='volume_imbalance',
                    lookahead=int(os.getenv("LOOKAHEAD_PERIODS", "100")),
                    tp_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.0005")),
                    sl_pct=float(os.getenv("STOP_LOSS_PCT", "0.00025"))
                )

                if df.empty or len(df) < 100:
                    logger.warning(f"❌ Not enough data for {symbol}. Skipping retraining.")
                    continue

                # Save dataset for training (keep only rows with primary signals)
                training_data = df[df['primary_signal'] == 1].copy()
                dataset_path = f"meta_labeling_dataset_{symbol}.csv"
                training_data.to_csv(dataset_path, index=False)
                logger.info(f"Saved fresh labeled dataset to {dataset_path} ({len(training_data)} samples)")

                # Train Model using the existing train_meta_model function
                new_model = self.train_meta_model(symbol, dataset_path)

                # Hot Swap (Update the live model in memory)
                if new_model:
                    with self.lock:
                        self.meta_models[symbol] = new_model
                    logger.info(f"✅ SUCCESSFULLY HOT-SWAPPED MODEL FOR {symbol}")

            except Exception as e:
                logger.error(f"Failed to retrain {symbol}: {e}")

    def retraining_scheduler_loop(self):
        """Background thread checking if it's time to retrain"""
        retrain_interval_hours = 72  # 3 Days
        initial_delay_hours = 72  # Wait 3 days before starting scheduler

        # Initial delay: wait 3 days after service start before beginning checks
        logger.info(f"⏰ Retraining Scheduler will start in {initial_delay_hours} hours (3 days)")
        time.sleep(initial_delay_hours * 3600)
        logger.info("📅 Retraining Scheduler now active (Checks every hour, triggers every 3 days)")

        while self.running:
            # Check model age
            for symbol in self.symbols:
                model_file = self.get_meta_model_file(symbol)
                if os.path.exists(model_file):
                    file_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                    age = datetime.now() - file_time

                    if age.total_seconds() > (retrain_interval_hours * 3600):
                        logger.info(f"Model for {symbol} is {age.days} days old. Triggering retraining...")
                        self.run_scheduled_retraining()
                        # Wait a bit after retraining to avoid spamming
                        time.sleep(3600)
                        break  # Restart loop to re-check fresh timestamps

            # Sleep 1 hour before checking again
            time.sleep(3600)

    # ------------------------------------------------------------------------
    # MAIN RUN
    # ------------------------------------------------------------------------
    def run(self):
        # 1. Load Models
        self.load_meta_models()

        # 2. Start Retraining Scheduler (Background Thread)
        self.retrain_thread = threading.Thread(target=self.retraining_scheduler_loop, daemon=True)
        self.retrain_thread.start()
        logger.info("📅 Retraining Scheduler thread started (Will activate after 3-day initial delay)")

        # 3. Pre-fill Buffers with Historical Data (Crucial for cold start)
        logger.info("🔄 Loading historical data to populate buffers...")
        hist_df = self.fetch_historical_data(days=2)
        if not hist_df.empty:
            for symbol in self.symbols:
                # Extract data for symbol, convert timestamp to string (simulating WS format if needed)
                # or keep as dicts. 'latest_data' expects list of dicts.
                sym_data = hist_df[hist_df['symbol'] == symbol.upper()].copy()
                if not sym_data.empty:
                    # Convert to records
                    records = sym_data.to_dict('records')
                    self.latest_data[symbol] = records[-2000:]  # Keep last 2000
                    logger.info(f"✅ Pre-filled {symbol} buffer with {len(self.latest_data[symbol])} records")
            
            # Mark buffers as ready - charts can now be generated safely
            with self.lock:
                self.buffers_ready = True
            logger.info("✅ All buffers populated with historical data - charts ready for generation")
        else:
            logger.warning("⚠️ Failed to fetch historical data - buffers will be populated from live WebSocket data")
            logger.warning("⚠️ Charts will not be available until sufficient live data is collected")

        # 4. Start Hedge Bot Server
        self.server_thread = threading.Thread(target=self.start_websocket_server, daemon=True)
        self.server_thread.start()

        # 5. Start Data Stream
        websocket.enableTrace(False)
        ws_headers = {}
        if AUTH_TOKEN:
            ws_headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

        ws_url = f"{WS_URL}?symbols={','.join(self.symbols)}"

        logger.info(f"Connecting to Data Stream: {WS_URL}")
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_open=self.on_open,
            on_error=self.on_error,
            on_close=self.on_close,
            header=ws_headers
        )
        self.ws.run_forever()


if __name__ == "__main__":
    predictor = MarketPredictor(SYMBOLS)
    predictor.run()