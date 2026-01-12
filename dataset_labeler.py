import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from enum import Enum
from dotenv import load_dotenv

# Performance Optimization
from numba import jit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8120"
AUTH_TOKEN = os.getenv("AUTH_BEARER_TOKEN", "")

# Triple Barrier Parameters
LOOKAHEAD_PERIODS = int(os.getenv("LOOKAHEAD_PERIODS", "100"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.004"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.002"))

PRIMARY_Z_SCORE_THRESHOLD = 2.0  # Trigger primary signal at 1.5 Std Dev

class TradeLabel(Enum):
    """Labels for Triple Barrier Method"""
    LONG_ENTRY = "LONG_ENTRY"
    SHORT_ENTRY = "SHORT_ENTRY"
    NO_ACTION = "NO_ACTION"


class MetaLabel(Enum):
    """Meta-Labels for Meta-Labeling (binary classification)"""
    SUCCESS = 1
    FAILURE = 0


# -----------------------------------------------------------------------------
# NUMBA OPTIMIZED KERNEL (High Performance TBM)
# -----------------------------------------------------------------------------
@jit(nopython=True)
def _apply_tbm_fast(close, high, low, lookahead, tp_pct, sl_pct):
    """
    JIT-compiled Triple Barrier Method.
    Returns integer array: 1 (Long), -1 (Short), 0 (No Action)
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)  # 0 default

    # Loop through each bar
    for i in range(n - lookahead):
        entry_price = close[i]

        # Define Barriers
        tp_price_long = entry_price * (1 + tp_pct)
        sl_price_long = entry_price * (1 - sl_pct)

        tp_price_short = entry_price * (1 - tp_pct)
        sl_price_short = entry_price * (1 + sl_pct)

        # Initialize hit indices (lookahead + 1 means not hit yet)
        first_tp_long = lookahead + 1
        first_sl_long = lookahead + 1
        first_tp_short = lookahead + 1
        first_sl_short = lookahead + 1

        # Scan the future window
        # Note: We start at 1 because we check the NEXT bars
        for j in range(1, lookahead + 1):
            future_idx = i + j
            h = high[future_idx]
            l = low[future_idx]

            # Check LONG barriers
            # Hit TP? (High touches TP)
            if h >= tp_price_long and first_tp_long > j:
                first_tp_long = j
            # Hit SL? (Low touches SL)
            if l <= sl_price_long and first_sl_long > j:
                first_sl_long = j

            # Check SHORT barriers
            # Hit TP? (Low touches TP)
            if l <= tp_price_short and first_tp_short > j:
                first_tp_short = j
            # Hit SL? (High touches SL)
            if h >= sl_price_short and first_sl_short > j:
                first_sl_short = j

            # Optimization: If both directions already hit a barrier, stop scanning
            long_resolved = (first_tp_long <= lookahead) or (first_sl_long <= lookahead)
            short_resolved = (first_tp_short <= lookahead) or (first_sl_short <= lookahead)
            if long_resolved and short_resolved:
                break

        # Determine Final Label

        # 1. Evaluate Long
        long_win = (first_tp_long < first_sl_long) and (first_tp_long <= lookahead)

        # 2. Evaluate Short
        short_win = (first_tp_short < first_sl_short) and (first_tp_short <= lookahead)

        if long_win:
            labels[i] = 1
        elif short_win:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels


class DatasetLabeler:
    """
    Professional Data Labeling Tool combining Triple Barrier Method and Meta-Labeling.
    Optimized for Orderflow/Tick Data.
    """

    def __init__(self):
        self.headers = {}
        if AUTH_TOKEN:
            self.headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        self.headers["Content-Type"] = "application/json"

    def fetch_historical_data(self, symbol, days=30):
        """Fetch historical data (Standard implementation)."""
        logger.info(f"Fetching {days} days of data for {symbol}...")
        all_data = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        chunk_days = 10
        current_start = start_time

        while current_start < end_time:
            current_end = min(current_start + timedelta(days=chunk_days), end_time)
            params = {
                'start': current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'end': current_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'symbols': symbol
            }
            try:
                headers = self.headers.copy()
                headers["Accept-Encoding"] = "gzip, deflate"
                response = requests.get(f"{BASE_URL}/download/trade_groups", params=params, headers=headers,
                                        stream=True)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            item = json.loads(line.decode('utf-8'))
                            all_data.append(item)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error fetching chunk for {symbol}: {e}")
            current_start = current_end

        df = pd.DataFrame(all_data)
        if not df.empty:
            if 'symbol' in df.columns:
                df = df[df['symbol'].str.lower() == symbol.lower()].copy()
            if 'start_ts' in df.columns:
                df['start_ts'] = pd.to_datetime(df['start_ts'], utc=True, format='mixed')
                df = df.sort_values('start_ts').reset_index(drop=True)
                cols = ['open', 'high', 'low', 'close', 'total_volume', 'taker_buy_volume']
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def apply_triple_barrier_labels(self, df, lookahead=50, tp_pct=0.02, sl_pct=0.01):
        """
        STEP 2: Apply Triple Barrier Method (Optimized with Numba).
        """
        # Prepare numpy arrays for Numba (much faster than Pandas Series)
        # Handle case where high/low might not exist (fallback to close)
        close_arr = df['close'].values.astype(np.float64)

        if 'high' in df.columns and 'low' in df.columns:
            high_arr = df['high'].values.astype(np.float64)
            low_arr = df['low'].values.astype(np.float64)
        else:
            logger.warning("High/Low columns missing, using Close for TBM (less accurate).")
            high_arr = close_arr
            low_arr = close_arr

        logger.info(f"Running Numba-optimized Triple Barrier Method on {len(df)} rows...")

        # Call the JIT function
        int_labels = _apply_tbm_fast(
            close_arr, high_arr, low_arr,
            lookahead, tp_pct, sl_pct
        )

        # Map integers back to Enums/Strings
        str_labels = []
        label_map = {
            1: TradeLabel.LONG_ENTRY.value,
            -1: TradeLabel.SHORT_ENTRY.value,
            0: TradeLabel.NO_ACTION.value
        }

        # Convert back to list/series (fast list comp)
        str_labels = [label_map[x] for x in int_labels]

        # Statistics
        unique, counts = np.unique(int_labels, return_counts=True)
        stats = dict(zip(unique, counts))
        logger.info(f"TBM Results: Longs={stats.get(1, 0)}, Shorts={stats.get(-1, 0)}, None={stats.get(0, 0)}")

        return str_labels

    def generate_primary_signals(self, df, strategy='volume_imbalance'):
        """
        STEP 1: Generate Primary Model Signals (High Recall Strategy).
        Improved logic for Orderflow data.
        """
        df = df.copy()
        df['primary_signal'] = 0
        df['primary_direction'] = 0

        if strategy == 'volume_imbalance':
            # ---------------------------------------------------------
            # IMPROVED STRATEGY FOR TICK/ORDERFLOW DATA
            # ---------------------------------------------------------

            # A. Calculate Delta (Net Buy Volume)
            if 'taker_buy_volume' in df.columns and 'total_volume' in df.columns:
                # Real Delta: Buy Volume - Sell Volume
                # Sell Volume = Total - Buy
                # Delta = Buy - (Total - Buy) = 2*Buy - Total
                df['delta'] = 2 * df['taker_buy_volume'] - df['total_volume']
                logger.info("Using REAL 'taker_buy_volume' for Delta calculation.")
            else:
                # Fallback: Tick Rule approximation
                logger.warning("Using Tick Rule approximation (Close vs Open) for Delta.")
                direction_proxy = np.where(df['close'] > df['open'], 1,
                                           np.where(df['close'] < df['open'], -1, 0))
                df['delta'] = df['total_volume'] * direction_proxy

            # B. Flow Momentum (Exponential Moving Average reacts faster than SMA)
            # Span=5 on 100-trade bars is very fast (last ~500 trades)
            df['flow_momentum'] = df['delta'].ewm(span=5, adjust=False).mean()

            # C. Dynamic Threshold (Z-Score)
            # We compare current momentum to the last 100 bars baseline
            rolling_window = 100
            df['flow_std'] = df['flow_momentum'].rolling(window=rolling_window).std()
            df['flow_mean'] = df['flow_momentum'].rolling(window=rolling_window).mean()

            # Avoid division by zero
            df['flow_std'] = df['flow_std'].replace(0, 0.0001)

            df['flow_z_score'] = (df['flow_momentum'] - df['flow_mean']) / df['flow_std']

            # D. Signal Generation
            # Threshold: 1.5 Std Dev deviation means "Significant Order Flow Event"

            threshold = PRIMARY_Z_SCORE_THRESHOLD

            # LONG
            long_condition = df['flow_z_score'] > threshold
            df.loc[long_condition, 'primary_signal'] = 1
            df.loc[long_condition, 'primary_direction'] = 1

            # SHORT
            short_condition = df['flow_z_score'] < -threshold
            df.loc[short_condition, 'primary_signal'] = 1
            df.loc[short_condition, 'primary_direction'] = -1

            logger.info(f"Strategy 'volume_imbalance': Generated signals using Z-Score > {threshold}")

        elif strategy == 'bb_squeeze':
            # Classic Volatility Breakout
            window = 20
            df['sma'] = df['close'].rolling(window).mean()
            df['std'] = df['close'].rolling(window).std()
            upper = df['sma'] + (2 * df['std'])
            lower = df['sma'] - (2 * df['std'])

            breakout_up = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
            df.loc[breakout_up, 'primary_signal'] = 1
            df.loc[breakout_up, 'primary_direction'] = 1

            breakout_down = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))
            df.loc[breakout_down, 'primary_signal'] = 1
            df.loc[breakout_down, 'primary_direction'] = -1

        # Fill NaNs
        df['primary_signal'] = df['primary_signal'].fillna(0)
        df['primary_direction'] = df['primary_direction'].fillna(0)

        return df

    def create_meta_labels(self, df, tbm_labels):
        """STEP 3: Create Meta-Labels (Success/Failure)."""
        df = df.copy()
        df['tbm_label'] = tbm_labels
        df['meta_label'] = np.nan

        has_signal = df['primary_signal'] == 1

        # Logic: Did the signal match the ground truth?
        long_success = has_signal & (df['primary_direction'] == 1) & (df['tbm_label'] == TradeLabel.LONG_ENTRY.value)
        short_success = has_signal & (df['primary_direction'] == -1) & (df['tbm_label'] == TradeLabel.SHORT_ENTRY.value)

        df.loc[long_success | short_success, 'meta_label'] = MetaLabel.SUCCESS.value

        # Failures
        has_signal_no_success = has_signal & df['meta_label'].isna()
        df.loc[has_signal_no_success, 'meta_label'] = MetaLabel.FAILURE.value

        return df

    def add_market_features(self, df):
        """Add features for the Meta-Model."""
        df = df.copy()

        # 1. Volatility
        df['returns'] = df['close'].pct_change()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['log_ret'].rolling(window=20).std()

        # 2. Volume / Flow Features
        if 'delta' in df.columns:
            df['delta_ma_10'] = df['delta'].rolling(window=10).mean()
            df['cum_delta'] = df['delta'].cumsum()  # Careful with stationarity here

        df['volume_ma_20'] = df['total_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['total_volume'] / df['volume_ma_20']

        # 3. Momentum
        df['rsi'] = self._calculate_rsi(df['close'], 14)

        return df

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def add_trend_features(self, df):
        """
        Add trend and momentum features required by the model.
        
        Features added:
        - momentum_5, momentum_10, momentum_20: Rate of Change (ROC) for different periods
        - price_to_ma20, price_to_ma50: Normalized distance of price from moving averages
        - ma_20_50_diff: Normalized spread between fast and slow moving averages
        - log_duration: Log of bar duration in milliseconds
        - trades_per_second: Trade frequency (100 trades / duration_in_seconds)
        - duration_z_score: Is this bar faster than average of last 50 bars?
        - delta_ratio: (Buy_Vol - Sell_Vol) / Total_Vol, range [-1, +1]
        - min_max_delta_ratio: Ratio of max_delta to min_delta (fight for zone)
        - cumulative_delta_divergence: Price rises but cumulative delta falls?
        - consecutive_delta: How many consecutive bars had positive/negative delta?
        - cvd_momentum: Change in cumulative volume delta
        - whale_dominance: Percentage of volume from largest trade
        - large_trade_ratio: Volume from large trades (>threshold) / Total volume
        - efficiency_ratio: (High - Low) / Total_Volume
        
        Args:
            df: DataFrame with at least a 'close' column
            
        Returns:
            DataFrame with added trend features
            
        Note: First 50 rows will contain NaNs due to rolling calculations.
              Consider using df.dropna() or df.bfill() based on pipeline policy.
        """
        df = df.copy()
        
        # ===== EXISTING MOMENTUM FEATURES =====
        # Momentum features using Rate of Change (ROC)
        df['momentum_5'] = (df['close'] / df['close'].shift(5)) - 1
        df['momentum_10'] = (df['close'] / df['close'].shift(10)) - 1
        df['momentum_20'] = (df['close'] / df['close'].shift(20)) - 1
        
        # Calculate Simple Moving Averages (SMA) for 20 and 50 periods
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Price to MA features
        df['price_to_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']
        df['price_to_ma50'] = (df['close'] - df['ma_50']) / df['ma_50']
        
        # MA Crossover
        df['ma_20_50_diff'] = (df['ma_20'] - df['ma_50']) / df['ma_50']
        
        # ===== NEW FEATURES (EXPANDED SET) =====
        
        # 1. LOG_DURATION: Logarithm of duration makes data "smoother" for ML
        if 'bar_duration_ms' in df.columns:
            df['log_duration'] = np.log1p(df['bar_duration_ms'])  # log1p avoids log(0)
        else:
            df['log_duration'] = 0
        
        # 2. TRADES_PER_SECOND: Trade frequency normalized by time
        if 'trade_count' in df.columns and 'bar_duration_ms' in df.columns:
            duration_seconds = df['bar_duration_ms'] / 1000.0
            duration_seconds = duration_seconds.replace(0, 0.001)  # Avoid division by zero
            df['trades_per_second'] = df['trade_count'] / duration_seconds
        else:
            # Fallback: Use 100 trades / 1 second (fixed bucket assumption)
            df['trades_per_second'] = 100.0
        
        # 3. DURATION_Z_SCORE: Is this bar faster/slower than recent average?
        if 'bar_duration_ms' in df.columns:
            rolling_window = 50
            df['duration_mean'] = df['bar_duration_ms'].rolling(window=rolling_window).mean()
            df['duration_std'] = df['bar_duration_ms'].rolling(window=rolling_window).std().replace(0, 1)
            df['duration_z_score'] = (df['bar_duration_ms'] - df['duration_mean']) / df['duration_std']
            df.drop(columns=['duration_mean', 'duration_std'], inplace=True)
        else:
            df['duration_z_score'] = 0
        
        # 4. DELTA_RATIO: Normalized delta in range [-1, +1]
        # Formula: (Buy - Sell) / Total = Delta / Total
        if 'delta' in df.columns and 'total_volume' in df.columns:
            total_vol_safe = df['total_volume'].replace(0, 0.0001)
            df['delta_ratio'] = df['delta'] / total_vol_safe
            df['delta_ratio'] = df['delta_ratio'].clip(-1, 1)  # Safety bounds
        else:
            df['delta_ratio'] = 0
        
        # 5. MIN_MAX_DELTA_RATIO: Fight for zone (max delta / min delta)
        # NOTE: API doesn't provide min/max delta per bar, so we approximate using rolling extremes
        if 'delta' in df.columns:
            window = 10
            df['rolling_max_delta'] = df['delta'].rolling(window=window).max()
            df['rolling_min_delta'] = df['delta'].rolling(window=window).min()
            # Avoid division by zero
            denominator = df['rolling_min_delta'].replace(0, 0.0001)
            df['min_max_delta_ratio'] = df['rolling_max_delta'] / denominator.abs()
            df.drop(columns=['rolling_max_delta', 'rolling_min_delta'], inplace=True)
        else:
            df['min_max_delta_ratio'] = 1
        
        # 6. CUMULATIVE_DELTA_DIVERGENCE: Price rises but cumulative delta falls?
        # Divergence = (Price_Change > 0) AND (CVD_Change < 0) OR vice versa
        if 'cum_delta' in df.columns:
            price_change = df['close'].diff(10)  # 10-bar price change
            cvd_change = df['cum_delta'].diff(10)  # 10-bar CVD change
            # Divergence signal: +1 (bullish divergence), -1 (bearish divergence), 0 (no divergence)
            df['cumulative_delta_divergence'] = np.where(
                (price_change > 0) & (cvd_change < 0), 1,  # Price up, CVD down = Hidden weakness
                np.where((price_change < 0) & (cvd_change > 0), -1, 0)  # Price down, CVD up = Hidden strength
            )
        else:
            df['cumulative_delta_divergence'] = 0
        
        # 7. CONSECUTIVE_DELTA: How many consecutive bars had positive/negative delta?
        if 'delta' in df.columns:
            delta_sign = np.sign(df['delta'])
            # Count consecutive occurrences of the same sign
            df['consecutive_delta'] = (delta_sign.groupby((delta_sign != delta_sign.shift()).cumsum()).cumcount() + 1) * delta_sign
        else:
            df['consecutive_delta'] = 0
        
        # 8. CVD_MOMENTUM: Rate of change in cumulative volume delta
        if 'cum_delta' in df.columns:
            df['cvd_momentum'] = df['cum_delta'].diff(5)  # 5-bar change
        else:
            df['cvd_momentum'] = 0
        
        # 9. WHALE_DOMINANCE: Percentage of volume from largest single trade
        if 'largest_trade_qty' in df.columns and 'total_volume' in df.columns:
            total_vol_safe = df['total_volume'].replace(0, 0.0001)
            df['whale_dominance'] = (df['largest_trade_qty'] / total_vol_safe) * 100
        else:
            df['whale_dominance'] = 0
        
        # 10. LARGE_TRADE_RATIO: Volume from trades > threshold / Total volume
        # NOTE: API provides 'large_trade_dominance' already, we can use it directly
        if 'large_trade_dominance' in df.columns:
            df['large_trade_ratio'] = df['large_trade_dominance'] / 100.0  # Convert percentage to ratio
        else:
            df['large_trade_ratio'] = 0
        
        # 11. EFFICIENCY_RATIO: (High - Low) / Total_Volume
        # Low value = High volume in tight range = Absorption/Fight
        if 'high' in df.columns and 'low' in df.columns and 'total_volume' in df.columns:
            price_range = df['high'] - df['low']
            total_vol_safe = df['total_volume'].replace(0, 0.0001)
            df['efficiency_ratio'] = price_range / total_vol_safe
        else:
            df['efficiency_ratio'] = 0
        
        # Note: close_location, avg_trade_size, delta_pct already provided by API
        # We ensure they exist, otherwise create them
        
        # 12. CLOSE_LOCATION: Where does price close in the bar? (Already in API data)
        if 'close_location' not in df.columns:
            if 'high' in df.columns and 'low' in df.columns:
                price_range = df['high'] - df['low']
                price_range_safe = price_range.replace(0, 0.0001)
                df['close_location'] = (df['close'] - df['low']) / price_range_safe
            else:
                df['close_location'] = 0.5  # Neutral
        
        # 13. AVG_TRADE_SIZE: Already in API data, but verify
        if 'avg_trade_size' not in df.columns:
            if 'total_volume' in df.columns and 'trade_count' in df.columns:
                trade_count_safe = df['trade_count'].replace(0, 1)
                df['avg_trade_size'] = df['total_volume'] / trade_count_safe
            else:
                df['avg_trade_size'] = 0
        
        # 14. DELTA_PCT: Already in API data (delta / volume)
        if 'delta_pct' not in df.columns:
            if 'delta' in df.columns and 'total_volume' in df.columns:
                total_vol_safe = df['total_volume'].replace(0, 0.0001)
                df['delta_pct'] = (df['delta'] / total_vol_safe) * 100
            else:
                df['delta_pct'] = 0
        
        return df

    def prepare_training_dataset(self, symbol, days=30, primary_strategy='volume_imbalance',
                                 lookahead=50, tp_pct=0.004, sl_pct=0.002):
        """Full Pipeline."""
        logger.info(f"--- STARTING PIPELINE for {symbol} ---")

        # 1. Data
        df = self.fetch_historical_data(symbol, days)
        if df.empty: return df

        # 2. Primary Signals
        df = self.generate_primary_signals(df, strategy=primary_strategy)

        # 3. TBM Labels (Ground Truth)
        tbm_labels = self.apply_triple_barrier_labels(df, lookahead, tp_pct, sl_pct)

        # 4. Meta Labels
        df = self.create_meta_labels(df, tbm_labels)

        # 5. Features
        df = self.add_market_features(df)
        df = self.add_trend_features(df)

        # Clean
        required_cols = ['primary_signal', 'meta_label']
        df_clean = df.dropna(subset=required_cols)
        # Keep only rows where a signal occurred (for training the Meta-Model)
        # Optional: Keep all rows if training a sequence model

        # ========================================================================
        # DROP "TOXIC" COLUMNS (Stationarity Check)
        # ========================================================================
        # Raw absolute value columns confuse tree-based models due to non-stationarity.
        # We keep only normalized/stationary features (ratios, percentages, z-scores, etc.)
        toxic_columns = [
            # Raw prices (non-stationary)
            'open', 'high', 'low', 'close',
            # Raw volumes (non-stationary)
            'total_volume', 'buy_volume', 'sell_volume', 'taker_buy_volume',
            # Raw absolute deltas (non-stationary cumulative)
            'cum_delta', 'cumulative_delta',
            # Raw moving averages (non-stationary)
            'ma_20', 'ma_50', 'sma',
            # Other raw absolute values
            'vwap', 'flow_mean', 'flow_std',
            # Intermediate calculation columns
            'largest_trade_qty', 'large_trade_dominance',
            # Trade counts (can be volatile/non-stationary)
            'trade_count', 'buy_trades', 'sell_trades',
            # Bar metadata
            'bar_duration_ms', 'price_range',
            # Symbol and timestamp (not features)
            'symbol', 'start_ts'
        ]
        
        # Drop only if they exist (avoid KeyError)
        existing_toxic = [col for col in toxic_columns if col in df_clean.columns]
        if existing_toxic:
            logger.info(f"Dropping {len(existing_toxic)} toxic columns: {existing_toxic}")
            df_clean = df_clean.drop(columns=existing_toxic)
        
        logger.info(f"Pipeline complete. Rows: {len(df_clean)}, Features: {len(df_clean.columns)}")
        return df_clean

    def test_chart_no_resample(self, symbol='btcusdt', days=2, num_bars=500, strategy='volume_imbalance'):
        """
        Visualizes the NATIVE Tick/Trade Bars without resampling.
        This shows exactly what the algorithm sees.
        """
        import mplfinance as mpf

        logger.info(f"Visualizing Native Tick Bars (Last {num_bars})...")

        df = self.fetch_historical_data(symbol, days)
        if df.empty: return

        # Apply strategy to get signals
        df = self.generate_primary_signals(df, strategy=strategy)

        # Slice last N bars
        df_plot = df.tail(num_bars).copy()

        # Set index for mplfinance
        # NOTE: Even if time gaps exist (nights/weekends), mplfinance plots rows sequentially
        # which is perfect for Tick Bars (Event Time).
        df_plot['Date'] = pd.to_datetime(df_plot['start_ts'])
        df_plot.set_index('Date', inplace=True)

        # Rename for mplfinance
        ohlc_df = df_plot[['open', 'high', 'low', 'close', 'total_volume']].copy()
        ohlc_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Markers
        long_mask = df_plot['primary_direction'] == 1
        short_mask = df_plot['primary_direction'] == -1

        long_markers = df_plot['low'] * 0.999
        long_markers[~long_mask] = np.nan

        short_markers = df_plot['high'] * 1.001
        short_markers[~short_mask] = np.nan

        apds = [
            mpf.make_addplot(long_markers, type='scatter', markersize=80, marker='^', color='green',
                             label='Long Signal'),
            mpf.make_addplot(short_markers, type='scatter', markersize=80, marker='v', color='red',
                             label='Short Signal')
        ]

        # Add Delta Panel if available
        if 'delta' in df_plot.columns:
            # Color bars for delta
            delta_colors = ['g' if v >= 0 else 'r' for v in df_plot['delta']]
            apds.append(mpf.make_addplot(df_plot['delta'], panel=2, type='bar', color=delta_colors, ylabel='Delta'))

        mpf.plot(
            ohlc_df,
            type='candle',
            style='charles',
            title=f'{symbol} - Tick Bars ({strategy})',
            ylabel='Price',
            volume=True,
            addplot=apds,
            panel_ratios=(3, 1, 1) if 'delta' in df_plot.columns else (3, 1),
            figsize=(14, 8)
        )


if __name__ == "__main__":
    labeler = DatasetLabeler()

    # List of symbols to process
    symbols = ['btcusdt', 'ethusdt', 'solusdt']
    #symbols = ['ethusdt']

    for symbol in symbols:
        # 1. Run Pipeline
        df = labeler.prepare_training_dataset(
            symbol=symbol,
            days=300,
            primary_strategy='volume_imbalance',
            lookahead=LOOKAHEAD_PERIODS,
            tp_pct=TAKE_PROFIT_PCT,
            sl_pct=STOP_LOSS_PCT
        )

        if not df.empty:
            # Save only the rows relevant for training (Signals)
            training_data = df[df['primary_signal'] == 1].copy()

            output_file = f"meta_labeling_dataset_{symbol}.csv"
            training_data.to_csv(output_file, index=False)

            logger.info(f"Saved {len(training_data)} training samples to {output_file}")

            # 2. Visualize
            labeler.test_chart_no_resample(symbol=symbol, days=1, num_bars=300)