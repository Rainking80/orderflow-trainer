import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from enum import Enum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8120"
# Endpoint to manage the golden dataset
GOLDEN_DATASET_ENDPOINT = f"{BASE_URL}/golden_dataset"
AUTH_TOKEN = os.getenv("AUTH_BEARER_TOKEN", "")

# Symbols to check
SYMBOLS = os.getenv("SYMBOLS", "btcusdt,ethusdt,solusdt").split(",")
SYMBOLS = [s.strip().lower() for s in SYMBOLS]

# Triple Barrier Parameters
LOOKAHEAD_PERIODS = int(os.getenv("LOOKAHEAD_PERIODS", "50"))  # Z periods
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))  # X% (2%)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.01"))  # Y% (1%)


class TradeLabel(Enum):
    """Labels for Triple Barrier Method"""
    LONG_ENTRY = "LONG_ENTRY"
    SHORT_ENTRY = "SHORT_ENTRY"
    NO_ACTION = "NO_ACTION"


class GoldenDatasetSelector:
    def __init__(self):
        self.headers = {}
        if AUTH_TOKEN:
            self.headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

        self.headers["Content-Type"] = "application/json"

    def fetch_historical_data(self, symbol, days=30):
        """
        Fetch historical data for a specific symbol.
        """
        logger.info(f"Fetching {days} days of data for {symbol}...")
        all_data = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Fetch in chunks for safety with large data
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
            # Filter by symbol in case API returns mixed data
            if 'symbol' in df.columns:
                df = df[df['symbol'].str.lower() == symbol.lower()].copy()

            if 'start_ts' in df.columns:
                df['start_ts'] = pd.to_datetime(df['start_ts'], utc=True, format='mixed')
                df = df.sort_values('start_ts').reset_index(drop=True)

                # Ensure numeric columns
                cols = ['open', 'high', 'low', 'close', 'total_volume']
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def apply_triple_barrier_labels(self, df, lookahead=50, tp_pct=0.02, sl_pct=0.01, debug_first_n=0):
        """
        Apply Triple Barrier Method to generate trading labels.

        Parameters:
        - df: DataFrame with 'close', 'high', 'low' prices
        - lookahead: How many bars to look into the future (Vertical Barrier / Z periods)
        - tp_pct: Take Profit percentage (X%)
        - sl_pct: Stop Loss percentage (Y%)
        - debug_first_n: Number of initial samples to debug log

        Returns:
        - List of labels: LONG_ENTRY, SHORT_ENTRY, or NO_ACTION
        """
        labels = []

        # Check if we have high/low data for intrabar analysis
        has_high_low = 'high' in df.columns and 'low' in df.columns

        # Diagnostic counters
        long_tp_hits = 0
        long_sl_hits = 0
        short_tp_hits = 0
        short_sl_hits = 0
        both_hit = 0
        neither_hit = 0

        for i in range(len(df) - lookahead):
            current_price = df['close'].iloc[i]

            # Calculate barrier prices for LONG
            tp_price_long = current_price * (1 + tp_pct)
            sl_price_long = current_price * (1 - sl_pct)

            # Calculate barrier prices for SHORT
            tp_price_short = current_price * (1 - tp_pct)
            sl_price_short = current_price * (1 + sl_pct)

            # Extract future window - check high/low if available, otherwise use close
            if has_high_low:
                future_high = df['high'].iloc[i + 1: i + 1 + lookahead]
                future_low = df['low'].iloc[i + 1: i + 1 + lookahead]

                # Check when barriers are hit for LONG (TP via high, SL via low)
                hit_tp_long = np.where(future_high >= tp_price_long)[0]
                hit_sl_long = np.where(future_low <= sl_price_long)[0]

                # Check when barriers are hit for SHORT (TP via low, SL via high)
                hit_tp_short = np.where(future_low <= tp_price_short)[0]
                hit_sl_short = np.where(future_high >= sl_price_short)[0]

                # Debug logging for first few samples
                if i < debug_first_n:
                    logger.info(f"  Sample {i}: current_price={current_price:.2f}")
                    logger.info(
                        f"    LONG barriers: TP={tp_price_long:.2f} (+{tp_pct * 100}%), SL={sl_price_long:.2f} (-{sl_pct * 100}%)")
                    logger.info(f"    Future window: high max={future_high.max():.2f}, low min={future_low.min():.2f}")
                    logger.info(f"    LONG hits: TP at bars {hit_tp_long.tolist()}, SL at bars {hit_sl_long.tolist()}")
                    logger.info(
                        f"    SHORT barriers: TP={tp_price_short:.2f} (-{tp_pct * 100}%), SL={sl_price_short:.2f} (+{sl_pct * 100}%)")
                    logger.info(
                        f"    SHORT hits: TP at bars {hit_tp_short.tolist()}, SL at bars {hit_sl_short.tolist()}")
            else:
                # Fallback to close prices only
                future_close = df['close'].iloc[i + 1: i + 1 + lookahead]

                # Check when barriers are hit for LONG
                hit_tp_long = np.where(future_close >= tp_price_long)[0]
                hit_sl_long = np.where(future_close <= sl_price_long)[0]

                # Check when barriers are hit for SHORT
                hit_tp_short = np.where(future_close <= tp_price_short)[0]
                hit_sl_short = np.where(future_close >= sl_price_short)[0]

            # First index (time point) of hitting for LONG
            first_tp_long = hit_tp_long[0] if len(hit_tp_long) > 0 else lookahead + 1
            first_sl_long = hit_sl_long[0] if len(hit_sl_long) > 0 else lookahead + 1

            # First index (time point) of hitting for SHORT
            first_tp_short = hit_tp_short[0] if len(hit_tp_short) > 0 else lookahead + 1
            first_sl_short = hit_sl_short[0] if len(hit_sl_short) > 0 else lookahead + 1

            # Track diagnostics
            if len(hit_tp_long) > 0:
                long_tp_hits += 1
            if len(hit_sl_long) > 0:
                long_sl_hits += 1
            if len(hit_tp_short) > 0:
                short_tp_hits += 1
            if len(hit_sl_short) > 0:
                short_sl_hits += 1
            if (len(hit_tp_long) > 0 or len(hit_tp_short) > 0) and (len(hit_sl_long) > 0 or len(hit_sl_short) > 0):
                both_hit += 1
            if len(hit_tp_long) == 0 and len(hit_sl_long) == 0 and len(hit_tp_short) == 0 and len(hit_sl_short) == 0:
                neither_hit += 1

            # Label Logic: LONG_ENTRY if TP hits before SL within lookahead
            if first_tp_long < first_sl_long and first_tp_long < lookahead:
                labels.append(TradeLabel.LONG_ENTRY.value)
            # Label Logic: SHORT_ENTRY if TP hits before SL within lookahead
            elif first_tp_short < first_sl_short and first_tp_short < lookahead:
                labels.append(TradeLabel.SHORT_ENTRY.value)
            else:
                labels.append(TradeLabel.NO_ACTION.value)

        # Log diagnostic summary
        total_samples = len(df) - lookahead
        logger.info(f"Triple Barrier Diagnostics (out of {total_samples} samples):")
        logger.info(f"  - LONG TP hits: {long_tp_hits} ({long_tp_hits / total_samples * 100:.1f}%)")
        logger.info(f"  - LONG SL hits: {long_sl_hits} ({long_sl_hits / total_samples * 100:.1f}%)")
        logger.info(f"  - SHORT TP hits: {short_tp_hits} ({short_tp_hits / total_samples * 100:.1f}%)")
        logger.info(f"  - SHORT SL hits: {short_sl_hits} ({short_sl_hits / total_samples * 100:.1f}%)")
        logger.info(f"  - Both barriers hit: {both_hit} ({both_hit / total_samples * 100:.1f}%)")
        logger.info(f"  - No barriers hit: {neither_hit} ({neither_hit / total_samples * 100:.1f}%)")

        # Fill remaining rows with NO_ACTION (they don't have enough future data)
        labels.extend([TradeLabel.NO_ACTION.value] * lookahead)
        return labels

    def get_existing_events(self, symbol):
        """Fetch already marked events for this symbol to avoid duplicates."""
        try:
            params = {'symbol': symbol}
            response = requests.get(GOLDEN_DATASET_ENDPOINT, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return []
        except Exception as e:
            logger.warning(f"Could not fetch existing events (API might not exist yet?): {e}")
        return []

    def calculate_event_window(self, df, signal_index, lookback_buckets=60, lookforward_buckets=120, safety_buffer_minutes=5):
        """
        Calculate start_time and end_time for an event based on bucket indices.
        
        Parameters:
        - df: DataFrame with 'start_ts' column containing timestamps
        - signal_index: Integer index where the signal was detected
        - lookback_buckets: Number of buckets to look back (default: 60)
        - lookforward_buckets: Number of buckets to look forward (default: 120)
        - safety_buffer_minutes: Safety buffer in minutes to add/subtract (default: 5)
        
        Returns:
        - (start_time, end_time) tuple or (None, None) if indices are out of bounds
        """
        # Calculate target indices with bounds checking
        start_index = signal_index - lookback_buckets
        end_index = signal_index + lookforward_buckets
        
        # Check if indices are within valid range
        if start_index < 0 or end_index >= len(df):
            logger.warning(
                f"Signal at index {signal_index} is too close to DataFrame boundary. "
                f"Need [{start_index}, {end_index}] but have [0, {len(df)-1}]"
            )
            return None, None
        
        # Get timestamps from the buckets
        start_bucket_time = df.iloc[start_index]['start_ts']
        end_bucket_time = df.iloc[end_index]['start_ts']
        
        # Apply safety buffer
        safety_delta = timedelta(minutes=safety_buffer_minutes)
        start_time = start_bucket_time - safety_delta
        end_time = end_bucket_time + safety_delta
        
        return start_time, end_time
    
    def submit_event(self, symbol, event_time, label, description, start_time, end_time):
        """
        Submit a new labeled event to the webservice.
        
        Parameters:
        - symbol: Trading symbol
        - event_time: Timestamp of the signal/event
        - label: Trade label (LONG_ENTRY, SHORT_ENTRY, etc.)
        - description: Human-readable description
        - start_time: Start of the data window (required, should be timestamp at i-60 bucket with safety buffer)
        - end_time: End of the data window (required, should be timestamp at i+120 bucket with safety buffer)
        """
        # Check if already exists to avoid duplicates
        existing_events = self.get_existing_events(symbol)
        for e in existing_events:
            try:
                e_reason = e.get('reason')
                e_time_str = e.get('event_time')
                if e_reason == label and e_time_str:
                    e_time = pd.to_datetime(e_time_str).replace(tzinfo=timezone.utc)

                    # Compare timestamps
                    if event_time.tzinfo is None:
                        event_time_cmp = event_time.replace(tzinfo=timezone.utc)
                    else:
                        event_time_cmp = event_time

                    diff = abs((e_time - event_time_cmp).total_seconds())
                    if diff < 60:  # 60 sec buffer
                        logger.info(f"Skipping duplicate event: {label} at {event_time}")
                        return
            except Exception as ex:
                logger.warning(f"Error checking duplicate: {ex}")
                continue

        # Construct payload
        payload = {
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "event_time": event_time.isoformat(),
            "reason": label,
            "description": description
        }

        logger.info(f"Submitting event for {symbol}: {label} at {event_time}")

        try:
            response = requests.post(GOLDEN_DATASET_ENDPOINT, json=payload, headers=self.headers)
            if response.status_code in [200, 201]:
                logger.info("Event submitted successfully.")
            elif response.status_code == 409:
                logger.info("Event already exists (Conflict).")
            else:
                logger.error(f"Failed to submit event: {response.text}")
        except Exception as e:
            logger.error(f"Error submitting event: {e}")

    def calculate_dynamic_lookahead(self, df, tp_pct, min_lookahead=50, max_lookahead=10000, sample_size=1000):
        """
        Calculate appropriate lookahead period based on symbol's price volatility.

        Parameters:
        - df: DataFrame with price data
        - tp_pct: Target take profit percentage
        - min_lookahead: Minimum lookahead to test
        - max_lookahead: Maximum lookahead to allow
        - sample_size: Number of samples to test

        Returns:
        - Optimal lookahead period
        """
        if len(df) < max_lookahead:
            max_lookahead = len(df) - 1

        # Test different lookaheads to find one where avg movement >= tp_pct
        test_lookaheads = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
        test_lookaheads = [l for l in test_lookaheads if l <= max_lookahead and l >= min_lookahead]

        for lookahead in test_lookaheads:
            movements = []
            step = max(1, (len(df) - lookahead) // sample_size)

            for i in range(0, min(len(df) - lookahead, sample_size * step), step):
                current_price = df['close'].iloc[i]
                future_window = df.iloc[i + 1: i + 1 + lookahead]

                max_up = ((future_window['high'].max() - current_price) / current_price)
                max_down = ((current_price - future_window['low'].min()) / current_price)

                movements.append(max(max_up, max_down))

            avg_movement = np.mean(movements) if movements else 0

            # We want avg movement to be at least 2x the TP target for good signal generation
            if avg_movement >= tp_pct * 2:
                logger.info(f"Selected lookahead={lookahead} (avg movement: {avg_movement * 100:.2f}%)")
                return lookahead

        # If no suitable lookahead found, use max available
        logger.warning(f"No ideal lookahead found. Using max_lookahead={max_lookahead}")
        return max_lookahead

    def analyze_symbol(self, symbol):
        """
        Analyze a symbol using Triple Barrier Method.
        Generate training labels and submit events for actionable trades.
        """
        # Fetch historical data
        df = self.fetch_historical_data(symbol, days=10)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return

        # Debug: Log data structure and sample
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        logger.info(f"Columns: {df.columns.tolist()}")
        if len(df) > 0:
            logger.info(f"Sample data (first row): {df.iloc[0].to_dict()}")
            if 'close' in df.columns:
                logger.info(f"Close price range: min={df['close'].min()}, max={df['close'].max()}")

        # Calculate dynamic lookahead based on symbol volatility
        lookahead = self.calculate_dynamic_lookahead(df, TAKE_PROFIT_PCT)
        logger.info(f"Using dynamic lookahead={lookahead} for {symbol}")

        # Apply Triple Barrier labels
        logger.info(f"Applying Triple Barrier Method to {symbol}...")
        labels = self.apply_triple_barrier_labels(
            df,
            lookahead=lookahead,
            tp_pct=TAKE_PROFIT_PCT,
            sl_pct=STOP_LOSS_PCT
        )

        # Add labels to dataframe
        df['label'] = labels

        # Debug: Count label distribution
        label_counts = df['label'].value_counts()
        logger.info(f"Label distribution for {symbol}:")
        logger.info(f"  - LONG_ENTRY: {label_counts.get(TradeLabel.LONG_ENTRY.value, 0)}")
        logger.info(f"  - SHORT_ENTRY: {label_counts.get(TradeLabel.SHORT_ENTRY.value, 0)}")
        logger.info(f"  - NO_ACTION: {label_counts.get(TradeLabel.NO_ACTION.value, 0)}")

        # Filter for actionable trades only (exclude NO_ACTION)
        actionable_df = df[df['label'] != TradeLabel.NO_ACTION.value].copy()

        if actionable_df.empty:
            logger.info(f"No actionable trades found for {symbol}")
            return

        logger.info(f"Found {len(actionable_df)} actionable trades for {symbol}")
        logger.info(f"  - LONG_ENTRY: {(actionable_df['label'] == TradeLabel.LONG_ENTRY.value).sum()}")
        logger.info(f"  - SHORT_ENTRY: {(actionable_df['label'] == TradeLabel.SHORT_ENTRY.value).sum()}")

        # Submit events for actionable trades
        # Use integer-based iteration to properly calculate window based on bucket indices
        for i in range(len(df)):
            label = df.iloc[i]['label']
            
            # Skip non-actionable signals
            if label == TradeLabel.NO_ACTION.value:
                continue
            
            event_time = df.iloc[i]['start_ts']
            current_price = df.iloc[i]['close']
            
            # Calculate dynamic window based on bucket indices (i-60 to i+120)
            start_time, end_time = self.calculate_event_window(
                df, 
                signal_index=i,
                lookback_buckets=60,
                lookforward_buckets=120,
                safety_buffer_minutes=5
            )
            
            # Skip if signal is too close to DataFrame boundaries
            if start_time is None or end_time is None:
                logger.info(f"Skipping signal at index {i} - insufficient data window")
                continue

            if label == TradeLabel.LONG_ENTRY.value:
                tp_price = current_price * (1 + TAKE_PROFIT_PCT)
                sl_price = current_price * (1 - STOP_LOSS_PCT)
                description = f"LONG: Entry={current_price:.4f}, TP={tp_price:.4f} (+{TAKE_PROFIT_PCT * 100}%), SL={sl_price:.4f} (-{STOP_LOSS_PCT * 100}%)"
            else:  # SHORT_ENTRY
                tp_price = current_price * (1 - TAKE_PROFIT_PCT)
                sl_price = current_price * (1 + STOP_LOSS_PCT)
                description = f"SHORT: Entry={current_price:.4f}, TP={tp_price:.4f} (-{TAKE_PROFIT_PCT * 100}%), SL={sl_price:.4f} (+{STOP_LOSS_PCT * 100}%)"

            self.submit_event(
                symbol=symbol,
                event_time=event_time,
                label=label,
                description=description,
                start_time=start_time,
                end_time=end_time
            )

    def run(self):
        logger.info("Starting Golden Dataset Selector with Triple Barrier Method...")
        logger.info(
            f"Parameters: Lookahead={LOOKAHEAD_PERIODS}, TP={TAKE_PROFIT_PCT * 100}%, SL={STOP_LOSS_PCT * 100}%")

        for symbol in SYMBOLS:
            try:
                self.analyze_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")

        logger.info("Golden Dataset Selector finished.")


if __name__ == "__main__":
    selector = GoldenDatasetSelector()
    selector.run()
