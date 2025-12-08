
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
# Endpoint to manage the golden dataset (Assumed based on requirements)
GOLDEN_DATASET_ENDPOINT = f"{BASE_URL}/golden_dataset"
AUTH_TOKEN = os.getenv("AUTH_BEARER_TOKEN", "")

# Symbols to check
SYMBOLS = os.getenv("SYMBOLS", "btcusdt,ethusdt,solusdt").split(",")
SYMBOLS = [s.strip().lower() for s in SYMBOLS]

class EventReason(Enum):
    VOLATILITY = "VOLATILITY"         # Price moved > X%
    VOLUME_SPIKE = "VOLUME_SPIKE"     # Volume > 2x 30-day average
    TREND_REVERSAL = "TREND_REVERSAL" # Local top/bottom
    SEASON_CHANGE = "SEASON_CHANGE"   # Alt/BTC season shift
    OTHER = "OTHER"

class GoldenDatasetSelector:
    def __init__(self):
        self.headers = {}
        if AUTH_TOKEN:
            self.headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        
        self.headers["Content-Type"] = "application/json"

    def fetch_historical_data(self, symbol, days=30):
        """
        Fetch historical data for a specific symbol.
        We fetch 30 days to calculate averages, even if we only scan the last 3 days for new events.
        """
        logger.info(f"Fetching {days} days of data for {symbol}...")
        all_data = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # We'll fetch in one go or chunks if needed. main.py uses 3 chunks.
        # Let's use a simplified chunking for the script.
        # Validating via chunks of 10 days to be safe with large data
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
                # Assuming raw trade groups download
                # Note: main.py uses 'gzip, deflate'
                headers = self.headers.copy()
                headers["Accept-Encoding"] = "gzip, deflate"
                
                response = requests.get(f"{BASE_URL}/download/trade_groups", params=params, headers=headers, stream=True)
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
            # Filter by symbol incase API returns mixed data
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

    def get_existing_events(self, symbol):
        """Fetch already marked events for this symbol to avoid duplicates."""
        try:
            params = {'symbol': symbol}
            response = requests.get(GOLDEN_DATASET_ENDPOINT, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json() # Assuming list of events
            elif response.status_code == 404:
                return []
        except Exception as e:
            logger.warning(f"Could not fetch existing events (API might not exist yet?): {e}")
        return []

    def submit_event(self, symbol, event_time, reason, description, start_time=None, end_time=None):
        """Submit a new event to the webservice."""
        # Defined window: T - 10h to T + 5h (defaults)
        if start_time is None:
            start_time = event_time - timedelta(hours=10)
        if end_time is None:
            end_time = event_time + timedelta(hours=5)
        
        # Check if already exists to satisfy "we dont send anything when the data was already marked"
        existing_events = self.get_existing_events(symbol)
        for e in existing_events:
            try:
                # Check for same reason and close timeframe (within 2 hours)
                e_reason = e.get('reason')
                e_time_str = e.get('event_time')
                if e_reason == reason.value and e_time_str:
                    e_time = pd.to_datetime(e_time_str).replace(tzinfo=timezone.utc)
                    # Use utc if event_time is utc
                    if event_time.tzinfo is None:
                        # Assume UTC if naive, usually pandas ts is naive or aware
                         pass
                    
                    # Ensure matching timezones for comparison
                    if e_time.tzinfo and event_time.tzinfo:
                         diff = abs((e_time - event_time).total_seconds())
                    elif e_time.tzinfo is None and event_time.tzinfo is None:
                         diff = abs((e_time - event_time).total_seconds())
                    else:
                         # Mix of aware and naive, strip valid one or localize naive
                         diff = abs((e_time.replace(tzinfo=None) - event_time.replace(tzinfo=None)).total_seconds())

                    if diff < 7200: # 2 hours buffer
                        logger.info(f"Skipping duplicate event: {reason.value} at {event_time} (Found similar at {e_time})")
                        return
            except Exception as ex:
                logger.warning(f"Error checking duplicate: {ex}")
                continue

        # Construct Payload
        payload = {
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "event_time": event_time.isoformat(), # The 'T' center
            "reason": reason.value,
            "description": description
        }
        
        logger.info(f"Submitting event for {symbol}: {reason.value} at {event_time} (Window: {start_time} - {end_time})")
        
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

    def analyze_symbol(self, symbol):
        # 1. Fetch 30 days of data for baseline comparisons
        df = self.fetch_historical_data(symbol, days=300)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return

        # Resample to 1H for Volatility/Volume analysis and 4H for Trends
        # Using 1H for granular volume checks
        df_1h = df.resample('1h', on='start_ts').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'total_volume': 'sum' # Volume sums up
        }).dropna()
        
        # Resample to Daily for Volatility baseline and Volume Baseline
        df_daily = df.resample('1D', on='start_ts').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'total_volume': 'sum'
        }).dropna()

        # Calculate Baselines (30-day)
        # We need simpler rolling stats
        df_daily['vol_ma_30'] = df_daily['total_volume'].rolling(window=30, min_periods=5).mean()
        
        # Identify Scan Window (All fetched data)
        if not df.empty:
            scan_start_time = df['start_ts'].min()
        else:
            scan_start_time = datetime.now(timezone.utc) - timedelta(days=3)
        
        # Collection of candidate events
        candidates = []

        # A. Volume Spikes (Daily)
        # "Volume > 2x the 30-day average"
        scan_daily = df_daily[df_daily.index >= scan_start_time]
        for ts, row in scan_daily.iterrows():
            if pd.notna(row['vol_ma_30']) and row['vol_ma_30'] > 0:
                if row['total_volume'] > 2.0 * row['vol_ma_30']:
                    desc = f"Volume Spike: {row['total_volume']:.0f} > 2x Avg ({row['vol_ma_30']:.0f})"
                    candidates.append({
                        'symbol': symbol,
                        'event_time': ts,
                        'reason': EventReason.VOLUME_SPIKE,
                        'description': desc
                    })

        # B. Price Volatility (Daily)
        # "Any day where price moved > X% (e.g., 5%)"
        for ts, row in scan_daily.iterrows():
            if row['open'] != 0:
                pct_move = abs((row['close'] - row['open']) / row['open'])
                if pct_move > 0.05:
                    desc = f"High Volatility: Daily move {pct_move:.1%}"
                    candidates.append({
                        'symbol': symbol,
                        'event_time': ts,
                        'reason': EventReason.VOLATILITY,
                        'description': desc
                    })

        # C. Trend Reversals (4H)
        # Resample to 4H
        df_4h = df.resample('4h', on='start_ts').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        # Check Local Tops/Bottoms in 4H
        window_neighbors = 2 # Check 2 neighbors on each side
        
        relevant_indices = [i for i, ts in enumerate(df_4h.index) if ts >= scan_start_time]
        
        for i in relevant_indices:
            if i < window_neighbors or i >= len(df_4h) - window_neighbors:
                continue
                
            curr = df_4h.iloc[i]
            prevs = df_4h.iloc[i-window_neighbors:i]
            nexts = df_4h.iloc[i+1:i+1+window_neighbors]
            
            # Local Top
            if (curr['high'] > prevs['high'].max()) and (curr['high'] > nexts['high'].max()):
                desc = f"Trend Reversal: Local Top at {curr['high']}"
                candidates.append({
                    'symbol': symbol,
                    'event_time': df_4h.index[i],
                    'reason': EventReason.TREND_REVERSAL,
                    'description': desc
                })
                
            # Local Bottom
            elif (curr['low'] < prevs['low'].min()) and (curr['low'] < nexts['low'].min()):
                desc = f"Trend Reversal: Local Bottom at {curr['low']}"
                candidates.append({
                    'symbol': symbol,
                    'event_time': df_4h.index[i],
                    'reason': EventReason.TREND_REVERSAL,
                    'description': desc
                })

        # Sort candidates by event_time to resolve overlaps
        candidates.sort(key=lambda x: x['event_time'])
        
        if not candidates:
            return

        # Assign initial default windows
        for c in candidates:
            c['start_time'] = c['event_time'] - timedelta(hours=2)
            c['end_time'] = c['event_time'] + timedelta(hours=5)

        # Resolve Overlaps
        # "calculate the exact middle between both events and use this as end of the first event and +1s for the beginning of the second event."
        for i in range(len(candidates) - 1):
            curr_event = candidates[i]
            next_event = candidates[i+1]
            
            # Check for overlap: Current End > Next Start
            if curr_event['end_time'] > next_event['start_time']:
                # Calculate middle of the EVENT_TIMES (timestamps of the events themselves)
                t1 = curr_event['event_time']
                t2 = next_event['event_time']
                
                # Careful with timezones
                midpoint = t1 + (t2 - t1) / 2
                
                # Update boundaries
                curr_event['end_time'] = midpoint
                next_event['start_time'] = midpoint + timedelta(seconds=1)
                
                logger.info(f"Resolved overlap between {t1} and {t2}. Split at {midpoint}.")

        # Submit Events
        for c in candidates:
            self.submit_event(
                symbol, 
                c['event_time'], 
                c['reason'], 
                c['description'], 
                start_time=c['start_time'], 
                end_time=c['end_time']
            )

        # D. Season Changes
        # This is harder to do per-symbol without global context.
        # Implemented as a TODO or requires aggregating all symbols first.
        # Since this runs per-symbol in this loop, we skip or do it separately.

    def run(self):
        logger.info("Starting Golden Dataset Selector...")
        
        # 0. Optional: Calculate Global Metrics for Season Changes (if needed)
        # For now, we process each symbol independently
        
        for symbol in SYMBOLS:
            try:
                self.analyze_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
        
        logger.info("Golden Dataset Selector finished.")

if __name__ == "__main__":
    selector = GoldenDatasetSelector()
    selector.run()
