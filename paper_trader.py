# Configure logging
import sys
import logging
import json
import os
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
import io
import threading
import time
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PaperTradeManager:
    def __init__(self,
                 initial_balance=1000.0,
                 stop_loss_pct=0.015,
                 trailing_stop_pct=0.015,
                 leverage=1.0,
                 telegram_callback=None,
                 on_trade_open_callback=None,
                 on_trade_close_callback=None,
                 state_file="paper_trader_state.json",
                 fee_pct=0.001,
                 entry_threshold=0.65,  # Standard Hürde
                 reversal_threshold=0.80,  # Reversal Hürde (Strenger)
                 cooldown_seconds=300
                 ):
        """
        Args:
            fee_pct: 0.001 = 0.1% Gebühr pro Seite (Entry + Exit).
                     Entspricht Standard Taker Fee auf vielen Exchanges + Slippage Buffer.
        """
        self.state_file = state_file
        self.SL_PCT = stop_loss_pct
        self.TSL_PCT = trailing_stop_pct
        self.FEE_PCT = fee_pct  # Gebühren-Konstante
        
        # Support both old and new callback patterns
        # If new callbacks are provided, use them. Otherwise fall back to telegram_callback
        self.on_trade_open_callback = on_trade_open_callback if on_trade_open_callback else telegram_callback
        self.on_trade_close_callback = on_trade_close_callback if on_trade_close_callback else telegram_callback
        
        self.leverage = leverage
        self.active_trades = {}
        self.trade_history = []

        # Decision Logic Settings
        self.ENTRY_THRESH = entry_threshold
        self.REVERSAL_THRESH = reversal_threshold
        self.COOLDOWN_SEC = cooldown_seconds
        
        # Orderflow data buffers for chart generation
        # Store bucketed trade data and 1-minute candles for each symbol
        self.bucketed_trades = {}  # symbol -> list of bucketed trade data
        self.minute_candles = {}  # symbol -> list of 1-minute OHLCV candles
        self.symbols = []  # List of symbols to track
        
        # Telegram configuration
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip('"')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip('"')
        
        # Log Telegram config status (partial for security)
        if self.telegram_bot_token and self.telegram_chat_id:
            logger.info(f"📸 Telegram chart sender configured (token: ...{self.telegram_bot_token[-8:]}, chat: {self.telegram_chat_id})")
        else:
            logger.warning("⚠️ Telegram chart sender NOT configured - missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        
        # Scheduler thread
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Load persisted state or use initial values
        loaded_state = self.load_state()
        if loaded_state:
            self.balance = loaded_state.get('balance', initial_balance)
            self.wins = loaded_state.get('wins', 0)
            self.losses = loaded_state.get('losses', 0)
            logger.info(f"💾 Loaded persisted state: Balance=${self.balance:.2f}, W/L={self.wins}/{self.losses}")
        else:
            self.balance = initial_balance
            self.wins = 0
            self.losses = 0
            logger.info(f"🆕 Starting with fresh state: Balance=${self.balance:.2f}")

    def load_state(self):
        """Load persisted wallet state from JSON file"""
        if not os.path.exists(self.state_file):
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception as e:
            logger.error(f"Failed to load state from {self.state_file}: {e}")
            return None
    
    def save_state(self):
        """Save current wallet state to JSON file"""
        state = {
            'balance': self.balance,
            'wins': self.wins,
            'losses': self.losses
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"💾 State saved: Balance=${self.balance:.2f}, W/L={self.wins}/{self.losses}")
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")

    def has_open_position(self, symbol):
        return symbol in self.active_trades

    def execute_signal(self, symbol, signal_side, price, timestamp, confidence, z_score=None):
        """
                Entscheidet basierend auf State, Cooldown und Confidence, ob gehandelt wird.
                Returns: String ('OPEN', 'FLIP', 'IGNORED', 'COOLDOWN') oder None
                """
        active_trade = self.active_trades.get(symbol)

        # --- LOGIK 1: COOLDOWN CHECK ---
        if active_trade:
            time_in_trade = (timestamp - active_trade['entry_time']).total_seconds()
            # Wenn Trade jünger als 5 Minuten ist -> Blockiere jegliche Änderung (auch Flips)
            if time_in_trade < self.COOLDOWN_SEC:
                # logger.info(f"⏳ COOLDOWN {symbol}: Trade too young ({time_in_trade:.0f}s).")
                return "COOLDOWN"

        # --- LOGIK 2: THRESHOLD BESTIMMUNG ---
        required_threshold = self.ENTRY_THRESH
        is_reversal = False

        if active_trade:
            if active_trade['side'] == signal_side:
                return "IGNORED"  # Wir sind schon in der Richtung positioniert
            else:
                # Wir sind Long, Signal ist Short (oder umgekehrt) -> Das ist ein Reversal!
                # Hürde erhöhen!
                required_threshold = self.REVERSAL_THRESH
                is_reversal = True

        # --- LOGIK 3: CONFIDENCE CHECK ---
        if confidence < required_threshold:
            return "WEAK_SIGNAL"

        # --- LOGIK 4: EXECUTION ---

        # Szenario A: Reversal (Flip)
        if is_reversal and active_trade:
            logger.info(
                f"🔄 SIGNAL FLIP {symbol}: {active_trade['side']} -> {signal_side} (Conf: {confidence:.1%} > {required_threshold:.2f})")

            # 1. Schließe alten Trade
            if active_trade['side'] == 'LONG':
                pnl = (price - active_trade['entry_price']) / active_trade['entry_price']
            else:
                pnl = (active_trade['entry_price'] - price) / active_trade['entry_price']

            self.close_trade(symbol, price, "SIGNAL_REVERSAL", pnl, timestamp)

            # 2. Öffne neuen Trade
            self.open_trade(symbol, signal_side, price, timestamp, confidence, z_score)
            return "FLIP"

        # Szenario B: Neuer Trade (Normal)
        elif not active_trade:
            self.open_trade(symbol, signal_side, price, timestamp, confidence, z_score)
            return "OPEN"

        return None

    def open_trade(self, symbol, side, price, timestamp, confidence=None, z_score=None):
        # Sicherheitscheck (falls direkt aufgerufen)
        if self.has_open_position(symbol):
            return

        if side == 'LONG':
            hard_sl = price * (1 - self.SL_PCT)
            trailing_sl = price * (1 - self.TSL_PCT)
        else:
            hard_sl = price * (1 + self.SL_PCT)
            trailing_sl = price * (1 + self.TSL_PCT)

        # Base position size (collateral)
        base_amount = 100.0
        # Leveraged position size
        leveraged_amount = base_amount * self.leverage

        self.active_trades[symbol] = {
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'entry_time': timestamp,
            'amount': base_amount,  # Collateral used
            'leveraged_amount': leveraged_amount,  # Actual position size
            'highest_price': price,
            'lowest_price': price,
            'stop_price': trailing_sl,
            'hard_stop': hard_sl,
            'leverage': self.leverage,
            'confidence': confidence,
            'z_score': z_score
        }
        logger.info(f"🔵 PAPER OPEN {side} {symbol} @ {price:.2f} (Leverage: {self.leverage}x, Fee: {self.FEE_PCT:.2%}) | Stop: {trailing_sl:.2f}")
        
        # Send Telegram notification for trade open
        if self.on_trade_open_callback:
            msg = f"🔵 <b>PAPER TRADE OPENED</b>\n" \
                  f"Symbol: {symbol.upper()}\n" \
                  f"Side: {side}\n" \
                  f"Leverage: {self.leverage}x\n" \
                  f"Collateral: ${base_amount:.2f}\n" \
                  f"Position Size: ${leveraged_amount:.2f}\n" \
                  f"Entry Price: ${price:.2f}\n" \
                  f"Stop Loss: ${trailing_sl:.2f}\n" \
                  f"Hard Stop: ${hard_sl:.2f}\n"
            
            # Add confidence if available
            if confidence is not None:
                msg += f"Confidence: {confidence:.1%}\n"
            
            # Add z-score if available
            if z_score is not None:
                msg += f"Z-Score: {z_score:.2f}\n"
            
            # Add entry timestamp
            msg += f"Entry Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.on_trade_open_callback(msg)

    def update_trade(self, symbol, current_price, timestamp):
        """Prüft Stop-Loss und Trailing-Stop bei jedem Preis-Tick"""
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        side = trade['side']
        exit_reason = None
        pnl = 0.0

        if side == 'LONG':
            # Trailing Update
            if current_price > trade['highest_price']:
                trade['highest_price'] = current_price
                new_stop = current_price * (1 - self.TSL_PCT)
                if new_stop > trade['stop_price']:
                    trade['stop_price'] = new_stop

            # Check Exits
            if current_price <= trade['hard_stop']:
                exit_reason = "HARD_STOP_LOSS"
            elif current_price <= trade['stop_price']:
                exit_reason = "TRAILING_STOP_HIT"

            pnl = (current_price - trade['entry_price']) / trade['entry_price']

        elif side == 'SHORT':
            # Trailing Update
            if current_price < trade['lowest_price']:
                trade['lowest_price'] = current_price
                new_stop = current_price * (1 + self.TSL_PCT)
                if new_stop < trade['stop_price']:
                    trade['stop_price'] = new_stop

            # Check Exits
            if current_price >= trade['hard_stop']:
                exit_reason = "HARD_STOP_LOSS"
            elif current_price >= trade['stop_price']:
                exit_reason = "TRAILING_STOP_HIT"

            pnl = (trade['entry_price'] - current_price) / trade['entry_price']

        if exit_reason:
            self.close_trade(symbol, current_price, exit_reason, pnl, timestamp)

    def close_trade(self, symbol, exit_price, reason, raw_pnl_pct, timestamp):
        trade = self.active_trades.pop(symbol)
        
        # 1. Get position values
        collateral = trade['amount']  # Base collateral used
        leveraged_position = trade['leveraged_amount']  # Actual position size with leverage
        leverage = trade['leverage']
        
        # 2. Berechne Gebühren (Entry + Exit) - based on LEVERAGED position
        # Entry Fee: Bezogen auf leveraged Entry-Preis
        entry_fee = leveraged_position * self.FEE_PCT
        
        # Exit Fee: Bezogen auf leveraged Exit-Wert (Position + Gewinn/Verlust)
        # Wenn wir 10% gewonnen haben, ist die Position jetzt 110$ wert -> höhere Fee
        exit_value = leveraged_position * (1 + raw_pnl_pct)
        exit_fee = exit_value * self.FEE_PCT
        
        total_fees = entry_fee + exit_fee
        
        # 3. Berechne Netto-Profit - P&L is on LEVERAGED position
        # Roh-Gewinn in Dollar (leveraged)
        raw_profit_abs = leveraged_position * raw_pnl_pct
        
        # Netto-Gewinn = Roh-Gewinn - Gebühren
        net_profit_abs = raw_profit_abs - total_fees
        
        # Netto-PnL in Prozent (based on collateral for ROI perspective)
        net_pnl_pct = net_profit_abs / collateral
        
        # 4. Update Balance (profit/loss applied to actual balance)
        self.balance += net_profit_abs
        
        # 5. Stats & Logging
        if net_profit_abs > 0: self.wins += 1
        else: self.losses += 1
        
        total = self.wins + self.losses
        win_rate = (self.wins / total * 100) if total > 0 else 0
        
        icon = "✅" if net_profit_abs > 0 else "❌"
        
        logger.info(
            f"{icon} PAPER CLOSE {trade['side']} {symbol} @ {exit_price:.2f}\n"
            f"   Leverage: {leverage}x | Collateral: ${collateral:.2f} | Position: ${leveraged_position:.2f}\n"
            f"   Reason: {reason}\n"
            f"   Raw PnL: {raw_pnl_pct:.2%} ({raw_profit_abs:+.2f}$)\n"
            f"   Fees: -{total_fees:.2f}$ (approx. {self.FEE_PCT*2:.2%})\n"
            f"   NET PnL: {net_pnl_pct:.2%} ({net_profit_abs:+.2f}$)\n"
            f"   Balance: {self.balance:.2f}$ | WR: {win_rate:.0f}%"
        )
        
        # History speichern
        trade['exit_price'] = exit_price
        trade['exit_time'] = timestamp
        trade['exit_reason'] = reason
        trade['raw_pnl_pct'] = raw_pnl_pct
        trade['net_pnl_pct'] = net_pnl_pct
        trade['fees_paid'] = total_fees
        trade['final_balance'] = self.balance
        self.trade_history.append(trade)
        
        # Persist state after trade closes
        self.save_state()
        
        # Send Telegram notification for trade close
        if self.on_trade_close_callback:
            result_text = "WIN" if net_profit_abs > 0 else "LOSS"
            
            # Calculate duration
            entry_time = trade.get('entry_time', timestamp)
            duration_seconds = (timestamp - entry_time).total_seconds()
            duration_minutes = duration_seconds / 60
            duration_hours = duration_minutes / 60
            
            # Format duration string
            if duration_hours >= 1:
                duration_str = f"{duration_hours:.2f}h"
            else:
                duration_str = f"{duration_minutes:.2f}m"
            
            msg = f"{icon} <b>PAPER TRADE CLOSED - {result_text}</b>\n" \
                  f"Symbol: {symbol.upper()}\n" \
                  f"Side: {trade['side']}\n" \
                  f"Leverage: {leverage}x\n" \
                  f"Collateral: ${collateral:.2f}\n" \
                  f"Position Size: ${leveraged_position:.2f}\n" \
                  f"Entry: ${trade['entry_price']:.2f}\n" \
                  f"Exit: ${exit_price:.2f}\n" \
                  f"Reason: {reason}\n" \
                  f"Raw PnL: {raw_pnl_pct:.2%} ({raw_profit_abs:+.2f}$)\n" \
                  f"Fees: -{total_fees:.2f}$\n" \
                  f"NET PnL: {net_pnl_pct:.2%} ({net_profit_abs:+.2f}$)\n" \
                  f"Balance: ${self.balance:.2f}\n" \
                  f"Win Rate: {win_rate:.0f}% ({self.wins}W/{self.losses}L)\n"
            
            # Add confidence if available
            if trade.get('confidence') is not None:
                msg += f"Confidence: {trade['confidence']:.1%}\n"
            
            # Add z-score if available
            if trade.get('z_score') is not None:
                msg += f"Z-Score: {trade['z_score']:.2f}\n"
            
            # Add timestamps
            msg += f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            msg += f"Exit Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            msg += f"Duration: {duration_str}"
            
            self.on_trade_close_callback(msg)
        
        # Resample bucketed trades to minute candles before chart generation
        try:
            self.resample_to_minute_candles(symbol)
        except Exception as e:
            logger.error(f"Failed to resample bucketed trades for {symbol}: {e}")
        
        # Generate and send hourly chart when trade exits
        self.generate_and_send_chart(symbol)
    
    # ========================================================================
    # ORDERFLOW DATA PROCESSING
    # ========================================================================
    
    def add_symbol(self, symbol):
        """Add a symbol to track for chart generation"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.bucketed_trades[symbol] = deque(maxlen=10000)  # Keep last 10k buckets
            self.minute_candles[symbol] = deque(maxlen=500)  # Keep last 500 minutes (~8 hours)
            logger.info(f"Added symbol {symbol} for orderflow tracking")
    
    def add_bucketed_trade(self, symbol, bucket_data):
        """
        Add a bucketed trade to the buffer.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            bucket_data: Dict containing:
                - timestamp: Unix timestamp or datetime
                - price: Average price of the bucket
                - volume: Total volume of the bucket
                - buy_volume: Buy side volume (optional)
                - sell_volume: Sell side volume (optional)
        """
        if symbol not in self.bucketed_trades:
            self.add_symbol(symbol)
        
        # Ensure timestamp is datetime
        if isinstance(bucket_data.get('timestamp'), (int, float)):
            bucket_data['timestamp'] = datetime.fromtimestamp(bucket_data['timestamp'])
        
        self.bucketed_trades[symbol].append(bucket_data)
    
    def add_minute_candle(self, symbol, candle_data):
        """
        Add a 1-minute candle directly (if already aggregated from external source).
        
        Args:
            symbol: Trading pair
            candle_data: Dict with keys: timestamp, open, high, low, close, volume
        """
        if symbol not in self.minute_candles:
            self.add_symbol(symbol)
        
        # Ensure timestamp is datetime
        if isinstance(candle_data.get('timestamp'), (int, float)):
            candle_data['timestamp'] = datetime.fromtimestamp(candle_data['timestamp'])
        elif isinstance(candle_data.get('timestamp'), str):
            candle_data['timestamp'] = pd.to_datetime(candle_data['timestamp'])
        
        self.minute_candles[symbol].append(candle_data)
    
    def resample_to_minute_candles(self, symbol):
        """
        Resample bucketed trades into 1-minute OHLCV candles.
        This is called when we have bucketed trade data that needs aggregation.
        """
        if symbol not in self.bucketed_trades or len(self.bucketed_trades[symbol]) == 0:
            return
        
        # Convert bucketed trades to DataFrame
        df = pd.DataFrame(list(self.bucketed_trades[symbol]))
        
        if 'timestamp' not in df.columns or 'price' not in df.columns:
            return
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to 1-minute candles
        ohlc_dict = {
            'price': 'ohlc',
            'volume': 'sum'
        }
        
        resampled = df.resample('1min').agg(ohlc_dict)
        resampled.columns = ['open', 'high', 'low', 'close', 'volume']
        resampled.dropna(inplace=True)
        
        # Update minute_candles buffer
        for timestamp, row in resampled.iterrows():
            candle = {
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            self.add_minute_candle(symbol, candle)
    
    # ========================================================================
    # CHART GENERATION WITH MPLFINANCE
    # ========================================================================
    
    def generate_hourly_chart(self, symbol, lookback_minutes=60):
        """
        Generate a 1-hour candlestick chart with trade entry/exit markers.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            lookback_minutes: Number of minutes to include in chart (default: 60)
        
        Returns:
            bytes: PNG image data, or None if insufficient data
        """
        if symbol not in self.minute_candles or len(self.minute_candles[symbol]) == 0:
            logger.warning(f"No candle data available for {symbol}")
            return None
        
        # Get last N minutes of candles
        candles = list(self.minute_candles[symbol])
        if len(candles) < 10:
            logger.warning(f"Insufficient candle data for {symbol}: {len(candles)} candles")
            return None
        
        # Take last lookback_minutes candles
        candles = candles[-lookback_minutes:]
        
        # Convert to DataFrame for mplfinance
        df = pd.DataFrame(candles)
        
        # Ensure proper datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Ensure OHLC columns exist and are numeric
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        if len(df) == 0:
            logger.warning(f"No valid candle data after cleaning for {symbol}")
            return None
        
        # Get time range for chart
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Find trades within this time range
        trade_markers = []
        
        # Check active trades
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            entry_time = trade.get('entry_time')
            if entry_time and start_time <= entry_time <= end_time:
                # Find closest candle timestamp
                closest_idx = df.index[df.index.get_indexer([entry_time], method='nearest')[0]]
                # Position marker below the candle
                candle_low = df.loc[closest_idx, 'low']
                price_offset = (df['high'].max() - df['low'].min()) * 0.01  # 1% of price range
                marker_price = candle_low - price_offset
                side = trade['side']
                marker_color = 'purple'
                marker_symbol = '^' if side == 'LONG' else 'v'
                trade_markers.append({
                    'timestamp': closest_idx,
                    'price': marker_price,
                    'type': 'entry',
                    'side': side,
                    'color': marker_color,
                    'marker': marker_symbol
                })
        
        # Check historical trades
        for trade in self.trade_history:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            
            # Entry marker
            if entry_time and start_time <= entry_time <= end_time:
                closest_idx = df.index[df.index.get_indexer([entry_time], method='nearest')[0]]
                # Position marker below the candle
                candle_low = df.loc[closest_idx, 'low']
                price_offset = (df['high'].max() - df['low'].min()) * 0.01  # 1% of price range
                marker_price = candle_low - price_offset
                side = trade['side']
                marker_color = 'purple'
                marker_symbol = '^' if side == 'LONG' else 'v'
                trade_markers.append({
                    'timestamp': closest_idx,
                    'price': marker_price,
                    'type': 'entry',
                    'side': side,
                    'color': marker_color,
                    'marker': marker_symbol
                })
            
            # Exit marker
            if exit_time and start_time <= exit_time <= end_time:
                closest_idx = df.index[df.index.get_indexer([exit_time], method='nearest')[0]]
                # Position marker below the candle
                candle_low = df.loc[closest_idx, 'low']
                price_offset = (df['high'].max() - df['low'].min()) * 0.01  # 1% of price range
                marker_price = candle_low - price_offset
                side = trade['side']
                # Exit markers are opposite color
                marker_color = 'purple'
                marker_symbol = 'v' if side == 'LONG' else '^'
                trade_markers.append({
                    'timestamp': closest_idx,
                    'price': marker_price,
                    'type': 'exit',
                    'side': side,
                    'color': marker_color,
                    'marker': marker_symbol
                })
        
        # Create addplot for trade markers
        addplots = []
        if trade_markers:
            # Create a series for markers
            marker_series = pd.Series(index=df.index, dtype=float)
            marker_colors = []
            marker_symbols = []
            
            for marker in trade_markers:
                marker_series[marker['timestamp']] = marker['price']
            
            # Group markers by color and symbol
            marker_groups = {}
            for marker in trade_markers:
                key = (marker['color'], marker['marker'])
                if key not in marker_groups:
                    marker_groups[key] = pd.Series(index=df.index, dtype=float)
                marker_groups[key][marker['timestamp']] = marker['price']
            
            # Create addplot for each marker group
            for (color, symbol), series in marker_groups.items():
                addplots.append(mpf.make_addplot(series, type='scatter', markersize=150, 
                                                  marker=symbol, color=color, secondary_y=False))
        
        # Use dark theme (binance or nightclouds)
        style = mpf.make_mpf_style(base_mpf_style='nightclouds',
                                     marketcolors=mpf.make_marketcolors(up='#00ff00', down='#ff0000',
                                                                         edge='inherit',
                                                                         wick='inherit',
                                                                         volume='in'))
        
        # Calculate appropriate y-axis limits based on actual price range
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        # Add 2% padding above and below for better visibility
        y_min = price_min - (price_range * 0.02)
        y_max = price_max + (price_range * 0.02)
        
        # Generate chart
        try:
            # Create figure with custom y-axis limits
            fig, axes = mpf.plot(df, type='candle', style=style, volume=True,
                                  title=f'{symbol.upper()} - 1H Chart (1-min candles)',
                                  ylabel='Price (USDT)',
                                  ylabel_lower='Volume',
                                  addplot=addplots if addplots else None,
                                  returnfig=True,
                                  figsize=(14, 8),
                                  ylim=(y_min, y_max))
            
            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Close all figures to prevent accumulation of markers/plots
            plt.close('all')
            
            return buf.read()
            
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            return None
    
    # ========================================================================
    # TELEGRAM INTEGRATION
    # ========================================================================
    
    def send_telegram_chart(self, symbol, chart_bytes, time_range="1H"):
        """
        Send a chart image to Telegram.
        
        Args:
            symbol: Trading pair
            chart_bytes: PNG image data (bytes)
            time_range: Time range description (e.g., "1H")
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured. Skipping chart send.")
            return False
        
        if not chart_bytes:
            logger.warning(f"No chart data to send for {symbol}")
            return False
        
        try:
            # Create caption with pair name and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            caption = f"<b>{symbol.upper()} - {time_range} Chart</b>\n"
            caption += f"Time: {current_time}\n"
            
            # Add trading stats if available
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                caption += f"\n🔵 <b>Active Position:</b>\n"
                caption += f"Side: {trade['side']}\n"
                caption += f"Entry: ${trade['entry_price']:.2f}\n"
                caption += f"Time: {trade['entry_time'].strftime('%H:%M')}\n"
            
            # Send via Telegram API
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
            files = {'photo': (f'{symbol}_chart.png', chart_bytes, 'image/png')}
            data = {
                'chat_id': self.telegram_chat_id,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            response_json = response.json()
            
            # Telegram API returns 200 for HTTP success, but actual result is in 'ok' field
            if response.status_code == 200 and response_json.get('ok'):
                logger.info(f"✅ Chart sent to Telegram for {symbol}")
                return True
            else:
                error_desc = response_json.get('description', 'Unknown error')
                logger.error(f"Failed to send chart to Telegram for {symbol}: {error_desc}")
                logger.error(f"Full response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending chart to Telegram: {e}")
            return False
    
    def generate_and_send_chart(self, symbol):
        """Generate and send chart for a symbol"""
        logger.info(f"Generating hourly chart for {symbol}...")
        chart_bytes = self.generate_hourly_chart(symbol)
        
        if chart_bytes:
            logger.info(f"Chart generated successfully ({len(chart_bytes)} bytes)")
            self.send_telegram_chart(symbol, chart_bytes, "1H")
        else:
            logger.warning(f"Failed to generate chart for {symbol}")
    
    # ========================================================================
    # HOURLY SCHEDULER
    # ========================================================================
    
    def start_hourly_scheduler(self):
        """Start the hourly chart generation scheduler"""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("📅 Hourly chart scheduler started")
    
    def stop_hourly_scheduler(self):
        """Stop the hourly scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("📅 Hourly chart scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop - runs every hour at minute 00"""
        logger.info("Scheduler loop started. Waiting for next hour...")
        
        while self.scheduler_running:
            try:
                now = datetime.now()
                
                # Calculate seconds until next hour (at :00 minutes)
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                seconds_until_next = (next_hour - now).total_seconds()
                
                logger.info(f"Next chart generation at {next_hour.strftime('%H:%M')} "
                           f"({seconds_until_next/60:.1f} minutes)")
                
                # Sleep until next hour
                sleep_interval = 60  # Check every minute
                while self.scheduler_running and datetime.now() < next_hour:
                    time.sleep(min(sleep_interval, seconds_until_next))
                    seconds_until_next = (next_hour - datetime.now()).total_seconds()
                
                if not self.scheduler_running:
                    break
                
                # Generate and send charts for all tracked symbols
                logger.info(f"\n{'='*60}")
                logger.info(f"🕐 HOURLY CHART GENERATION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                logger.info(f"{'='*60}")
                
                for symbol in self.symbols:
                    try:
                        self.generate_and_send_chart(symbol)
                        time.sleep(2)  # Small delay between charts
                    except Exception as e:
                        logger.error(f"Error generating chart for {symbol}: {e}")
                
                logger.info(f"{'='*60}\n")
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Wait a minute before retrying