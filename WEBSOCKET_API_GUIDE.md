# Hedge Fund WebSocket API Guide

This guide details the WebSocket endpoint provided by the Market Predictor Service for real-time hedge fund decision-making.

## Connection Details

- **URL:** `ws://localhost:8765`
- **Protocol:** WebSocket (WS)
- **Authentication:** Bearer Token (Header: `Authorization: Bearer <TOKEN>`)
- **Data Format:** JSON
- **Update Frequency:** Real-time (triggered on every incoming trade message processed by the predictor)

## Integration Overview

This endpoint acts as an "Alpha Oracle" for your hedge bot. It aggregates data from:
1.  **LightGBM Models:** Fast, reactive predictions based on order flow snapshots.
2.  **LSTM Models:** Temporal pattern recognition over the last 200 ticks.
3.  **Order Flow Imbalance (OFI):** Real-time buying/selling pressure.

It processes this data to provide **Global Market Regime** signals and **Relative Strength** metrics for pair trading.

---

## Payload Structure

The server broadcasts a JSON object with the following top-level keys:

```json
{
  "timestamp": "ISO-8601 Timestamp",
  "market_regime": { ... },
  "btc_dominance": { ... },
  "spread_matrix": { ... },
  "assets": { ... }
}
```

### 1. Market Regime (`market_regime`)
Describes the global state of the crypto market. Use this to determine your fund's overall exposure (Beta).

| Field | Type | Description |
| :--- | :--- | :--- |
| `season` | String | `BULL_MARKET`, `BEAR_MARKET`, `BTC_SEASON`, `ALT_SEASON`, `DIVERGENCE_RISK`, `NEUTRAL` |
| `score` | Float | **-1.0 to 1.0**. Global sentiment score. <br>• `> 0.5`: Strong Bull<br>• `< -0.5`: Strong Bear |
| `confidence` | Float | **0.0 to 1.0**. How strongly the models agree on this regime. |
| `details` | String | Human-readable explanation (e.g., "Money flowing into BTC"). |

**Strategy Tip:**
*   **BULL_MARKET:** Enable Long entries on all pairs.
*   **BEAR_MARKET:** Disable Long entries; tighten stops; consider Shorting.
*   **DIVERGENCE_RISK:** Reduce leverage/position size.

### 2. BTC Dominance (`btc_dominance`)
A proxy metric indicating capital flow between Bitcoin and Altcoins.

| Field | Type | Description |
| :--- | :--- | :--- |
| `score` | Float | **Positive** = BTC Stronger (Dominance Rising)<br>**Negative** = Alts Stronger (Dominance Falling) |
| `trend` | String | `RISING`, `FALLING`, `NEUTRAL` |

**Strategy Tip:**
*   **RISING:** Hedge Alts by Shorting Alts or Longing BTC.
*   **FALLING:** "Alt Season" mode. Aggressive on Altcoins.

### 3. Spread Matrix (`spread_matrix`)
The core metric for **Market Neutral Pair Trading**. Shows the relative strength difference between assets.

| Key | Value (Float) | Interpretation |
| :--- | :--- | :--- |
| `btc_vs_eth` | `0.15` | BTC is stronger than ETH. **Long BTC / Short ETH**. |
| `btc_vs_sol` | `-0.10` | SOL is stronger than BTC. **Long SOL / Short BTC**. |
| `eth_vs_sol` | `0.00` | Parity. No clear edge. |

**Formula:** `AssetA_Score - AssetB_Score`

### 4. Assets (`assets`)
Detailed metrics for individual assets (BTC, ETH, SOL). Use these for absolute ranking or filtering.

Each asset object contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `score` | Float | **Composite Score (-1.0 to 1.0)**. Weighted average of LightGBM (70%) and LSTM (30%). |
| `direction` | String | `PUMP`, `DUMP`, `NEUTRAL`. Derived from the composite score. |
| `lgbm_score` | Float | Raw LightGBM probability score (`PumpProb - DumpProb`). |
| `lgbm_direction`| String | Direction based purely on LightGBM. |
| `lstm_score` | Float | Raw LSTM probability score (`PumpProb - DumpProb`). |
| `lstm_direction`| String | Direction based purely on LSTM. |
| `ofi` | Float | **Order Flow Imbalance**. <br>• `> 0`: Net Buying Pressure<br>• `< 0`: Net Selling Pressure |

---

## Example Payload

```json
{
  "timestamp": "2025-12-02T23:15:00.123456",
  "market_regime": {
    "season": "BTC_SEASON",
    "score": 0.45,
    "confidence": 0.30,
    "details": "Money flowing into BTC (Dom Proxy: 0.25)"
  },
  "btc_dominance": {
    "score": 0.25,
    "trend": "RISING"
  },
  "spread_matrix": {
    "btc_vs_eth": 0.20,
    "btc_vs_sol": 0.15,
    "eth_vs_sol": -0.05
  },
  "assets": {
    "btc": {
      "score": 0.60,
      "direction": "PUMP",
      "lgbm_score": 0.65,
      "lgbm_direction": "PUMP",
      "lstm_score": 0.48,
      "lstm_direction": "PUMP",
      "ofi": 1.50
    },
    "eth": {
      "score": 0.40,
      "direction": "PUMP",
      "lgbm_score": 0.42,
      "lgbm_direction": "PUMP",
      "lstm_score": 0.35,
      "lstm_direction": "PUMP",
      "ofi": 0.20
    },
    "sol": {
      "score": 0.45,
      "direction": "PUMP",
      "lgbm_score": 0.48,
      "lgbm_direction": "PUMP",
      "lstm_score": 0.38,
      "lstm_direction": "PUMP",
      "ofi": 0.50
    }
  }
}
```

## Client Implementation Example (Python)

```python
import asyncio
import websockets
import json

async def hedge_bot():
    uri = "ws://localhost:8765"
    token = "4kEZu6v_W5y2oNf8oZg2L1b3bO8hJ8mX" # Replace with your actual token
    extra_headers = {"Authorization": f"Bearer {token}"}
    
    async with websockets.connect(uri, extra_headers=extra_headers) as websocket:
        print(f"Connected to {uri}")
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            # 1. Check Regime
            regime = data['market_regime']['season']
            if regime == "BEAR_MARKET":
                print("⚠️ BEAR MARKET DETECTED - HALTING LONGS")
                continue
                
            # 2. Check Spreads for Pair Trading
            spreads = data['spread_matrix']
            if spreads['btc_vs_eth'] > 0.15:
                print(f"🚀 SIGNAL: Long BTC / Short ETH (Spread: {spreads['btc_vs_eth']:.2f})")
            elif spreads['btc_vs_eth'] < -0.15:
                print(f"📉 SIGNAL: Long ETH / Short BTC (Spread: {spreads['btc_vs_eth']:.2f})")

asyncio.run(hedge_bot())
```
