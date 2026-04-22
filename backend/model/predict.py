"""
predict.py — NaijaFoodAI Prediction Engine
Loads per-commodity XGBoost models and returns NGN price predictions.

ROOT CAUSE OF THE BUG:
  The model was trained on log1p(usdprice). When predicting, we must:
    1. Get log-space prediction  →  model.predict(X)
    2. Inverse log              →  np.expm1(pred)   — this gives USD
    3. Multiply by NGN rate     →  usd * USD_TO_NGN

  If step 3 uses a stale, wrong, or missing rate (e.g. 1.0 or 0.00) the
  NGN output collapses to near-zero or explodes. We fix this by:
    a. Using the LAST KNOWN usdprice and price from the CSV to derive the
       real implied rate for that commodity/market — not a hardcoded value.
    b. Falling back to a sensible default only when no data is available.
"""

import os, json, pprint
import numpy as np
import pandas as pd
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../data/cleaned_prices.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "saved")
SUMMARY    = os.path.join(MODEL_DIR, "model_summary.json")

# ── Features (must match train.py exactly) ───────────────────────────────────
FEATURES = [
    "year", "month_sin", "month_cos", "season",
    "state_encoded", "market_encoded",
    "lag_1m", "lag_3m", "lag_6m",
    "roll_3m", "roll_6m",
    "price_change_pct",
]

# ── Fallback exchange rate (only used when CSV has no data for that row) ──────
FALLBACK_USD_TO_NGN = 1600.0   # update periodically


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(commodity: str) -> str:
    return (commodity.replace(" ", "_")
                     .replace("(", "").replace(")", "")
                     .replace(",", ""))


def _load_summary() -> dict:
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            return json.load(f)
    return {}


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_state_encodings(df: pd.DataFrame) -> dict:
    """Return {state_name: encoded_int} derived from the cleaned CSV."""
    states = sorted(df["admin1"].unique())
    return {s: i for i, s in enumerate(states)}


def _get_market_encodings(df: pd.DataFrame) -> dict:
    markets = sorted(df["market"].unique())
    return {m: i for i, m in enumerate(markets)}


def _month_features(month: int) -> tuple:
    sin = np.sin(2 * np.pi * month / 12)
    cos = np.cos(2 * np.pi * month / 12)
    season = 0 if month in [11, 12, 1, 2, 3] else 1
    return sin, cos, season


def _build_lag_row(df: pd.DataFrame, commodity: str, state: str,
                   year: int, month: int) -> dict | None:
    """
    Build a single feature row for (commodity, state, year, month).

    Strategy:
      - Find the most recent rows for this commodity in this state.
      - Use their usdprice values as lag_1m / lag_3m / lag_6m.
      - If fewer than 3 historical rows exist, fill with column median.
    """
    cdf = df[(df["commodity"] == commodity) & (df["admin1"] == state)].copy()
    cdf = cdf.sort_values("date").reset_index(drop=True)

    if cdf.empty:
        # State not seen → fall back to any data for this commodity
        cdf = df[df["commodity"] == commodity].sort_values("date").reset_index(drop=True)

    if cdf.empty:
        return None

    # ── log-transform the usdprice history (matches training) ────────────────
    prices_log = np.log1p(cdf["usdprice"].clip(lower=0).values)

    def safe_get(arr, idx):
        """Get element from end; return median if out of bounds."""
        pos = len(arr) - 1 - idx
        return arr[pos] if pos >= 0 else float(np.median(arr))

    lag_1m  = safe_get(prices_log, 0)   # most recent
    lag_3m  = safe_get(prices_log, 2)
    lag_6m  = safe_get(prices_log, 5)
    roll_3m = float(np.mean(prices_log[-3:])) if len(prices_log) >= 3 else lag_1m
    roll_6m = float(np.mean(prices_log[-6:])) if len(prices_log) >= 6 else lag_1m

    # ── price_change_pct (percent change from previous month) ─────────────────
    if len(prices_log) >= 2:
        price_change_pct = ((prices_log[-1] - prices_log[-2]) / abs(prices_log[-2]) * 100) if prices_log[-2] != 0 else 0.0
    else:
        price_change_pct = 0.0

    # ── last known NGN rate for this commodity/state ──────────────────────────
    last_row      = cdf.iloc[-1]
    last_usd      = float(last_row["usdprice"])
    last_ngn      = float(last_row["price"])
    last_date     = str(last_row["date"].date())

    # Implied exchange rate from actual data (most reliable)
    if last_usd > 0:
        implied_rate = last_ngn / last_usd
    else:
        implied_rate = FALLBACK_USD_TO_NGN

    return {
        "lag_1m":            lag_1m,
        "lag_3m":            lag_3m,
        "lag_6m":            lag_6m,
        "roll_3m":           roll_3m,
        "roll_6m":           roll_6m,
        "price_change_pct":  price_change_pct,
        "last_usd":          last_usd,
        "last_ngn":          last_ngn,
        "last_date":         last_date,
        "implied_rate":      implied_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_price(commodity: str, state: str, year: int, month: int) -> dict:
    """
    Predict the price of `commodity` in `state` for `year`/`month`.

    Returns a dict with:
      predicted_usd, predicted_ngn, confidence, trend, model_mape_pct,
      last_known_usd, last_known_ngn, last_data_date
    """
    safe = _safe_name(commodity)
    model_path = os.path.join(MODEL_DIR, f"model_{safe}.pkl")

    if not os.path.exists(model_path):
        return {"error": f"No model found for '{commodity}'"}

    model   = joblib.load(model_path)
    summary = _load_summary()
    df      = _load_data()

    state_enc   = _get_state_encodings(df)
    market_enc  = _get_market_encodings(df)

    # Pick the best matching market for this state
    state_markets = df[df["admin1"] == state]["market"].unique()
    if len(state_markets) == 0:
        return {"error": f"State '{state}' not found in data"}
    chosen_market = state_markets[0]

    s_enc = state_enc.get(state, 0)
    m_enc = market_enc.get(chosen_market, 0)

    # ── Build lag features ────────────────────────────────────────────────────
    lag_data = _build_lag_row(df, commodity, state, year, month)
    if lag_data is None:
        return {"error": f"No historical data for '{commodity}'"}

    sin, cos, season = _month_features(month)

    row = pd.DataFrame([{
        "year":             year,
        "month_sin":        sin,
        "month_cos":        cos,
        "season":           season,
        "state_encoded":    s_enc,
        "market_encoded":   m_enc,
        "lag_1m":           lag_data["lag_1m"],
        "lag_3m":           lag_data["lag_3m"],
        "lag_6m":           lag_data["lag_6m"],
        "roll_3m":          lag_data["roll_3m"],
        "roll_6m":          lag_data["roll_6m"],
        "price_change_pct": lag_data["price_change_pct"],
    }])

    # ── Predict (log space) and invert ───────────────────────────────────────
    log_pred      = model.predict(row[FEATURES])[0]
    predicted_usd = float(np.expm1(log_pred))

    # KEY FIX: use the implied rate derived from the actual data
    implied_rate  = lag_data["implied_rate"]
    predicted_ngn = round(predicted_usd * implied_rate, 2)

    # ── Trend vs last known ───────────────────────────────────────────────────
    last_usd = lag_data["last_usd"]
    if last_usd > 0:
        change_pct = (predicted_usd - last_usd) / last_usd * 100
    else:
        change_pct = 0.0

    trend = "stable"
    if change_pct >  3:  trend = "rising"
    if change_pct < -3:  trend = "falling"

    # ── Confidence from MAPE ─────────────────────────────────────────────────
    mape = summary.get(commodity, 999)
    confidence = "High" if mape < 15 else "Medium" if mape < 30 else "Low"

    return {
        "commodity":       commodity,
        "state":           state,
        "year":            year,
        "month":           month,
        "predicted_usd":   round(predicted_usd, 4),
        "predicted_ngn":   predicted_ngn,
        "last_known_usd":  round(last_usd, 4),
        "last_known_ngn":  round(lag_data["last_ngn"], 2),
        "last_data_date":  lag_data["last_date"],
        "implied_rate":    round(implied_rate, 2),
        "change_pct":      round(change_pct, 2),
        "trend":           trend,
        "confidence":      confidence,
        "model_mape_pct":  round(mape, 2),
    }


def predict_all(state: str, year: int, month: int) -> list[dict]:
    """Predict prices for all available commodities in a state."""
    summary = _load_summary()
    results = []
    for commodity in sorted(summary.keys()):
        result = predict_price(commodity, state, year, month)
        if "error" not in result:
            results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test (run directly: python model/predict.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Single Prediction Tests ===\n")
    tests = [
        ("Eggs",               "Lagos", 2025, 6),
        ("Rice (milled, local)", "Kano",  2025, 8),
        ("Maize",              "Kano",  2025, 9),
        ("Yam",                "Oyo",   2025, 4),
    ]
    for commodity, state, year, month in tests:
        result = predict_price(commodity, state, year, month)
        pprint.pprint(result)
        print()

    print("=== All Commodities for Lagos, June 2025 ===\n")
    all_results = predict_all("Lagos", 2025, 6)
    icons = {"rising": "📈", "falling": "📉", "stable": "➡️ "}
    for r in all_results:
        icon = icons.get(r["trend"], "➡️ ")
        print(f"  {icon}  {r['commodity']:<35} ₦{r['predicted_ngn']:>10,.2f}"
              f"   ({r['confidence']} confidence, rate: ₦{r['implied_rate']:,.0f}/$)")