"""
train_v2.py  —  NaijaAgro Predict · Updated Model Training
===========================================================
Key additions over train.py v1:

  NEW FEATURES added to every training row:
  ─────────────────────────────────────────
  1. transport_cost_factor
       = (distance_from_cheapest_source × litres_per_km × diesel_price) / bags_per_truck
       Computed from trade_economics.py for every (commodity, state, date) row.
       Captures fuel price and distance effects DIRECTLY as a feature.

  2. fuel_price_index
       = diesel price in that period normalised by 2000 baseline.
       Proxy for all fuel-related cost pressures across the 24-year dataset.
       WFP data goes back to 2002 when diesel was ~₦35/L vs ₦1,962 today.

  3. income_index
       = avg monthly income for that state / national average income.
       Captures purchasing power — richer states sustain higher prices.

  4. shock_dummy
       = 1 during known external shocks (COVID 2020, subsidy removal 2023)
       = 0 otherwise.
       Forces the model to learn shock behaviour explicitly.

  5. season_detailed
       = 0 dry (Nov–Mar), 1 early rain (Apr–Jun), 2 peak rain (Jul–Sep), 3 harvest (Oct)
       More granular than the binary season in v1.

Usage:
    python model/train_v2.py
"""

import pandas as pd
import numpy as np
import os, joblib, json, sys

# Make sure trade_economics is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../data/cleaned_prices.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature list (EXPANDED) ──────────────────────────────────────────────────
FEATURES_V2 = [
    # Time
    "year", "month_sin", "month_cos", "season_detailed",
    # Location
    "state_encoded", "market_encoded",
    # Price history (log-space)
    "lag_1m", "lag_3m", "lag_6m", "roll_3m", "roll_6m", "price_change_pct",
    # NEW: Supply chain & economic factors
    "transport_cost_factor",   # fuel × distance for this commodity/state
    "fuel_price_index",        # diesel price trend over 24 years
    "income_index",            # state purchasing power relative to national avg
    "shock_dummy",             # COVID 2020, subsidy removal 2023
]

# ── Fuel price history (approximate NGN/litre diesel, 2002–2026) ─────────────
# Source: NBS, DPR historical records, Resagratia
DIESEL_PRICE_HISTORY = {
    2002: 26,   2003: 30,   2004: 35,   2005: 55,   2006: 55,
    2007: 55,   2008: 70,   2009: 70,   2010: 130,  2011: 145,
    2012: 150,  2013: 155,  2014: 160,  2015: 175,  2016: 200,
    2017: 210,  2018: 220,  2019: 230,  2020: 230,  2021: 240,
    2022: 750,  2023: 900,  2024: 1600, 2025: 1800, 2026: 1962,
}
DIESEL_BASELINE = 200   # 2016 approximate

# ── Shock periods ────────────────────────────────────────────────────────────
SHOCK_PERIODS = {
    (2020, 3): 1,  (2020, 4): 1,  (2020, 5): 1,   # COVID lockdown
    (2020, 6): 1,  (2020, 7): 1,  (2020, 8): 1,   # COVID aftermath
    (2023, 6): 1,  (2023, 7): 1,  (2023, 8): 1,   # subsidy removal shock
    (2023, 9): 1,  (2023, 10):1,  (2023, 11):1,   # subsidy aftermath
    (2016, 2): 1,  (2016, 3): 1,  (2016, 4): 1,   # 2016 recession/FX crisis
}

# ── State income index ────────────────────────────────────────────────────────
NATIONAL_AVG_INCOME = 38000
STATE_INCOME = {
    "Lagos": 85000, "FCT": 78000, "Rivers": 72000, "Bayelsa": 60000,
    "Delta": 55000, "Anambra": 55000, "Akwa Ibom": 45000, "Oyo": 48000,
    "Edo": 47000, "Ogun": 50000, "Enugu": 45000, "Abia": 42000,
    "Imo": 40000, "Kaduna": 38000, "Kwara": 36000, "Kano": 35000,
    "Ondo": 35000, "Kogi": 30000, "Osun": 33000, "Ekiti": 32000,
    "Plateau": 32000, "Benue": 28000, "Nasarawa": 28000, "Ebonyi": 28000,
    "Niger": 27000, "Kebbi": 24000, "Adamawa": 24000, "Gombe": 25000,
    "Bauchi": 25000, "Taraba": 23000, "Sokoto": 23000, "Yobe": 21000,
    "Borno": 22000, "Zamfara": 22000, "Katsina": 26000, "Jigawa": 22000,
    "Cross River": 38000, "Taraba": 23000,
}

# ── Distance from cheapest source to each state (km) ────────────────────────
# Simplified lookup: for the training, we use the nearest cheapest source
# This is a simplified version of trade_economics.py for training use
NEAREST_SOURCE_DISTANCE = {
    # Grain commodities — nearest northern source
    "Maize":                  {"Lagos": 998, "Kano": 0,   "Abia": 1024, "Rivers": 1050, "Oyo": 780,  "FCT": 350,  "Anambra": 820, "Enugu": 840, "Imo": 1010, "Delta": 900},
    "Rice (milled, local)":   {"Lagos": 998, "Kano": 0,   "Abia": 1024, "Rivers": 1050, "Oyo": 780,  "FCT": 350,  "Anambra": 820, "Enugu": 840, "Imo": 1010, "Delta": 900},
    "Beans (white)":          {"Lagos": 998, "Kano": 0,   "Abia": 1024, "Rivers": 1050, "Oyo": 780,  "FCT": 350,  "Anambra": 820, "Enugu": 840, "Imo": 1010, "Delta": 900},
    # South-East commodities — source is local
    "Gari (white)":           {"Lagos": 580, "Kano": 1024,"Abia": 0,    "Rivers": 80,   "Oyo": 520,  "FCT": 520,  "Anambra": 152, "Enugu": 180, "Imo": 60,   "Delta": 200},
    "Oil (palm)":             {"Lagos": 580, "Kano": 1024,"Abia": 0,    "Rivers": 80,   "Oyo": 580,  "FCT": 520,  "Anambra": 152, "Enugu": 180, "Imo": 60,   "Delta": 200},
    "Yam":                    {"Lagos": 130, "Kano": 780, "Abia": 520,  "Rivers": 480,  "Oyo": 0,    "FCT": 400,  "Anambra": 461, "Enugu": 560, "Imo": 500,  "Delta": 340},
    # Imported goods — Lagos port is source
    "Rice (imported)":        {"Lagos": 0,   "Kano": 998, "Abia": 580,  "Rivers": 540,  "Oyo": 130,  "FCT": 530,  "Anambra": 461, "Enugu": 560, "Imo": 500,  "Delta": 340},
    "Oil (vegetable)":        {"Lagos": 0,   "Kano": 998, "Abia": 580,  "Rivers": 540,  "Oyo": 130,  "FCT": 530,  "Anambra": 461, "Enugu": 560, "Imo": 500,  "Delta": 340},
    "Sugar":                  {"Lagos": 0,   "Kano": 998, "Abia": 580,  "Rivers": 540,  "Oyo": 130,  "FCT": 530,  "Anambra": 461, "Enugu": 560, "Imo": 500,  "Delta": 340},
    # Fish — Rivers is source
    "Fish":                   {"Lagos": 540, "Kano": 1050,"Abia": 80,   "Rivers": 0,    "Oyo": 480,  "FCT": 520,  "Anambra": 170, "Enugu": 220, "Imo": 90,   "Delta": 160},
}
DEFAULT_DISTANCE = 500   # fallback for unknown commodity/state combos
LITRES_PER_KM    = 1.1
BAGS_PER_TRUCK   = 200


def get_transport_factor(commodity, state, year):
    """Compute transport cost per 50kg bag for this commodity/state/year."""
    diesel_price = DIESEL_PRICE_HISTORY.get(year, 200)
    dist_map     = NEAREST_SOURCE_DISTANCE.get(commodity, {})
    dist_km      = dist_map.get(state, DEFAULT_DISTANCE)
    fuel_litres  = dist_km * LITRES_PER_KM
    total_fuel   = fuel_litres * diesel_price
    cost_per_bag = total_fuel / BAGS_PER_TRUCK
    return round(cost_per_bag, 2)


def season_detailed(month):
    """0=dry, 1=early_rain, 2=peak_rain, 3=harvest"""
    if month in [11, 12, 1, 2, 3]:  return 0   # dry season
    if month in [4, 5, 6]:          return 1   # early rains
    if month in [7, 8, 9]:          return 2   # peak rains / growing
    return 3                                    # harvest (October)


def train():
    print("Loading cleaned data …")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ── Encode states and markets ────────────────────────────────────────────
    states  = sorted(df["admin1"].unique())
    markets = sorted(df["market"].unique())
    state_enc  = {s: i for i, s in enumerate(states)}
    market_enc = {m: i for i, m in enumerate(markets)}
    df["state_encoded"]  = df["admin1"].map(state_enc).fillna(0).astype(int)
    df["market_encoded"] = df["market"].map(market_enc).fillna(0).astype(int)

    # ── Time features ────────────────────────────────────────────────────────
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["season_detailed"] = df["month"].apply(season_detailed)

    # ── Existing lag features (already in cleaned CSV or re-derive) ──────────
    for col in ["lag_1m","lag_3m","lag_6m","roll_3m","roll_6m","price_change_pct"]:
        if col not in df.columns:
            df[col] = 0.0   # will be recomputed below for groups
    # Log-transform price and lags
    df["log_price"]          = np.log1p(df["price"].clip(lower=0))
    df["lag_1m"]             = np.log1p(df["lag_1m"].clip(lower=0))
    df["lag_3m"]             = np.log1p(df["lag_3m"].clip(lower=0))
    df["lag_6m"]             = np.log1p(df["lag_6m"].clip(lower=0))
    df["roll_3m"]            = np.log1p(df["roll_3m"].clip(lower=0))
    df["roll_6m"]            = np.log1p(df["roll_6m"].clip(lower=0))
    df["price_change_pct"]   = df["price_change_pct"].clip(-100, 500).fillna(0)

    # ── NEW FEATURES ─────────────────────────────────────────────────────────

    # 1. Transport cost factor
    print("  Computing transport cost factor …")
    df["transport_cost_factor"] = df.apply(
        lambda r: get_transport_factor(r["commodity"], r["admin1"], r["year"]),
        axis=1
    )
    df["transport_cost_factor"] = np.log1p(df["transport_cost_factor"])

    # 2. Fuel price index
    df["fuel_price_index"] = df["year"].map(
        lambda y: DIESEL_PRICE_HISTORY.get(y, 200) / DIESEL_BASELINE
    )

    # 3. Income index
    df["income_index"] = df["admin1"].map(
        lambda s: STATE_INCOME.get(s, NATIONAL_AVG_INCOME) / NATIONAL_AVG_INCOME
    ).fillna(1.0)

    # 4. Shock dummy
    df["shock_dummy"] = df.apply(
        lambda r: SHOCK_PERIODS.get((r["year"], r["month"]), 0),
        axis=1
    )

    # ── Drop rows with NaN in any feature ────────────────────────────────────
    df = df.dropna(subset=FEATURES_V2 + ["log_price"])

    # ── Train per commodity ──────────────────────────────────────────────────
    commodities = df["commodity"].unique()
    results = {}
    print(f"Training {len(commodities)} commodity models with {len(FEATURES_V2)} features …\n")

    for commodity in sorted(commodities):
        cdf = df[df["commodity"] == commodity].copy().reset_index(drop=True)
        if len(cdf) < 50:
            print(f"  Skip {commodity} — {len(cdf)} rows (need 50+)")
            continue

        X = cdf[FEATURES_V2]
        y = cdf["log_price"]

        split   = int(len(cdf) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators    = 500,
            learning_rate   = 0.035,
            max_depth       = 7,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            min_child_weight= 3,
            reg_alpha       = 0.1,
            reg_lambda      = 1.0,
            random_state    = 42,
            verbosity       = 0,
        )
        model.fit(X_train, y_train)

        preds   = np.expm1(model.predict(X_test))
        actuals = np.expm1(y_test)
        mape    = mean_absolute_percentage_error(actuals, preds) * 100

        safe = (commodity.replace(" ","_").replace("(","").replace(")","")
                         .replace(",","").replace("/","_"))
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_{safe}.pkl"))

        results[commodity] = round(mape, 2)
        icon = "🟢" if mape < 15 else "🟡" if mape < 30 else "🔴"
        print(f"  {icon}  {commodity:<35}  MAPE: {mape:.1f}%")

    # Save summary and feature list
    with open(os.path.join(MODEL_DIR, "model_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_list_v2.json"), "w") as f:
        json.dump(FEATURES_V2, f, indent=2)

    avg = np.mean(list(results.values()))
    print(f"\nDone — {len(results)} models saved.")
    print(f"Average MAPE: {avg:.1f}%")
    print(f"Features used: {FEATURES_V2}")


if __name__ == "__main__":
    train()