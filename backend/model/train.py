import pandas as pd
import numpy as np
import os, joblib, json
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../data/cleaned_prices.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "year", "month_sin", "month_cos", "season",
    "state_encoded", "market_encoded",
    "lag_1m", "lag_3m", "lag_6m",
    "roll_3m", "roll_6m",
    "price_change_pct"
]

def train():
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + ["price"])
    df = df[df["price"] > 0]

    # Log-transform price to handle inflation across 24 years
    df["log_price"] = np.log1p(df["price"])
    df["lag_1m"]    = np.log1p(df["lag_1m"].clip(lower=0))
    df["lag_3m"]    = np.log1p(df["lag_3m"].clip(lower=0))
    df["lag_6m"]    = np.log1p(df["lag_6m"].clip(lower=0))
    df["roll_3m"]   = np.log1p(df["roll_3m"].clip(lower=0))
    df["roll_6m"]   = np.log1p(df["roll_6m"].clip(lower=0))
    df["price_change_pct"] = df["price_change_pct"].clip(-100, 500).fillna(0)

    commodities = df["commodity"].unique()
    results = {}

    print(f"Training models for {len(commodities)} commodities...\n")

    for commodity in sorted(commodities):
        cdf = df[df["commodity"] == commodity].copy().reset_index(drop=True)

        if len(cdf) < 50:
            print(f"  ⚠ Skipping {commodity} — not enough data ({len(cdf)} rows)")
            continue

        X = cdf[FEATURES]
        y = cdf["log_price"]

        # Time-based split
        split   = int(len(cdf) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        # Predict and inverse log-transform
        log_preds  = model.predict(X_test)
        preds      = np.expm1(log_preds)
        actuals    = np.expm1(y_test)
        mape       = mean_absolute_percentage_error(actuals, preds) * 100

        safe_name  = commodity.replace(" ","_").replace("(","").replace(")","").replace(",","")
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_{safe_name}.pkl"))

        results[commodity] = round(mape, 2)
        status = "🟢" if mape < 15 else "🟡" if mape < 30 else "🔴"
        print(f"  {status} {commodity:<35} MAPE: {mape:.1f}%")

    with open(os.path.join(MODEL_DIR, "model_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    avg = np.mean(list(results.values()))
    print(f"\n✅ Done! {len(results)} models saved to backend/model/saved/")
    print(f"📊 Average MAPE: {avg:.1f}%")

if __name__ == "__main__":
    train()