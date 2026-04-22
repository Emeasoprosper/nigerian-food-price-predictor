import pandas as pd
import numpy as np
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PRICES_PATH  = os.path.join(BASE_DIR, "wfp_food_prices_nga.csv")
MARKETS_PATH = os.path.join(BASE_DIR, "wfp_markets_nga.csv")
OUTPUT_PATH  = os.path.join(BASE_DIR, "cleaned_prices.csv")

# Merge sparse variants into their parent commodity
COMMODITY_MAP = {
    "Cowpeas (brown)":          "Cowpeas",
    "Cowpeas (white)":          "Cowpeas",
    "Maize (white)":            "Maize",
    "Maize (yellow)":           "Maize",
    "Sorghum (brown)":          "Sorghum",
    "Sorghum (white)":          "Sorghum",
    "Groundnuts (shelled)":     "Groundnuts",
    "Beans (niebe)":            "Beans (white)",
    "Gari (white)":             "Cassava meal (gari, yellow)",
    "Rice (local)":             "Rice (milled, local)",
    "Yam (Abuja)":              "Yam",
}

def preprocess():
    print("Loading data...")
    prices  = pd.read_csv(PRICES_PATH)
    markets = pd.read_csv(MARKETS_PATH)

    df = prices.merge(markets[["market_id","latitude","longitude"]], on="market_id", how="left")

    # Apply commodity consolidation
    df["commodity"] = df["commodity"].replace(COMMODITY_MAP)

    # Date features
    df["date"]      = pd.to_datetime(df["date"])
    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"]    = df["month"].apply(lambda m: 0 if m in [11,12,1,2,3] else 1)

    # Clean
    df = df.dropna(subset=["price","usdprice","commodity","admin1"])
    df = df[df["price"] > 0]

    # Average if same commodity now has duplicates after merging
    df = df.groupby(["date","admin1","market","market_id","commodity",
                        "year","month","month_sin","month_cos","season"], as_index=False).agg(
                            price=("price","mean"),
                            usdprice=("usdprice","mean"))

    df = df.sort_values(["commodity","admin1","market","date"]).reset_index(drop=True)

    # Encode
    df["state_encoded"]     = df["admin1"].astype("category").cat.codes
    df["market_encoded"]    = df["market"].astype("category").cat.codes
    df["commodity_encoded"] = df["commodity"].astype("category").cat.codes

    # Lag features
# Keep one row per commodity+market+date (take mean if duplicates)
    df = df.groupby(["commodity","admin1","market","date",
                     "year","month","month_sin","month_cos","season",
                     "state_encoded","market_encoded","commodity_encoded"],
                     as_index=False).agg(price=("price","mean"), usdprice=("usdprice","mean"))

    df = df.sort_values(["commodity","market","date"]).reset_index(drop=True)

    # Lag features
    print("Engineering lag features...")
    grp = df.groupby(["commodity","market"])["price"]
    df["lag_1m"]  = grp.shift(1)
    df["lag_3m"]  = grp.shift(3)
    df["lag_6m"]  = grp.shift(6)
    df["roll_3m"] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["roll_6m"] = grp.transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["price_change_pct"] = grp.pct_change() * 100

    df = df.dropna(subset=["lag_1m","lag_3m"]).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Saved {len(df):,} rows to cleaned_prices.csv")
    print(f"Commodities after consolidation: {df['commodity'].nunique()}")
    print(f"Unique commodities: {sorted(df['commodity'].unique())}")

if __name__ == "__main__":
    preprocess()