import requests

# ── Live Exchange Rate ──────────────────────────────────────────────────
def get_usd_to_ngn():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        return r.json()["rates"]["NGN"]
    except:
        return 1551.70  # fallback from today's Pink Sheet

# ── World Bank Pink Sheet Data (April 2026) ────────────────────────────
# Source: World Bank Commodities Price Data, April 2, 2026
PINK_SHEET = {
    "Wheat":              {"global_usd_per_mt": 261.1,  "source": "US HRW"},
    "Maize":              {"global_usd_per_mt": 208.9,  "source": "US Yellow #2"},
    "Rice (milled, local)":{"global_usd_per_mt": 399.3, "source": "Thailand 5%"},
    "Sorghum":            {"global_usd_per_mt": None,   "source": "Not available"},
    "Groundnuts":         {"global_usd_per_mt": 1231.0, "source": "US Runners CFR"},
    "Oil (palm)":         {"global_usd_per_mt": 1049.0, "source": "Malaysia Crude"},
    "Oil (vegetable)":    {"global_usd_per_mt": 1140.0, "source": "Soybean Oil US"},
    "Sugar":              {"global_usd_per_mt": 320.0,  "source": "World ISA"},
    "Bananas":            {"global_usd_per_mt": 1210.0, "source": "US Import"},
    "Oranges":            {"global_usd_per_mt": 1010.0, "source": "Mediterranean"},
}

# Nigeria import markup multiplier (shipping + duties + trader margin)
NIGERIA_MARKUP = 1.30

def get_wheat_ngn_estimate():
    """Estimate current Nigerian wheat price per kg using global data."""
    ngn_rate     = get_usd_to_ngn()
    wheat_usd_kg = PINK_SHEET["Wheat"]["global_usd_per_mt"] / 1000
    base_ngn     = wheat_usd_kg * ngn_rate
    final_ngn    = base_ngn * NIGERIA_MARKUP
    return {
        "commodity":        "Wheat",
        "global_usd_per_mt": PINK_SHEET["Wheat"]["global_usd_per_mt"],
        "usd_to_ngn_rate":  round(ngn_rate, 2),
        "estimated_ngn_per_kg": round(final_ngn, 2),
        "note": "Estimate based on World Bank global price + live exchange rate + 30% import markup",
        "confidence": "Medium",
        "data_source": "World Bank Pink Sheet April 2026 + ExchangeRate-API"
    }

if __name__ == "__main__":
    result = get_wheat_ngn_estimate()
    print(f"\n🌾 Wheat Price Estimate for Nigeria")
    print(f"   Global price:    ${result['global_usd_per_mt']}/mt")
    print(f"   Exchange rate:   ₦{result['usd_to_ngn_rate']}/$")
    print(f"   Estimated price: ₦{result['estimated_ngn_per_kg']}/kg")
    print(f"   Note: {result['note']}")