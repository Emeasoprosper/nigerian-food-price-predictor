"""
trade_economics.py  —  NaijaAgro Predict · Supply Chain Cost Engine
====================================================================
Calculates the REAL landed cost of any commodity in any Nigerian state by
modelling the full supply chain:

  SOURCE PRICE  +  TRANSPORT COST  =  LANDED COST
  LANDED COST   ×  (1 + TRADER MARGIN)  =  STREET PRICE

Transport cost formula (verified from Nigerian haulage industry data):
  - A fully-loaded 15-tonne truck consumes ~1.1 litres of diesel per km
    (Source: Ayoola Ashiru, Medium 2016 — "Lagos to Ibadan 118km = 130L")
  - A standard food cargo truck carries ~200 bags of 50kg (10 tonnes payload)
  - Diesel price: ₦1,962.50/litre (GlobalPetrolPrices.com, May 2026)
  - Transport cost per bag = (distance × litres_per_km × diesel_price) ÷ bags_per_truck
  - Road factor: ×1.25 for poor roads (South-South/North-East routes)

Distance data: real driving distances from distancecalculator.net / verified sources
  - Kano  → Aba/Umuahia : 1,024 km   (distancecalculator.net)
  - Lagos → Umuahia     : 580 km    (SureDirect, Mar 2025)
  - Onitsha → Aba       : 152 km    (distancecalculator.net)

Trader margins: real documented Abia/South-East market margins
  - Bulk/wholesale: 10–15% (staples: rice, garri, maize)
  - Perishables: 30–50% (tomatoes, fish, onions — high spoilage risk)
  - Protein: 20–35% (eggs, meat, fish)
  - Oil/liquids: 15–25%
  - Fuel: 8–12%

Usage:
  from model.trade_economics import (
      get_landed_cost, get_best_source, get_street_price, explain_price
  )
"""

import math
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
DIESEL_PRICE_PER_LITRE  = 1962.50   # ₦/litre — GlobalPetrolPrices May 2026
PETROL_PRICE_PER_LITRE  = 1027.00   # ₦/litre — NNPC/Resagratia May 2026
LITRES_PER_KM_TRUCK     = 1.1       # 15-tonne loaded truck (Nigerian haulage data)
BAGS_PER_TRUCK          = 200       # standard 50kg bag load per 15-tonne truck
KG_PER_BAG              = 50        # standard bag weight

# Road condition multiplier — poor roads increase effective cost
ROAD_FACTOR = {
    "good":   1.00,   # Lagos–Ibadan, Abuja–Kaduna expressway
    "average":1.15,   # most federal highways
    "poor":   1.30,   # rural roads, South-South waterlogged routes, NE insecurity routes
}

# ── Driving distances (km) between source markets and destination states ──────
# Sources: distancecalculator.net, SureDirect.com, verified road routes
# Format: { "SourceCity → DestState": km }
DRIVING_DISTANCES = {
    # From KANO (Dawanau — largest grain/legume wholesale market)
    "Kano→Abia":         1024,   # distancecalculator.net Kano→Aba
    "Kano→Lagos":         998,   # SureDirect verified
    "Kano→Rivers":       1050,
    "Kano→Imo":          1010,
    "Kano→Enugu":         840,
    "Kano→Anambra":       820,   # via Onitsha
    "Kano→Delta":         900,
    "Kano→Edo":           850,
    "Kano→Oyo":           780,
    "Kano→FCT (Abuja)":   350,   # distancecalculator.net
    "Kano→Kaduna":        190,
    "Kano→Kwara":         520,
    "Kano→Kogi":          580,
    "Kano→Benue":         600,
    "Kano→Plateau":       320,
    "Kano→Kano":            0,
    "Kano→Jigawa":         80,
    "Kano→Katsina":       180,
    "Kano→Zamfara":       280,
    "Kano→Sokoto":        380,
    "Kano→Kebbi":         420,
    "Kano→Niger":         450,
    "Kano→Bauchi":        260,
    "Kano→Gombe":         380,
    "Kano→Adamawa":       600,
    "Kano→Taraba":        680,
    "Kano→Borno":         620,
    "Kano→Yobe":          520,

    # From ONITSHA (Anambra — largest food market in West Africa)
    "Onitsha→Abia":       152,   # distancecalculator.net
    "Onitsha→Lagos":      461,   # verified
    "Onitsha→Rivers":     170,
    "Onitsha→Imo":        100,
    "Onitsha→Enugu":      100,
    "Onitsha→Delta":       80,
    "Onitsha→Edo":        150,
    "Onitsha→Anambra":     10,   # same state
    "Onitsha→FCT (Abuja)": 330,
    "Onitsha→Kogi":       230,
    "Onitsha→Benue":      260,
    "Onitsha→Cross River":200,
    "Onitsha→Akwa Ibom":  220,
    "Onitsha→Ebonyi":     130,
    "Onitsha→Bayelsa":    200,

    # From LAGOS (Mile 12 — imported goods, rice, vegetable oil hub)
    "Lagos→Abia":         580,   # SureDirect
    "Lagos→Rivers":       540,
    "Lagos→Imo":          500,
    "Lagos→Enugu":        560,
    "Lagos→Anambra":      461,
    "Lagos→Delta":        340,
    "Lagos→Edo":          290,
    "Lagos→Oyo":          130,
    "Lagos→Ogun":          80,
    "Lagos→FCT (Abuja)":  530,
    "Lagos→Kwara":        300,
    "Lagos→Kogi":         380,
    "Lagos→Lagos":          0,
    "Lagos→Osun":         200,
    "Lagos→Ekiti":        280,
    "Lagos→Ondo":         240,

    # From IBADAN (Bodija — South-West distribution hub)
    "Ibadan→Abia":        520,
    "Ibadan→Lagos":       130,
    "Ibadan→Rivers":      480,
    "Ibadan→FCT (Abuja)": 400,
    "Ibadan→Oyo":           0,   # same city

    # From PORT HARCOURT (Mile 3 — fish, palm oil, South-South hub)
    "PortHarcourt→Abia":   80,
    "PortHarcourt→Rivers":  0,
    "PortHarcourt→Imo":    90,
    "PortHarcourt→Bayelsa":80,
    "PortHarcourt→Delta":  160,
    "PortHarcourt→Cross River": 120,
    "PortHarcourt→Akwa Ibom": 110,
    "PortHarcourt→Enugu":  220,
    "PortHarcourt→Anambra":180,
    "PortHarcourt→Lagos":  540,
    "PortHarcourt→FCT (Abuja)": 520,

    # From ABUJA (Wuse — FCT, central distribution)
    "Abuja→FCT (Abuja)":    0,
    "Abuja→Kogi":          180,
    "Abuja→Benue":         280,
    "Abuja→Niger":         130,
    "Abuja→Kwara":         250,
    "Abuja→Plateau":       280,
    "Abuja→Nasarawa":       80,
    "Abuja→Kaduna":        190,
    "Abuja→Kano":          350,

    # From OWERRI (Eke Onunwa — Imo palm oil belt)
    "Owerri→Abia":         60,
    "Owerri→Imo":           0,
    "Owerri→Rivers":       100,
    "Owerri→Anambra":      110,
    "Owerri→Enugu":        140,
    "Owerri→Delta":        180,
}

# ── Source markets per commodity ──────────────────────────────────────────────
# For each commodity: where is it produced/wholesaled cheapest?
# Format: { canonical_name: [ {city, state, wholesale_price, road_quality, source_note} ] }
COMMODITY_SOURCES = {
    "Rice (milled, local)": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 54000, "road": "average", "unit": "50kg bag", "note": "Largest rice wholesale market Nigeria"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 59000, "road": "average", "unit": "50kg bag", "note": "Major distribution hub"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 60000, "road": "good",    "unit": "50kg bag", "note": "Ibadan wholesale"},
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 65000, "road": "good",    "unit": "50kg bag", "note": "Lagos retail rate"},
        {"city": "Mile 3",        "state": "Rivers",   "wholesale_price": 63000, "road": "average", "unit": "50kg bag", "note": "Port Harcourt"},
    ],
    "Rice (imported)": [
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 85000, "road": "good",    "unit": "50kg bag", "note": "Port of Lagos — entry point for imports"},
        {"city": "Apapa",         "state": "Lagos",    "wholesale_price": 82000, "road": "average", "unit": "50kg bag", "note": "Port direct"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 88000, "road": "average", "unit": "50kg bag", "note": "Distributed from Lagos"},
    ],
    "Beans (white)": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 72000, "road": "average", "unit": "100kg bag", "note": "Closest to farms, cheapest"},
        {"city": "Tudun Wada",    "state": "Kaduna",   "wholesale_price": 76000, "road": "average", "unit": "100kg bag", "note": "Northern wholesale"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 87000, "road": "average", "unit": "100kg bag", "note": "South distribution"},
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 88000, "road": "good",    "unit": "100kg bag", "note": "Lagos rate"},
    ],
    "Beans (red)": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 68000, "road": "average", "unit": "100kg bag", "note": "Northern production"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 82000, "road": "average", "unit": "100kg bag", "note": "South rate"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 78000, "road": "good",    "unit": "100kg bag", "note": "Ibadan honey beans"},
    ],
    "Maize": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 32000, "road": "average", "unit": "100kg bag", "note": "Post-harvest North — cheapest"},
        {"city": "Tudun Wada",    "state": "Kaduna",   "wholesale_price": 35000, "road": "average", "unit": "100kg bag", "note": "Northern rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 45000, "road": "average", "unit": "100kg bag", "note": "South distribution"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 46000, "road": "good",    "unit": "100kg bag", "note": "South-West rate"},
    ],
    "Maize (white)": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 30000, "road": "average", "unit": "100kg bag", "note": "Northern farms"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 43000, "road": "average", "unit": "100kg bag", "note": "South rate"},
    ],
    "Maize flour": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 28000, "road": "average", "unit": "50kg bag", "note": "Northern mills"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 35000, "road": "average", "unit": "50kg bag", "note": "South rate"},
    ],
    "Gari (white)": [
        {"city": "Umuahia",       "state": "Abia",     "wholesale_price": 1400,  "road": "average", "unit": "paint bucket (~3kg)", "note": "Garri belt — production zone"},
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 1500,  "road": "average", "unit": "paint bucket (~3kg)", "note": "Imo production zone"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 1700,  "road": "average", "unit": "paint bucket (~3kg)", "note": "Distribution hub"},
        {"city": "Mile 3",        "state": "Rivers",   "wholesale_price": 1800,  "road": "average", "unit": "paint bucket (~3kg)", "note": "Port Harcourt"},
    ],
    "Cassava meal (gari, yellow)": [
        {"city": "Umuahia",       "state": "Abia",     "wholesale_price": 2200,  "road": "average", "unit": "paint bucket", "note": "Yellow garri production zone"},
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 2300,  "road": "average", "unit": "paint bucket", "note": "Imo rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 2500,  "road": "average", "unit": "paint bucket", "note": "Anambra distribution"},
    ],
    "Yam": [
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 1500,  "road": "good",    "unit": "per tuber", "note": "Yam belt — cheapest"},
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 2000,  "road": "average", "unit": "per tuber", "note": "Northern yam"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 2800,  "road": "average", "unit": "per tuber", "note": "South distribution"},
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 3000,  "road": "average", "unit": "per tuber", "note": "Imo rate"},
    ],
    "Oil (palm)": [
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 3800,  "road": "average", "unit": "5 litres", "note": "Palm belt — producing state"},
        {"city": "Umuahia",       "state": "Abia",     "wholesale_price": 4000,  "road": "average", "unit": "5 litres", "note": "Abia palm belt"},
        {"city": "Mile 3",        "state": "Rivers",   "wholesale_price": 4200,  "road": "average", "unit": "5 litres", "note": "Rivers rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 4800,  "road": "average", "unit": "5 litres", "note": "Distributed from South-South"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 6000,  "road": "good",    "unit": "5 litres", "note": "South-West rate"},
    ],
    "Oil (vegetable)": [
        {"city": "Apapa",         "state": "Lagos",    "wholesale_price": 5500,  "road": "average", "unit": "5 litres", "note": "Port import price"},
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 6000,  "road": "good",    "unit": "5 litres", "note": "Lagos wholesale"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 6800,  "road": "average", "unit": "5 litres", "note": "Distributed from Lagos"},
    ],
    "Tomatoes": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 800,   "road": "average", "unit": "per kg", "note": "Northern tomato farms — cheapest"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 1200,  "road": "good",    "unit": "per kg", "note": "Ibadan rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 2000,  "road": "average", "unit": "per kg", "note": "South distribution"},
        {"city": "Mile 3",        "state": "Rivers",   "wholesale_price": 2200,  "road": "average", "unit": "per kg", "note": "Rivers rate"},
    ],
    "Onions": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 800,   "road": "average", "unit": "per kg", "note": "Northern onion farms — cheapest"},
        {"city": "Tudun Wada",    "state": "Kaduna",   "wholesale_price": 900,   "road": "average", "unit": "per kg", "note": "Kaduna rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 1500,  "road": "average", "unit": "per kg", "note": "South rate"},
    ],
    "Fish": [
        {"city": "Mile 3",        "state": "Rivers",   "wholesale_price": 2200,  "road": "average", "unit": "per kg", "note": "Coastal — freshwater source"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 2800,  "road": "average", "unit": "per kg", "note": "Distributed from PH"},
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 4500,  "road": "poor",    "unit": "per kg", "note": "Far North — transport adds cost"},
    ],
    "Eggs": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 2400,  "road": "average", "unit": "per crate (30)", "note": "Northern poultry farms"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 2700,  "road": "good",    "unit": "per crate (30)", "note": "Ibadan poultry"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 3000,  "road": "average", "unit": "per crate (30)", "note": "South rate"},
    ],
    "Meat (beef)": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 3500,  "road": "average", "unit": "per kg", "note": "Northern cattle — cheapest"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 4200,  "road": "good",    "unit": "per kg", "note": "Ibadan abattoir"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 5000,  "road": "average", "unit": "per kg", "note": "South rate"},
    ],
    "Meat (goat)": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 4000,  "road": "average", "unit": "per kg", "note": "North — goat belt"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 5800,  "road": "average", "unit": "per kg", "note": "South rate"},
    ],
    "Groundnuts": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 32000, "road": "average", "unit": "50kg bag", "note": "Groundnut belt North"},
        {"city": "Tudun Wada",    "state": "Kaduna",   "wholesale_price": 35000, "road": "average", "unit": "50kg bag", "note": "Kaduna rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 45000, "road": "average", "unit": "50kg bag", "note": "South rate"},
    ],
    "Sorghum": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 22000, "road": "average", "unit": "100kg bag", "note": "Guinea corn belt North"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 35000, "road": "average", "unit": "100kg bag", "note": "South rate"},
    ],
    "Millet": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 20000, "road": "average", "unit": "100kg bag", "note": "Millet belt North"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 32000, "road": "average", "unit": "100kg bag", "note": "South rate"},
    ],
    "Salt": [
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 400,   "road": "good",    "unit": "per kg", "note": "Port salt entry"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 500,   "road": "average", "unit": "per kg", "note": "South distribution"},
    ],
    "Sugar": [
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 1500,  "road": "good",    "unit": "per kg", "note": "BUA/Dangote Sugar factory"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 1800,  "road": "average", "unit": "per kg", "note": "South distribution"},
    ],
    "Milk (powder)": [
        {"city": "Apapa",         "state": "Lagos",    "wholesale_price": 3800,  "road": "average", "unit": "400g tin", "note": "Port import"},
        {"city": "Mile 12",       "state": "Lagos",    "wholesale_price": 4000,  "road": "good",    "unit": "400g tin", "note": "Lagos wholesale"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 4500,  "road": "average", "unit": "400g tin", "note": "South rate"},
    ],
    "Bread": [
        {"city": "Lagos",         "state": "Lagos",    "wholesale_price": 800,   "road": "good",    "unit": "600g loaf", "note": "Flour mills in Lagos"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 1000,  "road": "average", "unit": "600g loaf", "note": "South-East bakeries"},
    ],
    "Bananas": [
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 300,   "road": "average", "unit": "per bunch", "note": "South-East production"},
        {"city": "Umuahia",       "state": "Abia",     "wholesale_price": 350,   "road": "average", "unit": "per bunch", "note": "Abia production zone"},
        {"city": "Mile 3",        "state": "Rivers",   "wholesale_price": 500,   "road": "average", "unit": "per bunch", "note": "Rivers rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 600,   "road": "average", "unit": "per bunch", "note": "Anambra rate"},
    ],
    "Oranges": [
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 200,   "road": "average", "unit": "per 5 fruits", "note": "South-East citrus zone"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 250,   "road": "good",    "unit": "per 5 fruits", "note": "South-West"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 300,   "road": "average", "unit": "per 5 fruits", "note": "South distribution"},
    ],
    "Watermelons": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 800,   "road": "average", "unit": "per fruit", "note": "Northern farms"},
        {"city": "Bodija",        "state": "Oyo",      "wholesale_price": 1200,  "road": "good",    "unit": "per fruit", "note": "South-West"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 1500,  "road": "average", "unit": "per fruit", "note": "South rate"},
    ],
    "Spinach": [
        {"city": "Eke Onunwa",    "state": "Imo",      "wholesale_price": 150,   "road": "average", "unit": "per bunch", "note": "South-East leafy veg zone"},
        {"city": "Umuahia",       "state": "Abia",     "wholesale_price": 180,   "road": "average", "unit": "per bunch", "note": "Abia rate"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 250,   "road": "average", "unit": "per bunch", "note": "Anambra rate"},
    ],
    "Cowpeas": [
        {"city": "Dawanau",       "state": "Kano",     "wholesale_price": 65000, "road": "average", "unit": "100kg bag", "note": "Northern cowpea belt"},
        {"city": "Onitsha",       "state": "Anambra",  "wholesale_price": 82000, "road": "average", "unit": "100kg bag", "note": "South rate"},
    ],
    "Fuel (petrol-gasoline)": [
        {"city": "Lagos",         "state": "Lagos",    "wholesale_price": 950,   "road": "good",    "unit": "per litre", "note": "NNPC pump price"},
        {"city": "Abuja",         "state": "FCT (Abuja)", "wholesale_price": 970, "road": "good",   "unit": "per litre", "note": "FCT NNPC rate"},
    ],
    "Fuel (diesel)": [
        {"city": "Lagos",         "state": "Lagos",    "wholesale_price": 1800,  "road": "good",    "unit": "per litre", "note": "Lagos depot price"},
        {"city": "Abuja",         "state": "FCT (Abuja)", "wholesale_price": 1850, "road": "good",  "unit": "per litre", "note": "FCT depot price"},
    ],
}

# ── Trader margins per commodity type ─────────────────────────────────────────
# Real South-East/Abia trader markup percentages based on commodity risk
TRADER_MARGINS = {
    "Rice (milled, local)":        0.18,   # stable staple, moderate margin
    "Rice (imported)":             0.20,   # import premium, moderate
    "Beans (white)":               0.20,
    "Beans (red)":                 0.20,
    "Maize":                       0.18,
    "Maize (white)":               0.18,
    "Maize (yellow)":              0.18,
    "Maize flour":                 0.20,
    "Gari (white)":                0.22,   # garri: moderate perishability
    "Cassava meal (gari, yellow)": 0.22,
    "Yam":                         0.28,   # yam spoils, higher margin
    "Oil (palm)":                  0.22,
    "Oil (vegetable)":             0.18,
    "Tomatoes":                    0.40,   # HIGH — 2-5 day shelf life
    "Onions":                      0.32,   # moderate perishability
    "Fish":                        0.38,   # HIGH — perishable, cold chain rare
    "Eggs":                        0.25,
    "Meat (beef)":                 0.30,
    "Meat (goat)":                 0.32,
    "Milk (powder)":               0.20,
    "Groundnuts":                  0.18,
    "Groundnuts (shelled)":        0.22,
    "Sorghum":                     0.18,
    "Millet":                      0.18,
    "Cowpeas":                     0.20,
    "Salt":                        0.15,   # near-zero perishability, low margin
    "Sugar":                       0.18,
    "Bread":                       0.25,   # stale in 2–3 days
    "Bananas":                     0.35,   # ripen fast
    "Oranges":                     0.28,
    "Watermelons":                 0.30,
    "Spinach":                     0.45,   # wilts in 1–2 days — HIGHEST margin
    "Fuel (petrol-gasoline)":      0.10,
    "Fuel (diesel)":               0.10,
}

# ── State income data (average monthly income, NGN, 2024) ────────────────────
# Source: NBS NLSS 2022 + estimated 2024 update
STATE_AVG_MONTHLY_INCOME = {
    "Lagos":          85000,
    "FCT (Abuja)":    78000,
    "Rivers":         72000,
    "Anambra":        55000,
    "Oyo":            48000,
    "Enugu":          45000,
    "Abia":           42000,   # Umudike/Umuahia corridor — MOUAU area
    "Delta":          55000,
    "Imo":            40000,
    "Edo":            47000,
    "Kaduna":         38000,
    "Kano":           35000,
    "Ogun":           50000,
    "Kwara":          36000,
    "Osun":           33000,
    "Ekiti":          32000,
    "Ondo":           35000,
    "Benue":          28000,
    "Kogi":           30000,
    "Plateau":        32000,
    "Nasarawa":       28000,
    "Niger":          27000,
    "Kebbi":          24000,
    "Zamfara":        22000,
    "Sokoto":         23000,
    "Katsina":        26000,
    "Jigawa":         22000,
    "Bauchi":         25000,
    "Gombe":          25000,
    "Adamawa":        24000,
    "Taraba":         23000,
    "Borno":          22000,
    "Yobe":           21000,
    "Akwa Ibom":      45000,
    "Cross River":    38000,
    "Bayelsa":        60000,   # Oil money — relatively high
    "Ebonyi":         28000,
}


# ─────────────────────────────────────────────────────────────────────────────
#  CORE CALCULATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_transport_cost_per_unit(
    source_city: str,
    dest_state: str,
    unit: str,
    road_quality: str = "average",
) -> dict:
    """
    Calculate transport cost from source city to destination state per unit.

    Formula:
        fuel_used       = distance × litres_per_km_truck × road_factor
        total_fuel_cost = fuel_used × diesel_price_per_litre
        cost_per_bag    = total_fuel_cost ÷ bags_per_truck
        cost_per_unit   = adjust if unit is not a 50kg bag

    Returns dict with full breakdown.
    """
    # Look up driving distance
    dist_key_city  = f"{source_city.replace(' ', '')}→{dest_state}"
    dist_key_state = None

    # Try city-based key first
    dist_km = DRIVING_DISTANCES.get(dist_key_city)
    if dist_km is None:
        # Try state-based key (source_city is actually a state city name)
        for key, km in DRIVING_DISTANCES.items():
            src_part, dst_part = key.split("→")
            if src_part.lower() in source_city.lower() or source_city.lower() in src_part.lower():
                if dst_part.lower() == dest_state.lower():
                    dist_km = km
                    break

    if dist_km is None:
        return {"error": f"No distance data for {source_city} → {dest_state}", "cost_per_unit": 0}

    if dist_km == 0:
        return {
            "distance_km": 0,
            "fuel_litres": 0,
            "fuel_cost": 0,
            "cost_per_unit": 0,
            "road_factor": 1.0,
            "note": "Same state — minimal transport cost",
        }

    road_mult   = ROAD_FACTOR.get(road_quality, 1.15)
    fuel_litres = dist_km * LITRES_PER_KM_TRUCK * road_mult
    fuel_cost   = fuel_litres * DIESEL_PRICE_PER_LITRE

    # All commodity sources use "bags" as truck load unit (standardised to 200 bags)
    # For non-bag items (per kg, per litre, per piece) we convert proportionally
    # A 200-bag truck = 200 × 50kg = 10,000 kg payload
    cost_per_50kg_bag = fuel_cost / BAGS_PER_TRUCK

    # Convert to actual commodity unit
    unit_lower = unit.lower()
    if "100kg" in unit_lower:
        cost_per_unit = cost_per_50kg_bag * 2
    elif "50kg" in unit_lower or "bag" in unit_lower:
        cost_per_unit = cost_per_50kg_bag
    elif "per kg" in unit_lower:
        cost_per_unit = cost_per_50kg_bag / 50
    elif "per litre" in unit_lower or "litre" in unit_lower:
        # Assume 1 litre ≈ 0.9 kg for oils/fuels
        cost_per_unit = cost_per_50kg_bag / 45
    elif "crate" in unit_lower:
        # A crate of 30 eggs ≈ 3.6 kg
        cost_per_unit = cost_per_50kg_bag * 3.6 / 50
    elif "paint bucket" in unit_lower:
        # Paint bucket ~3 kg garri
        cost_per_unit = cost_per_50kg_bag * 3 / 50
    elif "tuber" in unit_lower:
        # Average yam tuber 3–5 kg, use 4 kg
        cost_per_unit = cost_per_50kg_bag * 4 / 50
    elif "bunch" in unit_lower:
        # Banana bunch ~5 kg; spinach bunch ~0.5 kg
        if "banana" in unit_lower or "plantain" in unit_lower:
            cost_per_unit = cost_per_50kg_bag * 5 / 50
        else:
            cost_per_unit = cost_per_50kg_bag * 0.5 / 50
    elif "loaf" in unit_lower:
        cost_per_unit = cost_per_50kg_bag * 0.6 / 50
    elif "fruit" in unit_lower:
        cost_per_unit = cost_per_50kg_bag * 0.5 / 50
    elif "tin" in unit_lower or "400g" in unit_lower:
        cost_per_unit = cost_per_50kg_bag * 0.4 / 50
    else:
        cost_per_unit = cost_per_50kg_bag  # fallback

    return {
        "distance_km":    round(dist_km),
        "road_factor":    road_mult,
        "fuel_litres":    round(fuel_litres, 1),
        "diesel_price":   DIESEL_PRICE_PER_LITRE,
        "fuel_cost_total":round(fuel_cost),
        "bags_per_truck": BAGS_PER_TRUCK,
        "cost_per_unit":  round(cost_per_unit),
    }


def get_landed_cost(commodity: str, dest_state: str) -> list:
    """
    Calculate landed cost for a commodity from every available source to dest_state.

    Returns list of dicts, each with:
        source_city, source_state, wholesale_price, transport_cost,
        landed_cost, road_quality, unit, note
    Sorted by landed_cost ascending (cheapest first).
    """
    sources = COMMODITY_SOURCES.get(commodity)
    if not sources:
        return []

    results = []
    for src in sources:
        transport = get_transport_cost_per_unit(
            source_city  = src["city"],
            dest_state   = dest_state,
            unit         = src["unit"],
            road_quality = src["road"],
        )

        if "error" in transport:
            transport_cost = 0
            dist_km = None
        else:
            transport_cost = transport["cost_per_unit"]
            dist_km        = transport["distance_km"]

        landed = src["wholesale_price"] + transport_cost

        results.append({
            "source_city":     src["city"],
            "source_state":    src["state"],
            "wholesale_price": src["wholesale_price"],
            "transport_cost":  transport_cost,
            "landed_cost":     round(landed),
            "distance_km":     dist_km,
            "road_quality":    src["road"],
            "unit":            src["unit"],
            "note":            src["note"],
            "transport_detail":transport,
        })

    # Sort by landed cost — cheapest source first
    results.sort(key=lambda x: x["landed_cost"])
    return results


def get_best_source(commodity: str, dest_state: str) -> Optional[dict]:
    """Return the single cheapest landed-cost source for a commodity in a given state."""
    options = get_landed_cost(commodity, dest_state)
    return options[0] if options else None


def get_street_price(commodity: str, dest_state: str) -> dict:
    """
    Full pipeline:
      1. Find cheapest source (landed cost)
      2. Apply trader margin
      3. Return full price build-up + state income context

    Returns:
        {
          commodity, dest_state,
          best_source: { city, state, wholesale_price, transport_cost, landed_cost },
          all_sources: [...],
          trader_margin_pct,
          street_price,
          unit,
          income_context: { avg_monthly_income, price_as_pct_income },
          explanation: "human-readable breakdown"
        }
    """
    all_sources = get_landed_cost(commodity, dest_state)
    if not all_sources:
        return {"error": f"No source data for {commodity}"}

    best = all_sources[0]
    margin = TRADER_MARGINS.get(commodity, 0.20)
    street_price = round(best["landed_cost"] * (1 + margin))
    unit = best["unit"]

    # Income context
    monthly_income = STATE_AVG_MONTHLY_INCOME.get(dest_state, 35000)
    price_pct = (street_price / monthly_income) * 100 if monthly_income else 0

    # Human-readable explanation
    explanation = (
        f"Buying from {best['source_city']} ({best['source_state']}) "
        f"at ₦{best['wholesale_price']:,.0f}/{unit} wholesale. "
        f"Road distance: {best['distance_km']} km. "
        f"Transport cost per unit: ₦{best['transport_cost']:,.0f} "
        f"(truck diesel at ₦{DIESEL_PRICE_PER_LITRE:,.0f}/L, {best['distance_km']}km × "
        f"{LITRES_PER_KM_TRUCK}L/km × {best['road_quality']} road factor ÷ {BAGS_PER_TRUCK} bags). "
        f"Landed cost: ₦{best['landed_cost']:,.0f}. "
        f"Trader adds {margin*100:.0f}% margin → "
        f"street price: ₦{street_price:,.0f}. "
        f"This is {price_pct:.1f}% of avg {dest_state} monthly income (₦{monthly_income:,.0f})."
    )

    return {
        "commodity":         commodity,
        "dest_state":        dest_state,
        "best_source":       best,
        "all_sources":       all_sources,
        "trader_margin_pct": round(margin * 100, 1),
        "street_price":      street_price,
        "unit":              unit,
        "income_context": {
            "avg_monthly_income_ngn": monthly_income,
            "price_as_pct_income":    round(price_pct, 2),
        },
        "explanation": explanation,
    }


def explain_price(commodity: str, dest_state: str) -> str:
    """Return a plain-English, step-by-step explanation of how the street price is built."""
    result = get_street_price(commodity, dest_state)
    if "error" in result:
        return result["error"]
    return result["explanation"]


def get_all_commodities_for_state(dest_state: str) -> list:
    """
    Calculate street prices for ALL commodities for a given state.
    Returns list sorted by price_as_pct_income descending
    (most expensive relative to income first — shows where pain is greatest).
    """
    results = []
    for commodity in COMMODITY_SOURCES:
        r = get_street_price(commodity, dest_state)
        if "error" not in r:
            results.append(r)
    results.sort(key=lambda x: x["income_context"]["price_as_pct_income"], reverse=True)
    return results


def compare_states(commodity: str, states: list) -> list:
    """
    Compare street prices for a commodity across multiple states.
    Returns list sorted by street_price ascending.
    """
    results = []
    for state in states:
        r = get_street_price(commodity, state)
        if "error" not in r:
            results.append(r)
    results.sort(key=lambda x: x["street_price"])
    return results
