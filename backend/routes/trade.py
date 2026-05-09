"""
routes/trade.py  —  NaijaAgro Predict · Trade Economics Routes
==============================================================
Exposes the supply chain cost engine as REST endpoints.

Register in app.py with:
    from routes.trade import trade_bp
    app.register_blueprint(trade_bp, url_prefix="/api")

Endpoints:
    GET /api/trade/street-price
        ?commodity=Rice+(milled,+local)&state=Abia
        Returns full price build-up: source → transport → landed → margin → street price

    GET /api/trade/sources
        ?commodity=Rice+(milled,+local)&state=Abia
        Returns all source markets ranked cheapest landed cost first

    GET /api/trade/best-source
        ?commodity=Maize&state=Abia
        Returns single cheapest source with transport breakdown

    GET /api/trade/all
        ?state=Abia
        Returns street prices for ALL commodities in a state, sorted by
        affordability burden (% of average monthly income)

    GET /api/trade/compare
        ?commodity=Rice+(milled,+local)&states=Abia,Lagos,Kano,Rivers,Enugu
        Compares street price across multiple states

    GET /api/trade/explain
        ?commodity=Yam&state=Abia
        Returns human-readable step-by-step explanation of price formation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Blueprint, request, jsonify
from model.trade_economics import (
    get_street_price,
    get_landed_cost,
    get_best_source,
    get_all_commodities_for_state,
    compare_states,
    explain_price,
    DIESEL_PRICE_PER_LITRE,
    PETROL_PRICE_PER_LITRE,
    LITRES_PER_KM_TRUCK,
    BAGS_PER_TRUCK,
    STATE_AVG_MONTHLY_INCOME,
    TRADER_MARGINS,
)

trade_bp = Blueprint("trade", __name__)


def _require(param: str):
    """Return param or 400 error."""
    val = request.args.get(param, "").strip()
    if not val:
        return None, jsonify({"error": f"'{param}' is required"}), 400
    return val, None, None


@trade_bp.route("/trade/street-price", methods=["GET"])
def street_price():
    """
    GET /api/trade/street-price?commodity=Rice+(milled,+local)&state=Abia

    Full pipeline: source wholesale → transport → landed → margin → street price
    Also includes income context and all alternative source options.
    """
    commodity = request.args.get("commodity", "").strip()
    state     = request.args.get("state", "").strip()
    if not commodity:
        return jsonify({"error": "'commodity' is required"}), 400
    if not state:
        return jsonify({"error": "'state' is required"}), 400

    result = get_street_price(commodity, state)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@trade_bp.route("/trade/sources", methods=["GET"])
def sources():
    """
    GET /api/trade/sources?commodity=Rice+(milled,+local)&state=Abia

    All source markets ranked by cheapest landed cost.
    """
    commodity = request.args.get("commodity", "").strip()
    state     = request.args.get("state", "").strip()
    if not commodity or not state:
        return jsonify({"error": "'commodity' and 'state' are required"}), 400

    results = get_landed_cost(commodity, state)
    if not results:
        return jsonify({"error": f"No source data for '{commodity}'"}), 404

    return jsonify({
        "commodity":    commodity,
        "dest_state":   state,
        "n_sources":    len(results),
        "fuel_params": {
            "diesel_price_per_litre": DIESEL_PRICE_PER_LITRE,
            "litres_per_km":          LITRES_PER_KM_TRUCK,
            "bags_per_truck":         BAGS_PER_TRUCK,
        },
        "sources": results,
    })


@trade_bp.route("/trade/best-source", methods=["GET"])
def best_source():
    """
    GET /api/trade/best-source?commodity=Maize&state=Abia

    Returns the single cheapest source with full transport breakdown.
    """
    commodity = request.args.get("commodity", "").strip()
    state     = request.args.get("state", "").strip()
    if not commodity or not state:
        return jsonify({"error": "'commodity' and 'state' are required"}), 400

    result = get_best_source(commodity, state)
    if not result:
        return jsonify({"error": f"No source data for '{commodity}'"}), 404

    return jsonify({
        "commodity":  commodity,
        "dest_state": state,
        "best":       result,
    })


@trade_bp.route("/trade/all", methods=["GET"])
def all_commodities():
    """
    GET /api/trade/all?state=Abia

    Street prices for ALL commodities in a state, sorted by
    affordability burden (% of monthly income — most burdensome first).
    Useful for showing which goods hurt the most in that state.
    """
    state = request.args.get("state", "Abia").strip()
    results = get_all_commodities_for_state(state)
    income  = STATE_AVG_MONTHLY_INCOME.get(state, 35000)

    return jsonify({
        "state":              state,
        "avg_monthly_income": income,
        "n_commodities":      len(results),
        "commodities":        results,
    })


@trade_bp.route("/trade/compare", methods=["GET"])
def compare():
    """
    GET /api/trade/compare?commodity=Rice+(milled,+local)&states=Abia,Lagos,Kano,Rivers,Enugu

    Compares street price for a commodity across multiple states.
    """
    commodity  = request.args.get("commodity", "").strip()
    states_str = request.args.get("states", "").strip()
    if not commodity:
        return jsonify({"error": "'commodity' is required"}), 400

    states = [s.strip() for s in states_str.split(",") if s.strip()] if states_str else list(STATE_AVG_MONTHLY_INCOME.keys())
    results = compare_states(commodity, states)

    return jsonify({
        "commodity": commodity,
        "n_states":  len(results),
        "comparison": results,
    })


@trade_bp.route("/trade/explain", methods=["GET"])
def explain():
    """
    GET /api/trade/explain?commodity=Yam&state=Abia

    Human-readable step-by-step explanation of how the street price is built.
    """
    commodity = request.args.get("commodity", "").strip()
    state     = request.args.get("state", "").strip()
    if not commodity or not state:
        return jsonify({"error": "'commodity' and 'state' are required"}), 400

    text = explain_price(commodity, state)
    full = get_street_price(commodity, state)

    return jsonify({
        "commodity":   commodity,
        "state":       state,
        "explanation": text,
        "data":        full if "error" not in full else None,
    })


@trade_bp.route("/trade/config", methods=["GET"])
def config():
    """
    GET /api/trade/config

    Returns all constants used in the trade model — useful for frontend display
    and for verifying the model parameters.
    """
    return jsonify({
        "fuel": {
            "diesel_price_per_litre_ngn": DIESEL_PRICE_PER_LITRE,
            "petrol_price_per_litre_ngn": PETROL_PRICE_PER_LITRE,
            "source":                     "GlobalPetrolPrices.com May 2026",
        },
        "truck": {
            "litres_per_km":   LITRES_PER_KM_TRUCK,
            "bags_per_truck":  BAGS_PER_TRUCK,
            "kg_per_bag":      50,
            "source":          "Nigerian haulage data: 15t truck, 118km=130L (Ayoola Ashiru, 2016)",
        },
        "trader_margins": TRADER_MARGINS,
        "state_incomes":  STATE_AVG_MONTHLY_INCOME,
    })