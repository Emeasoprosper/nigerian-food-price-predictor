from flask import Blueprint, request, jsonify
import sys, os

# Allow imports from the parent backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.predict import predict_price, predict_all

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  /predict?commodity=Rice&state=Lagos&year=2025&month=6
    POST /predict  { "commodity": "...", "state": "...", "year": ..., "month": ... }

    Returns JSON with predicted_ngn, predicted_usd, trend, confidence, etc.
    """
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        commodity = body.get("commodity", "")
        state     = body.get("state", "")
        year      = int(body.get("year",  2025))
        month     = int(body.get("month", 6))
    else:
        commodity = request.args.get("commodity", "")
        state     = request.args.get("state", "")
        try:
            year  = int(request.args.get("year",  2025))
            month = int(request.args.get("month", 6))
        except ValueError:
            return jsonify({"error": "year and month must be integers"}), 400

    if not commodity:
        return jsonify({"error": "commodity is required"}), 400
    if not state:
        return jsonify({"error": "state is required"}), 400

    result = predict_price(commodity, state, year, month)

    if "error" in result:
        return jsonify(result), 404

    return jsonify(result)


@predict_bp.route("/predict/all", methods=["GET"])
def predict_all_route():
    """
    GET /predict/all?state=Lagos&year=2025&month=6
    Returns predictions for every available commodity in that state.
    """
    state = request.args.get("state", "Lagos")
    try:
        year  = int(request.args.get("year",  2025))
        month = int(request.args.get("month", 6))
    except ValueError:
        return jsonify({"error": "year and month must be integers"}), 400

    results = predict_all(state, year, month)
    return jsonify(results)


@predict_bp.route("/commodities", methods=["GET"])
def list_commodities():
    """
    GET /commodities  →  list of all commodities with trained models
    """
    import json, os
    summary_path = os.path.join(os.path.dirname(__file__), "..", "model", "saved", "model_summary.json")
    if not os.path.exists(summary_path):
        return jsonify({"error": "No model summary found. Run train.py first."}), 404
    with open(summary_path) as f:
        summary = json.load(f)
    return jsonify({
        "commodities": sorted(summary.keys()),
        "count": len(summary),
        "mape": summary
    })