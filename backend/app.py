"""
backend/app.py  —  NaijaAgro Predict · Flask Backend v2
"""
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow requests from React dev server AND production domain
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:3001", "http://127.0.0.1:3001",
    "http://localhost:3002", "http://127.0.0.1:3002",
    "https://your-domain.com"   # ← replace with your real frontend URL
]}})

# ── Blueprints ───────────────────────────────────────────────────────────────
from routes.predict import predict_bp
app.register_blueprint(predict_bp)          # keeps old routes working

from routes.trade import trade_bp
app.register_blueprint(trade_bp, url_prefix="/api")   # new trade routes


# ── Health check (keep the old one) ─────────────────────────────────────────
@app.route("/health")
def health():
    return {"status": "ok", "service": "NaijaAgro Predict API v2"}


# ── Root index (new) ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "service": "NaijaAgro Predict API v2",
        "trade_endpoints": {
            "street_price": "GET /api/trade/street-price?commodity=Rice+(milled,+local)&state=Abia",
            "all_sources":  "GET /api/trade/sources?commodity=Maize&state=Abia",
            "best_source":  "GET /api/trade/best-source?commodity=Maize&state=Abia",
            "all_prices":   "GET /api/trade/all?state=Abia",
            "compare":      "GET /api/trade/compare?commodity=Rice+(milled,+local)&states=Abia,Lagos,Kano",
            "explain":      "GET /api/trade/explain?commodity=Yam&state=Abia",
            "config":       "GET /api/trade/config",
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")