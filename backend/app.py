"""
backend/app.py  —  NaijaAgro Predict Flask API
Run:  python app.py   (from inside the backend/ folder)
"""

from flask import Flask
from flask_cors import CORS
from routes.predict import predict_bp

app = Flask(__name__)

# Allow requests from the React dev server (port 3000, 3001, 3002)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001", "http://localhost:3002", "http://127.0.0.1:3002"]}})

# Register blueprints
app.register_blueprint(predict_bp)


@app.route("/health")
def health():
    return {"status": "ok", "service": "NaijaAgro Predict API"}


if __name__ == "__main__":
    app.run(debug=True, port=5000)