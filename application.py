import os
import sys
import json
import threading
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    stream_with_context,
)
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from logger import logger
from exception import CustomException

app = Flask(__name__)
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
_training_in_progress = False
_training_logs = []
_model_loaded = False


def _check_model_loaded() -> bool:
    return (
        os.path.exists(os.path.join("artifacts", "model.pkl")) and
        os.path.exists(os.path.join("artifacts", "preprocessor.pkl"))
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page with dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    """Prediction form page."""
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for single-order prediction."""
    try:
        from src.pipeline.predict_pipeline import PredictPipeline, CustomData

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required = [
            "shipping_mode", "days_for_shipping_real",
            "days_for_shipment_scheduled", "market",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        custom_data = CustomData(
            shipping_mode=data.get("shipping_mode", "Standard Class"),
            days_for_shipping_real=float(data.get("days_for_shipping_real", 5)),
            days_for_shipment_scheduled=float(data.get("days_for_shipment_scheduled", 4)),
            order_item_quantity=int(data.get("order_item_quantity", 1)),
            order_item_discount_rate=float(data.get("order_item_discount_rate", 0.0)),
            order_item_product_price=float(data.get("order_item_product_price", 100.0)),
            sales_per_customer=float(data.get("sales_per_customer", 100.0)),
            market=data.get("market", "USCA"),
            order_region=data.get("order_region", "US Center"),
            customer_segment=data.get("customer_segment", "Consumer"),
            department_name=data.get("department_name", "Fan Shop"),
            category_name=data.get("category_name", "Sporting Goods"),
            order_status=data.get("order_status", "COMPLETE"),
            order_type=data.get("order_type", "DEBIT"),
            order_item_discount=float(data.get("order_item_discount", 0.0)),
            sales=float(data.get("sales", 100.0)),
            order_item_total=float(data.get("order_item_total", 100.0)),
            order_profit_per_order=float(data.get("order_profit_per_order", 20.0)),
            benefit_per_order=float(data.get("benefit_per_order", 20.0)),
            late_delivery_risk=int(data.get("late_delivery_risk", 0)),
            order_item_profit_ratio=float(data.get("order_item_profit_ratio", 0.2)),
            product_price=float(data.get("product_price", 100.0)),
            product_status=int(data.get("product_status", 0)),
            latitude=float(data.get("latitude", 40.0)),
            longitude=float(data.get("longitude", -74.0)),
        )

        df = custom_data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        result = pipeline.predict(df.iloc[0].to_dict())

        return jsonify({"success": True, "result": result})

    except FileNotFoundError:
        return jsonify({
            "error": "Model not trained yet. Please run training first.",
            "success": False,
        }), 503
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/train", methods=["POST"])
def train():
    """
    Trigger model training in a background thread.
    Streams logs back to client via Server-Sent Events.
    """
    global _training_in_progress, _training_logs

    if _training_in_progress:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json() or {}
    data_path = data.get("data_path", "data/supply_chain_data.csv")

    if not os.path.exists(data_path):
        return jsonify({
            "error": f"Dataset not found at: {data_path}. Please upload the dataset first.",
            "success": False,
        }), 404

    _training_logs = []
    _training_in_progress = True

    def run_training():
        global _training_in_progress, _training_logs
        try:
            from src.pipeline.train_pipeline import TrainPipeline
            _training_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started...")
            pipeline = TrainPipeline()
            metrics = pipeline.run_pipeline(data_path=data_path)
            _training_logs.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Training complete! "
                f"Best model: {metrics['best_model']} | "
                f"F1: {metrics['best_weighted_f1']:.4f}"
            )
        except Exception as e:
            _training_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error: {str(e)}")
            logger.error(f"Training thread error: {e}")
        finally:
            _training_in_progress = False

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({"success": True, "message": "Training started in background"})


@app.route("/train/status", methods=["GET"])
def train_status():
    """Return current training status and logs."""
    return jsonify({
        "in_progress": _training_in_progress,
        "logs": _training_logs[-50:],  # Last 50 log entries
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    model_ready = _check_model_loaded()
    return jsonify({
        "status": "ok",
        "model_loaded": model_ready,
        "training_in_progress": _training_in_progress,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return the latest training metrics."""
    metrics_path = os.path.join("artifacts", "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({
            "error": "No metrics available. Train the model first.",
            "available": False,
        }), 404

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        data["available"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features", methods=["GET"])
def feature_importance():
    """Return feature importance from the best model."""
    importance_path = os.path.join("artifacts", "feature_importance.json")
    if not os.path.exists(importance_path):
        return jsonify({
            "error": "Feature importance not available.",
            "available": False,
        }), 404

    try:
        with open(importance_path, "r") as f:
            data = json.load(f)

        # Return top 20 features
        top_features = dict(list(data.items())[:20])
        return jsonify({"features": top_features, "available": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return info about the trained model."""
    model_name_path = os.path.join("artifacts", "best_model_name.txt")
    if not os.path.exists(model_name_path):
        return jsonify({"available": False}), 404

    with open(model_name_path, "r") as f:
        best_model = f.read().strip()

    return jsonify({
        "best_model": best_model,
        "available": True,
        "artifacts": {
            "model": os.path.exists(os.path.join("artifacts", "model.pkl")),
            "preprocessor": os.path.exists(os.path.join("artifacts", "preprocessor.pkl")),
            "metrics": os.path.exists(os.path.join("artifacts", "metrics.json")),
        },
    })


# ── Error Handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "code": 404}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "code": 500}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    logger.info(f"Starting Amazon Supply Chain Intelligence on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
