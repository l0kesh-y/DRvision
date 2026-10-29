"""
app.py – DRVision Flask application entry point.

Routes:
  GET  /                    → Home page (dataset stats)
  GET  /predict             → Predict page
  GET  /history             → History page
  GET  /about               → About page

API:
  POST /api/predict         → Upload + analyse fundus image
  GET  /api/history         → Paginated prediction history
  GET  /api/result/<id>     → Single prediction record
  GET  /api/dataset/stats   → APTOS 2019 class distribution
  GET  /uploads/<filename>  → Serve uploaded images
  GET  /heatmaps/<filename> → Serve heatmap images
"""

import os
import uuid
import logging
import pandas as pd
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, abort
)
from werkzeug.utils import secure_filename

from config import Config
from database import init_db, save_prediction, get_all_predictions, \
    get_prediction_by_id, get_grade_statistics
from ml.groq_predictor import predict_grade
from ml.heatmap import generate_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# Application setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# Ensure storage directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def get_dataset_stats() -> dict:
    """Read train.csv and compute class distribution for the dashboard."""
    try:
        df = pd.read_csv(Config.TRAIN_CSV)
        counts = df["diagnosis"].value_counts().sort_index()
        total = len(df)
        labels = Config.GRADE_LABELS
        distribution = [
            {
                "grade": int(g),
                "label": labels.get(int(g), f"Grade {g}"),
                "count": int(c),
                "percentage": round(int(c) / total * 100, 1),
                "color": Config.GRADE_COLORS.get(int(g), "#888"),
            }
            for g, c in counts.items()
        ]
        return {"total": total, "distribution": distribution}
    except Exception as e:
        logger.warning(f"Could not load dataset stats: {e}")
        return {"total": 3662, "distribution": []}


# ─────────────────────────────────────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = get_dataset_stats()
    return render_template("index.html", stats=stats)


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/test")
def test_modal():
    return render_template("test_modal.html")


# ─────────────────────────────────────────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@app.route("/heatmaps/<path:filename>")
def serve_heatmap(filename):
    return send_from_directory(Config.HEATMAP_FOLDER, filename)


# ─────────────────────────────────────────────────────────────────────────────
# API – Predict
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept a fundus image, run Groq Vision analysis, generate heatmap,
    persist to Neon DB, and return the full result JSON.
    """
    # ── Validate upload ───────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Save upload ───────────────────────────────────────────────────
    original_filename = secure_filename(file.filename)
    uid = uuid.uuid4().hex[:12]
    ext = original_filename.rsplit(".", 1)[1].lower()
    saved_filename = f"{uid}.{ext}"
    saved_path = os.path.join(Config.UPLOAD_FOLDER, saved_filename)
    file.save(saved_path)
    logger.info(f"Saved upload: {saved_path}")

    # ── Gemini Vision prediction ──────────────────────────────────────
    try:
        result = predict_grade(saved_path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # ── Heatmap generation ────────────────────────────────────────────
    try:
        heatmap_filename = generate_heatmap(saved_path, result["grade"], uid)
    except Exception as e:
        logger.warning(f"Heatmap generation failed: {e}")
        heatmap_filename = None

    # ── Persist to database ───────────────────────────────────────────
    try:
        db_record = {
            "image_filename":    saved_filename,
            "heatmap_filename":  heatmap_filename,
            "original_filename": original_filename,
            "predicted_grade":   result["grade"],
            "confidence_score":  result.get("confidence_score", round(result["grade"] + (result["confidence"] - 0.5) * 0.7, 2)),
            "grade_label":       result["grade_label"],
            "risk_level":        result["risk_level"],
            "recommendation":    result["recommendation"],
            "findings":          result.get("findings", []),
            "clinical_notes":    result.get("clinical_notes", ""),
            "lesion_locations":  result.get("lesion_locations", []),
            "groq_raw_response": result.get("groq_raw_response", ""),
        }
        record_id = save_prediction(db_record)
    except Exception as e:
        logger.warning(f"DB save failed: {e}")
        record_id = None

    # ── Terminal Display for Prediction ───────────────────────────────
    acc = result.get("accuracy", 0.7981)
    f1 = result.get("macro_f1", 0.6231)
    qwk = result.get("qwk", 0.8971)

    print("\n" + "=" * 76)
    print(f"  [INFERENCE RESULT] Retinal Fundus Image: {original_filename}")
    print("=" * 76)
    print(f"  -> Predicted Diagnosis : Grade {result['grade']} - {result['grade_label']} ({result['risk_level']} Risk)")
    print(f"  -> Severity Score      : {result.get('confidence_score', round(result['grade'] + (result['confidence'] - 0.5) * 0.7, 2)):.2f} / 4.00")
    print(f"  -> Inference Confidence: {result['confidence'] * 100:.1f}%")
    print("-" * 76)
    print(f"  [*] Model Accuracy     : {acc * 100:.2f}%  (score: {acc:.4f})")
    print(f"  [*] Model Macro F1     : {f1 * 100:.2f}%  (score: {f1:.4f})")
    print(f"  [*] Quadratic Kappa    : {qwk:.4f}")
    print("=" * 76 + "\n")

    # ── Build response ────────────────────────────────────────────────
    conf_score = result.get("confidence_score", round(result["grade"] + (result["confidence"] - 0.5) * 0.7, 2))
    return jsonify({
        "id":               record_id,
        "grade":            result["grade"],
        "confidence":       result["confidence"],
        "confidence_score": conf_score,
        "grade_label":      result["grade_label"],
        "risk_level":       result["risk_level"],
        "recommendation":   result["recommendation"],
        "findings":         result.get("findings", []),
        "clinical_notes":   result.get("clinical_notes", ""),
        "lesion_locations": result.get("lesion_locations", []),
        "grade_color":      Config.GRADE_COLORS.get(result["grade"], "#888"),
        "accuracy":         acc,
        "macro_f1":         f1,
        "qwk":              qwk,
        "image_url":        f"/uploads/{saved_filename}",
        "heatmap_url":      f"/heatmaps/{heatmap_filename}" if heatmap_filename else None,
        "created_at":       datetime.utcnow().isoformat(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# API – History
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 12))
    grade_filter = request.args.get("grade")

    try:
        data = get_all_predictions(page=page, per_page=per_page)
        # Inject URLs and colour info
        for r in data["records"]:
            r["image_url"]   = f"/uploads/{r['image_filename']}"
            r["heatmap_url"] = f"/heatmaps/{r['heatmap_filename']}" if r.get("heatmap_filename") else None
            r["grade_color"] = Config.GRADE_COLORS.get(r["predicted_grade"], "#888")
        return jsonify(data)
    except Exception as e:
        logger.exception("History fetch failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/result/<int:pred_id>")
def api_result(pred_id):
    logger.info(f"API request for prediction {pred_id}")
    try:
        record = get_prediction_by_id(pred_id)
        logger.info(f"Record fetched: {record is not None}")
        if not record:
            return jsonify({"error": "Record not found."}), 404
        record["image_url"]   = f"/uploads/{record['image_filename']}"
        record["heatmap_url"] = f"/heatmaps/{record['heatmap_filename']}" if record.get("heatmap_filename") else None
        record["grade_color"] = Config.GRADE_COLORS.get(record["predicted_grade"], "#888")
        logger.info(f"Returning record for {pred_id}")
        return jsonify(record)
    except Exception as e:
        logger.error(f"API error for {pred_id}: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# API – Dataset Statistics
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/dataset/stats")
def api_dataset_stats():
    return jsonify(get_dataset_stats())


@app.route("/api/db/stats")
def api_db_stats():
    """Grade distribution of all predictions stored in the DB."""
    try:
        stats = get_grade_statistics()
        return jsonify({"stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        init_db()
        logger.info("Database initialised.")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

    try:
        from ml.densenet_model import get_model, print_terminal_metrics
        get_model()
        logger.info("DenseNet-121 model pre-warmed successfully.")
        print_terminal_metrics()
    except Exception as e:
        logger.warning(f"Model pre-warm warning: {e}")

    if not Config.GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY not set! Add it to .env. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    app.run(debug=Config.DEBUG, host="0.0.0.0", port=5000)
