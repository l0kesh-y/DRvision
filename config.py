"""
config.py – Centralised application configuration for DRVision.
Reads settings from .env file. No hardcoded secrets.
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── Flask ──────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "drvision-dev-secret-2024")
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # ── Google Gemini AI ───────────────────────────────────────────────
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_FALLBACK_MODEL = "gemini-flash-latest"

    # ── Neon PostgreSQL ────────────────────────────────────────────────
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/drvision")

    # ── Filesystem paths ───────────────────────────────────────────────
    UPLOAD_FOLDER     = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "uploads"))
    HEATMAP_FOLDER    = os.path.join(BASE_DIR, os.getenv("HEATMAP_FOLDER", "heatmaps"))
    DATASET_FOLDER    = os.path.join(BASE_DIR, os.getenv("DATASET_FOLDER", "dataset"))
    TRAIN_IMAGES_FOLDER = os.path.join(
        BASE_DIR, os.getenv("TRAIN_IMAGES_FOLDER", "dataset/train_images")
    )
    TRAIN_CSV = os.path.join(BASE_DIR, "dataset", "train.csv")

    # ── ML settings ────────────────────────────────────────────────────
    IMAGE_SIZE   = 384
    NUM_CLASSES  = 5

    # ── Allowed upload extensions ──────────────────────────────────────
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

    # ── DR Grade metadata ──────────────────────────────────────────────
    GRADE_LABELS = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }

    GRADE_RISK = {
        0: "Low",
        1: "Low–Moderate",
        2: "Moderate",
        3: "High",
        4: "Critical",
    }

    GRADE_RECOMMENDATIONS = {
        0: "No signs of diabetic retinopathy detected. Routine annual screening recommended.",
        1: "Mild non-proliferative changes observed. Consult an ophthalmologist within 12 months.",
        2: "Moderate non-proliferative changes detected. Ophthalmologist follow-up within 6 months.",
        3: "Severe non-proliferative changes detected. Urgent referral within 1 month required.",
        4: "Proliferative diabetic retinopathy detected. Immediate specialist consultation required.",
    }

    GRADE_COLORS = {
        0: "#22c55e",
        1: "#84cc16",
        2: "#f59e0b",
        3: "#f97316",
        4: "#ef4444",
    }

    GRADE_ICONS = {
        0: "✓",
        1: "◎",
        2: "⚠",
        3: "⚠⚠",
        4: "✕",
    }
