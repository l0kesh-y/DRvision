"""
ml/heatmap.py – Clinically-meaningful lesion attention heatmap generator.

Uses OpenCV multi-channel analysis to highlight:
  • Bright regions  → hard exudates / cotton-wool spots
  • Dark green spots → haemorrhages / microaneurysms
  • Red-dominant areas → haemorrhagic regions
  • Vascular abnormalities → enhanced vessel contrast

The intensity of highlighted regions scales with the predicted DR grade
so the heatmap is visually calibrated to clinical severity.
"""

import cv2
import numpy as np
import os
import uuid
import logging
from config import Config

logger = logging.getLogger(__name__)


def _build_circular_mask(h: int, w: int, shrink: float = 0.46) -> np.ndarray:
    """Return a float32 circular mask covering the retinal disc area."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * shrink)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    return mask


def generate_heatmap(image_path: str, grade: int, filename_prefix: str = "") -> str:
    """
    Generate a Grad-CAM-style lesion attention heatmap and save to disk.

    The heatmap is produced by:
      1. Brightness map  (L-channel in LAB)  → exudate candidates
      2. Green-channel inverse               → haemorrhage/MA candidates
      3. Red dominance map                   → haemorrhagic regions
      4. Vessel contrast                     → vascular abnormalities
      5. Weighted combination scaled by grade
      6. Gaussian smoothing + circular retinal mask
      7. JET colourmap overlay on the original image

    Args:
        image_path:      Path to the preprocessed or raw fundus image.
        grade:           Predicted DR grade (0–4).
        filename_prefix: Optional prefix for the saved filename.

    Returns:
        Filename of the saved heatmap (relative to HEATMAP_FOLDER).
    """
    SIZE = 512  # working resolution

    # ── Load and resize ───────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image for heatmap: {image_path}")

    img_bgr = cv2.resize(img_bgr, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_bgr.shape[:2]

    # ── Feature maps ──────────────────────────────────────────────────
    # 1. LAB L-channel → brightness (bright = exudates)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    brightness = lab[:, :, 0].astype(np.float32) / 255.0

    # 2. Green channel inverse → dark regions = haemorrhages/MAs
    green = img_rgb[:, :, 1].astype(np.float32) / 255.0
    hemorrhage_map = 1.0 - green

    # 3. Red dominance → R >> G in haemorrhagic regions
    red = img_rgb[:, :, 0].astype(np.float32) / 255.0
    blue = img_rgb[:, :, 2].astype(np.float32) / 255.0
    red_dominance = np.clip(red - green - blue * 0.3, 0, 1)

    # 4. Vessel contrast (green channel has best vessel visibility)
    #    High-contrast green areas = vessels / vascular abnormalities
    green_eq = cv2.equalizeHist(img_bgr[:, :, 1]).astype(np.float32) / 255.0
    vessel_map = np.abs(green_eq - green)

    # 5. Remove optic disc (very bright saturated circle → not a lesion)
    _, disc_mask_raw = cv2.threshold(
        lab[:, :, 0], 220, 255, cv2.THRESH_BINARY
    )
    disc_exclude = cv2.GaussianBlur(
        disc_mask_raw.astype(np.float32), (61, 61), 0
    ) / 255.0

    # ── Grade-weighted combination ────────────────────────────────────
    # Grade 0 → very low attention; Grade 4 → maximum attention
    severity = grade / 4.0  # 0.0 – 1.0

    # Weights that emphasise different lesion types per severity
    w_exudate    = 0.30 + severity * 0.25   # exudates more prominent at higher grades
    w_hemorrhage = 0.25 + severity * 0.30
    w_red_dom    = 0.20 + severity * 0.25
    w_vessel     = 0.10 + severity * 0.10

    attention = (
        brightness    * w_exudate
        + hemorrhage_map * w_hemorrhage
        + red_dominance  * w_red_dom
        + vessel_map     * w_vessel
    )

    # Suppress optic disc region
    attention = attention * (1.0 - disc_exclude * 0.80)

    # ── Incorporate authentic DenseNet-121 + CBAM Grad-CAM ─────────────
    try:
        from ml.densenet_model import compute_model_gradcam
        model_cam = compute_model_gradcam(image_path, target_size=SIZE)
        if model_cam is not None:
            # Blend true deep learning Grad-CAM with localized lesion features
            attention = 0.65 * model_cam + 0.35 * attention
            logger.info("Integrated authentic DenseNet-121 CBAM Grad-CAM into heatmap.")
    except Exception as dme:
        logger.warning(f"DenseNet Grad-CAM fallback: {dme}")

    # ── Circular retinal mask ─────────────────────────────────────────
    circle_mask = _build_circular_mask(h, w, shrink=0.46)
    attention = attention * circle_mask

    # ── Smooth and normalise ──────────────────────────────────────────
    attention = cv2.GaussianBlur(attention.astype(np.float32), (45, 45), 0)
    a_min, a_max = attention.min(), attention.max()
    if a_max > a_min:
        attention = (attention - a_min) / (a_max - a_min)
    else:
        attention = np.zeros_like(attention)

    # For Grade 0, cap maximum so the image doesn't look alarming
    if grade == 0:
        attention = attention * 0.35

    # ── Colourmap overlay ─────────────────────────────────────────────
    heatmap_uint8 = (attention * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: original image 50% + heatmap 50%
    alpha = 0.45 + severity * 0.10  # slightly more heatmap at higher grades
    overlay_rgb = cv2.addWeighted(
        img_rgb.astype(np.uint8), 1.0 - alpha,
        heatmap_rgb.astype(np.uint8), alpha, 0
    )

    # ── Save ──────────────────────────────────────────────────────────
    uid = filename_prefix or uuid.uuid4().hex[:12]
    heatmap_filename = f"heatmap_{uid}.jpg"
    save_path = os.path.join(Config.HEATMAP_FOLDER, heatmap_filename)

    os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
    result_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    logger.info(f"Heatmap saved: {save_path}")
    return heatmap_filename
