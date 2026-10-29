"""
ml/preprocessing.py – Retinal image preprocessing pipeline.

Steps:
  1. Circular mask crop  – remove black ocular margins
  2. Resize              – standardise to 384×384
  3. Ben Graham method   – Gaussian blur subtraction to flatten illumination
  4. CLAHE               – sharpen localised microvascular lesion edges
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_circular_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the retinal circle and zero-out pixels outside it.
    Handles images where the fundus circle may not be perfectly centred.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find the bounding box of all non-zero pixels
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img_bgr  # already black or blank — return as-is

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_bgr[y:y + h, x:x + w]

    # Build a circular mask for the cropped region
    ch, cw = cropped.shape[:2]
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cx, cy = cw // 2, ch // 2
    radius = int(min(ch, cw) * 0.48)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    return masked


def _ben_graham(img_bgr: np.ndarray, sigma_ratio: float = 0.3) -> np.ndarray:
    """
    Ben Graham's local Gaussian blur subtraction.
    Subtracts a blurred version of the image to flatten uneven illumination
    while preserving high-frequency vascular detail.

    Formula: output = img * α + blur(img) * (1-α) + β
    Simplified: output = img - GaussianBlur(img) + 128
    """
    h, w = img_bgr.shape[:2]
    sigma = int(min(h, w) * sigma_ratio) | 1  # must be odd
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    result = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return result


def _apply_clahe(img_bgr: np.ndarray,
                 clip_limit: float = 2.0,
                 tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the L-channel of LAB colour space.
    Improves contrast of microaneurysms and hard exudates without
    amplifying background noise globally.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str,
                     output_size: int = 384,
                     save_path: str | None = None) -> np.ndarray:
    """
    Full dual-stage preprocessing pipeline:
      circular crop → resize → Ben Graham illumination → CLAHE

    Args:
        image_path:  Path to the input fundus image.
        output_size: Target square resolution (default 384).
        save_path:   If provided, save the preprocessed image there.

    Returns:
        Preprocessed image as a BGR numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Step 1: Circular mask crop
    img = _crop_circular_mask(img)

    # Step 2: Resize to standard square
    img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

    # Step 3: Ben Graham illumination normalisation
    img = _ben_graham(img)

    # Step 4: CLAHE contrast enhancement
    img = _apply_clahe(img)

    # Clip to valid uint8 range
    img = np.clip(img, 0, 255).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, img)

    return img


def image_to_base64(image_path: str,
                    preprocess: bool = True,
                    output_size: int = 384) -> str:
    """
    Convert a fundus image to a base64-encoded JPEG string for the Groq API.

    Args:
        image_path:  Source image path.
        preprocess:  Apply the preprocessing pipeline before encoding.
        output_size: Resize target when preprocessing.

    Returns:
        Base64 string (no 'data:image/jpeg;base64,' prefix).
    """
    if preprocess:
        img_bgr = preprocess_image(image_path, output_size=output_size)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize((output_size, output_size), Image.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
