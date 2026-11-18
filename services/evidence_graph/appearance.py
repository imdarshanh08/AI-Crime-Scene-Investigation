"""Lightweight appearance features from detection crops (OpenCV only, no extra models).

Used for two things:
* colour-histogram similarity between two appearances, as *one* signal in
  cross-camera association (never as proof of identity);
* an approximate dominant colour, so a question such as "red backpack" can be
  answered with an explicit caveat instead of being ignored.

Both are sensitive to lighting, camera white balance, compression and occlusion.
Callers must present them as approximate.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

COLOUR_NAMES = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
                "black", "white", "grey", "brown")


def _raw_frame_path(evidence_path: str) -> str:
    """Prefer the unannotated keyframe (no drawn boxes/labels) when it exists."""
    p = Path(evidence_path)
    raw = p.with_name(p.stem + "_raw" + p.suffix)
    return str(raw) if raw.is_file() else str(p)


def parse_bbox(bbox_str) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(bbox_str, str) or not bbox_str.strip():
        return None
    try:
        coords = ast.literal_eval(bbox_str)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in coords)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


@lru_cache(maxsize=2048)
def _load_crop(evidence_path: str, bbox_str: str, _mtime: float) -> Optional[np.ndarray]:
    path = _raw_frame_path(evidence_path)
    img = cv2.imread(path)
    if img is None:
        return None
    box = parse_bbox(bbox_str)
    if box is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Inset the box so drawn borders / label text on annotated frames are excluded,
    # and so background at the box edge contributes less.
    top_inset = 0.18 if path == evidence_path else 0.08
    x1 = max(0, x1 + int(bw * 0.08))
    x2 = min(w, x2 - int(bw * 0.08))
    y1 = max(0, y1 + int(bh * top_inset))
    y2 = min(h, y2 - int(bh * 0.08))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2].copy()


def load_crop(evidence_path, bbox_str) -> Optional[np.ndarray]:
    if not evidence_path or not isinstance(evidence_path, str) or not Path(evidence_path).is_file():
        return None
    try:
        mtime = Path(evidence_path).stat().st_mtime
    except OSError:
        return None
    return _load_crop(evidence_path, bbox_str if isinstance(bbox_str, str) else "", mtime)


def colour_histogram(crop: np.ndarray) -> np.ndarray:
    """Normalised 2-D hue/saturation histogram (lighting-tolerant compared with RGB)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def histogram_similarity(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """1.0 = identical colour distribution, 0.0 = disjoint (Bhattacharyya-based)."""
    dist = cv2.compareHist(colour_histogram(crop_a), colour_histogram(crop_b), cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, min(1.0, 1.0 - dist)))


def dominant_colour(crop: np.ndarray) -> Tuple[str, float]:
    """Return (colour name, share of pixels) using coarse HSV bins."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    names = np.empty(len(hsv), dtype=object)
    achromatic = s < 45
    names[achromatic & (v < 60)] = "black"
    names[achromatic & (v >= 60) & (v < 190)] = "grey"
    names[achromatic & (v >= 190)] = "white"
    chroma = ~achromatic
    dark = chroma & (v < 50)
    names[dark] = "black"
    c = chroma & ~dark
    hue_bins = [
        (0, 8, "red"), (8, 20, "orange"), (20, 33, "yellow"), (33, 80, "green"),
        (80, 95, "cyan"), (95, 130, "blue"), (130, 150, "purple"), (150, 170, "pink"), (170, 181, "red"),
    ]
    for lo, hi, name in hue_bins:
        names[c & (h >= lo) & (h < hi)] = name
    # Low-value oranges read as brown.
    names[c & (h >= 8) & (h < 20) & (v < 150)] = "brown"
    values, counts = np.unique(names.astype(str), return_counts=True)
    idx = int(np.argmax(counts))
    return str(values[idx]), float(counts[idx] / counts.sum())
