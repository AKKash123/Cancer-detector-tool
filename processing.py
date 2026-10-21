"""OpenCV pipeline: enhance an oral photo and flag red / white suspicious patches."""
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class Params:
    clahe_clip: float = 2.0
    denoise: bool = True
    # Red patches (erythroplakia-like): redness (LAB a-channel) above the image's typical tissue
    red_delta: int = 18
    red_val_min: int = 50
    # White patches (leukoplakia-like): low saturation, bright
    white_sat_max: int = 45
    white_val_min: int = 170
    white_val_max: int = 245          # excludes specular glare above this
    kernel: int = 5
    min_area_pct: float = 0.3         # ignore regions smaller than this % of image


@dataclass
class Result:
    original: np.ndarray
    enhanced: np.ndarray
    red_mask: np.ndarray
    white_mask: np.ndarray
    annotated: np.ndarray
    regions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def decode_image(data: bytes, max_side: int = 720) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return resize_max(img, max_side)


def enhance(img: np.ndarray, p: Params) -> np.ndarray:
    out = cv2.bilateralFilter(img, 7, 40, 40) if p.denoise else img.copy()
    l, a, b = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _clean(mask: np.ndarray, k: int) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)


def _regions(mask, label, color, img_area, min_area_pct, canvas):
    found = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area / img_area * 100 < min_area_pct:
            continue
        perim = cv2.arcLength(c, True)
        hull_area = cv2.contourArea(cv2.convexHull(c)) or 1
        circ = float(np.clip(4 * np.pi * area / (perim ** 2 + 1e-6), 0, 1))
        solidity = area / hull_area
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(canvas, [c], -1, color, 2)
        cv2.putText(canvas, f"{label[0]}{len(found) + 1}", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        found.append({"type": label, "area_%": round(area / img_area * 100, 2),
                      "circularity": round(circ, 2), "solidity": round(solidity, 2),
                      "irregularity": round(1 - (circ + solidity) / 2, 2),
                      "bbox": (x, y, w, h)})
    return found


def analyze(img: np.ndarray, p: Params) -> Result:
    enh = enhance(img, p)
    hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    a_ch = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16)
    usable = v >= p.red_val_min
    baseline = np.median(a_ch[usable]) if usable.any() else np.median(a_ch)
    red = (a_ch > baseline + p.red_delta) & usable
    white = ((s <= p.white_sat_max) & (v >= p.white_val_min) & (v <= p.white_val_max))
    red_m = _clean(red.astype(np.uint8) * 255, p.kernel)
    white_m = _clean(white.astype(np.uint8) * 255, p.kernel)

    ann = img.copy()
    area = img.shape[0] * img.shape[1]
    regions = (_regions(red_m, "Red", (0, 0, 255), area, p.min_area_pct, ann)
               + _regions(white_m, "White", (255, 160, 0), area, p.min_area_pct, ann))

    red_pct = float((red_m > 0).sum() / area * 100)
    white_pct = float((white_m > 0).sum() / area * 100)
    irr = float(np.mean([r["irregularity"] for r in regions])) if regions else 0.0
    # Heuristic attention score (0-100): coverage, redness and border irregularity
    score = min(100.0, red_pct * 2.0 + white_pct * 1.0 + irr * 40 + len(regions) * 2)
    level = "Low" if score < 25 else "Moderate" if score < 55 else "High"
    stats = {"red_pct": round(red_pct, 2), "white_pct": round(white_pct, 2),
             "n_regions": len(regions), "mean_irregularity": round(irr, 2),
             "score": round(score, 1), "level": level}
    return Result(img, enh, red_m, white_m, ann, regions, stats)