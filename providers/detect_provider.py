from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import importlib.util

import numpy as np
from PIL import Image

from .types import DetectedRegion
from .utils import compute_image_hash, resize_for_processing, run_with_timeout, disk_cache_read, disk_cache_write

# Cache: (img_hash, terms_hash, thresholds) -> List[DetectedRegion]
_DET_CACHE: dict[tuple[str, str, str], List[DetectedRegion]] = {}


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = max(1e-6, area_a + area_b - inter)
    return inter / union


def _nms(regions: List[DetectedRegion], iou_thresh: float) -> List[DetectedRegion]:
    regions_sorted = sorted(regions, key=lambda r: r.score, reverse=True)
    kept: List[DetectedRegion] = []
    for reg in regions_sorted:
        if all(_iou_xyxy(reg.bbox_xyxy, k.bbox_xyxy) < iou_thresh for k in kept):
            kept.append(reg)
    return kept


@dataclass
class DetectionResult:
    label: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]


class GroundingDINOProvider:
    """Detection provider using GroundingDINO."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    @staticmethod
    def available() -> bool:
        # Check if groundingdino is available
        try:
            from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO as GDINO
            except Exception:
                raise ImportError("GroundingDINO not available")
            self._model = GDINO(device=self.device)

    def detect_terms(
        self,
        image: Image.Image,
        terms: Sequence[str],
        detection_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_boxes: int = 20,
        timeout_s: float = 20.0,
    ) -> List[DetectedRegion]:
        self._ensure_loaded()
        assert self._model is not None

        # Resize for processing and build cache key
        resized, sx, sy = resize_for_processing(image)
        img_hash = compute_image_hash(resized)
        terms_key = "|".join(sorted(set([t.strip().lower() for t in terms if t.strip()])))
        th_key = f"{detection_threshold:.3f}-{nms_iou_threshold:.3f}-{max_boxes}"
        cache_key = (img_hash, terms_key, th_key)
        if cache_key in _DET_CACHE:
            return _DET_CACHE[cache_key]
        disk_key = f"det:{cache_key}"
        disk_val = disk_cache_read(disk_key)
        if isinstance(disk_val, list):
            _DET_CACHE[cache_key] = disk_val
            return disk_val

        def _do_detect() -> List[DetectedRegion]:
            np_img = np.array(resized.convert("RGB"))
            regions: List[DetectedRegion] = []
            caption = ", ".join(terms)
            outputs = self._model.predict(
                image=np_img,
                caption=caption,
                box_threshold=detection_threshold,
                text_threshold=detection_threshold,
                iou_threshold=nms_iou_threshold,
            )
            for out in outputs:
                score = float(out.get("score", 0.0))
                if score < detection_threshold:
                    continue
                x1, y1, x2, y2 = int(out["x1"] * sx), int(out["y1"] * sy), int(out["x2"] * sx), int(out["y2"] * sy)
                label = str(out.get("label", "object")).strip()
                term = label if label else (terms[0] if terms else "object")
                regions.append(DetectedRegion(term=term, score=score, bbox_xyxy=(x1, y1, x2, y2)))
            regions = _nms(regions, iou_thresh=nms_iou_threshold)
            return regions[:max_boxes]

        regions = run_with_timeout(_do_detect, timeout_sec=timeout_s, default=[])
        _DET_CACHE[cache_key] = regions
        disk_cache_write(disk_key, regions)
        return regions


def _string_overlap_ratio(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union 