from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import importlib.util
import os

import numpy as np
from PIL import Image

from .types import Region, DetectedRegion
from .utils import compute_image_hash, resize_for_processing, run_with_timeout, disk_cache_read, disk_cache_write


@dataclass
class SegmentationMask:
    mask: np.ndarray  # boolean HxW
    score: float
    bbox_xyxy: Tuple[int, int, int, int]


# Cache: (img_hash, boxes_key, thresholds) -> List[SegmentationMask]
_SEG_CACHE: dict[tuple[str, str, str], List['SegmentationMask']] = {}


class SAMProvider:
    """Segmentation provider that prefers SAM2 if available; otherwise uses SAM (segment-anything).

    Interface accepts optional detection boxes to guide segmentation.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._sam2_predictor = None
        self._sam_predictor = None

    @staticmethod
    def _sam_checkpoint() -> Optional[str]:
        return (
            os.getenv("SAM_CHECKPOINT")
            or os.getenv("SAM_VIT_H_CHECKPOINT")
            or os.getenv("SAM_VIT_B_CHECKPOINT")
        )

    @classmethod
    def available(cls) -> bool:
        has_sam2 = importlib.util.find_spec("sam2") is not None
        has_sam = importlib.util.find_spec("segment_anything") is not None
        if has_sam2:
            return True
        if has_sam and cls._sam_checkpoint():
            return True
        return False

    def _ensure_loaded(self) -> None:
        if self._sam2_predictor is None and importlib.util.find_spec("sam2") is not None:
            try:
                from sam2.build_sam import build_sam2 as build_sam  # type: ignore
                from sam2.sam2_image_predictor import SAM2ImagePredictor as ImagePredictor  # type: ignore
                sam = build_sam(device=self.device)
                self._sam2_predictor = ImagePredictor(sam)
                return
            except Exception:
                self._sam2_predictor = None
        if self._sam_predictor is None and importlib.util.find_spec("segment_anything") is not None:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

            ckpt = self._sam_checkpoint()
            if not ckpt or not os.path.exists(ckpt):
                raise RuntimeError(
                    "Segment-Anything checkpoint not found. Set SAM_CHECKPOINT or SAM_VIT_H_CHECKPOINT/SAM_VIT_B_CHECKPOINT."
                )
            variant = "vit_h" if "vit_h" in os.path.basename(ckpt).lower() else "vit_b"
            sam = sam_model_registry[variant](checkpoint=ckpt)
            self._sam_predictor = SamPredictor(sam)

    def segment(
        self,
        image: Image.Image,
        boxes_xyxy: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        timeout_s: float = 20.0,
    ) -> List[SegmentationMask]:
        self._ensure_loaded()
        resized, sx, sy = resize_for_processing(image)
        img_hash = compute_image_hash(resized)
        boxes_key = "|".join([f"{b[0]}-{b[1]}-{b[2]}-{b[3]}" for b in (boxes_xyxy or [])])
        th_key = f"{timeout_s:.1f}"
        cache_key = (img_hash, boxes_key, th_key)
        if cache_key in _SEG_CACHE:
            return _SEG_CACHE[cache_key]
        disk_key = f"seg:{cache_key}"
        disk_val = disk_cache_read(disk_key)
        if isinstance(disk_val, list):
            _SEG_CACHE[cache_key] = disk_val
            return disk_val

        def _do_segment() -> List[SegmentationMask]:
            np_img = np.array(resized.convert("RGB"))

            if self._sam2_predictor is not None:
                pred = self._sam2_predictor
                pred.set_image(np_img)
                masks: List[SegmentationMask] = []
                if boxes_xyxy:
                    for box in boxes_xyxy:
                        # scale box to resized coords
                        rx1, ry1, rx2, ry2 = int(box[0] / sx), int(box[1] / sy), int(box[2] / sx), int(box[3] / sy)
                        m, scores, _ = pred.predict(box=np.array((rx1, ry1, rx2, ry2)))
                        if m is None:
                            continue
                        if m.ndim == 3:
                            idx = int(np.argmax(scores))
                            mask = m[idx]
                            score = float(scores[idx])
                        else:
                            mask = m
                            score = float(scores if np.isscalar(scores) else np.max(scores))
                        masks.append(
                            SegmentationMask(mask=mask.astype(bool), score=score, bbox_xyxy=box)
                        )
                else:
                    return []
                return masks

            if self._sam_predictor is not None:
                pred = self._sam_predictor
                pred.set_image(np_img)
                masks: List[SegmentationMask] = []
                if boxes_xyxy:
                    for box in boxes_xyxy:
                        rx1, ry1, rx2, ry2 = int(box[0] / sx), int(box[1] / sy), int(box[2] / sx), int(box[3] / sy)
                        masks_np, scores, _ = pred.predict(box=np.array((rx1, ry1, rx2, ry2)), point_coords=None, point_labels=None)
                        if masks_np is None:
                            continue
                        if masks_np.ndim == 3:
                            idx = int(np.argmax(scores))
                            mask = masks_np[idx]
                            score = float(scores[idx])
                        else:
                            mask = masks_np
                            score = float(scores if np.isscalar(scores) else np.max(scores))
                        masks.append(
                            SegmentationMask(mask=mask.astype(bool), score=score, bbox_xyxy=box)
                        )
                else:
                    return []
                return masks

            raise RuntimeError("No SAM or SAM2 implementation available.")

        masks = run_with_timeout(_do_segment, timeout_sec=timeout_s, default=[])
        _SEG_CACHE[cache_key] = masks
        disk_cache_write(disk_key, masks)
        return masks

    def to_regions(
        self,
        detected: List[DetectedRegion],
        masks: List[SegmentationMask],
        min_mask_area_px: int = 900,
        merge_iou_threshold: float = 0.8,
    ) -> List[Region]:
        regions: List[Region] = []
        for det, m in zip(detected, masks):
            area = int(m.mask.astype(np.uint8).sum())
            if area < min_mask_area_px:
                continue
            rle = mask_to_rle(m.mask)
            poly = mask_to_polygon(m.mask)
            regions.append(
                Region(
                    term=det.term,
                    score=det.score,
                    bbox_xyxy=det.bbox_xyxy,
                    mask_rle=rle,
                    polygon=poly,
                    area_px=area,
                )
            )
        merged: List[Region] = []
        for reg in sorted(regions, key=lambda r: r.score, reverse=True):
            if all(mask_iou(reg, k) < merge_iou_threshold for k in merged):
                merged.append(reg)
        return merged

    def clear(self) -> None:
        self._sam2_predictor = None
        self._sam_predictor = None


def mask_to_rle(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = mask.flatten(order="F")
    rle = []
    count = 0
    last = 0
    for val in flat:
        if val != last:
            rle.append(count)
            count = 1
            last = val
        else:
            count += 1
    rle.append(count)
    return {"size": [h, w], "counts": rle}


def mask_to_polygon(mask: np.ndarray) -> Optional[List[List[int]]]:
    try:
        import cv2

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: List[List[int]] = []
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            pts = cnt.reshape(-1, 2).tolist()
            flat = [int(x) for pt in pts for x in pt]
            if flat:
                polys.append(flat)
        return polys or None
    except Exception:
        return None


def mask_iou(a: Region, b: Region) -> float:
    if a.polygon is None or b.polygon is None:
        ax1, ay1, ax2, ay2 = a.bbox_xyxy
        bx1, by1, bx2, by2 = b.bbox_xyxy
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
    try:
        from shapely.geometry import Polygon

        def to_poly(poly_list: List[int]) -> Polygon:
            pts = [(poly_list[i], poly_list[i + 1]) for i in range(0, len(poly_list), 2)]
            return Polygon(pts)

        polys_a = [to_poly(p) for p in a.polygon]
        polys_b = [to_poly(p) for p in b.polygon]
        poly_a = polys_a[0]
        for p in polys_a[1:]:
            poly_a = poly_a.union(p)
        poly_b = polys_b[0]
        for p in polys_b[1:]:
            poly_b = poly_b.union(p)
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return float(inter / max(union, 1e-6))
    except Exception:
        return 0.0 