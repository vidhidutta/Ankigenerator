from __future__ import annotations

import os
import uuid
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .types import Region
from .ocr_provider import PaddleOCRProvider


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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


def _polygons_to_mask(polygons: List[List[int]], height: int, width: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for flat in polygons:
        if len(flat) >= 6:
            pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
            draw.polygon(pts, fill=255)
    return np.array(mask) > 0


def _apply_fill_mask(image: Image.Image, mask: np.ndarray, fill_color=(0, 0, 0), blur: bool = False) -> Image.Image:
    base = image.convert("RGBA")
    if blur:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=8)).convert("RGBA")
        alpha = Image.fromarray((mask * 255).astype(np.uint8))
        out = Image.composite(blurred, base, alpha)
        return out
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ys, xs = np.where(mask)
    if xs.size and ys.size:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        ov_draw.rectangle([x1, y1, x2, y2], fill=(*fill_color, 220))
    return Image.alpha_composite(base, overlay)


def _draw_outline(image: Image.Image, polygons: List[List[int]] | None, bbox_xyxy: Tuple[int, int, int, int]) -> Image.Image:
    out = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(out)
    if polygons:
        for flat in polygons:
            if len(flat) >= 6:
                pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
                draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=3)
    else:
        x1, y1, x2, y2 = bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
    return out


def _prefer_answer_text(img: Image.Image, region: Region, ocr_words: Optional[List[Dict[str, Any]]] = None) -> str:
    # If OCR available, compute IoU with OCR boxes and pick text if >= 0.5 IoU
    if not ocr_words:
        try:
            if PaddleOCRProvider.available() or PaddleOCRProvider.tesseract_available():
                ocr = PaddleOCRProvider()
                res = ocr.recognize(img, use_preprocess=True)
                ocr_words = [{"text": w.text, "bbox": w.bbox_xyxy} for w in res.words]
        except Exception:
            ocr_words = []
    best = None
    best_iou = 0.0
    for w in (ocr_words or []):
        iou = _bbox_iou(tuple(region.bbox_xyxy), tuple(w.get("bbox")))
        if iou >= 0.5 and iou > best_iou:
            best = w.get("text", "").strip()
            best_iou = iou
    return (best or (region.short_label or region.term or "")).strip()


def build_occlusion_items_for_image(
    image_path: str,
    regions: List[Region],
    output_dir: str = "out_cards",
    max_masks_per_image: int = 6,
    overlap_iou_threshold: float = 0.4,
    mask_style: str = "fill",  # or "blur"
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")

    kept: List[Region] = []
    skipped: List[Tuple[str, int]] = []
    for idx, reg in enumerate(regions):
        if len(kept) >= max_masks_per_image:
            break
        if any(_bbox_iou(reg.bbox_xyxy, k.bbox_xyxy) > overlap_iou_threshold for k in kept):
            skipped.append(("overlap>max_iou", idx))
            continue
        if reg.area_px <= 0:
            skipped.append(("too_small", idx))
            continue
        kept.append(reg)

    entries: List[dict] = []
    # OCR cache per image to avoid repeated calls
    ocr_cache: Optional[List[Dict[str, Any]]] = None
    for idx, reg in enumerate(kept):
        if reg.polygon:
            mask = _polygons_to_mask(reg.polygon, img.height, img.width)
        else:
            mask = np.zeros((img.height, img.width), dtype=bool)
            x1, y1, x2, y2 = reg.bbox_xyxy
            mask[y1:y2, x1:x2] = True
        front_img = _apply_fill_mask(img, mask, fill_color=(0, 0, 0), blur=(mask_style == "blur"))
        back_img = _draw_outline(img, reg.polygon, reg.bbox_xyxy)

        base = os.path.splitext(os.path.basename(image_path))[0]
        q_path = os.path.join(output_dir, f"{base}_occ_{idx}.png")
        a_path = os.path.join(output_dir, f"{base}_orig_{idx}.png")
        front_img.save(q_path)
        back_img.save(a_path)

        # Prefer OCR text if overlaps >= 0.5
        if ocr_cache is None:
            try:
                if PaddleOCRProvider.available() or PaddleOCRProvider.tesseract_available():
                    res = PaddleOCRProvider().recognize(img, use_preprocess=True)
                    ocr_cache = [{"text": w.text, "bbox": w.bbox_xyxy} for w in res.words]
                else:
                    ocr_cache = []
            except Exception:
                ocr_cache = []
        answer_text = _prefer_answer_text(img, reg, ocr_words=ocr_cache)
        rationale = reg.rationale or ""
        entries.append(
            {
                "type": "image_occlusion",
                "question_image_path": q_path,
                "answer_image_path": a_path,
                "alt_text": "Identify the covered structure",
                "answer_text": answer_text,
                "rationale": rationale,
                "_stats": {
                    "kept_mask_count": len(kept),
                    "skipped": skipped,
                    "mask_style": mask_style,
                    "max_overlap_iou": overlap_iou_threshold,
                },
            }
        )
    return entries 