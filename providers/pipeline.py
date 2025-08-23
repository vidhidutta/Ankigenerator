from __future__ import annotations

import os
from typing import List, Optional

from PIL import Image

from .types import Region, DetectedRegion
from .detect_provider import GroundingDINOProvider
from .segment_provider import SAMProvider
from .ocr_provider import PaddleOCRProvider
from .candidate_terms import CandidateTermGenerator
from .utils import resize_for_processing
from providers.vlm_provider import LocalQwen2VLProvider, LocalLLaVAOneVisionProvider, CloudVLMProvider


def _get_vlm():
    if LocalQwen2VLProvider.available():
        return LocalQwen2VLProvider()
    if LocalLLaVAOneVisionProvider.available():
        return LocalLLaVAOneVisionProvider()
    if CloudVLMProvider.available():
        return CloudVLMProvider()
    return None


def detect_segment_rank(
    image_path: str,
    slide_text: str = "",
    transcript_text: str = "",
    max_masks_per_image: int = 6,
    min_mask_area_px: int = 900,
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
) -> List[Region]:
    """Run detection→segmentation→ranking with fallbacks to OCR words.
    Returns a list of Regions ready for occlusion building.
    """
    image = Image.open(image_path).convert("RGB")

    # Candidate terms from OCR + text
    ocr_words: List[str] = []
    ocr_boxes: List[tuple[int, int, int, int]] = []
    regions: List[Region] = []

    # OCR (best-effort)
    if PaddleOCRProvider.available():
        try:
            ocr = PaddleOCRProvider()
            ocr_res = ocr.recognize(image, use_preprocess=True)
            for w in ocr_res.words:
                ocr_words.append(w.text)
                x1, y1, x2, y2 = w.bbox_xyxy
                ocr_boxes.append((x1, y1, x2, y2))
        except Exception:
            pass
    # Build candidate terms
    gen = CandidateTermGenerator()
    terms = gen.generate(
        # convert to OCRWord-compatible list is not necessary; we only need texts
        ocr_words=[type("W", (), {"text": t, "bbox_xyxy": (0, 0, 0, 0)}) for t in ocr_words],
        slide_text_bullets=[slide_text],
        transcript_window_texts=[transcript_text],
        top_k=30,
    ).terms

    # Detection
    dets: List[DetectedRegion] = []
    if GroundingDINOProvider.available() and terms:
        try:
            gd = GroundingDINOProvider(device="cpu")
            dets = gd.detect_terms(image=image, terms=terms, detection_threshold=detection_threshold, nms_iou_threshold=nms_iou_threshold, max_boxes=max_masks_per_image)
        except Exception:
            dets = []

    # Fallback: if no detection, use OCR boxes with top terms or VLM suggested labels
    if not dets:
        # try VLM to suggest labels
        vlm = _get_vlm()
        proposed = []
        if vlm:
            try:
                proposed = vlm.suggest_key_structures(image, k=6)
            except Exception:
                proposed = []
        labels = proposed or terms or ["structure"]
        # Map first few OCR boxes to labels in round-robin
        for i, b in enumerate(ocr_boxes[:max_masks_per_image]):
            term = labels[i % len(labels)].strip() if labels else "structure"
            dets.append(DetectedRegion(term=term, score=0.5, bbox_xyxy=b))

    # Segmentation (best-effort); if unavailable, return bbox-based regions
    if dets:
        if SAMProvider.available():
            try:
                sam = SAMProvider(device="cpu")
                boxes = [d.bbox_xyxy for d in dets]
                masks = sam.segment(image, boxes)
                regions = sam.to_regions(dets, masks, min_mask_area_px=min_mask_area_px, merge_iou_threshold=0.8)
            except Exception:
                regions = [Region(term=d.term, score=d.score, bbox_xyxy=d.bbox_xyxy, mask_rle=None, polygon=None, area_px=(d.bbox_xyxy[2]-d.bbox_xyxy[0])*(d.bbox_xyxy[3]-d.bbox_xyxy[1])) for d in dets]
        else:
            regions = [Region(term=d.term, score=d.score, bbox_xyxy=d.bbox_xyxy, mask_rle=None, polygon=None, area_px=(d.bbox_xyxy[2]-d.bbox_xyxy[0])*(d.bbox_xyxy[3]-d.bbox_xyxy[1])) for d in dets]

    # Rank via VLM (optional)
    vlm = _get_vlm()
    if vlm and regions:
        try:
            k = min(max_masks_per_image, len(regions))
            regions = vlm.rank_regions(image, regions, slide_text=slide_text, transcript_text=transcript_text, top_k=k)
        except Exception:
            pass
    return regions[:max_masks_per_image] 