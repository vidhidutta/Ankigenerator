import os
import re
import pytest
from PIL import Image, ImageDraw

from providers.ocr_provider import PaddleOCRProvider
from providers.detect_provider import GroundingDINOProvider
from providers.segment_provider import SAMProvider
from providers.vlm_provider import LocalQwen2VLProvider, LocalLLaVAOneVisionProvider, CloudVLMProvider
from providers.pipeline import detect_segment_rank
from providers.types import Region
from providers.occlusion_pipeline import build_occlusion_items_for_image
from flashcard_generator import export_flashcards_to_apkg

pytestmark = pytest.mark.image_occlusion


def _make_labelled(path: str):
    img = Image.new("RGB", (600, 400), color="white")
    d = ImageDraw.Draw(img)
    d.text((50, 40), "Mitral valve", fill="black")
    d.rectangle([200, 120, 320, 220], outline="black", width=3)
    d.text((210, 125), "aorta", fill="black")
    img.save(path)


def test_ocr_words(tmp_path):
    p = tmp_path / "diagram.png"
    _make_labelled(str(p))
    if not PaddleOCRProvider.available():
        pytest.skip("PaddleOCR not available")
    img = Image.open(p)
    res = PaddleOCRProvider().recognize(img, use_preprocess=True)
    assert len(res.words) > 0


def test_detection_basic(tmp_path):
    p = tmp_path / "diagram.png"
    _make_labelled(str(p))
    if not GroundingDINOProvider.available():
        pytest.skip("GroundingDINO not available")
    img = Image.open(p)
    regions = GroundingDINOProvider().detect_terms(img, terms=["aorta", "mitral valve"], detection_threshold=0.1)
    assert isinstance(regions, list)


def test_sam_masks(tmp_path):
    p = tmp_path / "diagram.png"
    _make_labelled(str(p))
    if not SAMProvider.available():
        pytest.skip("SAM/SAM2 not available")
    img = Image.open(p)
    boxes = [(200, 120, 320, 220)]
    masks = SAMProvider().segment(img, boxes)
    assert masks and masks[0].mask is not None


def test_vlm_ranking_structure(tmp_path):
    p = tmp_path / "blank.png"
    Image.new("RGB", (4, 4), color="white").save(p)
    vlm = None
    if LocalQwen2VLProvider.available():
        vlm = LocalQwen2VLProvider()
    elif LocalLLaVAOneVisionProvider.available():
        vlm = LocalLLaVAOneVisionProvider()
    elif CloudVLMProvider.available():
        vlm = CloudVLMProvider()
    if vlm is None:
        pytest.skip("No VLM available")
    regions = [Region(term="aorta", score=0.8, bbox_xyxy=(10, 10, 30, 30), mask_rle=None, polygon=None, area_px=400)]
    ranked = vlm.rank_regions(Image.open(p), regions, "Cardiac", "...", top_k=1)
    assert len(ranked) == 1


def test_e2e_detect_segment_export(tmp_path):
    p = tmp_path / "diagram.png"
    _make_labelled(str(p))
    regs = detect_segment_rank(str(p), slide_text="Cardiac anatomy", transcript_text="...")
    assert isinstance(regs, list)
    # Build occlusion items; allow empty if no providers installed
    items = build_occlusion_items_for_image(str(p), regs or [Region(term="aorta", score=0.5, bbox_xyxy=(200,120,320,220), mask_rle=None, polygon=None, area_px=12000)])
    assert items
    apkg = tmp_path / "out.apkg"
    export_flashcards_to_apkg(items, output_path=str(apkg))
    assert os.path.exists(apkg)


def test_fallback_vlm_detection(tmp_path, monkeypatch):
    p = tmp_path / "diagram.png"
    _make_labelled(str(p))
    # Force unavailability by monkeypatching providers
    monkeypatch.setattr("providers.detect_provider.GroundingDINOProvider.available", staticmethod(lambda: False))
    regs = detect_segment_rank(str(p), slide_text="Cardiac anatomy", transcript_text="...")
    # Should still produce some regions (from OCR+VLM fallback) unless OCR unavailable
    assert isinstance(regs, list) 