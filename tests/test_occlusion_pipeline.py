import os
import pytest
from PIL import Image, ImageDraw

from providers.types import Region
from providers.occlusion_pipeline import build_occlusion_items_for_image
from flashcard_generator import export_flashcards_to_apkg


def _make_simple(path: str):
    img = Image.new("RGB", (300, 200), color="white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "mitral valve", fill="black")
    d.rectangle([100, 60, 180, 140], outline="black", width=2)
    img.save(path)


def test_build_and_export_occlusion_items(tmp_path):
    img_path = str(tmp_path / "simple.png")
    _make_simple(img_path)

    regions = [
        Region(term="mitral valve", score=0.9, bbox_xyxy=(100, 60, 180, 140), mask_rle=None, polygon=None, area_px=6400, importance_score=0.9, short_label="Mitral valve", rationale="Clinically important valve"),
    ]

    items = build_occlusion_items_for_image(img_path, regions, output_dir=str(tmp_path), max_masks_per_image=3)
    assert len(items) == 1
    assert os.path.exists(items[0]["question_image_path"]) and os.path.exists(items[0]["answer_image_path"]) 

    # Try exporting to a tiny test deck
    apkg_path = str(tmp_path / "occlusion_test.apkg")
    export_flashcards_to_apkg(items, output_path=apkg_path)
    assert os.path.exists(apkg_path) 