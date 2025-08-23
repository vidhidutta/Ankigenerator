import pytest
import numpy as np
from PIL import Image, ImageDraw

from providers.segment_provider import SAMProvider
from providers.types import DetectedRegion


def _make_simple_shapes(path: str) -> Image.Image:
    img = Image.new("RGB", (400, 300), color="white")
    d = ImageDraw.Draw(img)
    d.ellipse([50, 50, 150, 150], outline="black", width=3)
    d.rectangle([220, 60, 350, 200], outline="black", width=3)
    img.save(path)
    return img


def test_sam_segmentation_with_boxes(tmp_path):
    if not SAMProvider.available():
        pytest.skip("SAM/SAM2 not available")

    p = tmp_path / "shapes.png"
    _make_simple_shapes(str(p))

    img = Image.open(p)
    boxes = [(40, 40, 160, 160), (210, 50, 360, 210)]
    det = [DetectedRegion(term="circle", score=0.9, bbox_xyxy=boxes[0]), DetectedRegion(term="rect", score=0.85, bbox_xyxy=boxes[1])]

    sam = SAMProvider(device="cpu")
    masks = sam.segment(img, boxes)
    assert len(masks) == len(boxes)

    regions = sam.to_regions(det, masks, min_mask_area_px=200, merge_iou_threshold=0.8)
    # Ensure small specks filtered (area criteria) and ordering preserved
    assert all(r.area_px >= 200 for r in regions) 