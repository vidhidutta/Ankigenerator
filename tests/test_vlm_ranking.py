import pytest
from PIL import Image

from providers.vlm_provider import LocalQwen2VLProvider, LocalLLaVAOneVisionProvider, CloudVLMProvider
from providers.types import Region


def _get_vlm():
    if LocalQwen2VLProvider.available():
        return LocalQwen2VLProvider()
    if LocalLLaVAOneVisionProvider.available():
        return LocalLLaVAOneVisionProvider()
    if CloudVLMProvider.available():
        return CloudVLMProvider()
    return None


def test_rank_regions_parsing(tmp_path):
    vlm = _get_vlm()
    if vlm is None:
        pytest.skip("No VLM available")

    # 1x1 white image placeholder
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (4, 4), color="white").save(img_path)
    img = Image.open(img_path)

    regions = [
        Region(term="mitral valve", score=0.8, bbox_xyxy=(10, 10, 50, 50), mask_rle=None, polygon=None, area_px=1600),
        Region(term="aorta", score=0.7, bbox_xyxy=(60, 20, 110, 70), mask_rle=None, polygon=None, area_px=2500),
    ]

    ranked = vlm.rank_regions(img, regions, "Cardiac anatomy", "...", top_k=2)

    # Ensure JSON parsed and optional fields populated if model returns them
    assert len(ranked) == len(regions)
    # Cannot guarantee model output; this just ensures no exception and structure is intact 