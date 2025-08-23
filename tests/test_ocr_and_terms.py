import os
from PIL import Image, ImageDraw

from providers.ocr_provider import PaddleOCRProvider
from providers.candidate_terms import CandidateTermGenerator
from providers.types import OCRWord


def _make_test_diagram(path: str) -> Image.Image:
    img = Image.new("RGB", (800, 500), color="white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 760, 120], outline="black", width=2)
    d.text((50, 60), "Cardiac Anatomy: Mitral Valve & Aorta", fill="black")
    d.text((50, 160), "Left ventricle pumps blood to aorta", fill="black")
    d.text((50, 200), "Mitral valve prevents backflow", fill="black")
    img.save(path)
    return img


def test_ocr_and_candidate_terms(tmp_path):
    # Prepare image
    p = tmp_path / "diagram.png"
    _make_test_diagram(str(p))

    # OCR
    if not PaddleOCRProvider.available():
        # If OCR backend not installed, skip test
        import pytest

        pytest.skip("PaddleOCR not available in this environment")

    ocr = PaddleOCRProvider(workdir=str(tmp_path))
    img = Image.open(p)
    result = ocr.recognize(img, use_preprocess=True)

    assert len(result.words) > 0, "OCR returned no words"

    # Candidate terms
    slide_bullets = [
        "Cardiac anatomy and physiology",
        "Left ventricle pumps blood",
        "Function of the mitral valve",
    ]
    transcript_window = [
        "We will discuss the left middle cerebral artery later, but for now focus on the mitral valve",
    ]

    gen = CandidateTermGenerator()
    terms = gen.generate(result.words, slide_bullets, transcript_window, top_k=30)

    # At least one domain-like phrase should appear
    joined = " ".join(terms.terms)
    assert any(
        phrase in joined for phrase in ["mitral valve", "left ventricle", "aorta"]
    ), f"Candidate terms missing expected domain phrases: {terms.terms}" 