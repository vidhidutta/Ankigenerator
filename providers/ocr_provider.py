from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import importlib.util
import os
import uuid

import cv2
import numpy as np
from PIL import Image

from .types import OCRWord, OcrResult
from .utils import compute_image_hash, resize_for_processing, run_with_timeout


@dataclass
class OCRTextBox:
    text: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


_OCR_CACHE: dict[tuple[str, bool, str], OcrResult] = {}


class PaddleOCRProvider:
    """OCR provider using PaddleOCR with optional preprocessing."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, workdir: str = ".") -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None  # type: ignore[var-annotated]
        self.workdir = workdir

    @staticmethod
    def available() -> bool:
        has_paddleocr = importlib.util.find_spec("paddleocr") is not None
        has_paddle_core = importlib.util.find_spec("paddle") is not None
        return bool(has_paddleocr and has_paddle_core)

    @staticmethod
    def tesseract_available() -> bool:
        return importlib.util.find_spec("pytesseract") is not None

    def _ensure_loaded(self) -> None:
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore

            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls)

    def _preprocess(self, image: Image.Image) -> tuple[Image.Image, Optional[str]]:
        # grayscale
        np_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        # contrast stretch using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        # adaptive threshold
        th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        pil = Image.fromarray(th)
        # save temp file for debugging/return
        out_dir = os.path.join(self.workdir, "debug_images")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ocr_pre_{uuid.uuid4().hex[:8]}.png")
        pil.save(out_path)
        return pil, out_path

    def recognize(self, image: Image.Image, use_preprocess: bool = True, timeout_s: float = 15.0) -> OcrResult:
        if self.available():
            provider = "paddle"
            self._ensure_loaded()
            assert self._ocr is not None
        elif self.tesseract_available():
            provider = "tesseract"
        else:
            provider = "none"

        # Resize for processing and hash cache key
        resized, sx, sy = resize_for_processing(image)
        cache_key = (compute_image_hash(resized), bool(use_preprocess), provider)
        if cache_key in _OCR_CACHE:
            return _OCR_CACHE[cache_key]

        def _do_ocr() -> OcrResult:
            if provider == "paddle":
                proc_img = resized
                pre_path: Optional[str] = None
                if use_preprocess:
                    proc_img, pre_path = self._preprocess(resized)

                np_img = np.array(proc_img.convert("RGB"))
                result = self._ocr.ocr(np_img, cls=self.use_angle_cls)

                words: List[OCRWord] = []
                for page in result or []:
                    for line in page:
                        bbox = line[0]
                        text = line[1][0]
                        score = float(line[1][1])
                        xs = [int(pt[0] * sx) for pt in bbox]
                        ys = [int(pt[1] * sy) for pt in bbox]
                        xyxy = (min(xs), min(ys), max(xs), max(ys))
                        if text:
                            words.append(OCRWord(text=text, bbox_xyxy=xyxy, confidence=score))
                return OcrResult(words=words, preprocessed_image_path=pre_path)

            if provider == "tesseract":
                import pytesseract as pt
                # Use tesseract to read words + boxes
                data = pt.image_to_data(resized, output_type=pt.Output.DICT)
                words: List[OCRWord] = []
                n = len(data.get("text", []))
                for i in range(n):
                    text = (data["text"][i] or "").strip()
                    if not text:
                        continue
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    x1, y1, x2, y2 = int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)
                    conf = float(data.get("conf", [0])[i] if isinstance(data.get("conf", None), list) else 0.0)
                    words.append(OCRWord(text=text, bbox_xyxy=(x1, y1, x2, y2), confidence=conf))
                return OcrResult(words=words, preprocessed_image_path=None)

            return OcrResult(words=[], preprocessed_image_path=None)

        result = run_with_timeout(_do_ocr, timeout_sec=timeout_s, default=OcrResult(words=[], preprocessed_image_path=None))
        _OCR_CACHE[cache_key] = result
        return result

    def clear(self) -> None:
        self._ocr = None

    @staticmethod
    def get_provider_name() -> str:
        if PaddleOCRProvider.available():
            return "paddle"
        if PaddleOCRProvider.tesseract_available():
            return "tesseract"
        return "none" 