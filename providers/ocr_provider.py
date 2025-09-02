from __future__ import annotations

import io
import importlib.util
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    """OCR provider using Google Vision API first, then Tesseract fallback."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, workdir: str = ".") -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None  # type: ignore[var-annotated]
        self.workdir = workdir

    @staticmethod
    def available() -> bool:
        # Check if Google Vision API is available
        has_google_vision = importlib.util.find_spec("google.cloud.vision") is not None
        has_credentials = bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        return has_google_vision and has_credentials

    @staticmethod
    def tesseract_available() -> bool:
        return importlib.util.find_spec("pytesseract") is not None

    @staticmethod
    def google_vision_available() -> bool:
        has_google_vision = importlib.util.find_spec("google.cloud.vision") is not None
        has_credentials = bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        return has_google_vision and has_credentials

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

    def _do_google_vision_ocr(self, resized: Image.Image, sx: float, sy: float) -> OcrResult:
        """Use Google Vision API for OCR"""
        print(f"[DEBUG] _do_google_vision_ocr called")
        try:
            from google.cloud import vision
            from google.cloud.vision_v1 import types
            print(f"[DEBUG] Google Vision imports successful")
            
            # Initialize Google Vision client
            client = vision.ImageAnnotatorClient()
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            resized.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create image object
            image = types.Image(content=img_byte_arr)
            
            # Perform text detection
            response = client.text_detection(image=image)
            texts = response.text_annotations
            
            words: List[OCRWord] = []
            if texts:
                # First element contains the entire text, skip it
                for text in texts[1:]:
                    text_content = text.description.strip()
                    if not text_content:
                        continue
                    
                    # Get bounding polygon
                    vertices = text.bounding_poly.vertices
                    if len(vertices) >= 4:
                        x_coords = [int(v.x * sx) for v in vertices]
                        y_coords = [int(v.y * sy) for v in vertices]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        # Use confidence from response if available
                        confidence = 0.9  # Google Vision doesn't provide per-word confidence
                        
                        words.append(OCRWord(
                            text=text_content,
                            bbox_xyxy=(x1, y1, x2, y2),
                            confidence=confidence
                        ))
            
            print(f"🔍 Google Vision API found {len(words)} words")
            return OcrResult(words=words, preprocessed_image_path=None)
            
        except Exception as e:
            print(f"⚠️ Google Vision API failed: {e}")
            return OcrResult(words=[], preprocessed_image_path=None)

    def _do_tesseract_ocr(self, resized: Image.Image, sx: float, sy: float) -> OcrResult:
        """Helper method to do Tesseract OCR"""
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

    def recognize(self, image: Image.Image, use_preprocess: bool = True, timeout_s: float = 15.0) -> OcrResult:
        # Try Google Vision API first, then Tesseract as fallback
        provider = "none"
        
        # Debug logging for provider selection
        google_vision_available = self.google_vision_available()
        tesseract_available = self.tesseract_available()
        
        print(f"[DEBUG] OCR Provider selection:")
        print(f"[DEBUG]   - Google Vision available: {google_vision_available}")
        print(f"[DEBUG]   - Tesseract available: {tesseract_available}")
        print(f"[DEBUG]   - GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        
        if google_vision_available:
            provider = "google_vision"
            print(f"[DEBUG] ✅ Using Google Vision API for OCR")
        elif tesseract_available:
            provider = "tesseract"
            print(f"[DEBUG] ⚠️ Google Vision not available, using Tesseract fallback")
        else:
            provider = "none"
            print(f"[DEBUG] ❌ No OCR providers available")

        # Resize for processing and hash cache key
        resized, sx, sy = resize_for_processing(image)
        cache_key = (compute_image_hash(resized), bool(use_preprocess), provider)
        if cache_key in _OCR_CACHE:
            return _OCR_CACHE[cache_key]

        def _do_ocr() -> OcrResult:
            if provider == "google_vision":
                # Try Google Vision API first
                result = self._do_google_vision_ocr(resized, sx, sy)
                
                # If Google Vision returns no words, fall back to Tesseract
                if not result.words and self.tesseract_available():
                    print("⚠️ Google Vision returned 0 words, falling back to Tesseract")
                    return self._do_tesseract_ocr(resized, sx, sy)
                
                return result

            if provider == "tesseract":
                return self._do_tesseract_ocr(resized, sx, sy)

            return OcrResult(words=[], preprocessed_image_path=None)

        result = run_with_timeout(_do_ocr, timeout_sec=timeout_s, default=OcrResult(words=[], preprocessed_image_path=None))
        _OCR_CACHE[cache_key] = result
        return result

    def clear(self) -> None:
        self._ocr = None

    @staticmethod
    def get_provider_name() -> str:
        if PaddleOCRProvider.google_vision_available():
            return "google_vision"
        if PaddleOCRProvider.tesseract_available():
            return "tesseract"
        return "none" 