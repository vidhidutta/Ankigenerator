from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class OCRWord:
    text: str
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float | None = None


@dataclass
class OcrResult:
    words: List[OCRWord]
    preprocessed_image_path: Optional[str] = None


@dataclass
class CandidateTerms:
    terms: List[str]


@dataclass
class DetectedRegion:
    term: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class Region:
    term: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]
    mask_rle: Dict[str, Any] | None
    polygon: Optional[List[List[int]]]
    area_px: int
    # Optional VLM ranking annotations
    importance_score: Optional[float] = None
    short_label: Optional[str] = None
    rationale: Optional[str] = None 