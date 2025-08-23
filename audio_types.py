from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    emphasis: float = 0.0

@dataclass
class SlideAudioWindow:
    slide_id: int
    window: Tuple[float, float]  # (start, end)
    confidence: float            # 0–1 alignment confidence
    segments: List[TranscriptSegment]
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    snr_db: Optional[float] = None
    snr_quality: Optional[str] = None

@dataclass
class AudioBundle:
    audio_path: str
    segments: List[TranscriptSegment]
    slide_windows: List[SlideAudioWindow]
    diarization_applied: bool = False 