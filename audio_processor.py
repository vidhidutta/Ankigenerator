#!/usr/bin/env python3
"""
Audio Processing Module for Enhanced Flashcard Generation
Handles audio transcription, temporal alignment, and emphasis detection
"""

import os
import tempfile
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import librosa
import whisper
from sklearn.preprocessing import StandardScaler
import json
import re
from datetime import datetime
from audio_types import TranscriptSegment, SlideAudioWindow, AudioBundle
from sentence_transformers import SentenceTransformer, util as st_util
import subprocess
import shutil
import yaml
import itertools

try:
    from faster_whisper import WhisperModel as FasterWhisperModel  # optional
except Exception:
    FasterWhisperModel = None

try:
    import whisper  # optional, fallback if faster-whisper not available
except Exception:
    whisper = None

@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata"""
    start_time: float
    end_time: float
    text: str
    confidence: float
    emphasis_score: float
    slide_number: Optional[int] = None
    keywords: List[str] = None

@dataclass
class AudioMetadata:
    """Metadata derived from audio analysis"""
    timestamp_start: float
    timestamp_end: float
    emphasis_score: float  # 0-1 based on lecturer emphasis
    time_allocation: float  # seconds spent on this concept
    confidence: float  # transcription confidence
    audio_segment_path: Optional[str] = None  # for audio flashcards
    keywords: List[str] = None

class AudioProcessor:
    def __init__(self, model_name: str = "base", sample_rate: int = 16000):
        """
        Initialize audio processor with Whisper or Faster-Whisper model
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        # Load config
        try:
            with open('config.yaml', 'r') as _f:
                _cfg = yaml.safe_load(_f) or {}
        except Exception:
            _cfg = {}
        ap_cfg = _cfg.get('audio_processing', {})
        self.sample_rate = int(ap_cfg.get('sample_rate', 16000))
        self.hop_length = 512
        self.frame_length = 2048
        self.emphasis_threshold = float(ap_cfg.get('emphasis_detection', {}).get('emphasis_threshold', 0.7))
        self.min_segment_duration = 2.0
        self.emphasis_enabled = bool(ap_cfg.get('emphasis_detection', {}).get('enabled', True))
        self.vad_filter = bool(ap_cfg.get('vad_filter', True))
        self.max_duration_sec = int(ap_cfg.get('max_duration_minutes', 120)) * 60
        self.allow_long_audio = bool(ap_cfg.get('allow_long_audio', False))
        # Clip bounds and tolerance
        self.clip_min_s = float(ap_cfg.get('clip_min_s', 6.0))
        self.clip_max_s = float(ap_cfg.get('clip_max_s', 15.0))
        self.clip_tol_s = float(ap_cfg.get('clip_duration_tolerance_s', 0.25))
        # Model selection
        cfg_model = ap_cfg.get('whisper_model', 'small')
        self.using_faster = False
        try:
            if FasterWhisperModel is not None:
                # Prefer faster-whisper if available
                self._fw_model = FasterWhisperModel(cfg_model if cfg_model else 'small', device='auto', compute_type='auto')
                self.using_faster = True
                self.model = None
            elif whisper is not None:
                self.model = whisper.load_model(model_name or cfg_model)
            else:
                raise RuntimeError("No speech-to-text backend available. Install faster-whisper or openai-whisper.")
        except Exception as e:
            raise RuntimeError(f"Audio model load failed: {e}. Try a smaller model or install faster-whisper.")
        
        # Lazy-loadable embedding model for semantic similarity
        try:
            self._st_model: Optional[SentenceTransformer] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            self._st_model = None
            self.logger.warning(f"SentenceTransformer model load failed: {e}")
        
        thr = ap_cfg.get('snr_thresholds', {})
        self.snr_good = float(thr.get('good_db', 20.0))
        self.snr_fair = float(thr.get('fair_db', 10.0))
        # Alignment penalties (defaults; can be overridden from config)
        self.align_penalty_same = 0.0
        self.align_penalty_next = -0.02
        self.align_penalty_skip = -0.10
        self.align_penalty_back = -0.50
        # Anchors
        self.anchor_bonus = 0.08
        self.anchors_map: Dict[str, List[str]] = {}
    
    def transcribe_audio(self, audio_path: str) -> List[AudioSegment]:
        """
        Transcribe audio file and return segments with timing using chunked processing to reduce RAM.
        Applies duration caps and optional VAD filtering.
        """
        try:
            if not os.path.exists(audio_path) or not os.path.isfile(audio_path):
                raise FileNotFoundError("Audio file not found or unreadable")
            # librosa>=0.10 uses path=, older versions use filename=
            try:
                duration_total = float(librosa.get_duration(path=audio_path))
            except TypeError:
                duration_total = float(librosa.get_duration(filename=audio_path))
            if duration_total > self.max_duration_sec and not self.allow_long_audio:
                raise RuntimeError(f"Audio too long ({duration_total/60:.1f} min). Increase max_duration or enable allow_long_audio in config.")
            segments: List[AudioSegment] = []
            # Faster-Whisper can stream directly
            if self.using_faster and self._fw_model is not None:
                try:
                    # Try with vad_filter (newer faster-whisper). Fallback without if unsupported.
                    try:
                        fw_segments, _info = self._fw_model.transcribe(
                            audio_path,
                            vad_filter=self.vad_filter,
                            language='en'
                        )
                    except TypeError:
                        fw_segments, _info = self._fw_model.transcribe(
                            audio_path,
                            language='en'
                        )
                    for seg in fw_segments:
                        start = float(seg.start)
                        end = float(seg.end)
                        text = (seg.text or '').strip()
                        if self.vad_filter and not text:
                            continue
                        # Emphasis score placeholder; compute later if needed
                        emphasis_score = 0.0
                        segments.append(AudioSegment(start_time=start, end_time=end, text=text, confidence=1.0, emphasis_score=emphasis_score))
                    # Compute emphasis using lightweight features per segment by streaming audio per segment window
                    try:
                        for s in segments:
                            # Load only the needed window to compute emphasis
                            audio_seg, sr = librosa.load(audio_path, sr=self.sample_rate, offset=s.start_time, duration=max(0.01, s.end_time - s.start_time))
                            s.emphasis_score = self._calculate_emphasis_score(audio_seg, sr, 0.0, max(0.01, s.end_time - s.start_time)) if self.emphasis_enabled else 0.0
                        return segments
                    except Exception:
                        return segments
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        raise RuntimeError("GPU out of memory. Switch to a smaller model or CPU mode.")
                    raise
            # Fallback: standard whisper in chunks
            if not self.using_faster and whisper is not None:
                pass
            else:
                # No fallback available
                raise RuntimeError("No compatible Whisper backend available for transcription.")
            # Process in 60s chunks with 5s overlap
            chunk_sec = 60.0
            overlap = 5.0
            t = 0.0
            while t < duration_total:
                dur = min(chunk_sec, duration_total - t)
                # VAD prefilter: skip very silent chunks
                audio_chunk, sr = librosa.load(audio_path, sr=self.sample_rate, offset=max(0.0, t), duration=max(0.01, dur))
                if self.vad_filter:
                    rms = float(np.mean(np.abs(audio_chunk))) if audio_chunk.size else 0.0
                    if rms < 1e-3:
                        t += max(1.0, chunk_sec - overlap)
                        continue
                try:
                    result = self.model.transcribe(audio_chunk, word_timestamps=False, language='en')
                    for seg in result.get('segments', []):
                        start = float(seg.get('start', 0.0)) + t
                        end = float(seg.get('end', 0.0)) + t
                        text = str(seg.get('text', '')).strip()
                        conf = float(seg.get('avg_logprob', 0.0))
                        # Compute emphasis score on the chunk portion
                        emphasis = 0.0
                        try:
                            # Map to chunk local times
                            local_start = max(0.0, float(seg.get('start', 0.0)))
                            local_end = max(local_start, float(seg.get('end', 0.0)))
                            seg_audio = audio_chunk[int(local_start * sr):int(local_end * sr)]
                            emphasis = self._calculate_emphasis_score(seg_audio, sr, 0.0, max(0.01, local_end - local_start)) if self.emphasis_enabled else 0.0
                        except Exception:
                            emphasis = 0.0
                        segments.append(AudioSegment(start_time=start, end_time=end, text=text, confidence=conf, emphasis_score=emphasis))
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        raise RuntimeError("GPU out of memory. Switch to a smaller model or CPU mode.")
                    raise
                t += max(1.0, chunk_sec - overlap)
            return segments
        except FileNotFoundError as e:
            self.logger.error(str(e))
            return []
        except RuntimeError as e:
            # Surface actionable message up the stack
            raise
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return []
    
    def _calculate_emphasis_score(self, audio: np.ndarray, sr: int, 
                                start_time: float, end_time: float) -> float:
        """
        Calculate emphasis score based on pitch, volume, and speaking rate
        
        Args:
            audio: Audio array
            sr: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Emphasis score (0-1)
        """
        if not self.emphasis_enabled:
            return 0.0
        try:
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio[start_sample:end_sample]
            
            if len(segment) == 0:
                return 0.0
            
            # Calculate pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(
                y=segment, sr=sr, hop_length=self.hop_length
            )
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 90)]
            avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            
            # Calculate volume (RMS energy)
            rms = librosa.feature.rms(
                y=segment, hop_length=self.hop_length
            ).flatten()
            avg_volume = np.mean(rms)
            
            # Calculate speaking rate (syllables per second approximation)
            onset_frames = librosa.onset.onset_detect(
                y=segment, sr=sr, hop_length=self.hop_length
            )
            speaking_rate = len(onset_frames) / (end_time - start_time)
            
            # Normalize features
            pitch_score = min(avg_pitch / 500, 1.0)  # Normalize to 0-1
            volume_score = min(avg_volume / 0.1, 1.0)  # Normalize to 0-1
            rate_score = min(speaking_rate / 10, 1.0)  # Normalize to 0-1
            
            # Combine scores (weighted average)
            emphasis_score = (0.4 * pitch_score + 0.4 * volume_score + 0.2 * rate_score)
            
            return min(emphasis_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating emphasis score: {e}")
            return 0.0  # Default neutral score when error
    
    def _extract_keywords(self, text) -> List[str]:
        """
        Extract medical keywords from text
        
        Args:
            text: Text to analyze (string or dict)
            
        Returns:
            List of keywords
        """
        # Handle both string and dictionary inputs
        if isinstance(text, dict):
            text_content = text.get('content', '')
        else:
            text_content = str(text)
        
        # Medical terminology patterns
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b(?:drug|medication|treatment|therapy|diagnosis|symptom|disease|patient|clinical|medical)\b',
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\d+(?:\.\d+)?(?:%|mg|ml|g|kg)\b',  # Measurements
        ]
        
        keywords = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))  # Remove duplicates
    
    def _generate_transcript_windows(self, segments: List[Any], window_sizes: List[int] = [45, 60, 90], stride: int = 15, lecturer_only: bool = False) -> List[Tuple[float, float, str, List[Any]]]:
        """
        Build rolling transcript windows over segments.
        Returns list of (start, end, text, segments_in_window)
        """
        if not segments:
            return []
        # Optionally filter to lecturer segments only
        if lecturer_only:
            segments = [s for s in segments if str(getattr(s, 'speaker', 'LECTURER')).upper() == 'LECTURER']
            if not segments:
                return []
        min_start = min(getattr(s, 'start_time', getattr(s, 'start', 0.0)) for s in segments)
        max_end = max(getattr(s, 'end_time', getattr(s, 'end', 0.0)) for s in segments)
        windows: List[Tuple[float, float, str, List[Any]]] = []
        t = min_start
        segs_sorted = sorted(segments, key=lambda s: getattr(s, 'start_time', getattr(s, 'start', 0.0)))
        while t < max_end:
            for w in window_sizes:
                start = t
                end = min(t + w, max_end)
                bucket: List[Any] = []
                for s in segs_sorted:
                    s_start = getattr(s, 'start_time', getattr(s, 'start', 0.0))
                    s_end = getattr(s, 'end_time', getattr(s, 'end', 0.0))
                    if s_end < start:
                        continue
                    if s_start > end:
                        break
                    bucket.append(s)
                if not bucket:
                    continue
                texts = []
                for s in bucket:
                    txt = getattr(s, 'text', '')
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt)
                text = " ".join(texts)
                if text.strip():
                    windows.append((start, end, text, bucket))
            t += stride
        return windows

    def _keyword_overlap_score(self, slide_text: str, window_text: str) -> float:
        slide_kw = set(self._extract_keywords(slide_text))
        window_kw = set(self._extract_keywords(window_text))
        if not slide_kw or not window_kw:
            return 0.0
        overlap = len(slide_kw & window_kw)
        denom = max(1, min(len(slide_kw), len(window_kw)))
        score = overlap / denom
        return float(np.clip(score, 0.0, 1.0))

    def _semantic_similarity_score(self, slide_text: str, window_text: str) -> float:
        if not self._st_model:
            return 0.0
        try:
            emb = self._st_model.encode(
                [slide_text, window_text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sim = st_util.cos_sim(emb[0], emb[1]).item()
            # cos_sim in [-1,1], map to [0,1]
            return float((sim + 1.0) / 2.0)
        except Exception as e:
            self.logger.warning(f"Semantic similarity failed: {e}")
            return 0.0

    def _normalize_text(self, text: str) -> str:
        return (text or "").lower()

    def _ensure_anchors(self) -> None:
        if self.anchors_map:
            return
        # Seed with hematology anchors and synonyms
        self.anchors_map = {
            "rdw": ["red cell distribution width", "red blood cell distribution width"],
            "mch": ["mean cell hemoglobin", "mean corpuscular hemoglobin"],
            "mchc": ["mean cell haemoglobin concentration", "mean corpuscular hemoglobin concentration"],
            "reticulocyte": ["retic", "reticulocytes"],
            "leukocyte": ["white blood cell", "wbc", "leucocyte", "white cell"],
            "neutrophil": ["neut", "neutrophils"],
            "lymphocyte": ["lymph", "lymphocytes"],
            "platelet": ["thrombocyte", "thrombocytes", "plt"],
            "hematocrit": ["hct"],
            "hemoglobin": ["haemoglobin", "hb"],
            "g6pd": ["glucose-6-phosphate dehydrogenase"],
            "eculizumab": ["anti-c5", "anti c5"],
        }

    def _extract_slide_anchors(self, slide_texts: List[str]) -> List[set]:
        self._ensure_anchors()
        out: List[set] = []
        for txt in slide_texts:
            t = self._normalize_text(str(txt))
            hits = set()
            for k, syns in self.anchors_map.items():
                if k in t:
                    hits.add(k)
                else:
                    for s in syns:
                        if s in t:
                            hits.add(k)
                            break
            out.append(hits)
        return out

    def _build_similarity_matrix(
        self,
        windows: List[Tuple[float, float, str, List[Any]]],
        slide_texts: List[str],
        slide_anchors: List[set],
    ) -> Tuple[np.ndarray, List[List[str]], Dict[int, List[str]]]:
        """
        Returns S[slide, time_bin], anchor_hits_by_bin[slide][bin] (terms), and anchor_hits_per_slide aggregated.
        """
        T = len(windows)
        N = len(slide_texts)
        S = np.zeros((N, T), dtype=np.float32)
        anchor_hits_per_slide: Dict[int, List[str]] = {i: [] for i in range(N)}
        anchor_hits_by_bin: List[List[str]] = [[ ] for _ in range(T)]
        for i, slide_text in enumerate(slide_texts):
            for t, (_, _, wtext, _) in enumerate(windows):
                sem = self._semantic_similarity_score(str(slide_text), wtext)
                kw = self._keyword_overlap_score(str(slide_text), wtext)
                score = 0.6 * sem + 0.4 * kw
                # anchors
                bonus = 0.0
                terms_hit: List[str] = []
                if slide_anchors[i]:
                    wt = self._normalize_text(wtext)
                    for k in list(slide_anchors[i]):
                        if k in wt or any(syn in wt for syn in self.anchors_map.get(k, [])):
                            bonus += float(self.anchor_bonus)
                            terms_hit.append(k)
                S[i, t] = float(np.clip(score + bonus, 0.0, 1.0))
                if terms_hit:
                    anchor_hits_by_bin[t].extend(terms_hit)
                    anchor_hits_per_slide[i].extend(terms_hit)
        # Deduplicate per slide hits
        for i in anchor_hits_per_slide.keys():
            anchor_hits_per_slide[i] = sorted(list(set(anchor_hits_per_slide[i])))
        return S, anchor_hits_by_bin, anchor_hits_per_slide

    def _align_with_slides_combined(
        self,
        segments: List[TranscriptSegment],
        slide_texts: List[str],
        lecturer_only: bool = False,
    ) -> List[SlideAudioWindow]:
        """Lightweight fallback: choose the single best transcript window per slide.

        Uses the same similarity matrix S but without DP. For each slide i, pick the
        time bin t with the highest score S[i, t] and return that window.
        """
        windows = self._generate_transcript_windows(segments, lecturer_only=lecturer_only)
        slide_windows: List[SlideAudioWindow] = []
        if not windows:
            return slide_windows
        slide_anchors = self._extract_slide_anchors(slide_texts)
        S, _hits_by_bin, _hits_per_slide = self._build_similarity_matrix(windows, slide_texts, slide_anchors)
        N, T = S.shape
        for i in range(N):
            if T == 0:
                continue
            t_best = int(np.argmax(S[i, :]))
            ws, we, _wtext, wsegs = windows[t_best]
            # Convert segments to TranscriptSegment if needed
            segs_ts = [
                TranscriptSegment(
                    start=getattr(s, 'start_time', getattr(s, 'start', 0.0)),
                    end=getattr(s, 'end_time', getattr(s, 'end', 0.0)),
                    text=getattr(s, 'text', ''),
                    emphasis=getattr(s, 'emphasis_score', getattr(s, 'emphasis', 0.0)),
                    speaker=getattr(s, 'speaker', None),
                )
                for s in wsegs
            ]
            conf = float(np.clip(S[i, t_best], 0.0, 1.0))
            slide_windows.append(
                SlideAudioWindow(slide_id=i, window=(float(ws), float(we)), confidence=conf, segments=segs_ts)
            )
        # Sort by slide id to keep deterministic order
        slide_windows.sort(key=lambda w: w.slide_id)
        return slide_windows

    def _align_monotonic_dp(
        self,
        segments: List[TranscriptSegment],
        slide_texts: List[str],
        lecturer_only: bool = False,
    ) -> Tuple[List[SlideAudioWindow], Dict[int, List[str]], Dict[str, float]]:
        windows = self._generate_transcript_windows(segments, lecturer_only=lecturer_only)
        slide_windows: List[SlideAudioWindow] = []
        if not windows:
            return slide_windows, {}, {}
        slide_anchors = self._extract_slide_anchors(slide_texts)
        S, anchor_hits_by_bin, anchor_hits_per_slide = self._build_similarity_matrix(windows, slide_texts, slide_anchors)
        N, T = S.shape
        # DP
        dp = np.full((N, T), -1e9, dtype=np.float32)
        prev = np.full((N, T), -1, dtype=np.int32)
        dp[:, 0] = S[:, 0]
        prev[:, 0] = -1
        psame = float(self.align_penalty_same)
        pnext = float(self.align_penalty_next)
        pskip = float(self.align_penalty_skip)
        pback = float(self.align_penalty_back)
        for t in range(1, T):
            for i in range(N):
                best = -1e9
                best_j = -1
                for j in range(N):
                    if dp[j, t-1] <= -1e8:
                        continue
                    if i == j:
                        pen = psame
                    elif i == j + 1:
                        pen = pnext
                    elif i > j + 1:
                        pen = pskip * (i - j - 1)
                    elif i == j - 1:
                        pen = pback
                    else:  # i < j-1 (multi backtrack)
                        pen = pback * (j - i)
                    val = dp[j, t-1] + pen
                    if val > best:
                        best = val
                        best_j = j
                dp[i, t] = best + S[i, t]
                prev[i, t] = best_j
        # Best end state
        end_i = int(np.argmax(dp[:, -1]))
        path = [end_i]
        for t in range(T-1, 0, -1):
            end_i = prev[end_i, t]
            if end_i < 0:
                end_i = path[-1]
            path.append(int(end_i))
        path = list(reversed(path))  # length T
        # Assign bins to slides
        bins_by_slide: Dict[int, List[int]] = {i: [] for i in range(N)}
        for t, i in enumerate(path):
            bins_by_slide[i].append(t)
        # Build windows
        for i in range(N):
            bins = sorted(bins_by_slide[i])
            if not bins:
                continue
            # contiguous merge: take min start to max end
            start_bin = bins[0]
            end_bin = bins[-1]
            start_time = windows[start_bin][0]
            end_time = windows[end_bin][1]
            # Confidence = mean S across bins
            conf = float(np.mean([S[i, b] for b in bins]))
            # Collect segments inside window
            segs = []
            for (_, _, _wtext, wsegs) in windows[start_bin:end_bin+1]:
                segs.extend(wsegs)
            segs_ts = [TranscriptSegment(start=getattr(s,'start_time', getattr(s,'start',0.0)), end=getattr(s,'end_time', getattr(s,'end',0.0)), text=getattr(s,'text',''), emphasis=getattr(s,'emphasis_score', getattr(s,'emphasis',0.0)), speaker=getattr(s,'speaker', None)) for s in segs]
            win = SlideAudioWindow(slide_id=i, window=(float(start_time), float(end_time)), confidence=conf, segments=segs_ts)
            # attach anchor hits
            hits = anchor_hits_per_slide.get(i, [])
            if hits:
                setattr(win, 'anchor_hits', hits)
            slide_windows.append(win)
        # Fallback for empty slides
        if len(slide_windows) < N:
            # compute per-slide best window fallback
            fallback = self._align_with_slides_combined(segments, slide_texts, lecturer_only)
            existing_ids = {w.slide_id for w in slide_windows}
            for w in fallback:
                if w.slide_id not in existing_ids:
                    slide_windows.append(w)
        # Sort by slide_id
        slide_windows.sort(key=lambda w: w.slide_id)
        penalties = {
            'same': psame,
            'next': pnext,
            'skip': pskip,
            'back': pback,
        }
        return slide_windows, anchor_hits_per_slide, penalties

    def _diarize_speakers(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """
        Try to diarize speakers. Returns list of (start, end, speaker_label).
        Fallback: single speaker 'LECTURER'.
        """
        try:
            from pyannote.audio import Pipeline  # optional dependency
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            diar = pipeline(audio_path)
            labeled: List[Tuple[float, float, str]] = []
            for turn, _, speaker in diar.itertracks(yield_label=True):
                labeled.append((float(turn.start), float(turn.end), str(speaker)))
            # Heuristic: choose the most speaking time speaker as LECTURER
            totals: Dict[str, float] = {}
            for s, e, sp in [(st, en, spk) for st, en, spk in labeled]:
                totals[sp] = totals.get(sp, 0.0) + (e - s)
            lecturer_speaker = max(totals.items(), key=lambda x: x[1])[0] if totals else 'SPEAKER_00'
            return [(s, e, 'LECTURER' if sp == lecturer_speaker else 'OTHER') for s, e, sp in labeled]
        except Exception:
            # Fallback: everything is lecturer
            return [(0.0, 1e9, 'LECTURER')]

    def align_with_slides(self, audio_segments: List[AudioSegment], 
                         slide_texts: List[str], slide_durations: Optional[List[float]] = None) -> Dict[int, List[AudioSegment]]:
        """
        Align audio segments with slides based on content similarity
        
        Args:
            audio_segments: List of audio segments
            slide_texts: List of slide texts
            slide_durations: Optional list of slide durations (if known)
            
        Returns:
            Dictionary mapping slide numbers to relevant audio segments
        """
        slide_audio_map = {}
        
        if slide_durations:
            # Use timing-based alignment
            current_time = 0.0
            for slide_num, duration in enumerate(slide_durations):
                slide_audio_map[slide_num] = []
                
                # Find segments that fall within this slide's time window
                for segment in audio_segments:
                    if (segment.start_time >= current_time and 
                        segment.start_time < current_time + duration):
                        segment.slide_number = slide_num
                        slide_audio_map[slide_num].append(segment)
                
                current_time += duration
        else:
            # Use content-based alignment
            for slide_num, slide_text in enumerate(slide_texts):
                slide_audio_map[slide_num] = []
                
                # Handle both string and dictionary slide_text
                if isinstance(slide_text, dict):
                    # Extract text content from semantic chunk
                    slide_text_content = slide_text.get('content', '')
                else:
                    slide_text_content = str(slide_text)
                
                # Ensure slide_text_content is a string before calling .lower()
                if isinstance(slide_text_content, dict):
                    slide_text_content = str(slide_text_content)
                
                # Find segments with keyword overlap
                slide_keywords = self._extract_keywords(slide_text_content.lower())
                
                for segment in audio_segments:
                    try:
                        raw_kws = getattr(segment, 'keywords', None)
                    except Exception:
                        raw_kws = None
                    if not raw_kws:
                        raw_kws = self._extract_keywords(getattr(segment, 'text', ''))
                    segment_keywords = [kw.lower() for kw in (raw_kws or [])]
                    
                    # Calculate keyword overlap
                    overlap = len(set(slide_keywords) & set(segment_keywords))
                    if overlap > 0:
                        segment.slide_number = slide_num
                        slide_audio_map[slide_num].append(segment)
        
        return slide_audio_map
    
    def calculate_content_weights(self, slide_audio_map: Dict[int, List[AudioSegment]]) -> Dict[int, float]:
        """
        Calculate content importance weights based on time allocation and emphasis
        
        Args:
            slide_audio_map: Dictionary mapping slides to audio segments
            
        Returns:
            Dictionary mapping slide numbers to importance weights
        """
        weights = {}
        
        for slide_num, segments in slide_audio_map.items():
            if not segments:
                weights[slide_num] = 0.5  # Default weight
                continue
            
            # Calculate total time spent on this slide
            total_time = sum(seg.end_time - seg.start_time for seg in segments)
            
            # Calculate average emphasis
            avg_emphasis = np.mean([seg.emphasis_score for seg in segments])
            
            # Calculate average confidence
            avg_confidence = np.mean([seg.confidence for seg in segments])
            
            # Combine factors for final weight
            time_weight = min(total_time / 60.0, 1.0)  # Normalize to 0-1
            weight = (0.4 * time_weight + 0.4 * avg_emphasis + 0.2 * avg_confidence)
            
            weights[slide_num] = weight
        
        return weights

    def build_audio_bundle(self, audio_path: str, slide_texts: List[str], diarization_enabled: bool = False, alignment_mode: str = "semantic+keyword") -> AudioBundle:
        """
        Build an AudioBundle from an audio file and slide texts.
        """
        segments_raw = self.transcribe_audio(audio_path)
        # Optionally diarize and mark speakers on raw segments
        diarized: List[Tuple[float, float, str]] = []
        diar_ok = False
        if diarization_enabled:
            try:
                diarized = self._diarize_speakers(audio_path)
                diar_ok = True
            except Exception:
                diarized = []
                diar_ok = False
        def label_for(t0: float, t1: float) -> Optional[str]:
            if not diarized:
                return 'LECTURER'
            best_label = 'OTHER'
            best_overlap = 0.0
            for ds, de, lab in diarized:
                overlap = max(0.0, min(t1, de) - max(t0, ds))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = lab
            return best_label
        
        # Convert to TranscriptSegment with speaker
        segments: List[TranscriptSegment] = []
        for seg in segments_raw:
            start = getattr(seg, 'start_time', getattr(seg, 'start', 0.0))
            end = getattr(seg, 'end_time', getattr(seg, 'end', start))
            text = getattr(seg, 'text', '')
            emphasis = getattr(seg, 'emphasis_score', getattr(seg, 'emphasis', 0.0))
            speaker = label_for(start, end) if diarization_enabled and diar_ok else None
            segments.append(TranscriptSegment(start=start, end=end, text=text, emphasis=emphasis, speaker=speaker))
        
        # Choose alignment mode
        if alignment_mode == "keyword":
            # Use existing keyword-based alignment via align_with_slides on raw segments
            # Need AudioSegment-like objects with start_time/end_time; map back from TranscriptSegment
            class _TmpSeg:
                def __init__(self, s):
                    self.start_time = s.start
                    self.end_time = s.end
                    self.text = s.text
                    self.emphasis_score = s.emphasis
                    try:
                        self.keywords = self._extract_keywords(getattr(s, 'text', ''))
                    except Exception:
                        self.keywords = []
            rawish = [_TmpSeg(s) for s in segments if (not diarization_enabled or str(getattr(s, 'speaker', 'LECTURER')).upper() == 'LECTURER')]
            alignment_map = self.align_with_slides(rawish, slide_texts)
            slide_windows: List[SlideAudioWindow] = []
            for slide_id, segs in alignment_map.items():
                if not segs:
                    continue
                starts = [getattr(s, 'start_time', 0.0) for s in segs]
                ends = [getattr(s, 'end_time', 0.0) for s in segs]
                window = (min(starts), max(ends))
                avg_emphasis = float(np.mean([getattr(s, 'emphasis_score', 0.0) for s in segs])) if segs else 0.0
                confidence = float(np.clip(avg_emphasis, 0.0, 1.0))
                segs_ts = [TranscriptSegment(start=getattr(s,'start_time',0.0), end=getattr(s,'end_time',0.0), text=getattr(s,'text',''), emphasis=getattr(s,'emphasis_score',0.0)) for s in segs]
                slide_windows.append(SlideAudioWindow(slide_id=slide_id, window=window, confidence=confidence, segments=segs_ts))
        else:
            # Monotonic DP alignment with anchors and penalties
            # Penalties may be loaded from config externally onto self
            slide_windows, anchor_hits_per_slide, penalties = self._align_monotonic_dp(segments, slide_texts, lecturer_only=(diarization_enabled and diar_ok))
            # Attach meta to bundle later
        
        for w in slide_windows:
            snr_db, quality = self._compute_window_snr(audio_path, w.window[0], w.window[1])
            w.snr_db = snr_db
            w.snr_quality = quality

        bundle = AudioBundle(audio_path=audio_path, segments=segments, slide_windows=slide_windows, diarization_applied=bool(diarization_enabled and diar_ok))
        setattr(bundle, 'alignment_method', 'monotonic_dp' if alignment_mode != 'keyword' else 'keyword')
        setattr(bundle, 'alignment_penalties', {'same': self.align_penalty_same, 'next': self.align_penalty_next, 'skip': self.align_penalty_skip, 'back': self.align_penalty_back})
        # Collect anchors used
        try:
            used = any(getattr(w, 'anchor_hits', None) for w in slide_windows)
            setattr(bundle, 'anchors_used', bool(used))
        except Exception:
            setattr(bundle, 'anchors_used', False)
        return bundle

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            import librosa
            try:
                return float(librosa.get_duration(path=audio_path))
            except TypeError:
                return float(librosa.get_duration(filename=audio_path))
        except Exception:
            return 0.0

    def _ffmpeg_available(self) -> bool:
        return shutil.which('ffmpeg') is not None

    def _clip_with_ffmpeg(self, src: str, start: float, end: float, out_path: str) -> bool:
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', f'{start:.3f}',
                '-to', f'{end:.3f}',
                '-i', src,
                '-vn', '-acodec', 'libmp3lame', '-q:a', '4',
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            self.logger.error(f"ffmpeg clip failed: {e}")
            return False

    def _probe_duration(self, path: str) -> float:
        # Prefer ffprobe for accuracy
        try:
            if shutil.which('ffprobe'):
                cmd = ['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1', path]
                out = subprocess.check_output(cmd).decode().strip()
                return float(out)
        except Exception:
            pass
        # Fallback to pydub
        try:
            from pydub import AudioSegment as _AS
            seg = _AS.from_file(path)
            return float(len(seg) / 1000.0)
        except Exception:
            pass
        # Last resort: librosa
        try:
            try:
                return float(librosa.get_duration(path=path))
            except TypeError:
                return float(librosa.get_duration(filename=path))
        except Exception:
            return 0.0

    def create_audio_clips_for_bundle(
        self,
        bundle: AudioBundle,
        output_dir: str,
        per_slide_max: int = 2,
        min_clip_sec: Optional[float] = None,
        max_clip_sec: Optional[float] = None,
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
        os.makedirs(output_dir, exist_ok=True)
        clips: Dict[int, List[str]] = {}
        warnings_map: Dict[int, List[str]] = {}
        # meta for report: per slide boundary types and lengths
        clip_meta: Dict[int, Dict[str, Any]] = {}
        duration = self._get_audio_duration(bundle.audio_path)
        ffmpeg_ok = self._ffmpeg_available()
        # Resolve bounds
        min_len = float(min_clip_sec if min_clip_sec is not None else self.clip_min_s)
        max_len = float(max_clip_sec if max_clip_sec is not None else self.clip_max_s)
        tol = float(self.clip_tol_s)
        # Learning config
        learning_cfg = getattr(self, 'learning_cfg', None)
        if learning_cfg is None:
            learning_cfg = {
                'enabled': True,
                'base_target_s': 7.0,
                'min_clip_s': 5.0,
                'max_clip_s': 15.0,
                'overflow_max_s': 1.0,
                'weight': {'semantic': 0.6, 'keyword': 0.3, 'emphasis': 0.1},
            }
        # Helper: pick target length given simple features (stub policy; can be replaced by bandit)
        def _choose_target_length(features: Dict[str, float]) -> float:
            base = float(learning_cfg.get('base_target_s', 7.0))
            # Simple adaptive tweak: faster speaking rate -> a bit longer, low SNR -> a bit shorter
            speaking = float(features.get('speaking_rate', 1.0))
            snr = float(features.get('snr_db', 20.0))
            emphasis = float(features.get('emphasis_mean', 0.5))
            adj = 0.0
            if speaking > 1.2:
                adj += 1.0
            if speaking < 0.8:
                adj -= 0.5
            if snr < 10.0:
                adj -= 0.5
            if emphasis > 0.7:
                adj += 0.5
            tgt = base + adj
            # Clamp target for Auto mode to a natural-sounding range [5, 12] s,
            # still subject to overall clip bounds downstream
            tgt_min = float(learning_cfg.get('min_clip_s', 5.0))
            tgt_cap_max = min(12.0, float(learning_cfg.get('max_clip_s', 15.0)))
            tgt = max(tgt_min, min(tgt_cap_max, tgt))
            return tgt
        # Helper: find nearest natural boundaries around [s,e] within slide window (pause/sentence boundaries stub)
        def _snap_to_boundaries(s: float, e: float, ws: float, we: float, target: float, win: SlideAudioWindow) -> Tuple[float, float, List[str]]:
            types: List[str] = []
            # Try to expand to hit target using window bounds as hard limits, allow slight overflow
            overflow_max = float(learning_cfg.get('overflow_max_s', 1.0))
            left = max(0.0, s)
            right = min(duration, e)
            # Prefer VAD-like pauses: gaps >= pause_min_ms between consecutive segments
            pause_thresh = float(learning_cfg.get('pause_min_ms', 250.0)) / 1000.0
            segs = sorted(getattr(win, 'segments', []) or [], key=lambda x: x.start)
            # nearest pause before left
            prev_pause = None
            for a, b in zip(segs, segs[1:]):
                gap = max(0.0, b.start - a.end)
                if gap >= pause_thresh and a.end <= left:
                    prev_pause = a.end
                if a.end <= left and (a.text or '').strip().endswith(('.', '!', '?')):
                    prev_pause = a.end
            if prev_pause is not None and prev_pause >= ws:
                left = max(ws, prev_pause)
                types.append('pause')
            # nearest pause after right
            next_pause = None
            for a, b in zip(segs, segs[1:]):
                gap = max(0.0, b.start - a.end)
                if gap >= pause_thresh and b.start >= right and next_pause is None:
                    next_pause = b.start
                    break
                if (a.text or '').strip().endswith(('.', '!', '?')) and a.end >= right and next_pause is None:
                    next_pause = a.end
                    break
            if next_pause is not None and next_pause <= we:
                right = min(we, next_pause)
                if 'pause' not in types:
                    types.append('pause')
            # Try sentence/pause boundaries if available later; for now, use segment edges and clamp
            need = target - (right - left)
            if need > 0:
                expand_each = need / 2.0
                new_left = max(ws, left - expand_each)
                new_right = min(we, right + expand_each)
                if (new_right - new_left) < target:
                    # allow slight overflow equally
                    space_left = max(0.0, left - ws)
                    space_right = max(0.0, we - right)
                    spill = min(overflow_max, target - (new_right - new_left))
                    new_left = max(0.0, new_left - min(spill/2.0, new_left))
                    new_right = min(duration, new_right + min(spill/2.0, duration - new_right))
                left, right = new_left, new_right
                types.append('expanded')
            # Note: boundary tags could include 'sentence'/'discourse'/'semantic_shift' when integrated
            return left, right, types
        if not ffmpeg_ok and not shutil.which('ffprobe'):
            self.logger.warning("ffmpeg/ffprobe not found. Audio clips may not be generated or validated.")
        audio_base = os.path.splitext(os.path.basename(bundle.audio_path))[0]
        safe_base = re.sub(r"[^A-Za-z0-9_-]+", "_", audio_base)
        for win in getattr(bundle, 'slide_windows', []) or []:
            segs_all = getattr(win, 'segments', []) or []
            segs_lect = [s for s in segs_all if str(getattr(s, 'speaker', 'LECTURER')).upper() == 'LECTURER']
            segs = segs_lect if segs_lect else segs_all
            if not segs:
                continue
            segs = sorted(segs, key=lambda s: (getattr(s, 'emphasis', 0.0), (s.end - s.start)), reverse=True)
            selected: List[Tuple[float, float]] = []
            for seg in segs:
                if len(selected) >= per_slide_max:
                    break
                s = float(seg.start)
                e = float(seg.end)
                ws, we = float(win.window[0]), float(win.window[1])
                s = max(s, ws)
                e = min(e, we)
                # Auto target via learning policy
                feats = {
                    'speaking_rate': max(0.1, (len(getattr(seg, 'text', '').split()) / max(1e-3, (seg.end - seg.start)))) ,
                    'snr_db': float(getattr(win, 'snr_db', 20.0)),
                    'emphasis_mean': float(getattr(seg, 'emphasis', 0.5)),
                    'window_length': float(we - ws),
                }
                target = _choose_target_length(feats) if learning_cfg.get('enabled', True) else (min_len + max_len) / 2.0
                # Expand to boundaries within slide window
                s2, e2, types = _snap_to_boundaries(s, e, ws, we, target, win)
                # Guard against invalid or negative-duration windows after boundary snapping
                if e2 <= s2:
                    msg = f"Skipped: invalid window after boundary snap"
                    warnings_map.setdefault(win.slide_id, []).append(msg)
                    self.logger.warning(f"{msg} (slide {win.slide_id})")
                    continue
                if e2 - s2 < min_len:
                    msg = f"Skipped: window {e2 - s2:.1f}s < min {min_len:.0f}s"
                    warnings_map.setdefault(win.slide_id, []).append(msg)
                    self.logger.warning(f"{msg} (slide {win.slide_id})")
                    continue
                s, e = s2, e2
                length = e - s
                if length > max_len:
                    mid = (s + e) / 2.0
                    s = max(0.0, mid - max_len / 2.0)
                    e = min(duration, s + max_len)
                if e - s <= 0.5:
                    continue
                start_ms = int(round(s * 1000))
                end_ms = int(round(e * 1000))
                k = len(selected) + 1
                fname = f"{safe_base}_aud_s{win.slide_id}_{k}_{start_ms}-{end_ms}.mp3"
                out_path = os.path.join(output_dir, fname)
                ok = False
                if ffmpeg_ok:
                    ok = self._clip_with_ffmpeg(bundle.audio_path, s, e, out_path)
                if not ok:
                    try:
                        import soundfile as sf
                        audio, sr = librosa.load(bundle.audio_path, sr=self.sample_rate)
                        start_sample = int(s * sr)
                        end_sample = int(e * sr)
                        segment = audio[start_sample:end_sample]
                        sf.write(out_path, segment, sr, format='MP3')
                        ok = True
                    except Exception as e2:
                        msg = f"Clipper unavailable; failed to create clip: {e2}"
                        warnings_map.setdefault(win.slide_id, []).append(msg)
                        self.logger.error(msg)
                        ok = False
                if ok:
                    clip_dur = self._probe_duration(out_path)
                    if not (min_len - tol <= clip_dur <= max_len + tol):
                        msg = f"Deleted: clip {clip_dur:.2f}s out of bounds [{min_len},{max_len}]"
                        warnings_map.setdefault(win.slide_id, []).append(msg)
                        self.logger.warning(f"{msg} (slide {win.slide_id})")
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass
                        continue
                    # Skip near-duplicate overlaps with already selected spans
                    overlap_too_high = False
                    for (ps, pe) in selected:
                        inter = max(0.0, min(e, pe) - max(s, ps))
                        cand_len = max(1e-6, e - s)
                        if inter / cand_len > 0.5:
                            overlap_too_high = True
                            break
                    if overlap_too_high:
                        msg = f"Skipped: overlaps existing clip >50%"
                        warnings_map.setdefault(win.slide_id, []).append(msg)
                        continue
                    selected.append((s, e))
                    clips.setdefault(win.slide_id, []).append(out_path)
                    meta = clip_meta.setdefault(win.slide_id, {'boundary_types': [], 'lengths_s': []})
                    meta['boundary_types'] = list(sorted(set(meta['boundary_types'] + types)))
                    meta['lengths_s'].append(float(clip_dur))
        # expose meta for report
        try:
            self.last_clip_meta = clip_meta
        except Exception:
            pass
        return clips, warnings_map

    def calculate_content_weights_from_bundle(self, bundle: AudioBundle) -> Dict[int, float]:
        """
        Calculate slide weights from an AudioBundle.
        """
        weights: Dict[int, float] = {}
        for win in bundle.slide_windows:
            if not win.segments:
                continue
            total_time = sum(max(s.end - s.start, 0.0) for s in win.segments)
            avg_emphasis = float(np.mean([s.emphasis for s in win.segments]))
            duration_score = np.tanh(total_time / 60.0)  # saturates after ~1 min
            # Boost by confidence as well
            weights[win.slide_id] = 0.3 * duration_score + 0.5 * avg_emphasis + 0.2 * float(np.clip(win.confidence, 0.0, 1.0))
        return weights
    
    def extract_audio_metadata(self, slide_num: int, segments: List[AudioSegment]) -> AudioMetadata:
        """
        Extract metadata for a specific slide from its audio segments
        
        Args:
            slide_num: Slide number
            segments: Audio segments for this slide
            
        Returns:
            AudioMetadata object
        """
        if not segments:
            return AudioMetadata(
                timestamp_start=0.0,
                timestamp_end=0.0,
                emphasis_score=0.5,
                time_allocation=0.0,
                confidence=0.0,
                keywords=[]
            )
        
        # Calculate metadata
        start_time = min(seg.start_time for seg in segments)
        end_time = max(seg.end_time for seg in segments)
        emphasis_score = np.mean([seg.emphasis_score for seg in segments])
        time_allocation = end_time - start_time
        confidence = np.mean([seg.confidence for seg in segments])
        
        # Combine keywords from all segments
        all_keywords = []
        for segment in segments:
            if segment.keywords:
                all_keywords.extend(segment.keywords)
        
        return AudioMetadata(
            timestamp_start=start_time,
            timestamp_end=end_time,
            emphasis_score=emphasis_score,
            time_allocation=time_allocation,
            confidence=confidence,
            keywords=list(set(all_keywords))  # Remove duplicates
        )
    
    def save_audio_segment(self, audio_path: str, start_time: float, 
                          end_time: float, output_path: str) -> bool:
        """
        Save a specific audio segment to a file
        
        Args:
            audio_path: Path to original audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the segment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio[start_sample:end_sample]
            
            # Save segment
            sf.write(output_path, segment, sr)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving audio segment: {e}")
            return False 

    def _compute_window_snr(self, audio_path: str, start: float, end: float) -> Tuple[float, str]:
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, offset=max(0.0, start), duration=max(0.01, end - start))
            if y.size == 0:
                return 0.0, 'Noisy'
            # RMS in dBFS-like
            rms = np.sqrt(np.mean(y ** 2)) + 1e-9
            # Estimate noise floor via 10th percentile of frame RMS
            frame = max(1, int(self.frame_length))
            hop = max(1, int(self.hop_length))
            rms_frames = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop).flatten()
            noise = np.percentile(rms_frames, 10) + 1e-9
            snr = 20.0 * np.log10(rms / noise)
            # Categorize
            if snr >= self.snr_good:
                q = 'Good'
            elif snr >= self.snr_fair:
                q = 'Fair'
            else:
                q = 'Noisy'
            return float(snr), q
        except Exception:
            return 0.0, 'Noisy' 