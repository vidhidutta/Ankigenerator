#!/usr/bin/env python3
"""
Simplified Audio Processing Module for Testing
Handles basic audio analysis without heavy dependencies
"""

import os
import tempfile
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import json
import re
from datetime import datetime
from audio_types import TranscriptSegment, SlideAudioWindow, AudioBundle

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

class SimpleAudioProcessor:
    def __init__(self, model_name: str = "mock"):
        """
        Initialize simplified audio processor for testing
        
        Args:
            model_name: Model type (currently only supports 'mock')
        """
        self.logger = logging.getLogger(__name__)
        
        # Audio analysis parameters
        self.sample_rate = 16000
        self.hop_length = 512
        self.frame_length = 2048
        
        # Emphasis detection parameters
        self.emphasis_threshold = 0.7
        self.min_segment_duration = 2.0  # seconds
        
    def transcribe_audio(self, audio_path: str) -> List[AudioSegment]:
        """
        Mock transcription for testing purposes
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of AudioSegment objects
        """
        try:
            # For testing, create mock segments based on file size
            file_size = os.path.getsize(audio_path)
            duration_estimate = file_size / (16000 * 2)  # Rough estimate
            
            # Create mock segments
            segments = []
            num_segments = max(3, int(duration_estimate / 30))  # 30-second segments
            
            for i in range(num_segments):
                start_time = i * 30.0
                end_time = (i + 1) * 30.0
                
                # Mock text based on segment number
                mock_texts = [
                    "Today we're going to talk about pharmacology, specifically beta blockers.",
                    "Beta blockers like propranolol work by blocking beta-1 receptors.",
                    "Side effects include bradycardia and fatigue.",
                    "Let's look at some clinical cases.",
                    "This is important for your clinical practice.",
                    "Remember these key points for the exam."
                ]
                
                text = mock_texts[i % len(mock_texts)]
                
                # Mock emphasis score (higher for important content)
                emphasis_score = 0.6 + (0.3 * (i % 3))  # Varies between 0.6-0.9
                
                # Extract keywords from mock text
                keywords = self._extract_keywords(text)
                
                audio_segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    confidence=0.9,  # Mock high confidence
                    emphasis_score=emphasis_score,
                    keywords=keywords
                )
                segments.append(audio_segment)
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Error in mock transcription: {e}")
            return []
    
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
        
        # Simple alignment: distribute segments across slides
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
                segment_keywords = [kw.lower() for kw in (segment.keywords or [])]
                
                # Calculate keyword overlap
                overlap = len(set(slide_keywords) & set(segment_keywords))
                if overlap > 0 or slide_num == len(slide_audio_map):  # Assign at least one segment per slide
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
        Mock audio segment saving for testing
        
        Args:
            audio_path: Path to original audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the segment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # For testing, just create a placeholder file
            with open(output_path, 'w') as f:
                f.write(f"Mock audio segment from {start_time}s to {end_time}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving audio segment: {e}")
            return False 

    def build_audio_bundle(self, audio_path: str, slide_texts: List[str]) -> AudioBundle:
        """
        Build an AudioBundle from an audio file and slide texts using mock transcription.
        """
        segments_raw = self.transcribe_audio(audio_path)
        segments: List[TranscriptSegment] = []
        for seg in segments_raw:
            start = getattr(seg, 'start_time', getattr(seg, 'start', 0.0))
            end = getattr(seg, 'end_time', getattr(seg, 'end', start))
            text = getattr(seg, 'text', '')
            emphasis = getattr(seg, 'emphasis_score', getattr(seg, 'emphasis', 0.0))
            segments.append(TranscriptSegment(start=start, end=end, text=text, emphasis=emphasis))
        
        # Simple alignment by keyword overlap
        def keyword_overlap(a: str, b: str) -> int:
            aset = set(re.findall(r"\w+", a.lower()))
            bset = set(re.findall(r"\w+", b.lower()))
            return len(aset & bset)
        
        slide_windows: List[SlideAudioWindow] = []
        for slide_id, slide_text in enumerate(slide_texts):
            segs_for_slide = [s for s in segments if keyword_overlap(s.text, slide_text) >= 1]
            if not segs_for_slide:
                continue
            starts = [s.start for s in segs_for_slide]
            ends = [s.end for s in segs_for_slide]
            window = (min(starts), max(ends))
            avg_emphasis = float(np.mean([s.emphasis for s in segs_for_slide]))
            confidence = float(np.clip(avg_emphasis, 0.0, 1.0))
            slide_windows.append(SlideAudioWindow(slide_id=slide_id, window=window, confidence=confidence, segments=segs_for_slide))
        
        return AudioBundle(audio_path=audio_path, segments=segments, slide_windows=slide_windows) 