# Audio Integration for Enhanced Flashcard Generation

## Overview

The audio integration feature enhances your Anki flashcard generation by analyzing lecture audio alongside PowerPoint slides. This allows the AI to understand the lecturer's emphasis, time allocation, and contextual nuances to create more contextually aware flashcards.

## How It Works

### 1. Audio Processing Pipeline

When you upload both a PowerPoint file and audio file, the system:

1. **Transcribes the audio** using OpenAI's Whisper model
2. **Analyzes emphasis patterns** by detecting:
   - Pitch variations (higher pitch = emphasis)
   - Volume changes (louder speech = emphasis)
   - Speaking rate (slower speech = important content)
3. **Aligns audio segments with slides** using keyword matching
4. **Calculates content weights** based on time allocation and emphasis
5. **Enhances flashcard generation** with audio-derived context

### 2. Emphasis Detection

The system analyzes three key audio features:

- **Pitch Analysis**: Higher fundamental frequency indicates emphasis
- **Volume Analysis**: RMS energy levels show volume emphasis
- **Speaking Rate**: Syllable detection reveals pacing changes

### 3. Content Weighting

Each slide receives a weight based on:
- **Time Allocation**: How long the lecturer spent on the content
- **Emphasis Score**: How much the lecturer emphasized the content
- **Confidence**: Transcription accuracy for that segment

## Benefits

### For Medical Students

1. **Context-Aware Flashcards**: Cards reflect what the lecturer actually emphasized
2. **Time-Based Prioritization**: Content that received more lecture time gets more detailed cards
3. **Emphasis Recognition**: Important concepts that the lecturer stressed are highlighted
4. **Better Retention**: Cards aligned with lecture flow improve learning efficiency

### Example Scenarios

**Scenario 1: Drug Mechanism**
- Lecturer spends 5 minutes explaining a drug mechanism
- System generates detailed cards about the mechanism
- Cards include clinical implications mentioned in lecture

**Scenario 2: Quick Reference**
- Lecturer briefly mentions a side effect
- System creates concise recall cards
- Cards focus on key facts without excessive detail

## Usage

### Upload Process

1. **Upload PowerPoint**: Your existing .pptx file
2. **Upload Audio** (Optional): Supported formats: .mp3, .wav, .m4a, .flac
3. **Configure Settings**: Choose your preferred flashcard settings
4. **Generate**: The system processes both inputs together

### Audio Requirements

- **Format**: MP3, WAV, M4A, or FLAC
- **Quality**: Clear speech, minimal background noise
- **Duration**: Should match lecture length
- **Language**: English (currently supported)

### Processing Time

- **Audio Processing**: 2-5 minutes for 1-hour lecture
- **Transcription**: Depends on audio length and quality
- **Analysis**: Real-time emphasis detection
- **Flashcard Generation**: Enhanced with audio context

## Technical Architecture

### Core Components

1. **AudioProcessor Class** (`audio_processor.py`)
   - Handles audio transcription with Whisper
   - Performs emphasis detection using librosa
   - Manages temporal alignment with slides

2. **Enhanced Flashcard Generation**
   - Integrates audio metadata into flashcard objects
   - Applies weighting based on audio analysis
   - Enhances prompts with audio context

3. **UI Integration** (`gradio_interface.py`)
   - Optional audio file upload
   - Progress tracking for audio processing
   - Enhanced status reporting

### Data Flow

```
PowerPoint + Audio → Audio Processing → Enhanced Analysis → Context-Aware Flashcards
     ↓                    ↓                    ↓                    ↓
Slide Text         Transcription        Emphasis Detection    Weighted Generation
Slide Images       Temporal Alignment   Content Weighting    Audio Metadata
```

## Configuration

### Audio Processing Settings (`config.yaml`)

```yaml
audio_processing:
  enabled: true
  whisper_model: "base"  # Model size for transcription
  emphasis_detection:
    pitch_weight: 0.4    # Weight for pitch analysis
    volume_weight: 0.4   # Weight for volume analysis
    rate_weight: 0.2     # Weight for speaking rate
    emphasis_threshold: 0.7  # Threshold for emphasis detection
  alignment:
    method: "content_based"  # How to align audio with slides
  weighting:
    max_weight_multiplier: 2.0  # Maximum emphasis multiplier
```

### New Configuration Keys

Add to `audio_processing` in `config.yaml`:

- clip_min_s: Minimum clip length in seconds (default 6)
- clip_max_s: Maximum clip length in seconds (default 15)
- clip_duration_tolerance_s: Post-write duration tolerance (default 0.25)

These bounds drive the UI sliders and are enforced during clipping. Clips outside bounds are deleted and noted in the UI/report.

### Runtime Warnings in UI/Report

- "Skipped: window X.Ys < min Zs": the selected window was too short.
- "Deleted: clip X.Ys out of bounds [min,max]": generated clip failed post-write duration validation.
- Orphan media warning: media files present but not referenced by any note.

The slide preview table shows Confidence, Semantic, Keyword, the selected window, and clip times, with low-confidence rows flagged.

### Flashcard Enhancement

The system enhances flashcards with:

- **Emphasis Weight**: 0.5-2.0 multiplier based on lecturer emphasis
- **Time Allocation**: Seconds spent on each concept
- **Audio Metadata**: Timestamps and confidence scores
- **Context Prompts**: Enhanced AI prompts with audio context

## Installation

### Dependencies

Add these to your `requirements.txt`:

```txt
# Audio processing dependencies
librosa==0.10.1
whisper==1.1.10
soundfile==0.12.1
torch==2.1.0
torchaudio==2.1.0
```

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download Whisper models (first run will download automatically):
   ```python
   import whisper
   model = whisper.load_model("base")
   ```

3. Configure audio settings in `config.yaml`

## Advanced Features

### Emphasis Detection Algorithm

The system uses a weighted combination of:

1. **Pitch Analysis**: Fundamental frequency detection
2. **Volume Analysis**: RMS energy calculation
3. **Speaking Rate**: Syllable onset detection

Formula: `emphasis_score = 0.4 * pitch_score + 0.4 * volume_score + 0.2 * rate_score`

### Content Alignment

Two alignment methods:

1. **Content-Based**: Keyword overlap between audio and slide text
2. **Timing-Based**: Known slide durations (if available)

### Weighting System

Content importance is calculated as:
```
weight = (0.4 * time_weight + 0.4 * emphasis + 0.2 * confidence) * 2
```

## Troubleshooting

### Common Issues

1. **Audio Not Processing**
   - Check file format (MP3, WAV, M4A, FLAC)
   - Ensure clear audio quality
   - Verify file size (max 100MB recommended)

2. **Poor Alignment**
   - Audio should match lecture content
   - Check for background noise
   - Ensure audio covers all slides

3. **Slow Processing**
   - Use smaller Whisper model ("tiny" or "base")
   - Reduce audio quality if needed
   - Check available RAM

### Performance Tips

- Use "base" Whisper model for good balance of speed/accuracy
- Process audio in chunks for very long lectures
- Ensure sufficient disk space for temporary files

## Future Enhancements

### Planned Features

1. **Multi-Language Support**: Additional language transcription
2. **Speaker Diarization**: Multiple speaker detection
3. **Emotion Detection**: Sentiment analysis for emphasis
4. **Audio Flashcards**: Include audio clips in cards
5. **Real-time Processing**: Live lecture processing

### Integration Opportunities

1. **Video Support**: Extract audio from video files
2. **Live Streaming**: Real-time lecture processing
3. **Collaborative Features**: Shared audio analysis
4. **Advanced Analytics**: Detailed emphasis reports

## API Reference

### AudioProcessor Class

```python
from audio_processor import AudioProcessor

# Initialize processor
processor = AudioProcessor(model_name="base")

# Transcribe audio
segments = processor.transcribe_audio("lecture.mp3")

# Align with slides
slide_audio_map = processor.align_with_slides(segments, slide_texts)

# Calculate weights
weights = processor.calculate_content_weights(slide_audio_map)
```

### Enhanced Flashcard Generation

```python
from flashcard_generator import generate_enhanced_flashcards_with_progress
from audio_processor import AudioProcessor

# Build AudioBundle first
ap = AudioProcessor(model_name="base")
audio_bundle = ap.build_audio_bundle("lecture.mp3", slide_texts)

# Generate with audio context
flashcards = generate_enhanced_flashcards_with_progress(
    slide_texts, slide_images, api_key, model, max_tokens, temperature,
    audio_bundle=audio_bundle
)
```

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies
3. Set up audio processing environment
4. Test with sample audio files

### Testing

- Use sample lectures for testing
- Verify emphasis detection accuracy
- Test alignment with various slide types
- Validate flashcard quality improvements

## Support

For issues related to audio processing:

1. Check the troubleshooting section
2. Verify audio file format and quality
3. Review configuration settings
4. Check system resources and dependencies

---

*This audio integration feature enhances the traditional slide-based flashcard generation by incorporating the lecturer's actual emphasis and time allocation, resulting in more contextually relevant and effective study materials.* 