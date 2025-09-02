#!/usr/bin/env python3
"""
Modified Gradio Interface with Audio Integration
Uses port 7864 by default to avoid conflicts
"""

import gradio as gr
import yaml
import os
import tempfile
import shutil
import sys
import time
from PIL import Image, ImageDraw
from ankigenerator.core.image_occlusion import detect_text_regions, mask_regions, make_qmask, make_omask
import hashlib
import logging
import glob

# Set up logging to a file
logging.basicConfig(filename="debug.log", level=logging.DEBUG)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Import our existing flashcard generator functions
from flashcard_generator import (
    extract_text_from_pptx,
    extract_images_from_pptx,
    filter_slides,
    should_skip_slide,
    generate_enhanced_flashcards_with_progress,
    remove_duplicate_flashcards,
    parse_flashcards,
    export_flashcards_to_apkg,
    stringify_dict,
    clean_cloze_text,  # NEW: for nicer cloze previews
    PROMPT_TEMPLATE,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    CATEGORY,
    EXAM,
    ORGANISATION,
    FEATURES,
    FLASHCARD_TYPE,
    ANSWER_FORMAT,
    CLOZE,
    OPENAI_API_KEY,
    Flashcard,
    QualityController,
    filter_relevant_images_for_occlusion,
    batch_generate_image_occlusion_flashcards
)

def find_audio_file(audio_path):
    """
    Handle audio file path that might be a directory and find the actual audio file.
    
    Args:
        audio_path: Path to audio file or directory containing audio files
        
    Returns:
        Path to the actual audio file, or None if not found
    """
    if not audio_path:
        return None
        
    # If it's already a file, return it
    if os.path.isfile(audio_path):
        return audio_path
        
    # If it's a directory, search for audio files
    if os.path.isdir(audio_path):
        # Supported audio formats
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']
        
        # Search for audio files in the directory
        for ext in audio_extensions:
            pattern = os.path.join(audio_path, f"*{ext}")
            audio_files = glob.glob(pattern)
            if audio_files:
                # Return the first audio file found
                return audio_files[0]
        
        # If no audio files found with common extensions, try any file
        all_files = os.listdir(audio_path)
        for file in all_files:
            file_path = os.path.join(audio_path, file)
            if os.path.isfile(file_path):
                # Check if it might be an audio file by extension
                _, ext = os.path.splitext(file.lower())
                if ext in audio_extensions:
                    return file_path
        
        print(f"[WARN] No audio files found in directory: {audio_path}")
        return None
    
    print(f"[WARN] Audio path is neither file nor directory: {audio_path}")
    return None

def sanitize_error_message(error_msg):
    """
    Convert technical error messages to user-friendly messages.
    
    Args:
        error_msg: Raw error message
        
    Returns:
        User-friendly error message
    """
    if not error_msg:
        return "An unknown error occurred."
    
    # Common error patterns and their user-friendly equivalents
    error_patterns = {
        "[Errno 21] Is a directory": "The system encountered a directory where it expected a file. This usually happens with audio files.",
        "[Errno 2] No such file or directory": "A required file was not found. Please check your uploads.",
        "expected string or bytes-like object, got 'dict'": "There was an issue processing the content format.",
        "too many values to unpack": "There was an issue with data processing. Please try again.",
        "cannot find loader": "Some images in your PowerPoint are in an unsupported format. The system will skip these and continue.",
        "ModuleNotFoundError": "A required component is missing. Please check your installation.",
        "API key not found": "Please check your API configuration.",
        "timeout": "The request took too long. Please try again.",
        "rate limit": "Too many requests. Please wait a moment and try again."
    }
    
    for pattern, friendly_msg in error_patterns.items():
        if pattern in error_msg:
            return friendly_msg
    
    # If no specific pattern matches, return a generic message
    return "An error occurred during processing. Please check your inputs and try again."

def validate_flashcard(flashcard):
    """
    Validate a flashcard object and return whether it should be included.
    
    Args:
        flashcard: Flashcard object or dict
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if flashcard is None:
        return False, "Flashcard is None"
    
    # Handle Flashcard objects
    if hasattr(flashcard, 'question') and hasattr(flashcard, 'answer'):
        question = flashcard.question
        answer = flashcard.answer
    # Handle dict objects
    elif isinstance(flashcard, dict):
        question = flashcard.get('question', '')
        answer = flashcard.get('answer', '')
    else:
        return False, f"Unknown flashcard type: {type(flashcard)}"
    
    # Check if both question and answer are non-empty
    if not question or not answer:
        return False, "Empty question or answer"
    
    if not question.strip() or not answer.strip():
        return False, "Question or answer contains only whitespace"
    
    return True, "Valid flashcard"

def flatten_flashcard_list(flashcards):
    """
    Flatten a list of flashcards and filter out invalid entries.
    
    Args:
        flashcards: List of flashcards (may contain nested lists or None values)
        
    Returns:
        Flat list of valid flashcards
    """
    flat_list = []
    skipped_count = 0
    
    def process_item(item):
        nonlocal skipped_count
        
        if item is None:
            skipped_count += 1
            print(f"[WARN] Skipped None flashcard entry")
            return
        
        if isinstance(item, list):
            # Recursively process nested lists
            for sub_item in item:
                process_item(sub_item)
        else:
            # Validate individual flashcard
            is_valid, reason = validate_flashcard(item)
            if is_valid:
                flat_list.append(item)
            else:
                skipped_count += 1
                print(f"[WARN] Skipped invalid flashcard: {reason}")
    
    # Process all items
    for item in flashcards:
        process_item(item)
    
    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} invalid flashcards")
    
    return flat_list

def create_occlusion(image_path, occlusion_boxes, output_path):
    img = Image.open(image_path).convert("RGB")
    # Instead of white rectangles, generate q/a masks and save individual previews
    for idx, box in enumerate(occlusion_boxes):
        # Generate red-box masks
        q = make_qmask(img, box)
        o = make_omask(img, box)
        # Save previews for Gradio to display
        q.save(os.path.join(output_path, f"qmask_{idx}.png"))
        o.save(os.path.join(output_path, f"omask_{idx}.png"))

def run_flashcard_generation(
    pptx_file,
    audio_file,  # New audio file parameter
    flashcard_level,
    question_style,
    use_cloze,
    # Removed parameters related to extra materials
    content_images,
    content_notes,
    enable_image_occlusion,
    occlusion_mode,
    occlusion_image,
    conf_threshold,
    max_masks_per_image,
    progress=gr.Progress(),
    card_types=None  # New parameter to handle multiple card types (list of selected card types)
):
    """
    Main function to generate flashcards from uploaded PowerPoint file
    """
    if not pptx_file:
        return "Please upload a PowerPoint file.", "No file uploaded", None
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set. Please check your .env file.", "API key not found", None
    
    # Initialize occluded_path before the try block
    occluded_path = None

    try:
        progress(0, desc="Starting flashcard generation...")
        status_msg = "Starting flashcard generation..."
        
        # Handle audio file path
        audio_path = None
        if audio_file:
            audio_path = find_audio_file(audio_file.name)
            if not audio_path:
                print(f"[WARN] Could not find audio file in: {audio_file.name}")
            else:
                print(f"[INFO] Using audio file: {audio_path}")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract text and images from PowerPoint
            progress(0.1, desc="Extracting content from PowerPoint...")
            slide_texts = extract_text_from_pptx(pptx_file)
            slide_images = extract_images_from_pptx(pptx_file, temp_dir)
            
            if not slide_texts:
                return "No text content found in the PowerPoint file.", "No content extracted", None
            
            progress(0.2, desc="Filtering relevant slides...")
            # Filter slides based on content relevance
            filtered_slide_texts, kept_indices = filter_slides(slide_texts, slide_images)
            
            if not filtered_slide_texts:
                return "No relevant content found after filtering.", "No relevant content", None
            
            # Filter slide_images to match the kept slides
            filtered_slide_images = [slide_images[i] for i in kept_indices] if kept_indices else slide_images
            
            progress(0.3, desc="Generating flashcards...")
            # Build AudioBundle if audio is provided
            audio_bundle = None
            if audio_path:
                try:
                    from audio_processor import AudioProcessor
                    ap = AudioProcessor(model_name="base")
                    audio_bundle = ap.build_audio_bundle(audio_path, filtered_slide_texts, diarization_enabled=diarization_cb.value, alignment_mode=alignment_mode.value)
                except RuntimeError as e:
                    user_msg = str(e)
                    return f"⚠️ Audio processing problem: {user_msg}", f"⚠️ Audio issue: {user_msg}", None
                except FileNotFoundError:
                    return "⚠️ Audio file unreadable.", "⚠️ Audio file unreadable.", None
                except Exception as e:
                    return f"⚠️ Audio error: {e}", f"⚠️ Audio error: {e}", None
            
            # Generate flashcards with progress tracking and audio context
            flashcard_previews = generate_enhanced_flashcards_with_progress(
                filtered_slide_texts,
                filtered_slide_images,  # Use filtered slide images
                OPENAI_API_KEY,
                MODEL_NAME,
                MAX_TOKENS,
                TEMPERATURE,
                progress=progress,
                use_cloze=use_cloze,
                question_style=question_style,
                audio_bundle=audio_bundle
            )
            
            # If audio is available and make clips enabled, create clips and attach to cards
            clips_map = {}
            if audio_bundle and make_clips_enabled:
                try:
                    from audio_processor import AudioProcessor
                    from flashcard_generator import Flashcard, AudioMetadataForCard
                    ap = AudioProcessor(model_name="base")
                    # Auto mode default: adaptive selection with boundary snapping; manual only when advanced used
                    try:
                        auto_mode = not bool(locals().get('use_manual_clip_len', False))
                    except Exception:
                        auto_mode = True
                    min_s = None if auto_mode else int(clip_length_s_val)
                    max_s = None if auto_mode else int(clip_length_s_val)
                    clips_map, warnings_map = ap.create_audio_clips_for_bundle(audio_bundle, output_dir=temp_dir, per_slide_max=int(max_clips_val), min_clip_sec=min_s, max_clip_sec=max_s)
                    # Attach to Flashcard objects by slide_number and propagate confidence
                    wins_by_id = {w.slide_id: w for w in getattr(audio_bundle, 'slide_windows', []) or []}
                    def attach_audio(cards):
                        out = []
                        for c in cards:
                            if isinstance(c, list):
                                out.append(attach_audio(c))
                            else:
                                try:
                                    if isinstance(c, Flashcard) and c.slide_number in clips_map:
                                        files = clips_map[c.slide_number]
                                        c.audio_metadata = AudioMetadataForCard(audio_files=files)
                                    if isinstance(c, Flashcard) and c.slide_number in wins_by_id:
                                        try:
                                            c.confidence = float(wins_by_id[c.slide_number].confidence)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                out.append(c)
                        return out
                    flashcard_previews = attach_audio(flashcard_previews)
                except Exception as e:
                    print(f"[WARN] Failed to create/attach audio clips: {e}")
            
            # Build preview table text
            preview_table = ""
            try:
                if audio_bundle:
                    wins = audio_bundle.slide_windows
                    preview_table += "\n\nSlide | Align conf | Window | #clips | Clip times | Emphasis avg\n"
                    preview_table += "----- | ---------- | ------ | ------ | ---------- | ------------\n"
                    for w in wins[:min(20, len(wins))]:
                        ws, we = w.window
                        times = []
                        if w.slide_id in clips_map:
                            for f in clips_map[w.slide_id]:
                                import re
                                m = re.search(r"_(\\d+)-(\\d+)\\.mp3$", f)
                                if m:
                                    s_ms, e_ms = int(m.group(1)), int(m.group(2))
                                    times.append(f"{s_ms//1000:02d}:{s_ms%1000:03d}-{e_ms//1000:02d}:{e_ms%1000:03d}")
                        import numpy as _np
                        avg_emp = 0.0
                        if w.segments:
                            avg_emp = float(_np.mean([getattr(s,'emphasis',0.0) for s in w.segments]))
                        row = f"{w.slide_id+1} | {w.confidence:.2f} | {int(ws//60):02d}:{int(ws%60):02d}-{int(we//60):02d}:{int(we%60):02d} | {len(clips_map.get(w.slide_id, []))} | {', '.join(times)} | {avg_emp:.2f}"
                        if w.confidence < 0.45:
                            row = "⚠️ " + row + "  (Low alignment confidence — consider different mode)"
                        preview_table += row + "\n"
            except Exception:
                pass
            
            if not flashcard_previews:
                return "No flashcards generated. Please check your content and settings.", "No flashcards generated", None
            
            # Flatten and validate flashcards
            progress(0.7, desc="Validating and flattening flashcards...")
            all_flashcards = flatten_flashcard_list(flashcard_previews)
            
            if not all_flashcards:
                return "No valid flashcards generated. Please check your content and settings.", "No valid flashcards", None
            
            progress(0.8, desc="Removing duplicates and optimizing...")
            # Remove duplicate flashcards
            unique_flashcards = remove_duplicate_flashcards(all_flashcards)
            
            progress(0.9, desc="Exporting to Anki format...")
            try:
                # Export to APKG format
                final_apkg_path = export_flashcards_to_apkg(unique_flashcards, temp_dir)
            except Exception as e:
                return f"⚠️ Export failed: {e}", f"⚠️ Media/export write failure: {e}", None
            
            # Handle image occlusion if enabled
            if enable_image_occlusion and filtered_slide_images:
                progress(0.85, desc="Processing image occlusion...")
                # Filter relevant images for occlusion
                relevant_images = filter_relevant_images_for_occlusion(
                    filtered_slide_images, 
                    filtered_slide_texts, 
                    OPENAI_API_KEY, 
                    MODEL_NAME
                )
                
                if relevant_images:
                    # Get Google Vision credentials path from config
                    io_config = config.get("image_occlusion", {})
                    credentials_path = io_config.get("google_credentials_path", "~/anki-flashcard-generator/google_credentials.json")
                    
                    # Flatten the list of lists into a single list of image paths
                    flat_relevant_images = []
                    for slide_images in relevant_images:
                        flat_relevant_images.extend(slide_images)
                    
                    flashcard_entries = batch_generate_image_occlusion_flashcards(
                        image_paths=flat_relevant_images,
                        export_dir=os.path.join(temp_dir, "occlusion_flashcards"),
                        conf_threshold=conf_threshold,
                        max_masks=max_masks_per_image,
                        use_google_vision=True,  # Use Google Vision API by default
                        credentials_path=os.path.expanduser(credentials_path)
                    )
                    logging.debug(f"[DEBUG] Generated flashcard entries: {flashcard_entries}")  # Debug statement to verify flashcard entries
                    
                    # Add image occlusion entries to the main flashcard list
                    if flashcard_entries:
                        # Validate and flatten image occlusion entries
                        valid_occlusion_entries = flatten_flashcard_list(flashcard_entries)
                        if valid_occlusion_entries:
                            unique_flashcards.extend(valid_occlusion_entries)
                            logging.debug(f"[DEBUG] Added {len(valid_occlusion_entries)} valid image occlusion flashcards to main list")
                        else:
                            logging.debug("[DEBUG] No valid image occlusion flashcards found")
                else:
                    logging.debug("[DEBUG] No images found for occlusion flashcards.")

            # Count different types of flashcards
            text_cards = sum(1 for card in unique_flashcards if hasattr(card, 'card_type') and card.card_type == 'text')
            occlusion_cards = sum(1 for card in unique_flashcards if hasattr(card, 'card_type') and card.card_type == 'occlusion')
            audio_cards = sum(1 for card in unique_flashcards if hasattr(card, 'audio_metadata') and card.audio_metadata)
            
            # Set final status message now that all processing is complete
            final_status = (
                f"✅ Complete! Generated {len(unique_flashcards)} quality-controlled flashcards "
                f"from {len(filtered_slide_texts)} slides."
            )
            
            # Print summary to terminal
            print(f"\n📊 Flashcard Generation Summary:")
            print(f"Flashcards generated: {len(unique_flashcards)} (Text: {text_cards}, Image Occlusion: {occlusion_cards}, Audio: {audio_cards})")
            print(f"Slides processed: {len(filtered_slide_texts)}")
            if audio_path:
                print(f"Audio file used: {os.path.basename(audio_path)}")
            
            # If available, show alignment confidences for first few slides
            try:
                if 'audio_bundle' in locals() and audio_bundle and getattr(audio_bundle, 'slide_windows', None):
                    wins = audio_bundle.slide_windows
                    print("\n🎯 Alignment confidence (first 5 slides):")
                    for w in wins[:5]:
                        print(f"  Slide {w.slide_id+1}: {w.window[0]:.1f}-{w.window[1]:.1f}s  conf={w.confidence:.2f}")
            except Exception:
                pass
            
            # Create flashcard summary
            flashcard_summary = f"Successfully generated {len(unique_flashcards)} flashcards!\n\n"
            flashcard_summary += "Sample flashcards:\n"
            for i, preview in enumerate(unique_flashcards[:5]):  # Show first 5
                flashcard_summary += f"\n{i+1}. {preview}\n"
            
            # Append alignment confidence table to the summary if available
            try:
                if 'audio_bundle' in locals() and audio_bundle and getattr(audio_bundle, 'slide_windows', None):
                    wins = audio_bundle.slide_windows
                    if wins:
                        flashcard_summary += "\nAlignment confidence (first 10 slides):\n"
                        flashcard_summary += "Slide | Start(s) | End(s) | Confidence\n"
                        flashcard_summary += "----- | -------- | ------ | ----------\n"
                        for w in wins[:10]:
                            flashcard_summary += f"{w.slide_id+1} | {w.window[0]:.1f} | {w.window[1]:.1f} | {w.confidence:.2f}\n"
            except Exception:
                pass
            
            if len(unique_flashcards) > 5:
                flashcard_summary += f"\n... and {len(unique_flashcards) - 5} more flashcards"
            if preview_table:
                flashcard_summary += preview_table
 
            return flashcard_summary, final_status, final_apkg_path
            
    except Exception as e:
        error_msg = str(e)
        user_friendly_error = sanitize_error_message(error_msg)
        
        # Log the full error for debugging
        print(f"[ERROR] Full error: {error_msg}")
        print(f"[ERROR] User-friendly message: {user_friendly_error}")
        
        return user_friendly_error, f"❌ Error: {user_friendly_error}", None

# Create the APEX.MED interface
def create_interface():
    # Custom CSS for dark theme
    custom_css = """
    .apex-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .apex-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    .apex-title {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .upload-section {
        background: rgba(45, 45, 45, 0.8);
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    
    .upload-button {
        background: linear-gradient(45deg, #ff4444, #cc0000);
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .upload-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 68, 68, 0.3);
    }
    
    .generate-button {
        background: linear-gradient(45deg, #444, #666);
        border: none;
        border-radius: 8px;
        padding: 12px 25px;
        color: white;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 15px;
    }
    
    .generate-button:hover {
        background: linear-gradient(45deg, #555, #777);
        transform: translateY(-1px);
    }
    
    .settings-panel {
        background: rgba(45, 45, 45, 0.9);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #333;
        height: fit-content;
    }
    
    .settings-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .setting-group {
        margin-bottom: 20px;
    }
    
    .setting-label {
        font-size: 1rem;
        font-weight: bold;
        color: #ccc;
        margin-bottom: 8px;
    }
    
    .footer {
        position: fixed;
        bottom: 20px;
        left: 20px;
        display: flex;
        gap: 20px;
    }
    
    .footer-link {
        color: #888;
        text-decoration: none;
        font-size: 0.9rem;
        transition: color 0.3s ease;
    }
    
    .footer-link:hover {
        color: #ccc;
    }
    """
    
    with gr.Blocks(css=custom_css, title="APEX.MED - Anki Flashcard Generator") as interface:
        gr.HTML("""
        <div class="apex-container">
            <div class="apex-header">
                <h1 class="apex-title">APEX.MED</h1>
                <p style="color: #ccc; font-size: 1.2rem; margin: 10px 0;">
                    Advanced Anki Flashcard Generator for Medical Students
                </p>
            </div>
        """)
        
        with gr.Row():
            # Left Section - Upload Area
            with gr.Column(scale=2):
                with gr.Column(elem_classes=["upload-section"]):
                    gr.HTML("""
                        <h2 style="color: white; margin-bottom: 20px;">
                            Upload to <span style="color: #ff4444;">uplearn</span>
                        </h2>
                        """)

                    # Upload button with custom styling
                    pptx_file = gr.File(
                        label="",
                        file_types=[".pptx"],
                        type="filepath",
                        elem_classes=["upload-button"],
                        container=False
                    )

                    # Audio file upload (optional)
                    gr.HTML("""
                    <div style="margin: 20px 0; padding: 15px; background: rgba(255, 68, 68, 0.1); border-radius: 10px; border: 1px solid rgba(255, 68, 68, 0.3);">
                        <h3 style="color: #ff4444; margin: 0 0 10px 0; font-size: 1.1rem;">🎵 Audio Enhancement (Optional)</h3>
                        <p style="color: #ccc; margin: 0; font-size: 0.9rem;">
                            Upload lecture audio to enhance flashcards with emphasis detection and time allocation analysis.
                        </p>
                    </div>
                    """)

                    audio_file = gr.File(
                        label="Upload Lecture Audio (Optional)",
                        file_types=[".mp3", ".wav", ".m4a", ".flac"],
                        type="filepath",
                        container=False
                    )
                    
                    # Diarization checkbox
                    diarization_cb = gr.Checkbox(label="Use diarization (ignore non-lecturer)", value=False)
                    make_clips_cb = gr.Checkbox(label="Make audio clips (Auto length)", value=True)
                    # Load audio clip bounds from config
                    try:
                        with open('config.yaml','r') as _cf:
                            _cfg = yaml.safe_load(_cf) or {}
                    except Exception:
                        _cfg = {}
                    ap_cfg = _cfg.get('audio_processing', {})
                    cfg_min = float(ap_cfg.get('clip_min_s', 6.0))
                    cfg_max = float(ap_cfg.get('clip_max_s', 15.0))
                    cfg_tol = float(ap_cfg.get('clip_duration_tolerance_s', 0.25))
                    # Advanced section for manual controls
                    with gr.Accordion("Advanced audio clipping", open=False):
                        use_manual_clip_len = gr.Checkbox(label="Use manual clip length", value=False)
                        clip_length_s = gr.Slider(cfg_min, cfg_max, value=min(max(7, cfg_min), max(7, min(cfg_max, 12)), step=1, label=f"Manual clip length (seconds) [{int(cfg_min)}–{int(cfg_max)}]")
                        max_clips = gr.Slider(0, 2, value=1, step=1, label="Max clips per slide")
                    alignment_mode = gr.Dropdown(choices=["keyword", "semantic+keyword"], value="semantic+keyword", label="Alignment mode")
                    low_conf_only = gr.Checkbox(label="Show only low-confidence (<0.5)", value=False)

                    # Generate button
                    generate_btn = gr.Button(
                        "Generate",
                        elem_classes=["generate-button"],
                        size="lg"
                    )
                
                # Right Section - Settings Panel
                with gr.Column(scale=1):
                    with gr.Column(elem_classes=["settings-panel"]):
                        gr.HTML('<div class="settings-title">Settings</div>')
                        
                        # Level Selection
                        with gr.Column(elem_classes=["setting-group"]):
                            gr.HTML('<div class="setting-label">Level</div>')
                            flashcard_level = gr.Radio(
                                choices=["L1", "L2", "Both"],
                                value="L1",
                                label="",
                                container=False,
                                elem_classes=["level-radio"]
                            )
                        
                        # Type Selection
                        with gr.Column(elem_classes=["setting-group"]):
                            gr.HTML('<div class="setting-label">Type</div>')
                            card_type = gr.CheckboxGroup(
                                choices=["Basic", "Cloze", "Image Occlusion"],
                                value=["Basic"],
                                label="",
                                container=False,
                                elem_classes=["type-checkbox"]
                            )
                        
                        # Question Style
                        with gr.Column(elem_classes=["setting-group"]):
                            gr.HTML('<div class="setting-label">Question Style</div>')
                            question_style = gr.Dropdown(
                                choices=["Word for Word", "Elaborated", "Simplified"],
                                value="Word for Word",
                                label="",
                                container=False,
                                elem_classes=["style-dropdown"]
                            )
                        
                        # Content Sources
                        with gr.Column(elem_classes=["setting-group"]):
                            gr.HTML('<div class="setting-label">Content Sources</div>')
                            content_text = gr.Checkbox(
                                label="Text",
                                value=True,
                                container=False,
                                elem_classes=["content-checkbox"]
                            )
                            content_images = gr.Checkbox(
                                label="Images",
                                value=True,
                                container=False,
                                elem_classes=["content-checkbox"]
                            )
                            content_notes = gr.Checkbox(
                                label="Speaker Notes",
                                value=True,
                                container=False,
                                elem_classes=["content-checkbox"]
                            )
                        
                        # Image Occlusion Settings (hidden by default)
                        conf_threshold_slider = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=config['conf_threshold'],
                            step=1,
                            label="Confidence Threshold",
                            visible=False
                        )
                        
                        max_masks_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=config['max_masks_per_image'],
                            step=1,
                            label="Max Masks per Image",
                            visible=False
                        )
            
            # Output Section
            with gr.Column(scale=1):
                flashcard_output = gr.Textbox(
                    label="Generated Flashcards",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                
                apkg_download = gr.File(
                    label="Download Anki Deck",
                    interactive=False
                )
                
                # Alignment preview and audio
                align_table = gr.Dataframe(headers=["Slide #","Title","Confidence","Semantic","Keyword","Quality","Window","Mode","#Clips","Clip times","Play"], label="Slide Alignment Preview", interactive=False)
                preview_audio = gr.Audio(label="Preview Clip", interactive=False)
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <a href="#" class="footer-link">Documentation</a>
            <a href="#" class="footer-link">Support</a>
            <a href="#" class="footer-link">About</a>
        </div>
        """)
        
        # Event handlers
        def update_occlusion_settings_visibility(card_type):
            if "Image Occlusion" in card_type:
                return gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)

        # Update image occlusion settings visibility based on card type
        card_type.change(
            fn=update_occlusion_settings_visibility,
            inputs=[card_type],
            outputs=[conf_threshold_slider, max_masks_slider]
        )

        # Update the function to receive the new parameter structure
        def run_flashcard_generation_updated(
            pptx_file,
            audio_file,  # Add audio file parameter
            flashcard_level,
            card_type,
            question_style,
            content_images,
            content_notes,
            conf_threshold,
            max_masks_per_image,
            low_conf_only,
            make_clips_enabled,
            clip_length_s_val,
            max_clips_val,
            progress=gr.Progress()
        ):
            # Map the new UI parameters to the existing function parameters
            # Handle multiple card types - generate all selected types
            use_cloze = "Cloze Deletion" in card_type
            enable_image_occlusion = "Image Occlusion" in card_type
            occlusion_mode = "AI-assisted (AI detects labels/regions)" if enable_image_occlusion else None
            occlusion_image = None  # Not used in AI-assisted mode
            
            # If no card types selected, default to Basic
            if not card_type:
                card_type = ["Basic"]
            
            res = run_flashcard_generation(
                pptx_file,
                audio_file,  # Add audio file parameter
                flashcard_level,  # This replaces the old flashcard_type parameter
                question_style,
                use_cloze,
                content_images,
                content_notes,
                enable_image_occlusion,
                occlusion_mode,
                occlusion_image,
                conf_threshold,
                max_masks_per_image,
                progress,
                card_type  # Pass the selected card types
            )
            # Ensure outputs length
            if isinstance(res, tuple):
                out = list(res)
            else:
                out = [res]
            while len(out) < 4:
                out.append([] if len(out) == 2 else None)
            # Create audio clips if requested
            try:
                flashcard_previews = out[0]
                audio_bundle = None  # simple variant may not expose; skip
                if audio_bundle and make_clips_enabled:
                    from audio_processor import AudioProcessor
                    from flashcard_generator import Flashcard, AudioMetadataForCard
                    ap = AudioProcessor(model_name="base")
                    auto_mode = True
                    min_s = None if auto_mode else int(clip_length_s_val)
                    max_s = None if auto_mode else int(clip_length_s_val)
                    clips_map, warnings_map = ap.create_audio_clips_for_bundle(audio_bundle, output_dir=temp_dir, per_slide_max=int(max_clips_val), min_clip_sec=min_s, max_clip_sec=max_s)
                    def attach_audio(cards):
                        out_cards = []
                        for c in cards:
                            if isinstance(c, list):
                                out_cards.append(attach_audio(c))
                            else:
                                try:
                                    if isinstance(c, Flashcard) and c.slide_number in clips_map:
                                        files = clips_map[c.slide_number]
                                        c.audio_metadata = AudioMetadataForCard(audio_files=files)
                                except Exception:
                                    pass
                                out_cards.append(c)
                        return out_cards
                    out[0] = attach_audio(flashcard_previews)
            except Exception:
                pass
            return tuple(out[:4])

        # Update the function call to use the new parameters
        generate_btn.click(
            fn=run_flashcard_generation_updated,
            inputs=[
                pptx_file,
                audio_file,  # Add audio file input
                flashcard_level,
                card_type,
                question_style,
                content_images,
                content_notes,
                conf_threshold_slider,
                max_masks_slider,
                low_conf_only,
                make_clips_cb,
                clip_length_s,
                max_clips
            ],
            outputs=[
                flashcard_output,
                status_output,
                align_table,
                preview_audio
            ]
        )
        
    return interface

if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Use port 7864 by default to avoid conflicts
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7864))
    for arg in sys.argv:
        if arg.startswith("--server_port"):
            parts = arg.split()
            if len(parts) == 2 and parts[1].isdigit():
                port = int(parts[1])
            elif '=' in arg:
                port = int(arg.split('=')[1])

    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        debug=True
    ) 