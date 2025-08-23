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
import re
import numpy as np
import pytesseract
import importlib.util

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
from providers.occlusion_pipeline import build_occlusion_items_for_image
from providers.pipeline import detect_segment_rank
from providers.segment_provider import SAMProvider
from providers.detect_provider import GroundingDINOProvider
from providers.vlm_provider import LocalQwen2VLProvider, LocalLLaVAOneVisionProvider, CloudVLMProvider

# Alignment mode normalization
ALLOWED_ALIGNMENT_MODES = {"keyword", "semantic+keyword"}

def _normalize_alignment_mode(value: str) -> tuple[str, str]:
    try:
        v = (value or "").strip()
    except Exception:
        v = ""
    if v in ALLOWED_ALIGNMENT_MODES:
        return v, ""
    return "semantic+keyword", "⚠️ Invalid alignment mode; defaulting to semantic+keyword."

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
    # Handle dict objects (support image-occlusion entries)
    elif isinstance(flashcard, dict):
        # Occlusion dicts are valid if they contain both image paths
        if 'question_image_path' in flashcard and 'answer_image_path' in flashcard:
            return True, "Valid occlusion"
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

# Helper to OCR images and append text per slide
def ocr_images_to_text(slide_images_per_slide):
    """
    slide_images_per_slide: List[List[str]] where inner list contains file paths for a slide
    Returns: List[str] of OCR text per slide (empty string if none)
    """
    ocr_texts = []
    for images in slide_images_per_slide or []:
        buf = []
        for path in images or []:
            try:
                txt = pytesseract.image_to_string(Image.open(path))
                txt = (txt or '').strip()
                if txt:
                    buf.append(txt)
            except Exception:
                continue
        ocr_texts.append("\n".join(buf))
    return ocr_texts

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
    card_types=None,  # New parameter to handle multiple card types (list of selected card types)
    diarization_enabled=False,
    use_emphasis_cb=True,
    ocr_images_cb=True,
    alignment_mode: str = "semantic+keyword",
    make_clips_enabled: bool = False,
    use_manual_clip_len: bool = False,
    clip_length_s_val: int = 9,
    max_clips_val: int = 2,
):
    """
    Main function to generate flashcards from uploaded PowerPoint file
    """
    if not pptx_file:
        return "Please upload a PowerPoint file.", "No file uploaded", None
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set. Please check your .env file.", "API key not found", None
    
    # Normalize alignment mode and prepare warning
    norm_mode, warn = _normalize_alignment_mode(alignment_mode)
    status_msg_warn = warn

    # Initialize occluded_path before the try block
    occluded_path = None

    try:
        progress(0, desc="Starting flashcard generation...")
        status_msg = "Starting flashcard generation...\n" + (status_msg_warn or "")
        
        # Handle audio file path (accepts gradio filepath string or file-like with .name)
        audio_path = None
        if audio_file:
            try:
                candidate = audio_file if isinstance(audio_file, (str, bytes, os.PathLike)) else getattr(audio_file, 'name', None)
                audio_path = find_audio_file(candidate)
                if not audio_path:
                    print(f"[WARN] Could not find audio file in: {candidate}")
                else:
                    print(f"[INFO] Using audio file: {audio_path}")
            except Exception as e:
                print(f"[WARN] Audio path resolution failed: {e}")
                audio_path = None
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract text and images from PowerPoint
            progress(0.1, desc="Extracting content from PowerPoint...")
            slide_texts = extract_text_from_pptx(pptx_file)
            # Extract images if image occlusion is enabled OR OCR is requested
            if enable_image_occlusion or ocr_images_cb:
                slide_images = extract_images_from_pptx(pptx_file, temp_dir)
            else:
                slide_images = []
            
            # If OCR requested, augment slide_texts with OCR text per slide
            if ocr_images_cb and slide_images:
                try:
                    ocr_texts = ocr_images_to_text(slide_images)
                    # Ensure 1:1 length
                    if len(ocr_texts) == len(slide_texts):
                        for i in range(len(slide_texts)):
                            otext = ocr_texts[i]
                            if not otext:
                                continue
                            cur = slide_texts[i]
                            if isinstance(cur, dict):
                                existing = str(cur.get('text', ''))
                                cur['text'] = (existing + ('\n' if existing else '') + otext).strip()
                                slide_texts[i] = cur
                            else:
                                existing = str(cur or '')
                                slide_texts[i] = (existing + ('\n' if existing else '') + otext).strip()
                except Exception as e:
                    print(f"[WARN] OCR on images failed: {e}")
            
            if not slide_texts:
                return "No text content found in the PowerPoint file.", "No content extracted", None
            
            progress(0.2, desc="Filtering relevant slides...")
            # Filter slides based on content relevance (will ignore images if list is empty)
            filtered_slide_texts, kept_indices = filter_slides(slide_texts, slide_images)
            
            # Lenient fallback: if nothing kept and image occlusion is OFF, keep all slides with non-empty text
            if not filtered_slide_texts and not enable_image_occlusion:
                kept_indices = [i for i, t in enumerate(slide_texts) if isinstance(t, str) and t.strip()]
                filtered_slide_texts = [slide_texts[i] for i in kept_indices]
                # slide_images is empty in this mode
                print("[WARN] Relevance filter removed all slides; falling back to all non-empty text slides.")
            
            if not filtered_slide_texts:
                return "No relevant content found after filtering.", "No relevant content", None
            
            # Filter slide_images to match the kept slides
            filtered_slide_images = [slide_images[i] for i in kept_indices] if (enable_image_occlusion and kept_indices) else (slide_images if enable_image_occlusion else [])
            
            progress(0.3, desc="Generating flashcards...")
            # Build AudioBundle if audio is provided
            audio_bundle = None
            if audio_path:
                try:
                    from audio_processor import AudioProcessor
                    ap = AudioProcessor(model_name="base")
                    # The processor reads emphasis_enabled from config; UI toggle will guide downstream usage
                    audio_bundle = ap.build_audio_bundle(audio_path, filtered_slide_texts, diarization_enabled=diarization_enabled, alignment_mode=norm_mode)
                    # Optional per-slide fallback: if semantic+keyword has low-conf windows, try keyword and replace when better
                    try:
                        if norm_mode != "keyword":
                            lows = [w for w in getattr(audio_bundle, 'slide_windows', []) if float(getattr(w, 'confidence', 0.0)) < 0.50]
                            if lows:
                                kw_bundle = ap.build_audio_bundle(audio_path, filtered_slide_texts, diarization_enabled=diarization_enabled, alignment_mode="keyword")
                                by_id = {w.slide_id: w for w in getattr(audio_bundle, 'slide_windows', [])}
                                for w2 in getattr(kw_bundle, 'slide_windows', []) or []:
                                    orig = by_id.get(w2.slide_id)
                                    if orig is None:
                                        continue
                                    c0 = float(getattr(orig, 'confidence', 0.0))
                                    c1 = float(getattr(w2, 'confidence', 0.0))
                                    if c1 > c0:
                                        # replace window with higher-confidence keyword alignment
                                        try:
                                            idx = next(i for i, ww in enumerate(audio_bundle.slide_windows) if ww.slide_id == w2.slide_id)
                                            # Mark modes for reporting/preview if needed
                                            setattr(w2, 'mode', 'keyword')
                                            audio_bundle.slide_windows[idx] = w2
                                            by_id[w2.slide_id] = w2
                                        except Exception:
                                            pass
                                    else:
                                        setattr(orig, 'mode', norm_mode)
                    except Exception as _e:
                        print(f"[WARN] Per-slide alignment fallback skipped: {_e}")
                    # attach flag for downstream
                    setattr(audio_bundle, 'use_emphasis', bool(use_emphasis_cb))
                    if diarization_enabled and not getattr(audio_bundle, 'diarization_applied', False):
                        print("[WARN] Diarization requested but unavailable; proceeding without diarization.")
                except RuntimeError as e:
                    user_msg = str(e)
                    return f"⚠️ Audio processing problem: {user_msg}", f"⚠️ Audio issue: {user_msg}", None
                except FileNotFoundError:
                    return "⚠️ Audio file unreadable.", "⚠️ Audio file unreadable.", None
                except Exception as e:
                    return f"⚠️ Audio error: {e}", f"⚠️ Audio error: {e}", None
            # Generate flashcards with progress tracking and audio context
            # Decide whether to generate text/cloze cards based on selected types
            try:
                allow_text = (not card_types) or any(str(t).lower() in ("basic", "cloze") for t in (card_types or []))
            except Exception:
                allow_text = True
            flashcard_previews = []
            if allow_text:
                flashcard_previews = generate_enhanced_flashcards_with_progress(
                filtered_slide_texts,
                filtered_slide_images,  # Use filtered slide images (empty when occlusion disabled)
                OPENAI_API_KEY,
                MODEL_NAME,
                MAX_TOKENS,
                TEMPERATURE,
                progress=progress,
                use_cloze=use_cloze,
                question_style=question_style,
                audio_bundle=audio_bundle
            )
            
            # Audio clips generation and preview table
            clips_map = {}
            warnings_map = {}
            wins_by_id = {w.slide_id: w for w in getattr(audio_bundle, 'slide_windows', []) or []} if audio_bundle else {}
            # Always attach alignment-derived metrics if available (even without clips)
            if audio_bundle:
                try:
                    from flashcard_generator import Flashcard
                    def attach_alignment(cards):
                        out = []
                        for c in cards:
                            if isinstance(c, list):
                                out.append(attach_alignment(c))
                            else:
                                try:
                                    if isinstance(c, Flashcard) and c.slide_number in wins_by_id:
                                        win = wins_by_id[c.slide_number]
                                        # Populate confidence
                                        try:
                                            c.confidence = float(getattr(win, 'confidence', 0.0))
                                        except Exception:
                                            pass
                                        # Populate emphasis_weight and time_allocation if not already set
                                        try:
                                            if getattr(win, 'segments', None):
                                                avg_emp = float(np.mean([getattr(s,'emphasis',0.0) for s in win.segments]))
                                                if getattr(c, 'emphasis_weight', 0.0) in (0.0, 1.0):
                                                    c.emphasis_weight = max(0.1, min(2.0, avg_emp * 2))
                                                ws, we = win.window
                                                duration = max(0.0, float(we - ws))
                                                if getattr(c, 'time_allocation', 0.0) <= 0.0:
                                                    c.time_allocation = duration
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                out.append(c)
                        return out
                    flashcard_previews = attach_alignment(flashcard_previews)
                except Exception:
                    pass
            # Create clips if enabled; add fallback preview clip if none found for a slide
            if audio_bundle and make_clips_enabled:
                try:
                    from audio_processor import AudioProcessor
                    from flashcard_generator import Flashcard, AudioMetadataForCard
                    ap = AudioProcessor(model_name="base")
                    # Auto mode: pass None for min/max to enable adaptive target; manual when accordion used
                    auto_mode = not bool(use_manual_clip_len)
                    min_s = None if auto_mode else int(clip_length_s_val)
                    max_s = None if auto_mode else int(clip_length_s_val)
                    clips_map, warnings_map = ap.create_audio_clips_for_bundle(
                        audio_bundle,
                        output_dir=temp_dir,
                        per_slide_max=int(max_clips_val),
                        min_clip_sec=min_s,
                        max_clip_sec=max_s
                    )
                    # Fallback: ensure at least one preview clip per slide window for listening
                    try:
                        audio_path = find_audio_file(audio_file.name) if audio_file else None
                    except Exception:
                        audio_path = None
                    if audio_path and os.path.isfile(audio_path):
                        for w in getattr(audio_bundle, 'slide_windows', []) or []:
                            if len(clips_map.get(w.slide_id, [])) == 0:
                                ws, we = w.window
                                mid = (ws + we) / 2.0
                                pv_len = max(5.0, min(12.0, (we - ws)))
                                s_pv = max(ws, mid - pv_len / 2.0)
                                e_pv = min(we, s_pv + pv_len)
                                try:
                                    tmp_name = f"preview_s{w.slide_id}_{int(s_pv*1000)}-{int(e_pv*1000)}.mp3"
                                    temp_preview = os.path.join(temp_dir, tmp_name)
                                    AudioProcessor(model_name="base")._clip_with_ffmpeg(audio_path, s_pv, e_pv, temp_preview)
                                    clips_map.setdefault(w.slide_id, []).append(temp_preview)
                                except Exception:
                                    pass
                    # Attach audio files and re-attach confidence for safety
                    def attach_audio(cards):
                        out = []
                        for c in cards:
                            if isinstance(c, list):
                                out.append(attach_audio(c))
                            else:
                                try:
                                    if isinstance(c, Flashcard) and c.slide_number in clips_map:
                                        files = clips_map[c.slide_number]
                                        if files:
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
                except RuntimeError as e:
                    print(f"[WARN] Audio clip error: {e}")
                except Exception as e:
                    print(f"[WARN] Failed to create/attach audio clips: {e}")
            
            # (alignment preview removed)
            preview_table = ""
            
            if not flashcard_previews and not enable_image_occlusion:
                return "No flashcards generated. Please check your content and settings.", "No flashcards generated", None
            
            # Flatten and validate flashcards (if any were generated)
            progress(0.7, desc="Validating and flattening flashcards...")
            all_flashcards = flatten_flashcard_list(flashcard_previews) if flashcard_previews else []
            
            # If user requested image occlusion, continue to occlusion even if no text/cloze cards yet
            if not all_flashcards and not enable_image_occlusion:
                return "No valid flashcards generated. Please check your content and settings.", "No valid flashcards", None
            
            progress(0.8, desc="Removing duplicates and optimizing...")
            # Remove duplicate flashcards
            unique_flashcards = remove_duplicate_flashcards(all_flashcards) if all_flashcards else []
            
            # (Export postponed to after image-occlusion so all cards are included)
            
            # Handle image occlusion if enabled
            if enable_image_occlusion and filtered_slide_images:
                progress(0.85, desc="Processing image occlusion...")
                # Filter relevant images for occlusion; fall back to all images if filter fails or returns none
                try:
                    relevant_images = filter_relevant_images_for_occlusion(
                        filtered_slide_images, 
                        filtered_slide_texts, 
                        OPENAI_API_KEY, 
                        MODEL_NAME
                    )
                except Exception as _e:
                    print(f"[WARN] Image relevance filter failed: {_e}; using all extracted images")
                    relevant_images = filtered_slide_images
                # Fallback if filter returns empty lists while we have images
                try:
                    any_filtered = any(len(lst) > 0 for lst in (relevant_images or []))
                    any_images = any(len(lst) > 0 for lst in (filtered_slide_images or []))
                    if not any_filtered and any_images:
                        print("[WARN] Relevance filter removed all images; falling back to all extracted images")
                        relevant_images = filtered_slide_images
                except Exception:
                    relevant_images = filtered_slide_images
                
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

            # Early exit if nothing was generated after occlusion (when text/cloze was disabled)
            if not unique_flashcards and not (enable_image_occlusion and filtered_slide_images):
                return "No valid flashcards generated. Please check your content and settings.", "No valid flashcards", None

            # Now export APKG with the final set of cards (including occlusions)
            progress(0.9, desc="Exporting to Anki format...")
            try:
                apkg_name = "generated_flashcards.apkg"
                try:
                    if pptx_file and getattr(pptx_file, 'name', None):
                        apkg_name = f"{os.path.splitext(os.path.basename(pptx_file.name))[0]}_flashcards.apkg"
                except Exception:
                    pass
                # First export to a temp path
                apkg_out_path = os.path.join(temp_dir, apkg_name)
                tmp_apkg_path = export_flashcards_to_apkg(unique_flashcards, apkg_out_path)
                # Persist outside the TemporaryDirectory so it survives after return
                outputs_dir = "/home/vidhidutta/anki-flashcard-generator/outputs"
                os.makedirs(outputs_dir, exist_ok=True)
                saved_path = os.path.join(outputs_dir, apkg_name)
                final_apkg_path = tmp_apkg_path
                try:
                    shutil.copy2(tmp_apkg_path, saved_path)
                    final_apkg_path = saved_path
                except Exception as _copy_err:
                    # Retry by exporting directly to outputs
                    try:
                        final_apkg_path = export_flashcards_to_apkg(unique_flashcards, saved_path)
                        if not final_apkg_path:
                            final_apkg_path = tmp_apkg_path  # Fallback to temp path if export returns None
                    except Exception as _reexp_err:
                        print(f"[WARN] Could not persist APKG to outputs: copy={_copy_err}; re-export={_reexp_err}")
                        final_apkg_path = tmp_apkg_path  # Fallback to temp path
                try:
                    if os.path.isfile(final_apkg_path):
                        print(f"[INFO] Saved APKG to: {final_apkg_path}")
                    else:
                        print(f"[WARN] APKG file not found at: {final_apkg_path}")
                except Exception as e:
                    print(f"[WARN] Error checking APKG file: {e}")
                print(f"[DEBUG] Final APKG path before return: {final_apkg_path}")
            except Exception as e:
                return f"⚠️ Export failed: {e}", f"⚠️ Media/export write failure: {e}", None
            
            # Build and save run report (JSON and CSV) next to the .apkg
            try:
                import json, csv
                apkg_dir = os.path.dirname(final_apkg_path) if final_apkg_path else temp_dir
                apkg_base = os.path.splitext(os.path.basename(final_apkg_path or 'generated_flashcards.apkg'))[0]
                json_path = os.path.join(apkg_dir, f"{apkg_base}_report.json")
                csv_path = os.path.join(apkg_dir, f"{apkg_base}_report.csv")
                report_rows = []
                diar_applied = bool(getattr(audio_bundle, 'diarization_applied', False)) if audio_bundle else False
                def _sf(v, default=None):
                    try:
                        if v is None:
                            return default
                        return float(v)
                    except Exception:
                        return default
                # Record image-understanding provider info
                try:
                    from providers.ocr_provider import PaddleOCRProvider
                    ocr_provider_name = PaddleOCRProvider.get_provider_name()
                except Exception:
                    ocr_provider_name = "none"
                detection_provider = "groundingdino" if importlib.util.find_spec('groundingdino_pytorch') else "none"
                segmentation_provider = "sam2" if importlib.util.find_spec('sam2') else ("sam" if importlib.util.find_spec('segment_anything') else "none")
                vlm_provider = "none"
                try:
                    from providers.vlm_provider import LocalQwen2VLProvider, LocalLLaVAOneVisionProvider, CloudVLMProvider
                    if LocalQwen2VLProvider.available():
                        vlm_provider = "qwen2-vl"
                    elif LocalLLaVAOneVisionProvider.available():
                        vlm_provider = "llava-onevision"
                    elif CloudVLMProvider.available():
                        vlm_provider = "cloud"
                except Exception:
                    pass
                for w in (audio_bundle.slide_windows if audio_bundle else []):
                    ws, we = w.window
                    title = ""
                    try:
                        raw = filtered_slide_texts[w.slide_id]
                        title = raw.splitlines()[0][:200]
                    except Exception:
                        title = ""
                    files = clips_map.get(w.slide_id, []) if 'clips_map' in locals() else []
                    emphasis_avg = 0.0
                    if getattr(w, 'segments', None):
                        try:
                            emphasis_avg = float(np.mean([_sf(getattr(s,'emphasis',0.0), 0.0) for s in w.segments]))
                        except Exception:
                            emphasis_avg = 0.0
                    kept_pct = None
                    if diar_applied:
                        ws_dur = max(0.0, we - ws)
                        lect_dur = 0.0
                        for s in w.segments:
                            if str(getattr(s, 'speaker', 'LECTURER')).upper() == 'LECTURER':
                                lect_dur += max(0.0, s.end - s.start)
                        kept_pct = (lect_dur / ws_dur * 100.0) if ws_dur > 0 else 0.0
                    report_rows.append({
                        'slide_id': int(w.slide_id),
                        'title': title,
                        'alignment_confidence': _sf(getattr(w,'confidence', 0.0), 0.0),
                        'semantic_score': _sf(getattr(w,'semantic_score', 0.0), 0.0),
                        'keyword_score': _sf(getattr(w,'keyword_score', 0.0), 0.0),
                        'window_start': _sf(ws, 0.0),
                        'window_end': _sf(we, 0.0),
                        'mode_used': norm_mode,
                        'auto_clip_used': (bool(min_s is None and max_s is None) if 'min_s' in locals() and 'max_s' in locals() else True),
                        'num_clips': int(len(files)),
                        'clip_filenames': [os.path.basename(f) for f in files],
                        'boundary_types': (getattr(getattr(ap, 'last_clip_meta', {}), 'get', lambda *_: {})() if False else (ap.last_clip_meta.get(w.slide_id, {}).get('boundary_types') if 'ap' in locals() and hasattr(ap, 'last_clip_meta') else None)),
                        'clip_lengths_s': (ap.last_clip_meta.get(w.slide_id, {}).get('lengths_s') if 'ap' in locals() and hasattr(ap, 'last_clip_meta') else None),
                        'emphasis_avg': _sf(emphasis_avg, 0.0),
                        'diarization_applied': bool(diar_applied),
                        'diarization_requested': bool(diarization_enabled),
                        'speech_kept_percent': (_sf(kept_pct, None) if kept_pct is not None else None),
                        'quality_rating': str(getattr(w,'snr_quality','Unknown')),
                        'image_understanding.ocr_provider': ocr_provider_name,
                        'image_understanding.detection_provider': detection_provider,
                        'image_understanding.segmentation_provider': segmentation_provider,
                        'image_understanding.vlm_provider': vlm_provider,
                    })
                with open(json_path, 'w') as jf:
                    json.dump(report_rows, jf, indent=2)
                if report_rows:
                    with open(csv_path, 'w', newline='') as cf:
                        writer = csv.DictWriter(cf, fieldnames=list(report_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(report_rows)
            except Exception as e:
                print(f"[WARN] Failed to write run report: {e}")

            # Count different types of flashcards
            text_cards = sum(1 for card in unique_flashcards if hasattr(card, 'card_type') and str(getattr(card, 'card_type', '')).lower() in ('text','basic'))
            occlusion_cards = sum(1 for card in unique_flashcards if (hasattr(card, 'card_type') and card.card_type == 'occlusion') or (isinstance(card, dict) and 'question_image_path' in card and 'answer_image_path' in card))
            audio_cards = sum(1 for card in unique_flashcards if (hasattr(card, 'audio_metadata') and card.audio_metadata) or (isinstance(card, dict) and bool(card.get('audio_files'))))
            
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
            # Nicely formatted preview lines
            try:
                from flashcard_generator import Flashcard as _FC
                for i, preview in enumerate(unique_flashcards[:5]):  # Show first 5
                    if isinstance(preview, _FC):
                        q = (preview.question or '').strip()
                        a = (preview.answer or '').strip()
                        q_disp = (q[:120] + ("…" if len(q) > 120 else ""))
                        a_disp = (a[:140] + ("…" if len(a) > 140 else ""))
                        s_num = int(getattr(preview, 'slide_number', 0)) + 1
                        conf = float(getattr(preview, 'confidence', 0.0))
                        ctype = getattr(preview, 'card_type', '') or ('cloze' if getattr(preview, 'is_cloze', False) else 'basic')
                        flashcard_summary += f"\n{i+1}. Flashcard(question='{q_disp}', answer='{a_disp}', slide={s_num}, type='{ctype}', confidence={conf:.2f})\n"
                    else:
                        flashcard_summary += f"\n{i+1}. {preview}\n"
            except Exception:
                for i, preview in enumerate(unique_flashcards[:5]):
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
            
            # Per-slide summary: number of cards and clips
            try:
                cards_per_slide = {}
                for c in unique_flashcards:
                    try:
                        sid = int(getattr(c, 'slide_number', -1))
                    except Exception:
                        continue
                    if sid >= 0:
                        cards_per_slide[sid] = cards_per_slide.get(sid, 0) + 1
                # hide per-slide alignment summary in public UI
            except Exception:
                pass
 
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
            header_info = f"\nClip bounds: {int(cfg_min)}–{int(cfg_max)}s, tol ±{cfg_tol:.2f}s | Emphasis: {'on' if use_emphasis_cb else 'off'}\n"
            if preview_table:
                flashcard_summary += header_info + preview_table
            else:
                flashcard_summary += header_info
            # Include report paths and saved deck path
            try:
                flashcard_summary += f"\nReport saved: {json_path}\nCSV: {csv_path}\n"
                if final_apkg_path:
                    flashcard_summary += f"APKG: {final_apkg_path}\n"
            except Exception:
                pass
            # Warn if PaddleOCR missing and Tesseract fallback used
            try:
                from providers.ocr_provider import PaddleOCRProvider
                if not PaddleOCRProvider.available() and PaddleOCRProvider.tesseract_available():
                    status_output = f"⚠️ PaddleOCR unavailable; using basic OCR fallback (Tesseract).\n" + (status_output or "")
            except Exception:
                pass
            # Disable audio preview row interactions based on tools; guard if component not present
            try:
                tools_ok = shutil.which('ffmpeg') or shutil.which('ffprobe')
            except Exception:
                tools_ok = False
            # Include warning summary
            if warnings_map:
                warn_lines = []
                for sid, msgs in warnings_map.items():
                    for m in msgs:
                        warn_lines.append(f"Slide {sid+1}: {m}")
                flashcard_summary += "\nWarnings:\n" + "\n".join(warn_lines)
 
            # Return the concrete file path for the File component
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
        color: #ff4444;
    }
    
    .progress-container {
        background: rgba(45, 45, 45, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .status-text {
        color: #ccc;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    """
    
    with gr.Blocks(
        title="APEX.MED - Medical Flashcard Generator", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Main container with dark theme
        with gr.Column(elem_classes=["apex-container"]):
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="apex-header">
                    <h1 class="apex-title">APEX.MED</h1>
                </div>
                """)
            
            with gr.Row():
                # Left Section - Upload and Generation
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
                        with gr.Accordion("Advanced audio clipping", open=False):
                            use_manual_clip_len = gr.Checkbox(label="Use manual clip length", value=False)
                            clip_length_s = gr.Slider(5, 15, value=7, step=1, label=f"Manual clip length (seconds) [5–15]")
                            max_clips = gr.Slider(0, 2, value=1, step=1, label="Max clips per slide")
                        alignment_mode = gr.Dropdown(choices=["keyword", "semantic+keyword"], value="keyword", label="Alignment mode")
                        low_conf_only = gr.Checkbox(label="Show only low-confidence (<0.5)", value=False)
                        use_emphasis_cb = gr.Checkbox(label="Use emphasis weighting", value=True)
                        
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
                            text_src = gr.Checkbox(value=True, label="Text")
                            img_src = gr.Checkbox(value=False, label="Images")
                            notes_src = gr.Checkbox(value=True, label="Speaker Notes")
                            ocr_images_cb = gr.Checkbox(value=True, label="Extract text from images (OCR)")
                            keep_image_only_cb = gr.Checkbox(value=True, label="Always keep image-only slides")
                            ai_image_understanding_cb = gr.Checkbox(value=False, label="Use AI image understanding (OCR + detection + SAM + VLM)", visible=False)
                            clear_cache_btn = gr.Button("Clear cache")
                            # Backward-compatible aliases
                            content_images = img_src
                            content_notes = notes_src
                        
                        # Image Occlusion Settings (hidden by default)
                        # Adaptive AI Configuration Status
                        adaptive_status = gr.HTML(
                            value="🤖 <strong>Adaptive AI Configuration Enabled</strong><br>"
                                  "The AI automatically analyzes each image and optimizes parameters for best results.<br>"
                                  "No manual configuration needed!",
                            label="AI Configuration Status",
                            visible=False
                        )
                        
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
            # Progress and Results Section
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes=["progress-container"]):
                        gr.HTML('<div class="status-text">Status</div>')
                        status_output = gr.Textbox(
                            label="",
                            value="Ready to generate flashcards...",
                            interactive=False,
                            lines=2,
                            container=False
                        )
                        
                        progress_bar = gr.Progress()
                        
                        flashcard_output = gr.Textbox(
                            label="Generated Flashcards",
                            value="Flashcards will appear here after generation...",
                            interactive=False,
                            lines=10,
                            container=False
                        )
                        
                        # Download field for exported deck
                        gr.HTML('<div style="margin-top: 20px; font-weight: bold; color: #ff4444; font-size: 18px;">📥 DOWNLOAD GENERATED FLASHCARDS</div>')
                        apkg_file = gr.File(
                            label="Download APKG", 
                            interactive=False,
                            file_count="single",
                            height=100,
                            container=True,
                            scale=1
                        )
                        gr.HTML('<div style="margin-top: 10px; color: #888; font-size: 12px;">The APKG file will appear here after generation. Click to download.</div>')

            # (Removed occlusion review/edit UI)
 
            # Footer
            gr.HTML("""
            <div class="footer">
                <a href="#" class="footer-link">About</a>
                <a href="#" class="footer-link">FAQs</a>
                <a href="#" class="footer-link">Contact Us</a>
            </div>
            """)

        # Function to update visibility of image occlusion settings
        def update_occlusion_settings_visibility(card_type):
            if "Image Occlusion" in card_type:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        # Update image occlusion settings visibility based on card type
        card_type.change(
            fn=update_occlusion_settings_visibility,
            inputs=[card_type],
            outputs=[adaptive_status, conf_threshold_slider, max_masks_slider]
        )

        # Show AI image understanding when image occlusion is enabled or audio is provided
        def _toggle_ai_img_visibility(audio_fp, card_type):
            try:
                has_audio = bool(audio_fp)
                has_image_occlusion = "Image Occlusion" in card_type if card_type else False
                should_show = has_audio or has_image_occlusion
            except Exception:
                should_show = False
            return gr.update(visible=should_show, value=(False if not should_show else None))

        # Update visibility when either audio or card type changes
        audio_file.change(fn=_toggle_ai_img_visibility, inputs=[audio_file, card_type], outputs=[ai_image_understanding_cb])
        card_type.change(fn=_toggle_ai_img_visibility, inputs=[audio_file, card_type], outputs=[ai_image_understanding_cb])
        
        # Add medical vision AI status indicator
        def check_medical_vision_status():
            try:
                from providers.medical_vision_provider import GoogleMedicalVisionProvider
                if GoogleMedicalVisionProvider.available():
                    return "✅ Google Medical Vision AI Available"
                else:
                    return "❌ Google Medical Vision AI Not Available"
            except Exception:
                return "❌ Google Medical Vision AI Not Available"
        
        medical_vision_status = gr.HTML(
            value=check_medical_vision_status(),
            label="Medical Vision AI Status"
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
            use_emphasis_cb,
            ocr_images_cb,
            keep_image_only_cb,
            ai_image_understanding_cb,
            alignment_mode,
            diarization_cb,
            make_clips_cb,
            use_manual_clip_len,
            clip_length_s,
            max_clips,
            progress=gr.Progress()
        ):
            print(f"[DEBUG] Function called with pptx_file={pptx_file}, audio_file={audio_file}")
            # Map the new UI parameters to the existing function parameters
            use_cloze = "Cloze" in card_type
            enable_image_occlusion = "Image Occlusion" in card_type
            occlusion_mode = "AI-assisted (AI detects labels/regions)" if enable_image_occlusion else None
            occlusion_image = None
            if not card_type:
                card_type = ["Basic"]
            try:
                res = run_flashcard_generation(
                    pptx_file,
                    audio_file,
                    flashcard_level,
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
                    card_type,
                    diarization_enabled=bool(diarization_cb),
                    use_emphasis_cb=use_emphasis_cb,
                    ocr_images_cb=ocr_images_cb,
                    alignment_mode=alignment_mode,
                    make_clips_enabled=make_clips_cb,
                    clip_length_s_val=int(clip_length_s),
                    max_clips_val=int(max_clips),
                )
                # Normalize result and map to outputs in correct order
                summary, status, apkg_path = None, None, None
                if isinstance(res, tuple):
                    if len(res) >= 1:
                        summary = res[0]
                    if len(res) >= 2:
                        status = res[1]
                    if len(res) >= 3:
                        apkg_path = res[2]
                else:
                    summary = res
                # Prepare occlusion state: use full pipeline if toggled ON and images available
                state = {"images": [], "proposals": {}, "cached_images": set()}
                try:
                    if enable_image_occlusion and pptx_file and ai_image_understanding_cb:
                        tmpdir = tempfile.mkdtemp()
                        imgs_by_slide = extract_images_from_pptx(pptx_file, tmpdir)
                        flat_imgs = [p for lst in imgs_by_slide for p in lst if os.path.isfile(p)]
                        # Try Google Medical Vision AI first, fallback to other providers
                        try:
                            from providers.medical_vision_provider import create_medical_vision_provider
                            medical_vision = create_medical_vision_provider()
                            
                            if medical_vision.available():
                                print(f"[INFO] Using Google Medical Vision AI for intelligent medical image analysis")
                                proposals = {}
                                for ip in flat_imgs[:50]:
                                    try:
                                        # Use medical vision AI for intelligent analysis
                                        medical_regions = medical_vision.analyze_medical_image(ip)
                                        rows = []
                                        for region in medical_regions:
                                            x, y, w, h = region.bbox
                                            rows.append([
                                                True, 
                                                region.text, 
                                                region.rationale, 
                                                region.importance_score, 
                                                int(w*h), 
                                                int(x), int(y), int(x+w), int(y+h)
                                            ])
                                        proposals[ip] = rows
                                        print(f"[INFO] Medical Vision AI found {len(rows)} testable regions in {os.path.basename(ip)}")
                                    except Exception as e:
                                        print(f"[WARN] Medical Vision AI failed for {ip}: {e}")
                                        continue
                            else:
                                raise Exception("Medical Vision AI not available")
                                
                        except Exception as medical_error:
                            print(f"[INFO] Medical Vision AI unavailable, using fallback providers: {medical_error}")
                        # Decide lightweight mode automatically if heavy providers unavailable
                        heavy_available = bool(GroundingDINOProvider.available() or SAMProvider.available() or LocalQwen2VLProvider.available() or LocalLLaVAOneVisionProvider.available() or CloudVLMProvider.available())
                        # Proposals
                        proposals = {}
                        for ip in flat_imgs[:50]:
                            try:
                                if heavy_available:
                                    regs = detect_segment_rank(ip, slide_text="", transcript_text="", )
                                    rows = []
                                    for ridx, r in enumerate(regs):
                                        x1,y1,x2,y2 = r.bbox_xyxy
                                        rows.append([True, (r.short_label or r.term or ""), (r.rationale or ""), float(r.importance_score or 0.5), int(r.area_px), int(x1), int(y1), int(x2), int(y2)])
                                    proposals[ip] = rows
                                else:
                                    # Lightweight fast boxes using detect_text_regions
                                    im = Image.open(ip)
                                    regs = detect_text_regions(im, conf_threshold=conf_threshold, use_blocks=config.get('image_occlusion',{}).get('use_blocks', False))
                                    rows = []
                                    for r in regs[:max_masks_per_image]:
                                        x,y,w,h = r
                                        rows.append([True, "", "", 0.5, int(w*h), int(x), int(y), int(x+w), int(y+h)])
                                    proposals[ip] = rows
                            except Exception:
                                continue
                        state = {"images": flat_imgs, "proposals": proposals, "cached_images": set()}
                        # Return with exactly 3 outputs: text, status, file path
                        print(f"[DEBUG] Returning: summary='{summary}', status='{status}', apkg_path='{apkg_path}'")
                        # Ensure apkg_path is a valid file path for Gradio
                        if apkg_path and os.path.exists(apkg_path):
                            print(f"[DEBUG] APKG file exists at: {apkg_path}")
                        else:
                            print(f"[DEBUG] APKG file not found or invalid: {apkg_path}")
                        return (
                            summary or "",
                            status or "",
                            apkg_path,
                        )
                except Exception:
                    state = {"images": [], "proposals": {}, "cached_images": set()}
                # Build outputs: [status_output, flashcard_output, apkg_file]
                print(f"[DEBUG] Final return: summary='{summary}', status='{status}', apkg_path='{apkg_path}'")
                # Ensure apkg_path is a valid file path for Gradio
                if apkg_path and os.path.exists(apkg_path):
                    print(f"[DEBUG] APKG file exists at: {apkg_path}")
                else:
                    print(f"[DEBUG] APKG file not found or invalid: {apkg_path}")
                outputs = [
                    summary or "",
                    status or "",
                    apkg_path,
                ]
                return tuple(outputs)
            except Exception as e:
                msg = f"⚠️ Error: {e}"
                return msg, msg, None

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
                use_emphasis_cb,
                ocr_images_cb,
                keep_image_only_cb,
                ai_image_understanding_cb,
                alignment_mode,
                diarization_cb,
                make_clips_cb,
                use_manual_clip_len,
                clip_length_s,
                max_clips,
            ],
            outputs=[
                status_output,
                flashcard_output,
                apkg_file,
            ]
        )

        # (occlusion review helpers removed)

        # (occlusion review helpers removed)

        # (occlusion review components removed)

        # (occlusion review helpers removed)

        # (occlusion review components removed)

        # (occlusion review helpers removed)

        # (occlusion review components removed)

        # (occlusion review helpers removed)

        # (occlusion review components removed)

        # (occlusion review helpers removed)

        # (occlusion review components removed)
 
        def clear_cache():
            # Clear in-memory caches (best-effort)
            try:
                import providers.ocr_provider as ocrp
                ocrp._OCR_CACHE.clear()
            except Exception:
                pass
            try:
                import providers.detect_provider as detp
                detp._DET_CACHE.clear()
            except Exception:
                pass
            try:
                import providers.segment_provider as segp
                segp._SEG_CACHE.clear()
            except Exception:
                pass
            return "Caches cleared"

        clear_cache_btn.click(fn=clear_cache, inputs=[], outputs=[status_output])
 
    return interface

if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Allow server_port override via CLI or env
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7866))
    for arg in sys.argv:
        if arg.startswith("--server_port"):
            parts = arg.split()
            if len(parts) == 2 and parts[1].isdigit():
                port = int(parts[1])
            elif '=' in arg:
                port = int(arg.split('=')[1])

    # Create and launch the interface
    interface = create_interface()
    
    # Try to find an available port
    import socket
    def find_free_port(start_port):
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # fallback
    
    available_port = find_free_port(port)
    if available_port != port:
        print(f"⚠️ Port {port} is busy, using port {available_port} instead")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=available_port,
        share=True,
        debug=True
    ) 