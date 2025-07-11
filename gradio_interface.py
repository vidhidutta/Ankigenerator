import gradio as gr
import yaml
import os
import tempfile
import shutil
import sys
import time
from PIL import Image, ImageDraw
from utils.image_occlusion import detect_text_regions, mask_regions, make_qmask, make_omask
import hashlib
import logging

# Set up logging to a file
logging.basicConfig(filename="debug.log", level=logging.DEBUG)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def update_occlusion_image_editor(enable_image_occlusion, occlusion_mode, slide_images):
    if enable_image_occlusion and occlusion_mode == "Semi-automated (user selects regions)" and slide_images and slide_images[0]:
        return gr.update(visible=True, value=slide_images[0][0])
    else:
        return gr.update(visible=False, value=None)





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
    save_text_to_pdf,
    build_extra_materials_prompt,
    call_api_for_extra_materials,
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
    category,
    exam,
    flashcard_type,
    answer_format,
    use_cloze,
    generate_glossary,
    generate_topic_map,
    generate_summary,
    quality_control,
    anti_repetition,
    conciseness,
    context_enrichment,
    depth_consistency,
    auto_cloze,
    content_images,
    content_notes,
    enable_image_occlusion,
    occlusion_mode,
    occlusion_image,
    conf_threshold,
    max_masks_per_image,
    progress=gr.Progress()
):
    """
    Main function to generate flashcards from uploaded PowerPoint file
    """
    if not pptx_file:
        return "Please upload a PowerPoint file.", "No file uploaded", None, None
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set. Please check your .env file.", "API key not found", None, None
    
    # Initialize occluded_path before the try block
    occluded_path = None

    try:
        progress(0, desc="Starting flashcard generation...")
        status_msg = "Starting flashcard generation..."
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy uploaded file to temp directory
            temp_pptx_path = os.path.join(temp_dir, "uploaded_presentation.pptx")
            shutil.copy2(pptx_file.name, temp_pptx_path)
            
            # Extract text (and notes)
            progress(0.1, desc="Extracting text from PowerPoint...")
            status_msg = "Extracting text from PowerPoint..."
            slide_data = extract_text_from_pptx(temp_pptx_path)
            # Prepend notes to slide text if present and if content_notes is True
            slide_texts = []
            for entry in slide_data:
                notes = entry.get('notes_text', '').strip()
                slide_text = entry.get('slide_text', '').strip()
                if content_notes and notes:
                    combined = f"[NOTES]\n{notes}\n[SLIDE]\n{slide_text}" if slide_text else f"[NOTES]\n{notes}"
                else:
                    combined = slide_text
                slide_texts.append(combined)
            
            # Extract images if requested
            if content_images:
                progress(0.2, desc="Extracting images from PowerPoint...")
                status_msg = "Extracting images from PowerPoint..."
                slide_images = extract_images_from_pptx(temp_pptx_path, os.path.join(temp_dir, "slide_images"))
                print(f"[DEBUG] Extracted image paths: {str(slide_images)[:100]}...")  # Truncate long lists
            else:
                slide_images = [[] for _ in slide_texts]
            
            # Ensure image_paths is correctly set
            image_paths = [img for slide in slide_images for img in slide]  # Flatten the list of lists
            print(f"[DEBUG] Flattened image paths: {str(image_paths)[:100]}...")  # Truncate long lists
            
            if not slide_texts:
                return "Error: No text found in PowerPoint file or error occurred during extraction.", "No text found", None, None
            
            # Filter out slides that should be skipped
            progress(0.3, desc="Filtering slides to remove navigation content...")
            status_msg = "Filtering slides to remove navigation content..."
            filtered_slide_texts, kept_slide_indices = filter_slides(slide_texts, slide_images)
            print(f"[DEBUG] Slides kept after filtering: {len(filtered_slide_texts)}")
            filtered_slide_images = [slide_images[i] for i in kept_slide_indices]
            print(f"[DEBUG] Images kept after filtering: {sum(len(images) for images in filtered_slide_images)}")
            print(f"[DEBUG] Filtered slide texts: {len(filtered_slide_texts)}, Filtered slide images: {len(filtered_slide_images)}")
            logging.debug(f"Processing {len(filtered_slide_texts)} slides (filtered from {len(slide_texts)} total slides)")
            
            # Filter slide_images to match filtered_slide_texts
            # kept_slide_indices = [] # This line is now redundant as filter_slides handles it
            # for i, slide_text in enumerate(slide_texts):
            #     if not should_skip_slide(slide_text):
            #         kept_slide_indices.append(i)
            
            # filtered_slide_images = [slide_images[i] for i in kept_slide_indices]
            
            # Update configuration based on UI inputs
            current_category = category if category else CATEGORY
            current_exam = exam if exam else EXAM
            current_flashcard_type = {"level_1": flashcard_type == "Level 1", "level_2": flashcard_type == "Level 2", "both": flashcard_type == "Both"}
            current_answer_format = answer_format if answer_format else ANSWER_FORMAT
            current_cloze = "yes" if use_cloze else "no"
            current_features = {
                "topic_map": generate_topic_map,
                "glossary": generate_glossary,
                "summary_review_sheet": generate_summary,
                "index": False,
                "none": False
            }
            
            # Apply quality control settings
            if quality_control:
                progress(0.35, desc="Applying quality control settings...")
                status_msg = "Applying quality control settings..."
                
                # Update configuration with quality control settings
                quality_settings = {
                    "anti_repetition": anti_repetition,
                    "conciseness": conciseness,
                    "context_enrichment": context_enrichment,
                    "depth_consistency": depth_consistency,
                    "auto_cloze": auto_cloze
                }
                logging.debug(f"Quality control settings: {quality_settings}")
            
            # Generate flashcards with progress tracking
            progress(0.4, desc="Generating flashcards with AI...")
            status_msg = f"Generating flashcards with AI for {len(filtered_slide_texts)} slides..."
            result = generate_enhanced_flashcards_with_progress(
                filtered_slide_texts, 
                filtered_slide_images, 
                OPENAI_API_KEY, 
                MODEL_NAME, 
                MAX_TOKENS, 
                TEMPERATURE,
                progress,
                use_cloze=use_cloze
            )
            
            # Handle the new return format (flashcards, analysis_data)
            if isinstance(result, tuple):
                all_flashcards, analysis_data = result
            else:
                all_flashcards = result
                analysis_data = None
            
            if not all_flashcards:
                return "No flashcards were generated. Please check your PowerPoint content.", "No flashcards generated", None, None
            
            # After quality control, ensure we keep Flashcard objects, not tuples
            # Remove any conversion to (question, answer) tuples after quality control
            # Use the Flashcard objects directly for export
            flashcard_objs_for_export = all_flashcards
            # Only convert to dicts for display/summary, not for export
            # Build concise previews for the UI (skip internal metadata)
            flashcard_previews = []
            for card in flashcard_objs_for_export:
                # Skip image-occlusion placeholders – they cannot be previewed meaningfully
                if isinstance(card, dict) and card.get("type") == "image_occlusion":
                    continue  # do not add to preview list

                if hasattr(card, "is_cloze") and getattr(card, "is_cloze", False):
                    # Prefer the pre-computed preview text; fallback to cleaned cloze text
                    preview = getattr(card, "preview_text", "") or clean_cloze_text(getattr(card, "cloze_text", ""))
                    flashcard_previews.append(preview.strip())
                else:
                    # Basic Q-A format
                    q = getattr(card, "question", "")
                    a = getattr(card, "answer", "")
                    flashcard_previews.append(f"Q: {q}\nA: {a}")

            # Fallback to stringify_dict for logging/debug, not user display
            all_flashcards_dicts = [stringify_dict(card, use_cloze=use_cloze) for card in flashcard_objs_for_export]
            
            # If image occlusion is enabled and AI-assisted mode, filter and process relevant images
            occlusion_flashcards = []
            total_occlusion_flashcards = 0
            if enable_image_occlusion and occlusion_mode == "AI-assisted (AI detects labels/regions)" and filtered_slide_images:
                progress(0.85, desc="Filtering images for relevance...")
                status_msg = "Filtering images for relevance..."
                
                # Filter images to only include relevant medical/clinical content
                relevant_images = filter_relevant_images_for_occlusion(
                    filtered_slide_images, 
                    filtered_slide_texts, 
                    OPENAI_API_KEY, 
                    MODEL_NAME
                )
                
                # Process relevant images for occlusion
                progress(0.87, desc="Processing relevant images for occlusion...")
                status_msg = "Processing relevant images for occlusion..."
                
                processed_hashes = set() # Initialize a set to track processed hashes
                for slide_idx, images in enumerate(relevant_images):
                    logging.debug(f"[DEBUG] Processing slide {slide_idx + 1} with images: {images}")  # Debug statement to verify images being processed
                    # Initialize placeholder before iterating
                    image_path = None
                    for img_idx, img_path in enumerate(images):
                        image_path = img_path  # Keep for logging context
                        try:
                            img = Image.open(img_path).convert("RGB")
                            # Create occlusion for this image
                            occluded_path = os.path.join(temp_dir, f"occluded_slide{slide_idx+1}_img{img_idx+1}.png")

                            # Open and process the image
                            regions = detect_text_regions(img, conf_threshold=conf_threshold)

                            # Check if regions are empty
                            if not regions:
                                logging.debug(f"[DEBUG] No regions detected for image: {img_path}, skipping save.")
                                continue

                            # Generate a hash of the regions
                            regions_hash = hashlib.md5(str(regions).encode()).hexdigest()

                            # Check if this hash has already been processed
                            if regions_hash in processed_hashes:
                                logging.debug(f"[DEBUG] Duplicate regions detected for image: {img_path}, skipping save.")
                                continue

                            # Add the hash to the set of processed hashes
                            processed_hashes.add(regions_hash)

                            # Generate red-box mask layer
                            qmask_img = make_qmask(img, regions[0])  # assume first region for overlay

                            # Composite mask over the original slide so content shows through
                            base_rgba = img.convert("RGBA")
                            q_overlay = Image.alpha_composite(base_rgba, qmask_img)

                            # Save the composited image
                            q_overlay.save(occluded_path)

                            # Add debug statement for unique saves
                            logging.debug(f"[DEBUG] Saved occluded image: {occluded_path} with {len(regions)} masks")
                            
                            # Append in dict format understood by the exporter
                            occlusion_flashcards.append({
                                "type": "image_occlusion",
                                "question_image_path": occluded_path,  # masked image (front)
                                "answer_image_path": img_path          # original image (back)
                            })
                            total_occlusion_flashcards += 1
                            logging.debug(f"✅ Created occlusion dict for slide {slide_idx + 1}, image {img_idx + 1}")
                        except Exception as e:
                            logging.debug(f"⚠️ Error on {img_path}: {e}")
                
                logging.debug(f"✅ Created {total_occlusion_flashcards} image occlusion flashcards from relevant images")
            
            # The per-image append already occurs inside the loop; no extra append here

            # Merge occlusion flashcards with the main list
            all_cards_for_export = flashcard_objs_for_export + occlusion_flashcards

            # Add debug print statement
            print(f"[DEBUG] Total occlusion flashcards added: {len(occlusion_flashcards)}")

            # --------------------
            # DE-DUPLICATE CARDS
            # --------------------
            seen = set()
            unique_cards = []
            for card in all_cards_for_export:
                if hasattr(card, "question") and isinstance(card.question, str):
                    key = card.question.strip().lower().rstrip('.?!')
                    if key in seen:
                        continue
                    seen.add(key)
                unique_cards.append(card)
            all_cards_for_export = unique_cards

            # Export flashcards to APKG using de-duplicated cards
            progress(0.9, desc="Exporting flashcards to Anki .apkg...")
            status_msg = "Exporting flashcards to Anki .apkg..."
            apkg_path = os.path.join(temp_dir, "generated_flashcards.apkg")
            logging.debug(f"[DEBUG] First 3 card dicts to be exported: {all_flashcards_dicts[:3]}")
            # Filter out any Flashcard-class objects; keep only our image-occlusion dicts
            from flashcard_generator import export_flashcards_to_apkg
            logging.debug(f"[DEBUG] Exporting {len(all_cards_for_export)} total cards (text + image occlusion)")
            export_flashcards_to_apkg(all_cards_for_export, apkg_path, pptx_filename=pptx_file.name)
            logging.debug(f"[DEBUG] Wrote APKG to: {apkg_path}")

            # Generate extra materials if requested
            extra_materials = ""
            pdf_path = None
            if any([generate_glossary, generate_topic_map, generate_summary]):
                progress(0.95, desc="Generating extra materials...")
                status_msg = "Generating extra materials (glossary, topic map, summary)..."
                extra_prompt = build_extra_materials_prompt(filtered_slide_texts, current_features)
                if extra_prompt:
                    extra_response = call_api_for_extra_materials(
                        extra_prompt, 
                        OPENAI_API_KEY, 
                        MODEL_NAME, 
                        1000, 
                        TEMPERATURE
                    )
                    if not extra_response.startswith("__API_ERROR__"):
                        extra_materials = extra_response
                        pdf_path = os.path.join(temp_dir, "lecture_notes.pdf")
                        save_text_to_pdf(extra_response, pdf_path)
                    else:
                        extra_materials = f"Error generating extra materials: {extra_response}"
            
            progress(1.0, desc="Finalizing...")
            status_msg = "Finalizing and preparing downloads..."
            
            # Prepare output
            flashcard_summary = f"Successfully generated {len(all_flashcards)} flashcards!\n\n"
            
            if quality_control:
                flashcard_summary += "🎯 Quality Control Applied:\n"
                if anti_repetition:
                    flashcard_summary += "• Anti-repetition: Removed duplicate cards\n"
                if conciseness:
                    flashcard_summary += "• Conciseness: Split wordy answers into focused cards\n"
                if context_enrichment:
                    flashcard_summary += "• Context enrichment: Added clinical context\n"
                if depth_consistency:
                    flashcard_summary += "• Depth consistency: Ensured appropriate complexity\n"
                if auto_cloze:
                    flashcard_summary += "• Auto-cloze: Identified cloze opportunities\n"
                flashcard_summary += "\n"
            
            flashcard_summary += "Sample flashcards:\n"
            for i, preview in enumerate(flashcard_previews[:5]):  # Show first 5 concise previews
                flashcard_summary += f"\n{i+1}. {preview}\n"

            if len(flashcard_previews) > 5:
                flashcard_summary += f"\n... and {len(flashcard_previews) - 5} more flashcards"
            
            # Copy files to accessible location
            final_apkg_path = "generated_flashcards.apkg"
            final_pdf_path = "lecture_notes.pdf" if extra_materials else None
            
            shutil.copy2(apkg_path, final_apkg_path)
            if extra_materials and pdf_path and os.path.exists(pdf_path):
                shutil.copy2(pdf_path, "lecture_notes.pdf")
            
            final_status = f"✅ Complete! Generated {len(flashcard_previews)} quality-controlled flashcards from {len(filtered_slide_texts)} slides."
            
            # Call batch_generate_image_occlusion_flashcards with correct image_paths
            if image_paths:
                # Load configuration
                with open('config.yaml', 'r') as file:
                    config = yaml.safe_load(file)
                conf_threshold_slider = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=config['conf_threshold'],
                    step=1,
                    label="Confidence Threshold"
                )
                max_masks_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=config['max_masks_per_image'],
                    step=1,
                    label="Max Masks per Image"
                )
                # Update the call to batch_generate_image_occlusion_flashcards
                flashcard_entries = batch_generate_image_occlusion_flashcards(
                    image_paths,
                    os.path.join(temp_dir, "occlusion_flashcards"),
                    conf_threshold=conf_threshold,
                    mask_method='rectangle'
                )
                logging.debug(f"[DEBUG] Generated flashcard entries: {flashcard_entries}")  # Debug statement to verify flashcard entries
            else:
                logging.debug("[DEBUG] No images found for occlusion flashcards.")
            
            return flashcard_summary, final_status, final_apkg_path, final_pdf_path
            
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        return error_msg, f"❌ Error: {str(e)}", None, None

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Medical Flashcard Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧠 Medical Flashcard Generator")
        gr.Markdown("Upload your PowerPoint lecture and generate high-quality Anki flashcards for medical studies.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload")
                pptx_file = gr.File(
                    label="Upload PowerPoint (.pptx)",
                    file_types=[".pptx"],
                    type="filepath"
                )
                
                gr.Markdown("### ⚙️ Settings")
                category = gr.Dropdown(
                    choices=["Medical", "Dental", "Nursing", "Pharmacy", "Other"],
                    value="Medical",
                    label="Subject Category"
                )
                
                exam = gr.Dropdown(
                    choices=["Year 1 MBBS", "Year 2 MBBS", "Year 3 MBBS", "Year 4 MBBS", "Year 5 MBBS", "Finals", "PLAB", "USMLE", "Other"],
                    value="Year 2 MBBS",
                    label="Exam Level"
                )
                
                flashcard_type = gr.Radio(
                    choices=["Level 1", "Level 2", "Both"],
                    value="Both",
                    label="Flashcard Level",
                    info="Level 1: Basic recall | Level 2: Interpretation & application"
                )
                
                answer_format = gr.Dropdown(
                    choices=["minimal", "bullet_points", "short_paragraph", "mixture", "best"],
                    value="best",
                    label="Answer Format"
                )
                
                use_cloze = gr.Checkbox(
                    label="Use Cloze Deletions",
                    value=False,
                    info="Automatically convert suitable cards to cloze format"
                )
                
                gr.Markdown("### 🎯 Quality Control")
                quality_control = gr.Checkbox(
                    label="Enable Quality Control",
                    value=True,
                    info="Apply advanced quality control to improve flashcard quality"
                )
                
                with gr.Accordion("Quality Control Options", open=False):
                    anti_repetition = gr.Checkbox(
                        label="Remove Duplicates",
                        value=True,
                        info="Detect and remove repetitive flashcards"
                    )
                    
                    conciseness = gr.Checkbox(
                        label="Improve Conciseness",
                        value=True,
                        info="Split wordy answers into focused cards"
                    )
                    
                    context_enrichment = gr.Checkbox(
                        label="Enrich Context",
                        value=True,
                        info="Add clinical context to shallow cards"
                    )
                    
                    depth_consistency = gr.Checkbox(
                        label="Enforce Depth Consistency",
                        value=True,
                        info="Ensure appropriate complexity per level"
                    )
                    
                    auto_cloze = gr.Checkbox(
                        label="Auto-Detect Cloze Opportunities",
                        value=True,
                        info="Automatically identify cards suitable for cloze format"
                )
                
                gr.Markdown("### 📚 Extra Materials")
                generate_glossary = gr.Checkbox(
                    label="Generate Glossary",
                    value=True,
                    info="Create a glossary of key terms"
                )
                
                generate_topic_map = gr.Checkbox(
                    label="Generate Topic Map",
                    value=True,
                    info="Create an outline of main lecture themes"
                )
                
                generate_summary = gr.Checkbox(
                    label="Generate Summary Sheet",
                    value=True,
                    info="Create a compressed revision version"
                )
                
                gr.Markdown("### 📦 Content Sources")
                content_text = gr.Checkbox(
                    label="Text (slide content)",
                    value=True,
                    interactive=False,
                    info="Main slide text is always included"
                )
                content_images = gr.Checkbox(
                    label="Images (tables, graphs, diagrams)",
                    value=True,
                    info="Include images, charts, and diagrams from slides"
                )
                content_notes = gr.Checkbox(
                    label="Speaker Notes",
                    value=True,
                    info="Include speaker notes if available"
                )
                # Optionally, add future toggles here (e.g., audio)
                # content_audio = gr.Checkbox(label="Audio (future)", value=False, interactive=False)
                
                gr.Markdown("### 🖼️ Image Occlusion")
                enable_image_occlusion = gr.Checkbox(
                    label="Enable Image Occlusion Flashcards",
                    value=False,
                    info="Generate flashcards by hiding parts of diagrams/images (image occlusion)"
                )
                occlusion_mode = gr.Dropdown(
                    choices=["AI-assisted (AI detects labels/regions)"],
                    value="AI-assisted (AI detects labels/regions)",
                    label="Occlusion Mode",
                    interactive=False,  # Only one option, so not interactive
                    info="Image occlusion is fully AI-assisted. Manual annotation is currently disabled."
                )
                
                occlusion_image = gr.Image(
                    label="Occlusion Image (hidden)",
                    visible=False
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Flashcards",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Progress & Results")
                status_output = gr.Textbox(
                    label="Status",
                    value="Ready to generate flashcards...",
                    interactive=False,
                    lines=2
                )
                
                # Remove label argument for compatibility
                progress_bar = gr.Progress()
                
                flashcard_output = gr.Textbox(
                    label="Generated Flashcards",
                    value="Flashcards will appear here after generation...",
                    interactive=False,
                    lines=15
                )
                
                with gr.Row():
                    apkg_download = gr.File(
                        label="Download Anki .apkg",
                        visible=True
                    )
                    pdf_download = gr.File(
                        label="Download PDF Notes",
                        visible=True
                    )
        
        # Add a state to hold slide_images for dynamic UI
        slide_images_state = gr.State([])

        # Add a dependency to update the occlusion image when toggles change
        enable_image_occlusion.change(
            fn=update_occlusion_image_editor,
            inputs=[enable_image_occlusion, occlusion_mode, slide_images_state],
            outputs=occlusion_image
        )
        occlusion_mode.change(
            fn=update_occlusion_image_editor,
            inputs=[enable_image_occlusion, occlusion_mode, slide_images_state],
            outputs=occlusion_image
        )

        # Expose sliders in the UI
        conf_threshold_slider = gr.Slider(
            minimum=10,
            maximum=100,
            value=config['conf_threshold'],
            step=1,
            label="Confidence Threshold"
        )

        max_masks_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=config['max_masks_per_image'],
            step=1,
            label="Max Masks per Image"
        )

        # Update the function to receive slider values directly
        generate_btn.click(
            fn=run_flashcard_generation,
            inputs=[
                pptx_file,
                category,
                exam,
                flashcard_type,
                answer_format,
                use_cloze,
                generate_glossary,
                generate_topic_map,
                generate_summary,
                quality_control,
                anti_repetition,
                conciseness,
                context_enrichment,
                depth_consistency,
                auto_cloze,
                content_images,
                content_notes,
                enable_image_occlusion,
                occlusion_mode,
                occlusion_image,
                conf_threshold_slider,
                max_masks_slider
            ],
            outputs=[
                flashcard_output,
                status_output,
                apkg_download,
                pdf_download
            ]
        )
        
        gr.Markdown("---")
        gr.Markdown("### 📖 How to Use")
        gr.Markdown("""
        1. **Upload** your PowerPoint lecture file (.pptx)
        2. **Configure** settings based on your study needs
        3. **Enable Quality Control** for better flashcard quality
        4. **Click Generate** and wait for processing
        5. **Download** the CSV file and import into Anki
        
        ### 🎯 Quality Control Features
        - **Anti-Repetition**: Removes duplicate cards testing the same concept
        - **Conciseness**: Splits wordy answers into focused, atomic cards
        - **Context Enrichment**: Adds clinical context to shallow memorization cards
        - **Depth Consistency**: Ensures Level 1 cards are basic and Level 2 cards require reasoning
        - **Auto-Cloze**: Automatically converts suitable cards to cloze format
        """)
        
        gr.Markdown("### 🔧 Technical Details")
        gr.Markdown("""
        - Uses OpenAI GPT-4o for intelligent flashcard generation
        - Semantic processing for better content understanding
        - TF-IDF similarity for duplicate detection
        - NLTK for text processing and quality analysis
        - Automatic confidence scoring for card quality
        """)
    
    return interface

# Add a function to update the occlusion image visibility and value
def update_occlusion_image(enable_image_occlusion, occlusion_mode, slide_images):
    if enable_image_occlusion and occlusion_mode == "Semi-automated (user selects regions)" and slide_images and slide_images[0]:
        return gr.update(visible=True, value=slide_images[0][0])
    else:
        return gr.update(visible=False, value=None)

if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    ) 