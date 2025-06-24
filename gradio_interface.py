import gradio as gr
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import sys
import time

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
    tuple_to_dict
)

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
    progress=gr.Progress()
):
    """
    Main function to generate flashcards from uploaded PowerPoint file
    """
    if not pptx_file:
        return "Please upload a PowerPoint file.", "No file uploaded", None, None, None
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set. Please check your .env file.", "API key not found", None, None, None
    
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
            else:
                slide_images = [[] for _ in slide_texts]
            
            if not slide_texts:
                return "Error: No text found in PowerPoint file or error occurred during extraction.", "No text found", None, None, None
            
            # Filter out slides that should be skipped
            progress(0.3, desc="Filtering slides to remove navigation content...")
            status_msg = "Filtering slides to remove navigation content..."
            filtered_slide_texts = filter_slides(slide_texts)
            print(f"Processing {len(filtered_slide_texts)} slides (filtered from {len(slide_texts)} total slides)")
            
            # Filter slide_images to match filtered_slide_texts
            kept_slide_indices = []
            for i, slide_text in enumerate(slide_texts):
                if not should_skip_slide(slide_text):
                    kept_slide_indices.append(i)
            
            filtered_slide_images = [slide_images[i] for i in kept_slide_indices]
            
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
                print(f"Quality control settings: {quality_settings}")
            
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
                return "No flashcards were generated. Please check your PowerPoint content.", "No flashcards generated", None, None, None
            
            # After quality control, ensure we keep Flashcard objects, not tuples
            # Remove any conversion to (question, answer) tuples after quality control
            # Use the Flashcard objects directly for export
            flashcard_objs_for_export = all_flashcards
            # Only convert to dicts for display/summary, not for export
            all_flashcards_dicts = [tuple_to_dict(card, use_cloze=use_cloze) for card in flashcard_objs_for_export]
            
            # Export flashcards to APKG using Flashcard objects, not dicts
            progress(0.9, desc="Exporting flashcards to Anki .apkg...")
            status_msg = "Exporting flashcards to Anki .apkg..."
            apkg_path = os.path.join(temp_dir, "generated_flashcards.apkg")
            print(f"[DEBUG] First 3 card dicts to be exported: {all_flashcards_dicts[:3]}")
            print(f"[DEBUG] Card types: {[d.get('type') for d in all_flashcards_dicts]}")
            print(f"[DEBUG] Calling export_flashcards_to_apkg with {len(flashcard_objs_for_export)} cards, output: {apkg_path}")
            try:
                export_flashcards_to_apkg(flashcard_objs_for_export, apkg_path)
                print(f"[DEBUG] export_flashcards_to_apkg completed. File exists: {os.path.exists(apkg_path)}")
            except Exception as e:
                print(f"[DEBUG] Error in export_flashcards_to_apkg: {e}")
            
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
            for i, d in enumerate(all_flashcards_dicts[:5]):  # Show first 5
                flashcard_summary += f"\n{i+1}. Q: {d['question']}\n   A: {d['answer']}\n"
            
            if len(all_flashcards_dicts) > 5:
                flashcard_summary += f"\n... and {len(all_flashcards_dicts) - 5} more flashcards"
            
            # Copy files to accessible location
            final_csv_path = None
            final_apkg_path = "generated_flashcards.apkg"
            final_pdf_path = "lecture_notes.pdf" if extra_materials else None
            
            shutil.copy2(apkg_path, final_apkg_path)
            if extra_materials and pdf_path and os.path.exists(pdf_path):
                shutil.copy2(pdf_path, "lecture_notes.pdf")
            
            final_status = f"✅ Complete! Generated {len(all_flashcards_dicts)} quality-controlled flashcards from {len(filtered_slide_texts)} slides."
            return flashcard_summary, final_status, final_csv_path, final_apkg_path, final_pdf_path
            
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        return error_msg, f"❌ Error: {str(e)}", None, None, None

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
        
        # Update the function call to include quality control parameters
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
                content_notes
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
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Changed to port 7860
        share=True,             # Create a public link
        debug=True
    ) 