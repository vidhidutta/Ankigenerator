import gradio as gr
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Import our existing flashcard generator functions
from flashcard_generator import (
    extract_text_from_pptx,
    extract_images_from_pptx,
    filter_slides,
    should_skip_slide,
    generate_multimodal_flashcards_http,
    parse_flashcards,
    export_flashcards_to_csv,
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
    OPENAI_API_KEY
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
    generate_summary
):
    """
    Main function to generate flashcards from uploaded PowerPoint file
    """
    if not pptx_file:
        return "Please upload a PowerPoint file.", None, None
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set. Please check your .env file.", None, None
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy uploaded file to temp directory
            temp_pptx_path = os.path.join(temp_dir, "uploaded_presentation.pptx")
            shutil.copy2(pptx_file.name, temp_pptx_path)
            
            # Extract text and images
            print("Extracting text and images from PowerPoint...")
            slide_texts = extract_text_from_pptx(temp_pptx_path)
            slide_images = extract_images_from_pptx(temp_pptx_path, os.path.join(temp_dir, "slide_images"))
            
            if not slide_texts:
                return "Error: No text found in PowerPoint file or error occurred during extraction.", None, None
            
            # Filter out slides that should be skipped
            print("Filtering slides to remove navigation and empty content...")
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
            
            # Generate flashcards
            print("Generating multimodal flashcards with AI (text + images) via HTTP API...")
            all_flashcards = generate_multimodal_flashcards_http(
                filtered_slide_texts, 
                filtered_slide_images, 
                OPENAI_API_KEY, 
                MODEL_NAME, 
                MAX_TOKENS, 
                TEMPERATURE
            )
            
            if not all_flashcards:
                return "No flashcards were generated. Please check your PowerPoint content.", None, None
            
            # Export flashcards to CSV
            csv_path = os.path.join(temp_dir, "generated_flashcards.csv")
            export_flashcards_to_csv(all_flashcards, csv_path)
            
            # Generate extra materials if requested
            extra_materials = ""
            if any([generate_glossary, generate_topic_map, generate_summary]):
                print("Generating extra materials...")
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
            
            # Prepare output
            flashcard_summary = f"Successfully generated {len(all_flashcards)} flashcards!\n\n"
            flashcard_summary += "Sample flashcards:\n"
            for i, (question, answer) in enumerate(all_flashcards[:5]):  # Show first 5
                flashcard_summary += f"\n{i+1}. Q: {question}\n   A: {answer}\n"
            
            if len(all_flashcards) > 5:
                flashcard_summary += f"\n... and {len(all_flashcards) - 5} more flashcards"
            
            # Copy files to accessible location
            final_csv_path = "generated_flashcards.csv"
            final_pdf_path = "lecture_notes.pdf" if extra_materials else None
            
            shutil.copy2(csv_path, final_csv_path)
            if extra_materials and os.path.exists(os.path.join(temp_dir, "lecture_notes.pdf")):
                shutil.copy2(os.path.join(temp_dir, "lecture_notes.pdf"), "lecture_notes.pdf")
            
            return flashcard_summary, final_csv_path, final_pdf_path
            
    except Exception as e:
        return f"Error during processing: {str(e)}", None, None

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
                    file_count="single"
                )
                
                gr.Markdown("### ⚙️ Settings")
                category = gr.Dropdown(
                    choices=["Pharmacology", "Neuroscience", "Psychiatry", "Cardiology", "Other"],
                    value="Pharmacology",
                    label="Medical Category"
                )
                
                exam = gr.Dropdown(
                    choices=["Year 1 MBBS", "Year 2 MBBS", "Year 3 MBBS", "Year 4 MBBS", "Year 5 MBBS", "Finals", "PLAB", "USMLE", "Other"],
                    value="Year 2 MBBS",
                    label="Exam Level"
                )
                
                flashcard_type = gr.Radio(
                    choices=["Level 1", "Level 2", "Both"],
                    value="Level 1",
                    label="Flashcard Type"
                )
                
                answer_format = gr.Dropdown(
                    choices=["minimal", "bullet_points", "short_paragraph", "mixture", "best"],
                    value="bullet_points",
                    label="Answer Format"
                )
                
                use_cloze = gr.Checkbox(
                    label="Use Cloze Deletions",
                    value=False
                )
                
                gr.Markdown("### 📚 Extra Materials")
                generate_glossary = gr.Checkbox(
                    label="Generate Glossary",
                    value=True
                )
                
                generate_topic_map = gr.Checkbox(
                    label="Generate Topic Map",
                    value=True
                )
                
                generate_summary = gr.Checkbox(
                    label="Generate Summary Sheet",
                    value=False
                )
                
                generate_btn = gr.Button("🚀 Generate Flashcards", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Results")
                output_text = gr.Textbox(
                    label="Generated Content",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
                
                with gr.Row():
                    csv_download = gr.File(
                        label="Download Flashcards (CSV)",
                        file_types=[".csv"],
                        interactive=False
                    )
                    
                    pdf_download = gr.File(
                        label="Download Notes (PDF)",
                        file_types=[".pdf"],
                        interactive=False
                    )
        
        # Instructions
        with gr.Accordion("ℹ️ Instructions", open=False):
            gr.Markdown("""
            **How to use:**
            1. Upload your PowerPoint lecture file (.pptx)
            2. Configure your settings (category, exam level, etc.)
            3. Click "Generate Flashcards"
            4. Download the generated CSV file and import it into Anki
            
            **Tips:**
            - The tool automatically filters out navigation slides and irrelevant content
            - Only medical/exam-relevant content is converted to flashcards
            - Images in your slides are analyzed for additional content
            - Generated flashcards are optimized for medical exam preparation
            """)
        
        # Connect the button to the function
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
                generate_summary
            ],
            outputs=[output_text, csv_download, pdf_download]
        )
    
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
        server_port=8080,       # Changed to port 8080
        share=True,             # Create a public link
        debug=True
    ) 