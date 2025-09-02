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
    clean_cloze_text,
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

def create_interface():
    # Custom CSS for dark theme
    custom_css = """
    .apex-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-height: 100vh;
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
                        
                        with gr.Row():
                            apkg_download = gr.File(
                                label="Download Anki .apkg",
                                visible=True,
                                container=False
                            )
            
            # Footer
            gr.HTML("""
            <div class="footer">
                <a href="#" class="footer-link">About</a>
                <a href="#" class="footer-link">FAQs</a>
                <a href="#" class="footer-link">Contact Us</a>
            </div>
            """)

        # Simple test function
        def test_generation(pptx_file, flashcard_level, card_type, question_style, content_images, content_notes):
            if not pptx_file:
                return "Please upload a PowerPoint file.", "No file uploaded", None
            
            return f"Test: File uploaded successfully!", "✅ Test completed", None

        # Connect the generate button
        generate_btn.click(
            fn=test_generation,
            inputs=[
                pptx_file,
                flashcard_level,
                card_type,
                question_style,
                content_images,
                content_notes
            ],
            outputs=[
                flashcard_output,
                status_output,
                apkg_download
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

    # Allow server_port override via CLI or env
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7862))
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