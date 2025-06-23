#!/usr/bin/env python3
"""
Demonstration script for progress tracking in flashcard generation
"""

import time
import gradio as gr

def simulate_flashcard_generation(progress=gr.Progress()):
    """Simulate the flashcard generation process with progress tracking"""
    
    # Step 1: Starting
    progress(0, desc="Starting flashcard generation...")
    time.sleep(0.5)
    
    # Step 2: Extract text
    progress(0.1, desc="Extracting text from PowerPoint...")
    time.sleep(0.5)
    
    # Step 3: Extract images
    progress(0.2, desc="Extracting images from PowerPoint...")
    time.sleep(0.5)
    
    # Step 4: Filter slides
    progress(0.3, desc="Filtering slides to remove navigation content...")
    time.sleep(0.5)
    
    # Step 5: AI Generation (simulate multiple slides)
    total_slides = 5
    for i in range(total_slides):
        progress_percent = 0.4 + (0.45 * (i / total_slides))
        progress(progress_percent, desc=f"Processing slide {i+1}/{total_slides}...")
        time.sleep(1)  # Simulate AI processing time
        
        # Update with flashcard count
        cards_generated = (i + 1) * 3  # Simulate 3 cards per slide
        progress(progress_percent, desc=f"Slide {i+1}/{total_slides}: Generated 3 cards (Total: {cards_generated})")
    
    # Step 6: Export
    progress(0.9, desc="Exporting flashcards to CSV...")
    time.sleep(0.5)
    
    # Step 7: Extra materials
    progress(0.95, desc="Generating extra materials...")
    time.sleep(0.5)
    
    # Step 8: Complete
    progress(1.0, desc="Finalizing...")
    time.sleep(0.5)
    
    return f"""✅ Flashcard Generation Complete!

📊 Summary:
• Processed {total_slides} slides
• Generated {total_slides * 3} flashcards
• Extracted text and images
• Created CSV export
• Generated extra materials

🎯 Progress tracking worked perfectly!"""

def create_demo_interface():
    with gr.Blocks(title="Progress Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎯 Progress Tracking Demonstration")
        gr.Markdown("This demo shows how the progress tracking works during flashcard generation.")
        
        demo_btn = gr.Button("🚀 Start Demo", variant="primary", size="lg")
        
        # Progress bar
        progress_bar = gr.Progress()
        
        # Status display
        status_text = gr.Textbox(
            label="Status",
            value="Ready to demonstrate progress tracking",
            interactive=False,
            lines=2
        )
        
        # Output
        output_text = gr.Textbox(
            label="Demo Results",
            interactive=False,
            lines=10
        )
        
        demo_btn.click(
            fn=simulate_flashcard_generation,
            outputs=[output_text]
        )
        
        # Instructions
        with gr.Accordion("ℹ️ How Progress Tracking Works", open=False):
            gr.Markdown("""
            **Progress Tracking Phases:**
            
            1. **Starting (0%)**: Initialization and setup
            2. **Text Extraction (10%)**: Extracting text from PowerPoint slides
            3. **Image Extraction (20%)**: Extracting images and diagrams
            4. **Content Filtering (30%)**: Removing navigation and empty slides
            5. **AI Generation (40-85%)**: Processing each slide with AI
               - Shows "Slide X/Y: Generated N cards (Total: M)"
            6. **Export (90%)**: Saving flashcards to CSV
            7. **Extra Materials (95%)**: Generating glossary, topic map, etc.
            8. **Complete (100%)**: Finalization and file preparation
            
            **Benefits:**
            - Real-time feedback for large files
            - Clear indication of processing time
            - Easy to identify where issues occur
            - Professional user experience
            """)
    
    return interface

if __name__ == "__main__":
    interface = create_demo_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8082,
        share=False,
        debug=True
    ) 