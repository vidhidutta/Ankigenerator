#!/usr/bin/env python3
"""
Test script for progress tracking functionality
"""

import time
import gradio as gr

def test_progress_function(progress=gr.Progress()):
    """Test function to demonstrate progress tracking"""
    progress(0, desc="Starting test...")
    time.sleep(1)
    
    progress(0.2, desc="Processing step 1...")
    time.sleep(1)
    
    progress(0.4, desc="Processing step 2...")
    time.sleep(1)
    
    progress(0.6, desc="Processing step 3...")
    time.sleep(1)
    
    progress(0.8, desc="Processing step 4...")
    time.sleep(1)
    
    progress(1.0, desc="Complete!")
    
    return "✅ Progress tracking test completed successfully!"

def create_test_interface():
    with gr.Blocks(title="Progress Test", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧪 Progress Tracking Test")
        
        test_btn = gr.Button("🚀 Test Progress", variant="primary")
        
        progress_bar = gr.Progress()
        
        status_text = gr.Textbox(
            label="Status",
            value="Ready to test progress tracking",
            interactive=False,
            lines=2
        )
        
        output_text = gr.Textbox(
            label="Output",
            interactive=False,
            lines=5
        )
        
        test_btn.click(
            fn=test_progress_function,
            outputs=[output_text]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_test_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8081,
        share=False,
        debug=True
    ) 