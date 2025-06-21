#!/usr/bin/env python3
"""
Simple launcher for the Medical Flashcard Generator UI
"""

import subprocess
import sys
import os

def main():
    print("🧠 Medical Flashcard Generator UI")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("gradio_interface.py"):
        print("❌ Error: gradio_interface.py not found in current directory")
        print("Please run this script from the anki-flashcard-generator directory")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected")
        print("Please activate your virtual environment first:")
        print("source venv/bin/activate")
        print()
    
    # Check if OpenAI API key is set
    if not os.path.exists(".env"):
        print("❌ Error: .env file not found")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Starting Gradio interface...")
    print("📱 The interface will be available at: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch the Gradio interface
        subprocess.run([sys.executable, "gradio_interface.py"])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")

if __name__ == "__main__":
    main() 