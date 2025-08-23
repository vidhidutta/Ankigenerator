#!/usr/bin/env python3
"""
Simple launcher for the Anki Flashcard Generator
Always uses a different port to avoid conflicts
"""

import os
import sys
import subprocess

def main():
    """Main launcher function"""
    print("🚀 Starting Anki Flashcard Generator on port 7865...")
    
    # Set environment variable to use port 7865
    env = os.environ.copy()
    env['GRADIO_SERVER_PORT'] = '7865'
    
    # Kill any existing Gradio processes
    try:
        subprocess.run(['pkill', '-f', 'gradio'], check=False)
        print("🔄 Killed existing Gradio processes")
    except:
        pass
    
    # Start the application
    try:
        print("🌐 Starting server on http://localhost:7865")
        print("📝 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the gradio interface with the environment variable set
        subprocess.run([
            sys.executable, 'gradio_interface.py'
        ], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 