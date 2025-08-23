#!/usr/bin/env python3
"""
Launcher script for the Anki Flashcard Generator
Starts the application on a different port to avoid conflicts
"""

import os
import sys
import subprocess
import time

def find_available_port(start_port=7862):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Main launcher function"""
    print("🚀 Starting Anki Flashcard Generator...")
    
    # Find an available port
    port = find_available_port(7862)
    if not port:
        print("❌ Error: No available ports found")
        sys.exit(1)
    
    print(f"✅ Using port: {port}")
    
    # Set environment variables
    env = os.environ.copy()
    env['GRADIO_SERVER_PORT'] = str(port)
    
    # Kill any existing Gradio processes
    try:
        subprocess.run(['pkill', '-f', 'gradio'], check=False)
        time.sleep(2)
    except:
        pass
    
    # Start the application
    try:
        print(f"🌐 Starting server on http://localhost:{port}")
        print("📝 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the gradio interface
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