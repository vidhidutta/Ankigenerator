#!/usr/bin/env python3
"""
Server monitoring script to ensure backend and frontend stay connected
"""

import requests
import time
import subprocess
import sys
import os

def check_backend():
    """Check if backend server is responding"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_frontend():
    """Check if frontend server is responding"""
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        return response.status_code == 200
    except:
        return False

def restart_backend():
    """Restart backend server"""
    print("🔄 Restarting backend server...")
    try:
        # Kill existing processes
        subprocess.run(["pkill", "-f", "uvicorn"], check=False)
        time.sleep(2)
        
        # Start new server
        os.chdir("/home/vidhidutta/anki-flashcard-generator")
        subprocess.Popen([
            "bash", "-c", 
            "source venv/bin/activate && uvicorn api_server:app --host 0.0.0.0 --port 8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ Backend server restarted")
        return True
    except Exception as e:
        print(f"❌ Failed to restart backend: {e}")
        return False

def restart_frontend():
    """Restart frontend server"""
    print("🔄 Restarting frontend server...")
    try:
        # Kill existing processes
        subprocess.run(["pkill", "-f", "npm run dev"], check=False)
        time.sleep(2)
        
        # Start new server
        os.chdir("/home/vidhidutta/anki-flashcard-generator/ojamed-web")
        subprocess.Popen([
            "bash", "-c", 
            "npm run dev -- --host 0.0.0.0"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ Frontend server restarted")
        return True
    except Exception as e:
        print(f"❌ Failed to restart frontend: {e}")
        return False

def main():
    """Main monitoring loop"""
    print("🚀 Starting server monitoring...")
    print("📊 Backend: http://localhost:8000")
    print("🌐 Frontend: http://localhost:5173")
    print("⏰ Monitoring every 30 seconds...")
    print("-" * 50)
    
    while True:
        backend_ok = check_backend()
        frontend_ok = check_frontend()
        
        status_backend = "✅" if backend_ok else "❌"
        status_frontend = "✅" if frontend_ok else "❌"
        
        print(f"{time.strftime('%H:%M:%S')} - Backend: {status_backend} | Frontend: {status_frontend}")
        
        if not backend_ok:
            print("⚠️  Backend server down, attempting restart...")
            restart_backend()
            time.sleep(5)  # Wait for restart
        
        if not frontend_ok:
            print("⚠️  Frontend server down, attempting restart...")
            restart_frontend()
            time.sleep(5)  # Wait for restart
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        sys.exit(1)


