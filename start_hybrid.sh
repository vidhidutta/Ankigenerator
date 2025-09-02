#!/bin/bash
# Hybrid startup script for ojamed-web + Python API backend

echo "🚀 Starting Hybrid Image Occlusion System"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ FastAPI not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "🌐 Starting FastAPI backend server..."
echo "   API will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo ""

# Start the FastAPI server
python api_server.py &

# Store the PID
API_PID=$!

echo "✅ FastAPI server started (PID: $API_PID)"
echo ""
echo "📱 Next steps:"
echo "   1. Open a new terminal"
echo "   2. cd ojamed-web"
echo "   3. npm run dev"
echo "   4. Open http://localhost:5173 in your browser"
echo ""
echo "🔗 The frontend will automatically connect to the API at http://localhost:8000"
echo ""
echo "⏹️  To stop the API server, run: kill $API_PID"
echo ""

# Wait for user to stop
echo "Press Ctrl+C to stop the API server..."
wait $API_PID



