FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy package files and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole repo so flashcard_generator.py and friends are inside the image
COPY . .

# Debug: Check what's in the frontend directories
RUN echo "=== Contents of frontend-dist ===" && ls -la frontend-dist/ || echo "frontend-dist directory not found" && echo "=== Contents of ojamed-web ===" && ls -la ojamed-web/ || echo "ojamed-web directory not found"

# Ensure /app is on PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8000

# Keep your existing CMD, or use Render's PORT with a sane default
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
