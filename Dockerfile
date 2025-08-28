FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole repo so flashcard_generator.py and friends are inside the image
COPY . .

# Ensure /app is on PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8000

# Keep your existing CMD, or use Render's PORT with a sane default
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
