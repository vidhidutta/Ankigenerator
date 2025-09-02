FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Node.js for building the frontend
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy package files and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole repo so flashcard_generator.py and friends are inside the image
COPY . .

# Build the React frontend
WORKDIR /app/ojamed-web
RUN npm install && npm run build

# Go back to app directory
WORKDIR /app

# Ensure /app is on PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8000

# Keep your existing CMD, or use Render's PORT with a sane default
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
