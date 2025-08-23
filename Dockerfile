FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

# Render will inject $PORT; default to 8000 locally
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
