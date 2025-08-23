import os
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.pipeline import run_pipeline

# ---- Basic knobs (adjust via env later) ----
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")]

app = FastAPI(title="OjaMed API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def _stream_to_temp(upload: UploadFile, max_mb: int) -> str:
    """Stream uploaded file to a temp location with size guard."""
    size = 0
    with tempfile.NamedTemporaryFile(prefix="ojamed_", suffix=f"_{upload.filename}", delete=False) as tmp:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File too large (>{max_mb} MB)")
            tmp.write(chunk)
        return tmp.name

def _zip_outputs(paths: list[str]) -> str:
    zip_path = tempfile.mktemp(prefix="ojamed_", suffix=".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            arc = Path(p).name
            z.write(p, arcname=arc if arc else Path(p).name)
    return zip_path

def _cleanup(paths: list[str]):
    for p in paths:
        try:
            if Path(p).is_file():
                os.remove(p)
            elif Path(p).is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

@app.post("/convert")
def convert(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # 1) Save to temp with size guard
    input_path = _stream_to_temp(file, MAX_FILE_MB)

    # 2) Run your real generator (stubbed now)
    apkg_path, csv_path = run_pipeline(input_path)

    # 3) Package both into a single ZIP to simplify browser download
    zip_path = _zip_outputs([apkg_path, csv_path])

    # 4) Clean up temp files after response is sent
    background.add_task(_cleanup, [input_path, apkg_path, csv_path, zip_path])

    # 5) Return the ZIP
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="ojamed_deck.zip",
    )
