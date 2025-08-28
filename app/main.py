import os
import csv
import tempfile
import zipfile
import genanki
import shutil
import traceback
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.pipeline import run_pipeline

# ---- Basic knobs (adjust via env later) ----
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://.pages.dev"
).split(",")]

app = FastAPI(title="OjaMed API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True}

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
        try:
            if len(paths) > 0 and paths[0]:
                z.write(paths[0], arcname="deck.apkg")
            if len(paths) > 1 and paths[1]:
                z.write(paths[1], arcname="deck.csv")
        except Exception:
            # Best effort: fall back to original names if indexing fails
            for p in paths or []:
                z.write(p, arcname=Path(p).name)
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
async def convert(background: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pptx', '.ppt', '.pdf')):
            raise HTTPException(status_code=400, detail="Only .pptx, .ppt, and .pdf files are supported")
        
        # Save uploaded file to temp location
        input_path = tempfile.mktemp(prefix="ojamed_input_", suffix=Path(file.filename).suffix)
        try:
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            file.file.close()
        
        # Demo short-circuit
        if os.getenv("OJAMED_FORCE_DEMO") == "1":
            import csv
            import genanki
            demo_cards = [
                ("What drug class is furosemide?", "Loop diuretic"),
                ("Main adverse effect?", "Hypokalemia"),
                ("Contraindicated with?", "Sulfa allergy (relative)"),
            ]
            # write CSV
            tmp_dir = tempfile.mkdtemp(prefix="ojamed_demo_")
            csv_path = os.path.join(tmp_dir, "deck.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["question","answer"])
                w.writerows(demo_cards)
            # write APKG (Basic model)
            model = genanki.Model(
                1607392319, "Basic (OjaMed)",
                fields=[{"name":"Question"},{"name":"Answer"}],
                templates=[{"name":"Card 1","qfmt":"{{Question}}","afmt":"{{FrontSide}}<hr id='answer'>{{Answer}}"}],
            )
            deck = genanki.Deck(2059400110, "OjaMed Demo Deck")
            for q,a in demo_cards:
                deck.add_note(genanki.Note(model=model, fields=[q,a]))
            apkg_path = os.path.join(tmp_dir, "deck.apkg")
            genanki.Package(deck).write_to_file(apkg_path)
            # zip as stable names
            zip_path = os.path.join(tmp_dir, "ojamed_deck.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(apkg_path, arcname="deck.apkg")
                z.write(csv_path,  arcname="deck.csv")
            return FileResponse(zip_path, media_type="application/zip", filename="ojamed_deck.zip")

        # 2) Run generator
        try:
            apkg_path, csv_path = run_pipeline(input_path)
        except Exception as e:
            # Bubble error in debug mode
            if os.getenv("OJAMED_DEBUG") == "1":
                raise HTTPException(status_code=500, detail=str(e))
            # Non-debug: return 500 with generic message
            raise HTTPException(status_code=500, detail="Generation failed; check logs.")
        
        # 3) Zip the outputs
        zip_path = _zip_outputs([apkg_path, csv_path])
        
        # 4) Clean up temp files in background
        background.add_task(_cleanup, [input_path, apkg_path, csv_path])
        
        return FileResponse(zip_path, media_type="application/zip", filename="ojamed_deck.zip")
    except Exception as e:
        tb = traceback.format_exc()
        print("[OjaMed][ERROR] /convert failed:", repr(e))
        print(tb)
        # If debug flag is on, return the traceback so we can see it from curl
        if os.getenv("OJAMED_DEBUG") == "1":
            return PlainTextResponse(tb, status_code=500)
        # Otherwise keep a generic 500
        return PlainTextResponse("Internal Server Error", status_code=500)


@app.get("/diag")
def diag():
    import app.pipeline as pipeline
    return {
        "zip_names": "deck.* enabled",
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "demo": os.getenv("OJAMED_FORCE_DEMO") == "1",
        "debug": os.getenv("OJAMED_DEBUG") == "1",
        "pipeline_tag": getattr(pipeline, "PIPELINE_TAG", "unknown"),
    }
