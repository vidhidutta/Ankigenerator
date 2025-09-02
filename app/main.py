import os
import tempfile
import zipfile
import traceback
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import app.pipeline as pipeline

app = FastAPI(title="OjaMed API", version="2.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (React frontend)
if os.path.exists("ojamed-web/dist"):
    app.mount("/assets", StaticFiles(directory="ojamed-web/dist/assets"), name="assets")
    app.mount("/static", StaticFiles(directory="ojamed-web/dist"), name="static")

@app.get("/")
def read_root():
    # Serve the React frontend
    if os.path.exists("ojamed-web/dist/index.html"):
        return FileResponse("ojamed-web/dist/index.html")
    return {"message": "OjaMed Flashcard Generator API", "version": "2.2.0"}

@app.get("/diag")
def diag():
    import app.pipeline as pipeline
    return {
        "zip_names": "deck.* enabled",
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "demo": os.getenv("OJAMED_FORCE_DEMO") == "1",
        "debug": os.getenv("OJAMED_DEBUG") == "1",
        "holistic_analysis": os.getenv("OJAMED_HOLISTIC_ANALYSIS", "1") == "1",
        "pipeline_tag": getattr(pipeline, "PIPELINE_TAG", "unknown"),
    }

@app.post("/convert")
async def convert(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Demo short-circuit
        if os.getenv("OJAMED_FORCE_DEMO") == "1":
            demo_cards = [
                ("What drug class is furosemide?", "Loop diuretic"),
                ("Main adverse effect?", "Hypokalemia"),
                ("Contraindicated with?", "Sulfa allergy (relative)"),
            ]
            
            # Create temp directory for demo files
            tmp_dir = tempfile.mkdtemp(prefix="ojamed_demo_")
            csv_path = os.path.join(tmp_dir, "deck.csv")
            apkg_path = os.path.join(tmp_dir, "deck.apkg")
            
            # Write demo CSV
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["question","answer"])
                w.writerows(demo_cards)
            
            # Write demo APKG
            import genanki
            model = genanki.Model(
                1607392319, "Basic (OjaMed)",
                fields=[{"name":"Question"},{"name":"Answer"}],
                templates=[{"name":"Card 1","qfmt":"{{Question}}","afmt":"{{FrontSide}}<hr id='answer'>{{Answer}}"}],
            )
            deck = genanki.Deck(2059400110, "OjaMed Demo Deck")
            for q,a in demo_cards:
                deck.add_note(genanki.Note(model=model, fields=[q,a]))
            genanki.Package(deck).write_to_file(apkg_path)
            
            # Create demo comprehensive notes PDF
            demo_pdf_path = os.path.join(tmp_dir, "demo_comprehensive_notes.pdf")
            try:
                from medical_notes_pdf_generator import MedicalNotesPDFGenerator
                # Create a simple demo analysis object
                class DemoAnalysis:
                    lecture_title = "Demo Lecture"
                    main_topics = ["Demo Topic 1", "Demo Topic 2"]
                    concepts = []
                    mind_maps = []
                    knowledge_gaps = []
                    filled_gaps = []
                    learning_objectives = ["Understand demo concepts", "Apply demo knowledge"]
                    clinical_pearls = ["Demo clinical insight 1", "Demo clinical insight 2"]
                    glossary = {"Demo Term": "Demo definition"}
                    cross_references = []
                
                demo_analysis = DemoAnalysis()
                generator = MedicalNotesPDFGenerator(demo_pdf_path)
                generator.generate_comprehensive_notes(demo_analysis)
                print(f"[OjaMed][DEMO] Demo PDF generated: {demo_pdf_path}")
            except Exception as e:
                print(f"[OjaMed][DEMO] Demo PDF generation failed: {e}")
                demo_pdf_path = None
            
            # Zip demo files
            zip_path = os.path.join(tmp_dir, "ojamed_deck.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(apkg_path, arcname="deck.apkg")
                z.write(csv_path, arcname="deck.csv")
                if demo_pdf_path:
                    z.write(demo_pdf_path, arcname="demo_comprehensive_notes.pdf")
            
            return FileResponse(zip_path, media_type="application/zip", filename="ojamed_deck.zip")
        
        # Save uploaded file
        upload_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"[OjaMed][MAIN] File uploaded: {upload_path}")
        
        # Run pipeline
        pipeline_result = pipeline.run_pipeline(upload_path)
        
        # Create ZIP with all outputs
        zip_path = os.path.join(pipeline_result['temp_dir'], "ojamed_deck.zip")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            # Always include flashcards
            z.write(pipeline_result['apkg_path'], arcname="deck.apkg")
            z.write(pipeline_result['csv_path'], arcname="deck.csv")
            
            # Include comprehensive notes PDF if generated
            if pipeline_result['comprehensive_notes_pdf']:
                pdf_filename = os.path.basename(pipeline_result['comprehensive_notes_pdf'])
                z.write(pipeline_result['comprehensive_notes_pdf'], arcname=pdf_filename)
                print(f"[OjaMed][MAIN] 📄 Added comprehensive notes to ZIP: {pdf_filename}")
            else:
                print("[OjaMed][MAIN] No comprehensive notes PDF to include in ZIP")
        
        # Clean up uploaded file
        background_tasks.add_task(os.unlink, upload_path)
        
        print(f"[OjaMed][MAIN] ZIP created successfully: {zip_path}")
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
