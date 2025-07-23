# 🧠 CONTEXT: My Anki Automation App (Built for Medical Students)

I'm building an AI-powered Python app that takes PowerPoint (`.pptx`) lecture slides and automatically generates **Anki flashcards**, including **text-based cards**, **cloze deletions**, and **image occlusion cards**. The goal is to help **medical students** (like myself) understand material better and save hours every week.

---

## 🔄 NEW CORE VISION (WITH LECTURE RECORDINGS)

This app isn't just about slides — it's about **contextual intelligence**.

Students often rely not just on slides, but on what the **lecturer says** — tone, emphasis, repetition, side explanations — to figure out what’s actually important.

So this app will also allow students to **upload their lecture recordings**, which will be transcribed and analysed alongside the slides to generate flashcards that reflect:

- The **lecturer’s key points**
- Emphasised or repeated concepts
- Definitions, examples, and explanations provided during speech
- Higher-level understanding not always found in the slides

The vision is to replicate what a top-performing student would do manually:
1. Listen to the lecture
2. Connect it with slides
3. Extract and structure important information for Anki

Except this app does it automatically — and much faster.

---

## 🎯 GOALS (For You, Agent)
You're here to help me:
1. Fix bugs (especially image occlusion errors)
2. Improve performance (avoid duplicate images, improve mask detection)
3. Add new features (like integrating lecture audio analysis)
4. Test & refactor my codebase (clean architecture and scalability)
5. Support business launch (landing page, user onboarding, feedback loops)

---

# 🔧 HOW THE APP WORKS (Current Flow)

1. **Input**: User uploads `.pptx` lecture slides *(and soon audio files)*.
2. **Text Extraction**: Extracts slide content using `python-pptx`.
3. **Chunking + Processing**: Uses semantic processing (TF-IDF or OpenAI API) to generate:
   - Basic Q&A cards
   - Cloze deletions
   - "Level 1", "Level 2", "Level 3" flashcards (increasing depth)
4. **Image Occlusion**:
   - Extracts images from slides
   - Applies OCR (pytesseract) and masking
   - Saves debug masks and exports to Anki-compatible CSV
5. **Export**: Outputs `.csv` or `.apkg` files for Anki import

---

# 🚧 CURRENT ISSUES (Need Help With)

### 🖼️ Image Occlusion
- Masks are often **irrelevant or overlapping**, and don’t isolate meaningful regions
- Too many **duplicate versions** of the same image are saved
- Some slides process **shapes with no images**, wasting memory
- Final export sometimes includes **zero occlusion flashcards**

### ⚙️ Architecture Problems
- Need help **structuring** the app better (consider `src/` layout or modular architecture)
- Organising `config.yaml` (e.g. confidence threshold, max masks per image)
- Add **unit tests** or **better logging/debugging**

---

# 🛠 STACK

- Python 3.x
- `python-pptx`
- `pytesseract` for OCR
- `matplotlib`, `PIL`, `OpenCV` for image processing
- `openai` or `transformers` for language model processing
- CSV output (currently), APKG support (planned)
- Gradio frontend (in-progress)
- Git for version control

---

# 📈 FUTURE ROADMAP

### Feature Ideas
- Lecture audio transcription (via Whisper or OpenAI)
- Integrating audio + slide alignment for smarter flashcards
- Cloze + image occlusion hybrid cards
- Auto-tagging flashcards by topic
- Multi-input: support for extra readings or textbook chapters
- Better user configuration (confidence slider, max flashcards per slide)
- Plugin system (user-defined prompt templates)
- Sync to Anki via AnkiConnect or add-on

### Business + UX
- Pricing ideas (freemium? 50 free cards/month?)
- Marketing to med students (TikTok, societies, reps)
- Creating a landing page & onboarding flow
- Referral + feedback system

---

# 📂 FILE STRUCTURE (Typical)
You may see files like:


anki-flashcard-generator/
 │
 ├── app.py
 ├── config.yaml
 ├── gradio_interface.py
 ├── image_occlusion.py
 ├── semantic_processing.py
 ├── pptx_extractor.py
 ├── anki_exporter.py
 ├── utils/
 │ └── helpers.py
 ├── debug/
 ├── output/
 ├── templates/
 ├── README.md
 └── requirements.txt

---

# 🤖 AGENT TASKS (Suggested First Steps)

1. Diagnose and fix image occlusion bugs.
2. Refactor image masking logic to reduce duplication and improve accuracy.
3. Help structure the app more cleanly (e.g. Python modules and CLI).
4. Prepare for adding lecture audio transcription using Whisper or OpenAI.
5. Write unit tests for `extract_text`, `generate_masks`, etc.
6. Suggest a go-to-market plan for beta testing among UK medical students.

