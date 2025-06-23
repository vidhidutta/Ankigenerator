# 🧠 CONTEXT: My Anki Automation App (Built for Medical Students)

Convert PowerPoint slides to high-yield Anki flashcards using AI (OpenAI GPT-4o) with advanced semantic processing!

## 🚀 New Features

### **Semantic Processing & AI Enhancement**
- **Intelligent Content Chunking**: Breaks slides into meaningful semantic units
- **Slide-Level Embeddings**: Groups similar slides for better context
- **Duplicate Detection**: Automatically removes similar flashcards
- **Medical Term Extraction**: Identifies and preserves key medical terminology
- **Cross-Slide Context**: Uses related slide content for enhanced prompts

### **Enhanced Progress Tracking**
- **Real-time Progress Bars**: Visual feedback during processing
- **Detailed Status Updates**: Shows current operation and statistics
- **Content Analysis**: Displays processing metrics and quality scores

## Features
- Extracts text and images from PowerPoint slides
- Uses AI to generate exam-relevant, well-formatted flashcards
- **Semantic chunking** for optimal content processing
- **Embedding-based similarity analysis** for better context
- **Automatic duplicate removal** using semantic similarity
- Exports to CSV for easy Anki import
- Gradio web UI for no-code use
- CLI for advanced users

## Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/vidhidutta/Ankigenerator.git
   cd Ankigenerator
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI API key** to a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Gradio Web UI (Recommended)
1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Launch the UI:
   ```bash
   python3 gradio_interface.py
   ```

3. Open the provided link in your browser (e.g., http://localhost:8080)

4. Upload your PowerPoint, configure settings, and generate flashcards!

### Command Line Interface (CLI)
1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the tool:
   ```bash
   python flashcard_generator.py your_presentation.pptx
   ```

## 🧪 Testing

### Test Semantic Processing
```bash
python3 test_semantic.py
```

### Test Progress Tracking
```bash
python3 test_progress.py
```

### Demo Progress Features
```bash
python3 demo_progress.py
```

## Importing to Anki
- In Anki, go to **File > Import** and select the generated `generated_flashcards.csv` file.
- **Important:** When prompted, set the field separator to **comma**.
- Map the fields to "Front" (Question) and "Back" (Answer).

## Configuration

### Semantic Processing Settings (`config.yaml`)
```yaml
semantic_processing:
  enabled: true
  chunk_size: 500          # Max characters per chunk
  overlap: 50              # Character overlap between chunks
  similarity_threshold: 0.7 # Threshold for grouping similar slides
  embedding_model: all-MiniLM-L6-v2
  duplicate_removal: true  # Remove duplicate flashcards
  duplicate_threshold: 0.8 # Similarity threshold for duplicates
```

## Requirements
- Python 3.8+
- PowerPoint file (.pptx)
- OpenAI API key
- Internet connection
- **New Dependencies:**
  - sentence-transformers (for embeddings)
  - scikit-learn (for clustering)
  - nltk (for text processing)

## Troubleshooting
- **API key not set:** Make sure your `.env` file contains `OPENAI_API_KEY=your_key`.
- **No flashcards generated:** Check your internet connection and API key, and ensure your slides contain medical content.
- **Anki import issues:** Set the field separator to comma during import.
- **Semantic processing errors:** The system will automatically fall back to basic processing if semantic features fail.

## 🎯 Enhanced Processing Pipeline

1. **Content Extraction**: Extract text and images from slides
2. **Semantic Chunking**: Break content into meaningful units
3. **Embedding Analysis**: Compute slide similarities
4. **Context Enhancement**: Add related slide information
5. **AI Generation**: Generate flashcards with enhanced prompts
6. **Duplicate Removal**: Remove similar flashcards
7. **Quality Analysis**: Provide processing statistics

## License
MIT License

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

