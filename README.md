# Anki Flashcard Generator

Convert PowerPoint slides to high-yield Anki flashcards using AI (OpenAI GPT-4o)!

## Features
- Extracts text and images from PowerPoint slides
- Uses AI to generate exam-relevant, well-formatted flashcards
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

## Importing to Anki
- In Anki, go to **File > Import** and select the generated `generated_flashcards.csv` file.
- **Important:** When prompted, set the field separator to **comma**.
- Map the fields to "Front" (Question) and "Back" (Answer).

## Requirements
- Python 3.8+
- PowerPoint file (.pptx)
- OpenAI API key
- Internet connection

## Troubleshooting
- **API key not set:** Make sure your `.env` file contains `OPENAI_API_KEY=your_key`.
- **No flashcards generated:** Check your internet connection and API key, and ensure your slides contain medical content.
- **Anki import issues:** Set the field separator to comma during import.

## License
MIT License

---

Created by Vidhi Dutta. For questions or suggestions, open an issue on GitHub!
