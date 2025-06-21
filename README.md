# Anki Flashcard Generator

Convert PowerPoint slides to Anki flashcards using AI!

## Setup

1. **Install dependencies** (in your virtual environment):
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add your OpenRouter API key** to a `.env` file:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

1. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Run the tool** with your PowerPoint file:
   ```bash
   python flashcard_generator.py your_presentation.pptx
   ```

3. **Import into Anki**:
   - Open Anki
   - Go to File > Import
   - Select the generated `flashcards.csv` file
   - Choose your deck and card type

## Example

```bash
# Generate flashcards from a presentation
python flashcard_generator.py lecture_slides.pptx

# Specify custom output file
python flashcard_generator.py lecture_slides.pptx --output my_flashcards.csv
```

## How it works

1. **Extracts text** from all slides in your PowerPoint
2. **Uses AI** to generate high-quality question-answer pairs
3. **Exports** to CSV format for easy Anki import

## Requirements

- Python 3.7+
- PowerPoint file (.pptx format)
- OpenRouter API key
- Internet connection for AI processing

## Troubleshooting

- **"API key not set"**: Make sure your `.env` file contains `OPENROUTER_API_KEY=your_key`
- **"File not found"**: Check the path to your PowerPoint file
- **"No flashcards generated"**: Check your internet connection and API key 