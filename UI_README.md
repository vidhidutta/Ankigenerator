# 🧠 Medical Flashcard Generator - Web Interface

A user-friendly web interface for generating high-quality Anki flashcards from PowerPoint lectures.

## 🚀 Quick Start

### Option 1: Simple Launcher (Recommended)
```bash
# Make sure you're in the project directory
cd anki-flashcard-generator

# Activate virtual environment
source venv/bin/activate

# Launch the UI
python launch_ui.py
```

### Option 2: Direct Launch
```bash
# Activate virtual environment
source venv/bin/activate

# Launch directly
python gradio_interface.py
```

## 🌐 Access the Interface

Once launched, open your web browser and go to:
- **Local**: http://localhost:7860
- **Network**: http://your-ip-address:7860

## 📋 How to Use

### 1. Upload Your Lecture
- Click "Upload PowerPoint (.pptx)" 
- Select your PowerPoint lecture file
- Only .pptx files are supported

### 2. Configure Settings
- **Medical Category**: Choose the subject area (Pharmacology, Neuroscience, etc.)
- **Exam Level**: Select your current year/exam (Year 1 MBBS, Finals, etc.)
- **Flashcard Type**: Level 1 (basic), Level 2 (advanced), or Both
- **Answer Format**: Choose how detailed you want answers to be
- **Use Cloze Deletions**: Enable for fill-in-the-blank style cards

### 3. Extra Materials (Optional)
- **Generate Glossary**: Creates a list of key terms with definitions
- **Generate Topic Map**: Creates an outline of main lecture themes
- **Generate Summary Sheet**: Creates a compressed revision version

### 4. Generate Flashcards
- Click the "🚀 Generate Flashcards" button
- Wait for processing (this may take a few minutes)
- View sample flashcards in the results area

### 5. Download Results
- **CSV File**: Download the flashcards for import into Anki
- **PDF File**: Download extra materials (if generated)

## 📱 Interface Features

### 🎨 Modern Design
- Clean, intuitive interface
- Responsive layout
- Medical-themed styling

### ⚡ Smart Processing
- Automatic slide filtering (removes navigation/empty slides)
- Content filtering (removes irrelevant "setting the scene" content)
- Image analysis (extracts information from diagrams/charts)

### 🔧 Customizable Settings
- Multiple medical categories
- Different exam levels
- Various flashcard types and formats
- Optional extra materials

### 📊 Real-time Feedback
- Progress updates during processing
- Sample flashcards preview
- Error handling and user guidance

## 🛠️ Technical Details

### Backend Integration
- Uses your existing `flashcard_generator.py` backend
- Maintains all the advanced filtering and processing logic
- Supports multimodal analysis (text + images)

### File Handling
- Secure temporary file processing
- Automatic cleanup after generation
- Support for large PowerPoint files

### API Integration
- Uses OpenAI GPT-4o for flashcard generation
- HTTP API for multimodal support
- Robust error handling

## 🔧 Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Make sure you have a `.env` file with your API key
- Format: `OPENAI_API_KEY=your_api_key_here`

**"Virtual environment not detected"**
- Activate your virtual environment: `source venv/bin/activate`

**"No flashcards generated"**
- Check that your PowerPoint contains medical content
- Ensure slides aren't just navigation/title slides
- Verify your API key is valid and has credits

**Interface won't load**
- Check if port 7860 is available
- Try a different port in the code if needed
- Ensure all dependencies are installed

### Getting Help
- Check the console output for detailed error messages
- Verify your PowerPoint file format (.pptx only)
- Ensure your OpenAI API key has sufficient credits

## 🎯 Tips for Best Results

1. **Use well-structured lectures** with clear medical content
2. **Include relevant images/diagrams** for better analysis
3. **Choose appropriate category and exam level** for better filtering
4. **Generate extra materials** for comprehensive study resources
5. **Review generated flashcards** before importing to Anki

## 🔄 Importing to Anki

1. Download the generated CSV file
2. Open Anki
3. Go to File → Import
4. Select the CSV file
5. Choose your deck
6. Set field mapping: Field 1 = Question, Field 2 = Answer
7. Click Import

## 📈 Performance

- **Processing time**: 1-5 minutes depending on lecture size
- **Flashcard quality**: High-quality, exam-relevant content only
- **Filtering efficiency**: Automatically removes 30-50% of irrelevant content
- **Memory usage**: Optimized for large PowerPoint files

---

**Happy studying! 📚✨** 