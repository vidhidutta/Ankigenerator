# Slide-to-Image Conversion for Image Occlusion

## Overview

The flashcard generator now includes automated slide-to-image conversion, which captures **ALL slide content** (shapes, text, diagrams, embedded images) as PNG images for processing by the occlusion algorithm.

## How It Works

1. **PowerPoint → PDF**: Uses LibreOffice headless to convert PPTX to PDF
2. **PDF → PNG**: Uses pdf2image to convert each slide to individual PNG images
3. **Combined Extraction**: Combines slide PNGs with embedded images for maximum coverage
4. **Occlusion Processing**: Processes the full slide images for text region detection

## Benefits

- ✅ **Captures everything**: Shapes, text boxes, diagrams, embedded images
- ✅ **Better occlusion results**: More content available for masking
- ✅ **Automated workflow**: No manual steps required
- ✅ **Backward compatible**: Still extracts embedded images as fallback

## Dependencies Required

### System Dependencies
```bash
# Install LibreOffice (for PPTX → PDF conversion)
sudo apt-get install libreoffice

# Install poppler-utils (required by pdf2image)
sudo apt-get install poppler-utils
```

### Python Dependencies
```bash
# Install pdf2image
pip install pdf2image

# Or update requirements.txt and install all dependencies
pip install -r requirements.txt
```

## Testing

Run the test script to verify everything works:
```bash
python test_slide_conversion.py
```

This will:
1. Check if all dependencies are available
2. Test the conversion with a sample PowerPoint file
3. Show detailed results

## Expected Output

When processing a PowerPoint file, you should see:
```
[DEBUG] Converting slides to PNG images...
[DEBUG] Converting PPTX to PDF: libreoffice --headless --convert-to pdf --outdir /tmp/... file.pptx
[DEBUG] Converted slide 1 to: /path/to/slide1_full.png
[DEBUG] Converted slide 2 to: /path/to/slide2_full.png
[DEBUG] Successfully converted 25 slides to PNG
[DEBUG] Extracting embedded images...
[DEBUG] Processing slide 1 with 3 shapes
[DEBUG] Slide 1, Shape 1: type=PLACEHOLDER (14), has_image=False
[DEBUG] Slide 1, Shape 2: type=PICTURE (13), has_image=True
Extracted embedded images for 25 slides (total images: 8)
Extracted images for 25 slides (total images: 33)
```

## Troubleshooting

### LibreOffice Not Found
```bash
sudo apt-get install libreoffice
```

### pdf2image Import Error
```bash
pip install pdf2image
```

### poppler-utils Missing
```bash
sudo apt-get install poppler-utils
```

### Conversion Fails
- Check that LibreOffice can open your PowerPoint file manually
- Ensure the file is not corrupted
- Try with a simpler PowerPoint file first

## Performance Notes

- **Conversion time**: ~1-2 seconds per slide
- **File size**: PNG files are larger than embedded images
- **Quality**: 150 DPI provides good balance of quality vs size
- **Memory**: Each slide is processed individually to minimize memory usage

## Configuration

The conversion uses these default settings:
- **DPI**: 150 (good quality for OCR)
- **Format**: PNG (lossless, good for text detection)
- **Fallback**: If conversion fails, falls back to embedded image extraction

## Integration

The new functionality is automatically integrated into the existing pipeline:
1. Upload PowerPoint file
2. Convert slides to PNG images
3. Extract embedded images as supplement
4. Process all images for occlusion flashcards
5. Generate flashcards with improved content coverage

No changes to the UI or workflow are required - it's completely transparent to the user. 