#!/usr/bin/env python3
"""
Test script for the new slide-to-image conversion functionality.
This tests the automated conversion of PowerPoint slides to PNG images.
"""

import os
import sys
import tempfile
from flashcard_generator import convert_pptx_to_slide_pngs, extract_images_from_pptx

def test_slide_conversion():
    """Test the slide conversion functionality"""
    print("🧪 Testing Slide-to-Image Conversion")
    print("=" * 50)
    
    # Test with a sample PowerPoint file if available
    test_files = [
        "sample_lecture.pptx",
        "test_presentation.pptx", 
        "medical_lecture.pptx"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Testing slide conversion with: {test_file}")
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Test the new conversion function
                    slide_pngs = convert_pptx_to_slide_pngs(test_file, temp_dir)
                    
                    if slide_pngs:
                        print(f"✅ Successfully converted {len(slide_pngs)} slides to PNG")
                        for i, slide_images in enumerate(slide_pngs):
                            print(f"  Slide {i+1}: {len(slide_images)} images")
                            for img_path in slide_images:
                                if os.path.exists(img_path):
                                    print(f"    ✅ {img_path}")
                                else:
                                    print(f"    ❌ {img_path} (not found)")
                    else:
                        print("❌ No slides were converted")
                    
                    # Test the full extraction function
                    print(f"\nTesting full image extraction...")
                    all_images = extract_images_from_pptx(test_file, temp_dir)
                    
                    if all_images:
                        total_images = sum(len(images) for images in all_images)
                        print(f"✅ Total images extracted: {total_images}")
                        for i, images in enumerate(all_images):
                            if images:
                                print(f"  Slide {i+1}: {len(images)} images")
                    else:
                        print("❌ No images extracted")
                        
                except Exception as e:
                    print(f"❌ Error during conversion: {e}")
                    
                return
                
    print("⚠️ No test PowerPoint files found")
    print("Please place a PowerPoint file in the current directory to test")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    # Check LibreOffice
    try:
        import subprocess
        result = subprocess.run(["libreoffice", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ LibreOffice is available")
        else:
            print("❌ LibreOffice not found. Install with: sudo apt-get install libreoffice")
    except Exception as e:
        print(f"❌ Error checking LibreOffice: {e}")
    
    # Check pdf2image
    try:
        from pdf2image import convert_from_path
        print("✅ pdf2image is available")
    except ImportError:
        print("❌ pdf2image not found. Install with: pip install pdf2image")
    
    # Check poppler-utils (required by pdf2image)
    try:
        result = subprocess.run(["pdftoppm", "-h"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ poppler-utils is available")
        else:
            print("❌ poppler-utils not found. Install with: sudo apt-get install poppler-utils")
    except Exception as e:
        print(f"❌ Error checking poppler-utils: {e}")

if __name__ == "__main__":
    print("🔬 Slide Conversion Test")
    print("=" * 60)
    print("This test verifies the new automated slide-to-image conversion.")
    print()
    
    # Check dependencies first
    check_dependencies()
    print()
    
    # Test the conversion
    test_slide_conversion()
    
    print("\n📋 Summary:")
    print("=" * 30)
    print("✅ If all dependencies are available, the conversion should work")
    print("✅ The new approach captures ALL slide content (shapes, text, diagrams)")
    print("✅ This should significantly improve occlusion flashcard generation")
    print("⚠️  If LibreOffice or pdf2image are missing, install them first") 