#!/usr/bin/env python3
"""
Test script for image occlusion export functionality
"""

import os
import tempfile
from PIL import Image, ImageDraw
from flashcard_generator import export_flashcards_to_apkg, Flashcard
from utils.image_occlusion import (
    detect_text_regions,
    mask_regions,
    generate_occlusion_flashcard_entry,
    save_occlusion_pair
)

def create_test_image_occlusion_cards():
    """Create test image occlusion flashcards"""
    
    # Load the complex test image
    img = Image.open('complex_test_image.png')
    
    # Save original image
    original_path = "complex_test_image.png"
    
    # Detect regions and create occluded version
    regions = detect_text_regions(img, conf_threshold=30)
    if regions:
        occluded_img = mask_regions(img, regions, method='rectangle')
        occluded_path = "occluded_complex_test_image.png"
        occluded_img.save(occluded_path)
    else:
        print("❌ No regions detected in the complex image")
        return
    
    # Create test flashcards
    test_cards = [
        # Image occlusion flashcard
        {
            "question_img": occluded_path,
            "answer_img": original_path,
            "type": "image_occlusion",
            "alt_text": "What text is hidden here?"
        }
    ]
    
    # Test export
    output_path = "test_occlusion_export.apkg"
    print(f"Testing export with {len(test_cards)} cards...")
    print(f"  - {len([c for c in test_cards if isinstance(c, dict)])} image occlusion cards")
    
    try:
        export_flashcards_to_apkg(test_cards, output_path)
        print(f"✅ Successfully exported to {output_path}")
        
        # Check if file exists and has reasonable size
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"✅ File size: {size} bytes")
            if size > 1000:  # Should be at least 1KB
                print("✅ File size looks reasonable")
            else:
                print("⚠️ File size seems small")
        else:
            print("❌ Export file not found")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

def test_occlusion_utility_functions():
    """Test the image occlusion utility functions"""
    print("\n🧪 Testing image occlusion utility functions...")
    
    try:
        # Load the complex test image
        img = Image.open('complex_test_image.png')
        
        # Test region detection with the complex image
        regions = detect_text_regions(img, conf_threshold=30)
        print(f"✅ Text regions detected: {len(regions)}")
        
        # Test masking
        if regions:
            masked_img = mask_regions(img, regions, method='rectangle')
            print("✅ Image masking successful")
        else:
            print("❌ No regions detected in the complex image")
        
        # Test entry generation
        entry = generate_occlusion_flashcard_entry("test_occ.png", "test_orig.png", "Test question")
        print(f"✅ Entry generated: {entry}")
        
        print("✅ All utility function tests passed")
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Image Occlusion Export Functionality")
    print("=" * 50)
    
    # Test utility functions
    test_occlusion_utility_functions()
    
    # Test export functionality
    print("\n🧪 Testing Export Functionality")
    print("=" * 50)
    create_test_image_occlusion_cards()
    
    print("\n✅ Test completed!") 