#!/usr/bin/env python3
"""
Test script for image occlusion export functionality
"""

import os
import tempfile
from PIL import Image, ImageDraw
from flashcard_generator import export_flashcards_to_apkg, Flashcard

def create_test_image_occlusion_cards():
    """Create test image occlusion flashcards"""
    
    # Create a temporary directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create a simple test image (white background with black text)
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Test Image", fill='black')
        draw.text((50, 100), "This is a sample", fill='black')
        draw.text((50, 150), "for testing", fill='black')
        
        # Save original image
        original_path = os.path.join(temp_dir, "original_test.png")
        img.save(original_path)
        
        # Create occluded version (white rectangle over text)
        occluded_img = img.copy()
        draw_occ = ImageDraw.Draw(occluded_img)
        draw_occ.rectangle([50, 50, 200, 200], fill='white')
        
        # Save occluded image
        occluded_path = os.path.join(temp_dir, "occluded_test.png")
        occluded_img.save(occluded_path)
        
        # Create test flashcards
        test_cards = [
            # Regular text flashcards
            Flashcard(
                question="What is the capital of France?",
                answer="Paris",
                level=1,
                slide_number=1
            ),
            Flashcard(
                question="What is 2 + 2?",
                answer="4",
                level=1,
                slide_number=1
            ),
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
        print(f"  - {len([c for c in test_cards if isinstance(c, Flashcard)])} text cards")
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
        from utils.image_occlusion import (
            detect_text_regions,
            mask_regions,
            generate_occlusion_flashcard_entry,
            save_occlusion_pair
        )
        print("✅ All utility functions imported successfully")
        
        # Test with a simple image
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), "Test", fill='black')
        
        # Test region detection (might not find much with simple image)
        regions = detect_text_regions(img, conf_threshold=30)
        print(f"✅ Text regions detected: {len(regions)}")
        
        # Test masking
        if regions:
            masked_img = mask_regions(img, regions, method='rectangle')
            print("✅ Image masking successful")
        else:
            # Test with a dummy region
            masked_img = mask_regions(img, [(20, 40, 50, 20)], method='rectangle')
            print("✅ Image masking successful (with dummy region)")
        
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