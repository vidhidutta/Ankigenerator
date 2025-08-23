#!/usr/bin/env python3
"""
Debug script to identify and fix the errors in flashcard generation
"""

import os
import sys
import tempfile
from flashcard_generator import (
    extract_text_from_pptx,
    extract_images_from_pptx,
    filter_slides,
    generate_enhanced_flashcards_with_progress,
    remove_duplicate_flashcards,
    export_flashcards_to_apkg,
    Flashcard
)

def test_semantic_processor():
    """Test if semantic processor is working"""
    try:
        from semantic_processor import SemanticProcessor
        processor = SemanticProcessor()
        print("✅ Semantic processor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Semantic processor failed: {e}")
        return False

def test_image_processing():
    """Test image processing functions"""
    print("\n=== Testing Image Processing ===")
    
    # Test with the actual PowerPoint file in the directory
    test_files = ["diuretics.pptx", "sample_lecture.pptx", "test_presentation.pptx"]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Testing with {test_file}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Extract text and images
                    slide_texts = extract_text_from_pptx(test_file)
                    slide_images = extract_images_from_pptx(test_file, temp_dir)
                    
                    print(f"Extracted {len(slide_texts)} slides with text")
                    print(f"Extracted {len(slide_images)} slide image sets")
                    
                    # Test image path validation
                    for i, images in enumerate(slide_images):
                        print(f"Slide {i+1} has {len(images)} images:")
                        for img_path in images:
                            if os.path.isfile(img_path):
                                print(f"  ✅ {img_path} (file)")
                            elif os.path.isdir(img_path):
                                print(f"  ❌ {img_path} (directory - this is the problem!)")
                            else:
                                print(f"  ❓ {img_path} (neither file nor directory)")
                    
                    return True
                    
                except Exception as e:
                    print(f"Error processing {test_file}: {e}")
    
    print("No test files found")
    return False

def test_flashcard_generation():
    """Test flashcard generation with dummy data"""
    print("\n=== Testing Flashcard Generation ===")
    
    # Create dummy flashcards
    dummy_flashcards = [
        Flashcard(
            question="What is the function of the heart?",
            answer="The heart pumps blood throughout the body.",
            level=1,
            slide_number=1
        ),
        Flashcard(
            question="What are the main chambers of the heart?",
            answer="The heart has four chambers: left atrium, right atrium, left ventricle, right ventricle.",
            level=2,
            slide_number=1
        )
    ]
    
    try:
        # Test duplicate removal
        print("Testing duplicate removal...")
        result = remove_duplicate_flashcards(dummy_flashcards)
        print(f"Original: {len(dummy_flashcards)}, After deduplication: {len(result)}")
        
        # Test export
        print("Testing export...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_deck.apkg")
            export_flashcards_to_apkg(dummy_flashcards, output_path)
            print(f"✅ Export successful: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flashcard generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Debugging Flashcard Generator Errors")
    print("=" * 50)
    
    # Test semantic processor
    semantic_ok = test_semantic_processor()
    
    # Test image processing
    image_ok = test_image_processing()
    
    # Test flashcard generation
    flashcard_ok = test_flashcard_generation()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Semantic Processor: {'✅' if semantic_ok else '❌'}")
    print(f"Image Processing: {'✅' if image_ok else '❌'}")
    print(f"Flashcard Generation: {'✅' if flashcard_ok else '❌'}")
    
    if not all([semantic_ok, image_ok, flashcard_ok]):
        print("\n🚨 Issues detected! Check the error messages above.")
    else:
        print("\n✅ All tests passed!")

if __name__ == "__main__":
    main() 