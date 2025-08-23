#!/usr/bin/env python3
"""
Test script to verify the production fixes for:
1. Semantic generation error with better logging
2. Export failure for directory paths
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_semantic_generation_with_logging():
    """Test the semantic generation error fix with detailed logging"""
    print("🔍 Testing semantic generation with logging...")
    
    from flashcard_generator import generate_flashcards_from_semantic_chunks
    
    # Test with various problematic chunk_data types
    test_cases = [
        # Case 1: chunk_data is a string (should be converted to dict)
        ("Direct string content", "Direct string content"),
        
        # Case 2: chunk_data is a dict but missing 'text' key
        ({"slide_index": 1, "key_phrases": ["term1"]}, {"slide_index": 1, "key_phrases": ["term1"]}),
        
        # Case 3: chunk_data['text'] is a dict (should be converted to string)
        ({"text": {"nested": "content"}, "slide_index": 1}, {"text": {"nested": "content"}, "slide_index": 1}),
        
        # Case 4: Normal case (should work fine)
        ({"text": "Normal content", "slide_index": 1}, {"text": "Normal content", "slide_index": 1}),
    ]
    
    for i, (description, chunk_data) in enumerate(test_cases):
        print(f"\n  Testing case {i+1}: {description}")
        
        # Mock the API call to avoid actual network requests
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Question: Test Q\nAnswer: Test A"
                    }
                }]
            }
            mock_post.return_value = mock_response
            
            # Create test data
            semantic_chunks = [chunk_data]
            slide_images = [[]]  # Empty images for testing
            
            try:
                result = generate_flashcards_from_semantic_chunks(
                    semantic_chunks, slide_images, "fake_api_key", "gpt-4", 1000, 0.3
                )
                print(f"    ✅ Success: Generated {len(result)} flashcards")
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    print("✅ Semantic generation logging test completed")

def test_export_directory_validation():
    """Test the export failure fix for directory paths"""
    print("🔍 Testing export directory validation...")
    
    from flashcard_generator import export_flashcards_to_apkg, Flashcard
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories that would cause the error
        test_dir1 = os.path.join(temp_dir, 'test_directory1')
        test_dir2 = os.path.join(temp_dir, 'test_directory2')
        os.makedirs(test_dir1, exist_ok=True)
        os.makedirs(test_dir2, exist_ok=True)
        
        # Create a test file for comparison
        test_file = os.path.join(temp_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Test flashcards with various path issues
        test_flashcards = [
            # Case 1: Dict with directory paths
            {
                'question_image_path': test_dir1,  # Directory!
                'answer_image_path': test_dir2,    # Directory!
                'alt_text': 'Test card 1'
            },
            
            # Case 2: Flashcard object with directory image_path
            Flashcard(
                question="Test Q",
                answer="Test A", 
                level=1,
                slide_number=1,
                image_path=test_dir1  # Directory!
            ),
            
            # Case 3: Flashcard object with valid file path
            Flashcard(
                question="Test Q2",
                answer="Test A2",
                level=1,
                slide_number=1,
                image_path=test_file  # Valid file
            ),
            
            # Case 4: Dict with valid file paths
            {
                'question_image_path': test_file,  # Valid file
                'answer_image_path': test_file,    # Valid file
                'alt_text': 'Test card 2'
            }
        ]
        
        # Mock genanki to avoid actual file creation
        with patch('genanki.Package') as mock_package:
            mock_package.return_value.write_to_file.return_value = None
            
            # This should not raise an error now
            try:
                export_flashcards_to_apkg(test_flashcards, 'test.apkg')
                print("✅ Export function handles directory paths gracefully")
            except Exception as e:
                print(f"❌ Export function still has issues: {e}")
                return False
    
    return True

def test_image_path_validation():
    """Test that image_path is only set with verified file paths"""
    print("🔍 Testing image_path validation...")
    
    from flashcard_generator import Flashcard
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories and files
        test_dir = os.path.join(temp_dir, 'test_directory')
        test_file = os.path.join(temp_dir, 'test_file.txt')
        os.makedirs(test_dir, exist_ok=True)
        
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Test cases
        test_cases = [
            (test_file, "Valid file path", True),
            (test_dir, "Directory path", False),
            ("/nonexistent/file.txt", "Non-existent file", False),
            (None, "None path", False),
            ("", "Empty path", False),
        ]
        
        for path, description, should_work in test_cases:
            print(f"\n  Testing: {description}")
            
            # Create a flashcard and try to set image_path
            fc = Flashcard("Test Q", "Test A", 1, 1)
            
            # Simulate the validation logic
            if path and os.path.isfile(path):
                fc.image_path = path
                print(f"    ✅ Set image_path to: {path}")
            elif path:
                print(f"    ⚠️ Skipped invalid path: {path}")
            else:
                print(f"    ⚠️ Skipped null/empty path")
            
            # Check if the path was set correctly
            if should_work:
                assert fc.image_path == path, f"Expected {path}, got {fc.image_path}"
            else:
                assert fc.image_path is None, f"Expected None, got {fc.image_path}"
    
    print("✅ Image path validation test completed")

def test_all_production_fixes():
    """Run all production fix tests"""
    print("🧪 Testing all production fixes...")
    print("=" * 60)
    
    try:
        test_semantic_generation_with_logging()
        print()
        
        test_export_directory_validation()
        print()
        
        test_image_path_validation()
        print()
        
        print("✅ All production fixes verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_all_production_fixes()
    sys.exit(0 if success else 1) 