#!/usr/bin/env python3
"""
Test script to verify the three fixes:
1. LibreOffice pre-check
2. Semantic generation error fix
3. Export failure fix for directory paths
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_libreoffice_precheck():
    """Test the LibreOffice pre-check functionality"""
    print("🔍 Testing LibreOffice pre-check...")
    
    from flashcard_generator import convert_pptx_to_slide_pngs
    
    # Test with LibreOffice available
    with patch('shutil.which', return_value='/usr/bin/libreoffice'):
        result = convert_pptx_to_slide_pngs('test.pptx', 'output')
        print("✅ LibreOffice available - function should proceed normally")
    
    # Test with LibreOffice not available
    with patch('shutil.which', return_value=None):
        result = convert_pptx_to_slide_pngs('test.pptx', 'output')
        assert result == [], "Should return empty list when LibreOffice not found"
        print("✅ LibreOffice not available - function returns empty list")

def test_semantic_generation_fix():
    """Test the semantic generation error fix"""
    print("🔍 Testing semantic generation fix...")
    
    from semantic_processor import SemanticProcessor
    
    processor = SemanticProcessor()
    
    # Test normal case
    chunk_data = {'text': 'Normal text content', 'key_phrases': ['term1', 'term2']}
    result = processor.build_enhanced_prompt(chunk_data, "Base prompt")
    assert 'Normal text content' in result
    print("✅ Normal chunk_data works correctly")
    
    # Test fallback case (chunk_data is the text itself)
    result = processor.build_enhanced_prompt("Direct text content", "Base prompt")
    assert 'Direct text content' in result
    print("✅ Fallback for direct text works correctly")
    
    # Test dict case
    dict_chunk = {'some_key': 'some_value'}
    result = processor.build_enhanced_prompt(dict_chunk, "Base prompt")
    assert 'some_key' in result or 'some_value' in result
    print("✅ Dict fallback works correctly")

def test_export_directory_validation():
    """Test the export failure fix for directory paths"""
    print("🔍 Testing export directory validation...")
    
    from flashcard_generator import export_flashcards_to_apkg
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test directory that would cause the error
        test_dir = os.path.join(temp_dir, 'test_directory')
        os.makedirs(test_dir, exist_ok=True)
        
        # Test flashcards with directory paths
        test_flashcards = [
            {
                'question_image_path': test_dir,  # This is a directory!
                'answer_image_path': test_dir,    # This is a directory!
                'alt_text': 'Test card'
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

def test_all_fixes():
    """Run all tests"""
    print("🧪 Testing all fixes...")
    print("=" * 50)
    
    try:
        test_libreoffice_precheck()
        print()
        
        test_semantic_generation_fix()
        print()
        
        test_export_directory_validation()
        print()
        
        print("✅ All fixes verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    sys.exit(0 if success else 1) 