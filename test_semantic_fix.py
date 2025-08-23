#!/usr/bin/env python3
"""
Test to verify the semantic processor fix for the dictionary error
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_processor import SemanticProcessor

def test_semantic_processor_fix():
    """Test that the semantic processor handles dictionary data correctly"""
    print("🧪 Testing Semantic Processor Fix")
    print("=" * 50)
    
    # Initialize semantic processor
    processor = SemanticProcessor()
    
    # Test data that might cause the dictionary error
    test_chunk_data = {
        'text': 'Beta blockers competitively inhibit beta-adrenergic receptors',
        'key_phrases': ['beta blockers', 'receptors', 'sympathetic'],
        'related_slides': [
            'Slide 1: Introduction to beta blockers',
            'Slide 2: Mechanism of action'
        ],
        'slide_index': 1,
        'group_size': 1
    }
    
    # Test with problematic data that might be dictionaries
    problematic_chunk_data = {
        'text': {'content': 'Some text content'},  # Dictionary instead of string
        'key_phrases': {'terms': ['term1', 'term2']},  # Dictionary instead of list
        'related_slides': {'slides': ['slide1', 'slide2']},  # Dictionary instead of list
        'slide_index': 1,
        'group_size': 1
    }
    
    # Test with None values
    none_chunk_data = {
        'text': None,
        'key_phrases': None,
        'related_slides': None,
        'slide_index': 1,
        'group_size': 1
    }
    
    test_cases = [
        ("Normal data", test_chunk_data),
        ("Problematic data", problematic_chunk_data),
        ("None data", none_chunk_data)
    ]
    
    for test_name, chunk_data in test_cases:
        print(f"\nTesting: {test_name}")
        try:
            # This should not raise the "expected string or bytes-like object, got 'dict'" error
            enhanced_prompt = processor.build_enhanced_prompt(chunk_data, "Test prompt template")
            print(f"✅ Success: Generated prompt of length {len(enhanced_prompt)}")
            print(f"   Preview: {enhanced_prompt[:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
            if "expected string or bytes-like object, got 'dict'" in str(e):
                print("   This is the error we're trying to fix!")
            return False
    
    print("\n✅ All tests passed! The semantic processor fix is working.")
    return True

if __name__ == "__main__":
    success = test_semantic_processor_fix()
    sys.exit(0 if success else 1) 