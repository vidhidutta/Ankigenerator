#!/usr/bin/env python3
"""
Test script to verify enhanced medical term grouping
"""

import os
import sys

def test_medical_term_grouping():
    """Test the enhanced medical term grouping logic"""
    
    # Set environment variable
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/vidhidutta/anki-flashcard-generator/google_credentials.json"
    
    try:
        from providers.medical_vision_provider import GoogleMedicalVisionProvider
        
        # Create provider
        provider = GoogleMedicalVisionProvider()
        
        if not provider.available():
            print("❌ Google Medical Vision AI not available")
            return
        
        print("✅ Google Medical Vision AI available")
        
        # Test the grouping logic directly
        print("\n🧪 Testing medical term grouping logic:")
        
        # Test cases from your biochemical pathway
        test_cases = [
            ("glucose", "6-phosphate"),
            ("6-phosphate", "dehydrogenase"),
            ("glucose 6-phosphate", "dehydrogenase"),
            ("fructose", "6-phosphate"),
            ("fructose 6-phosphate", "1,6-diphosphate"),
            ("hexose", "monophosphate"),
            ("monophosphate", "shunt"),
            ("hexose monophosphate", "shunt"),
            ("nadp", "nadph"),
            ("atp", "adp"),
        ]
        
        for prev_text, current_text in test_cases:
            should_group = provider._should_always_group(prev_text, current_text)
            is_complete = provider._is_complete_medical_term(prev_text + " " + current_text)
            is_part = provider._is_part_of_medical_concept(prev_text, current_text)
            
            status = "✅" if should_group or is_complete or is_part else "❌"
            print(f"  {status} '{prev_text}' + '{current_text}'")
            print(f"    Always group: {should_group}")
            print(f"    Complete term: {is_complete}")
            print(f"    Part of concept: {is_part}")
            print()
        
        print("🎯 Key biochemical terms that should be grouped:")
        key_terms = [
            "glucose 6-phosphate dehydrogenase",
            "fructose 1,6-diphosphate", 
            "hexose monophosphate shunt",
            "nadp nadph",
            "atp adp"
        ]
        
        for term in key_terms:
            is_complete = provider._is_complete_medical_term(term)
            status = "✅" if is_complete else "❌"
            print(f"  {status} '{term}' -> Complete term: {is_complete}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_medical_term_grouping()










