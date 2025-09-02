#!/usr/bin/env python3
"""
Test script to verify medical term grouping in the enhanced medical vision provider
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import tempfile

def create_test_medical_image():
    """Create a test image with medical terms to test grouping"""
    
    # Create a white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a default font, fallback to basic if not available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw medical terms that should be grouped together
    medical_terms = [
        ("Glucose", (100, 100), "blue"),
        ("6-phosphate", (200, 100), "blue"), 
        ("dehydrogenase", (350, 100), "blue"),
        ("Fructose", (100, 200), "red"),
        ("1,6-diphosphate", (200, 200), "red"),
        ("Hexose", (100, 300), "green"),
        ("monophosphate", (200, 300), "green"),
        ("shunt", (350, 300), "green"),
        ("NADP", (500, 150), "purple"),
        ("NADPH", (500, 200), "purple"),
        ("ATP", (600, 100), "orange"),
        ("ADP", (600, 150), "orange")
    ]
    
    # Draw the terms
    for text, pos, color in medical_terms:
        draw.text(pos, text, fill=color, font=font)
    
    # Save the test image
    test_image_path = "test_medical_terms.png"
    image.save(test_image_path)
    print(f"✅ Created test image: {test_image_path}")
    
    return test_image_path

def test_medical_term_grouping():
    """Test the enhanced medical vision provider with our test image"""
    
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
        
        # Create test image
        test_image = create_test_medical_image()
        
        # Analyze the image
        print(f"🔍 Analyzing test image: {test_image}")
        regions = provider.analyze_medical_image(test_image)
        
        print(f"📊 Found {len(regions)} regions:")
        for i, region in enumerate(regions):
            print(f"  Region {i+1}:")
            print(f"    Text: '{region.text}'")
            print(f"    Type: {region.region_type}")
            print(f"    Score: {region.importance_score:.2f}")
            print(f"    Rationale: {region.rationale}")
            print(f"    BBox: {region.bbox}")
            print()
        
        # Check for specific groupings we expect
        expected_groupings = [
            "glucose 6-phosphate dehydrogenase",
            "fructose 1,6-diphosphate", 
            "hexose monophosphate shunt",
            "nadp nadph",
            "atp adp"
        ]
        
        print("🎯 Checking for expected term groupings:")
        for expected in expected_groupings:
            found = any(expected.lower() in region.text.lower() for region in regions)
            status = "✅" if found else "❌"
            print(f"  {status} '{expected}'")
        
        # Clean up
        if os.path.exists(test_image):
            os.remove(test_image)
            print(f"🧹 Cleaned up test image")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_medical_term_grouping()










