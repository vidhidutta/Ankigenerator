#!/usr/bin/env python3
"""
Test script for Google Medical Vision AI provider
"""

import os
import sys

def test_medical_vision():
    """Test the medical vision provider"""
    print("🧪 Testing Google Medical Vision AI Provider...")
    
    try:
        from providers.medical_vision_provider import GoogleMedicalVisionProvider
        
        # Check if available
        if GoogleMedicalVisionProvider.available():
            print("✅ Google Medical Vision AI is available!")
            
            # Test provider creation
            provider = GoogleMedicalVisionProvider()
            print(f"✅ Provider created successfully")
            print(f"✅ Client initialized: {provider.client is not None}")
            
            # Test with a sample image if available
            test_image = "test_image.png"  # You can replace this with an actual image
            if os.path.exists(test_image):
                print(f"✅ Testing with image: {test_image}")
                regions = provider.analyze_medical_image(test_image)
                print(f"✅ Found {len(regions)} testable regions")
                
                for i, region in enumerate(regions[:3]):  # Show first 3
                    print(f"  Region {i+1}: {region.text} (Score: {region.importance_score:.2f})")
                    print(f"    Type: {region.region_type}")
                    print(f"    Rationale: {region.rationale}")
            else:
                print("⚠️ No test image found, skipping analysis test")
                
        else:
            print("❌ Google Medical Vision AI is not available")
            print("   Make sure you have:")
            print("   1. GOOGLE_APPLICATION_CREDENTIALS environment variable set")
            print("   2. google-cloud-vision package installed")
            print("   3. Valid service account JSON file")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install required packages: pip install google-cloud-vision")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_medical_vision()






