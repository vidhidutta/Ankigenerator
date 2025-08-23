#!/usr/bin/env python3
"""
Test script to simulate audio upload functionality
"""

import os
import tempfile
import requests
import json

def create_test_files():
    """Create test PowerPoint and audio files"""
    print("📁 Creating test files...")
    
    # Create a simple test PowerPoint content (simulated)
    pptx_content = """
    Slide 1: Introduction to Pharmacology
    - Drug classification
    - Mechanism of action
    - Clinical applications
    
    Slide 2: Beta Blockers
    - Propranolol
    - Mechanism: Beta-1 receptor blockade
    - Indications: Hypertension, Angina
    
    Slide 3: Side Effects
    - Bradycardia
    - Fatigue
    - Sexual dysfunction
    """
    
    # Create a test audio file (simulated)
    audio_content = "Mock audio content for testing emphasis detection"
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(pptx_content)
        pptx_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mp3', delete=False) as f:
        f.write(audio_content)
        audio_file = f.name
    
    print(f"✅ Created test files:")
    print(f"  - PowerPoint: {pptx_file}")
    print(f"  - Audio: {audio_file}")
    
    return pptx_file, audio_file

def test_audio_processing():
    """Test the audio processing functionality"""
    print("\n🎵 Testing Audio Processing...")
    
    try:
        from audio_processor_simple import SimpleAudioProcessor
        
        # Create test files
        pptx_file, audio_file = create_test_files()
        
        # Initialize processor
        processor = SimpleAudioProcessor()
        print("✅ AudioProcessor initialized")
        
        # Test transcription
        segments = processor.transcribe_audio(audio_file)
        print(f"✅ Transcribed {len(segments)} segments")
        
        # Test slide texts
        slide_texts = [
            "Slide 1: Introduction to Pharmacology",
            "Slide 2: Beta Blockers and their mechanisms",
            "Slide 3: Side Effects and clinical considerations"
        ]
        
        # Test alignment
        slide_audio_map = processor.align_with_slides(segments, slide_texts)
        print(f"✅ Aligned audio with {len(slide_audio_map)} slides")
        
        # Test weighting
        weights = processor.calculate_content_weights(slide_audio_map)
        print(f"✅ Calculated weights: {weights}")
        
        # Clean up
        os.unlink(pptx_file)
        os.unlink(audio_file)
        
        print("✅ Audio processing test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

def test_gradio_interface():
    """Test that the Gradio interface is accessible"""
    print("\n🌐 Testing Gradio Interface...")
    
    try:
        response = requests.get("http://localhost:7863", timeout=10)
        
        if response.status_code == 200:
            print("✅ Gradio interface is accessible")
            print(f"   - URL: http://localhost:7863")
            print(f"   - Status: {response.status_code}")
            return True
        else:
            print(f"❌ Gradio interface returned status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Could not access Gradio interface: {e}")
        return False

def main():
    """Run all tests"""
    print("🎵 Audio Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Audio Processing", test_audio_processing),
        ("Gradio Interface", test_gradio_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Audio integration is working correctly.")
        print("\n🚀 You can now:")
        print("1. Visit http://localhost:7863 to use the interface")
        print("2. Upload PowerPoint files with optional audio")
        print("3. Generate enhanced flashcards with audio context")
        print("4. Test the audio emphasis detection and weighting")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 