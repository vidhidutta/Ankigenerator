#!/usr/bin/env python3
"""
Test script for Holistic Medical Analysis

This script demonstrates how the AI acts as a medical expert educator,
comprehensively analyzing lectures and generating both flashcards and
comprehensive notes with mind maps.
"""

import os
import sys
import tempfile

def test_holistic_analysis():
    """Test the holistic medical analysis system"""
    
    print("🧠 Testing OjaMed Holistic Medical Analysis System")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        print("   You can test the system by setting OJAMED_FORCE_DEMO=1")
        return False
    
    print("✅ OpenAI API key found")
    
    # Sample lecture content (respiratory medicine)
    sample_lecture = """
    SLIDE 1: Interstitial Lung Disease (ILD)
    
    Definition: A group of disorders affecting the lung interstitium, characterized by inflammation and fibrosis.
    
    Types:
    - Idiopathic pulmonary fibrosis (IPF)
    - Sarcoidosis
    - Hypersensitivity pneumonitis
    - Connective tissue disease-associated ILD
    
    SLIDE 2: Pathophysiology
    
    The interstitium is the space between alveoli containing:
    - Capillaries
    - Lymphatic vessels
    - Connective tissue
    
    In ILD, this space becomes thickened due to:
    - Inflammation
    - Fibrosis
    - Granuloma formation
    
    SLIDE 3: Clinical Presentation
    
    Symptoms:
    - Progressive dyspnea
    - Dry cough
    - Fatigue
    - Weight loss
    
    Physical findings:
    - Bibasilar crackles
    - Clubbing
    - Reduced chest expansion
    
    SLIDE 4: Diagnostic Approach
    
    Imaging:
    - Chest X-ray: Reticular or nodular patterns
    - High-resolution CT: Ground glass opacities, honeycombing
    
    Pulmonary function tests:
    - Restrictive pattern: Reduced FVC and TLC
    - Reduced DLCO (diffusing capacity)
    
    SLIDE 5: Treatment Options
    
    Pharmacological:
    - Corticosteroids (prednisone)
    - Immunosuppressants (azathioprine, mycophenolate)
    - Antifibrotic agents (pirfenidone, nintedanib)
    
    Non-pharmacological:
    - Oxygen therapy
    - Pulmonary rehabilitation
    - Lung transplantation
    """
    
    print("\n📚 Sample Lecture Content:")
    print("Topic: Interstitial Lung Disease (ILD)")
    print(f"Length: {len(sample_lecture)} characters")
    
    try:
        # Import the holistic analyzer
        from holistic_medical_analyzer import HolisticMedicalAnalyzer
        
        print("\n🧠 Initializing Holistic Medical Analyzer...")
        analyzer = HolisticMedicalAnalyzer(api_key)
        
        print("\n🔍 Performing comprehensive analysis...")
        analysis = analyzer.analyze_lecture_holistically(sample_lecture, "Interstitial Lung Disease")
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Results:")
        print(f"  • {len(analysis.concepts)} medical concepts identified")
        print(f"  • {len(analysis.mind_maps)} mind maps generated")
        print(f"  • {len(analysis.knowledge_gaps)} knowledge gaps identified")
        print(f"  • {len(analysis.filled_gaps)} gaps filled with expert knowledge")
        print(f"  • {len(analysis.learning_objectives)} learning objectives created")
        print(f"  • {len(analysis.clinical_pearls)} clinical pearls extracted")
        print(f"  • {len(analysis.glossary)} terms in medical glossary")
        
        # Show some examples
        if analysis.concepts:
            print(f"\n🎯 Sample Concepts:")
            for i, concept in enumerate(analysis.concepts[:3]):
                print(f"  {i+1}. {concept.name} ({concept.category})")
                print(f"     Importance: {concept.importance}")
                print(f"     Clinical: {concept.clinical_relevance[:100]}...")
        
        if analysis.clinical_pearls:
            print(f"\n💎 Sample Clinical Pearls:")
            for i, pearl in enumerate(analysis.clinical_pearls[:2]):
                print(f"  {i+1}. {pearl}")
        
        if analysis.learning_objectives:
            print(f"\n🎓 Sample Learning Objectives:")
            for i, objective in enumerate(analysis.learning_objectives[:2]):
                print(f"  {i+1}. {objective}")
        
        # Test PDF generation
        print(f"\n📄 Testing PDF Generation...")
        try:
            from medical_notes_pdf_generator import generate_medical_notes_pdf
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                pdf_path = tmp.name
            
            pdf_path = generate_medical_notes_pdf(analysis, pdf_path)
            
            if os.path.exists(pdf_path):
                print(f"✅ PDF generated successfully: {pdf_path}")
                print(f"   File size: {os.path.getsize(pdf_path)} bytes")
                
                # Clean up
                os.unlink(pdf_path)
            else:
                print("❌ PDF generation failed")
                
        except Exception as e:
            print(f"⚠️ PDF generation test failed: {e}")
        
        print(f"\n🎉 Holistic analysis test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demo_mode():
    """Test the demo mode functionality"""
    
    print("\n🎭 Testing Demo Mode")
    print("=" * 30)
    
    # Set demo environment variable
    os.environ["OJAMED_FORCE_DEMO"] = "1"
    
    try:
        from app.adapter import extract_cards_from_ppt
        
        # Create a dummy file path
        dummy_path = "demo_lecture.pptx"
        
        print("📝 Testing demo card extraction...")
        cards = extract_cards_from_ppt(dummy_path)
        
        if cards:
            print(f"✅ Demo mode working: {len(cards)} cards generated")
            for i, (q, a) in enumerate(cards, 1):
                print(f"  {i}. Q: {q}")
                print(f"     A: {a}")
        else:
            print("❌ Demo mode failed")
            
    except Exception as e:
        print(f"❌ Demo mode test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up environment
        if "OJAMED_FORCE_DEMO" in os.environ:
            del os.environ["OJAMED_FORCE_DEMO"]

if __name__ == "__main__":
    print("🚀 OjaMed Holistic Medical Analysis Test Suite")
    print("=" * 60)
    
    # Test demo mode first (doesn't require API key)
    test_demo_mode()
    
    # Test full holistic analysis (requires API key)
    print("\n" + "=" * 60)
    test_holistic_analysis()
    
    print("\n" + "=" * 60)
    print("🏁 Test suite completed!")
    print("\nTo use the system:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Upload a PowerPoint lecture via the API")
    print("3. Receive both flashcards AND comprehensive medical notes")
    print("4. The AI will act as a medical expert, filling knowledge gaps")
    print("5. Generate visual mind maps showing concept relationships")
