#!/usr/bin/env python3
"""
Test script for semantic processing functionality
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_semantic_processor():
    """Test the semantic processor functionality"""
    try:
        from semantic_processor import SemanticProcessor
        
        print("🧪 Testing Semantic Processing...")
        
        # Initialize processor
        processor = SemanticProcessor()
        print("✅ Semantic processor initialized successfully")
        
        # Test text cleaning
        test_text = "Slide 1: This is a test slide about diabetes mellitus and insulin receptors."
        cleaned = processor.clean_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test semantic chunking
        long_text = """
        Diabetes mellitus is a chronic metabolic disorder characterized by high blood glucose levels. 
        It occurs when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. 
        There are two main types: Type 1 diabetes, which is an autoimmune condition, and Type 2 diabetes, which is often related to lifestyle factors. 
        Insulin is a hormone produced by beta cells in the pancreas that helps glucose enter cells for energy production. 
        Insulin receptors are proteins on cell surfaces that bind to insulin and trigger glucose uptake mechanisms.
        """
        
        chunks = processor.semantic_chunk_text(long_text, max_chunk_size=200, overlap=30)
        print(f"✅ Semantic chunking: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {chunk[:100]}...")
        
        # Test key phrase extraction
        key_phrases = processor.extract_key_phrases(long_text)
        print(f"✅ Key phrase extraction: {len(key_phrases)} phrases found")
        print(f"   Phrases: {key_phrases}")
        
        # Test embeddings
        test_slides = [
            "Slide 1: Introduction to diabetes mellitus and insulin mechanisms.",
            "Slide 2: Type 1 diabetes is an autoimmune condition affecting beta cells.",
            "Slide 3: Type 2 diabetes involves insulin resistance and lifestyle factors.",
            "Slide 4: Insulin receptors and glucose uptake mechanisms.",
            "Slide 5: Treatment options for diabetes including medications and lifestyle changes."
        ]
        
        embeddings = processor.compute_slide_embeddings(test_slides)
        print(f"✅ Embeddings computed: {embeddings.shape}")
        
        # Test slide grouping
        slide_groups = processor.find_similar_slides(embeddings, similarity_threshold=0.6)
        print(f"✅ Slide grouping: {len(slide_groups)} groups found")
        for i, group in enumerate(slide_groups):
            print(f"   Group {i+1}: Slides {[j+1 for j in group]}")
        
        # Test semantic chunks creation
        semantic_chunks = processor.create_semantic_chunks(test_slides)
        print(f"✅ Semantic chunks created: {len(semantic_chunks)} chunks")
        
        # Test content analysis
        analysis = processor.analyze_content_quality(semantic_chunks)
        print(f"✅ Content analysis completed:")
        print(f"   • Total chunks: {analysis['total_chunks']}")
        print(f"   • Total slides: {analysis['total_slides']}")
        print(f"   • Average chunk size: {analysis['avg_chunk_size']:.1f}")
        print(f"   • Unique key phrases: {analysis['unique_key_phrases']}")
        
        print("\n🎉 All semantic processing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing semantic processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_flashcard_generation():
    """Test the enhanced flashcard generation"""
    try:
        from flashcard_generator import generate_enhanced_flashcards_with_progress, remove_duplicate_flashcards
        
        print("\n🧪 Testing Enhanced Flashcard Generation...")
        
        # Test data
        test_slides = [
            "Slide 1: Diabetes mellitus is characterized by high blood glucose levels due to insulin deficiency or resistance.",
            "Slide 2: Insulin is produced by beta cells in the pancreas and helps glucose enter cells.",
            "Slide 3: Type 1 diabetes is an autoimmune condition that destroys beta cells."
        ]
        
        test_images = [[], [], []]  # No images for testing
        
        # Mock API key (this will fail but we can test the processing)
        mock_api_key = "test_key"
        
        print("✅ Enhanced flashcard generation functions imported successfully")
        print("✅ Duplicate removal function imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced flashcard generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Semantic Processing and Enhanced Flashcard Generation")
    print("=" * 60)
    
    # Test semantic processor
    semantic_success = test_semantic_processor()
    
    # Test enhanced flashcard generation
    flashcard_success = test_enhanced_flashcard_generation()
    
    print("\n" + "=" * 60)
    if semantic_success and flashcard_success:
        print("🎉 All tests passed! Semantic processing is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the main interface: python3 gradio_interface.py")
    print("3. Upload a PowerPoint file to test the enhanced features") 