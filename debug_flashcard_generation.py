#!/usr/bin/env python3
"""
Debug script to test flashcard generation and see what's happening
"""

import os
import tempfile
from flashcard_generator import (
    generate_enhanced_flashcards_with_progress,
    parse_flashcards,
    Flashcard
)
from gradio_interface import validate_flashcard, flatten_flashcard_list

def test_parse_flashcards():
    """Test the parse_flashcards function with sample AI responses"""
    print("🔍 Testing parse_flashcards function")
    print("=" * 50)
    
    # Test cases
    test_responses = [
        # Good response
        """Question: What is the mechanism of action of beta blockers?
Answer: Beta blockers competitively inhibit beta-adrenergic receptors, reducing sympathetic nervous system activity.

Question: What are the main side effects of beta blockers?
Answer: Bradycardia, fatigue, and sexual dysfunction.""",
        
        # Bad response (empty questions/answers)
        """Question: 
Answer: 

Question: What is a drug?
Answer: """,
        
        # Mixed response
        """Question: What is pharmacology?
Answer: The study of drugs and their effects on the body.

Question: 
Answer: 

Question: What are receptors?
Answer: Proteins that bind to specific molecules and trigger cellular responses."""
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nTest case {i+1}:")
        print(f"Response: {response[:100]}...")
        
        flashcards, _ = parse_flashcards(response, use_cloze=False, slide_number=1, level=1)
        print(f"Parsed {len(flashcards)} flashcards:")
        
        for j, card in enumerate(flashcards):
            is_valid, reason = validate_flashcard(card)
            print(f"  Card {j+1}: {is_valid} - {reason}")
            if is_valid:
                print(f"    Q: {card.question}")
                print(f"    A: {card.answer}")
    
    print()

def test_flashcard_validation():
    """Test the flashcard validation function"""
    print("🔍 Testing flashcard validation")
    print("=" * 50)
    
    test_cards = [
        Flashcard("What is the heart?", "The heart pumps blood.", 1, 1),
        Flashcard("", "Empty question", 1, 1),
        Flashcard("Empty answer", "", 1, 1),
        Flashcard("   ", "Whitespace question", 1, 1),
        Flashcard("Whitespace answer", "   ", 1, 1),
        None,
        {"question": "Dict card", "answer": "Dict answer"},
        {"question": "", "answer": "Empty dict question"},
    ]
    
    for i, card in enumerate(test_cards):
        is_valid, reason = validate_flashcard(card)
        print(f"Card {i+1}: {is_valid} - {reason}")
        if is_valid:
            if hasattr(card, 'question'):
                print(f"  Q: '{card.question}'")
                print(f"  A: '{card.answer}'")
            else:
                print(f"  Q: '{card.get('question', 'N/A')}'")
                print(f"  A: '{card.get('answer', 'N/A')}'")
    
    print()

def test_flatten_flashcard_list():
    """Test the flatten_flashcard_list function"""
    print("🔍 Testing flashcard list flattening")
    print("=" * 50)
    
    # Create test flashcards
    valid_card1 = Flashcard("Q1", "A1", 1, 1)
    valid_card2 = Flashcard("Q2", "A2", 1, 1)
    invalid_card = Flashcard("", "A3", 1, 1)
    
    test_lists = [
        [valid_card1, valid_card2],  # All valid
        [valid_card1, None, valid_card2],  # With None
        [valid_card1, invalid_card, valid_card2],  # With invalid
        [[valid_card1], [valid_card2]],  # Nested lists
        [],  # Empty list
    ]
    
    for i, card_list in enumerate(test_lists):
        print(f"Test case {i+1}: {len(card_list)} items")
        flattened = flatten_flashcard_list(card_list)
        print(f"  Result: {len(flattened)} valid cards")
        for j, card in enumerate(flattened):
            print(f"    Card {j+1}: {card.question} -> {card.answer}")
        print()
    
    print()

def test_basic_generation():
    """Test basic flashcard generation"""
    print("🔍 Testing basic flashcard generation")
    print("=" * 50)
    
    # Mock slide texts
    slide_texts = [
        "Slide 1: Introduction to Pharmacology\n- Drug classification\n- Mechanism of action",
        "Slide 2: Beta Blockers\n- Propranolol\n- Mechanism: Beta-1 receptor blockade\n- Indications: Hypertension, Angina",
        "Slide 3: Side Effects\n- Bradycardia\n- Fatigue\n- Sexual dysfunction"
    ]
    
    # Mock slide images (empty for this test)
    slide_images = [[], [], []]
    
    try:
        # Test basic generation
        flashcards = generate_enhanced_flashcards_with_progress(
            slide_texts=slide_texts,
            slide_images=slide_images,
            api_key="test_key",  # This will fail, but we can see the error
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.3,
            use_cloze=False,
            question_style="Word for word"
        )
        
        print(f"Generated {len(flashcards)} flashcards")
        
    except Exception as e:
        print(f"Error in basic generation: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 Debugging Flashcard Generation")
    print("=" * 50)
    
    test_parse_flashcards()
    test_flashcard_validation()
    test_flatten_flashcard_list()
    test_basic_generation()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    main() 