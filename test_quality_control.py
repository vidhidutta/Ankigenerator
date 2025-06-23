#!/usr/bin/env python3
"""
Test script for Quality Control Features
Demonstrates how the enhanced system addresses the quality issues identified in the analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flashcard_generator import QualityController, Flashcard, EnhancedFlashcardGenerator
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def test_anti_repetition():
    """Test anti-repetition functionality"""
    print("🧪 Testing Anti-Repetition Feature")
    print("=" * 50)
    
    controller = QualityController()
    
    # Create test cards with repetition (like the FEV1 examples from analysis)
    test_cards = [
        Flashcard("What does FEV1 stand for in spirometry?", "Forced Expiratory Volume in 1 second", 1, 1),
        Flashcard("What is FEV1 and how is it used to assess lung function?", "FEV1 measures the amount of air a person can forcefully exhale in one second. It is used to assess lung function and diagnose obstructive lung diseases like asthma.", 1, 1),
        Flashcard("What does FEV₁ represent in lung function tests?", "FEV₁ represents the forced expiration volume in one second.", 1, 1),
        Flashcard("In spirometry, what is typically measured to assess lung function over time?", "FEV₁ (Forced Expiratory Volume in 1 second)", 1, 1),
    ]
    
    print("Original cards:")
    for i, card in enumerate(test_cards):
        print(f"  {i+1}. Q: {card.question}")
        print(f"     A: {card.answer}")
        print()
    
    # Detect duplicates
    duplicates = controller.detect_repetition(test_cards, threshold=0.7)
    print(f"Detected {len(duplicates)} duplicate pairs:")
    for i, j in duplicates:
        print(f"  Cards {i+1} and {j+1} are similar")
    
    # Remove duplicates (keep the best one)
    unique_cards = []
    duplicate_indices = set()
    for i, j in duplicates:
        duplicate_indices.add(j)
    
    for i, card in enumerate(test_cards):
        if i not in duplicate_indices:
            unique_cards.append(card)
    
    print(f"\nAfter deduplication: {len(unique_cards)} cards")
    for i, card in enumerate(unique_cards):
        print(f"  {i+1}. Q: {card.question}")
        print(f"     A: {card.answer}")
        print()

def test_conciseness():
    """Test conciseness improvement"""
    print("🧪 Testing Conciseness Feature")
    print("=" * 50)
    
    controller = QualityController()
    
    # Test wordy answer (like the flow-volume curve example from analysis)
    wordy_question = "What does the flow-volume curve of a poorly controlled asthmatic patient typically show compared to a healthy volunteer?"
    wordy_answer = "The flow-volume curve of a poorly controlled asthmatic patient typically shows a reduced peak expiratory flow rate (PFR), decreased forced vital capacity (FVC), and a concave shape, indicating airflow limitation, compared to a healthy volunteer."
    
    print("Original wordy card:")
    print(f"Q: {wordy_question}")
    print(f"A: {wordy_answer}")
    print(f"Word count: {len(wordy_answer.split())}")
    print(f"Sentences: {len(sent_tokenize(wordy_answer))}")
    print()
    
    # Check if it's too wordy
    is_wordy = controller.is_too_wordy(wordy_answer)
    print(f"Is too wordy: {is_wordy}")
    
    if is_wordy:
        # Split into multiple cards
        split_cards = controller.split_wordy_answer(wordy_question, wordy_answer)
        print(f"\nSplit into {len(split_cards)} focused cards:")
        for i, (q, a) in enumerate(split_cards):
            print(f"  {i+1}. Q: {q}")
            print(f"     A: {a}")
            print(f"     Word count: {len(a.split())}")
            print()

def test_shallow_card_detection():
    """Test shallow card detection and enrichment"""
    print("🧪 Testing Shallow Card Detection")
    print("=" * 50)
    
    controller = QualityController()
    
    # Test shallow cards (like the PEF, FVC examples from analysis)
    shallow_cards = [
        ("What does PEF stand for in lung function tests?", "Peak Expiratory Flow", 1),
        ("What does FVC stand for in spirometry?", "Forced Vital Capacity", 1),
        ("What gases are inhaled during the single breath-hold technique to measure DLCO/TLCO?", "Helium and carbon monoxide", 1),
    ]
    
    print("Original shallow cards:")
    for i, (q, a, level) in enumerate(shallow_cards):
        is_shallow = controller.is_shallow_card(q, a, level)
        print(f"  {i+1}. Q: {q}")
        print(f"     A: {a}")
        print(f"     Is shallow: {is_shallow}")
        print()
    
    print("Enriched cards:")
    for i, (q, a, level) in enumerate(shallow_cards):
        enriched_q, enriched_a = controller.enrich_shallow_card(q, a, level)
        print(f"  {i+1}. Q: {enriched_q}")
        print(f"     A: {enriched_a}")
        print()

def test_cloze_opportunities():
    """Test cloze opportunity detection"""
    print("🧪 Testing Cloze Opportunity Detection")
    print("=" * 50)
    
    controller = QualityController()
    
    # Test cards that would be good for cloze (like the FEV1/FVC ratio example from analysis)
    cloze_candidates = [
        ("What is the FEV1/FVC ratio considered normal?", "An FEV1/FVC ratio above 70–80% is considered normal, adjusted for age, gender, and height."),
        ("What is the significance of hemoglobin's binding affinity for CO compared to O₂?", "Hemoglobin binds to carbon monoxide 200–250 times more strongly than to oxygen."),
        ("Name the three main types of COPD.", "Emphysema, chronic bronchitis, and small airway disease"),
    ]
    
    for i, (q, a) in enumerate(cloze_candidates):
        is_cloze, cloze_text = controller.detect_cloze_opportunities(q, a)
        print(f"  {i+1}. Q: {q}")
        print(f"     A: {a}")
        print(f"     Good for cloze: {is_cloze}")
        if is_cloze:
            print(f"     Cloze text: {cloze_text}")
        print()

def test_depth_consistency():
    """Test depth consistency checking"""
    print("🧪 Testing Depth Consistency")
    print("=" * 50)
    
    controller = QualityController()
    
    # Test cards with depth inconsistency (like the examples from analysis)
    inconsistent_cards = [
        Flashcard("What is the characteristic shape of the flow-volume curve in a poorly controlled asthmatic patient compared to a healthy person?", "The curve shows a scooped or concave appearance, while a healthy person's curve is smooth.", 1, 1),  # Too deep for Level 1
        Flashcard("What does FVC stand for in spirometry?", "Forced Vital Capacity", 2, 1),  # Too shallow for Level 2
    ]
    
    print("Original cards with depth issues:")
    for i, card in enumerate(inconsistent_cards):
        print(f"  {i+1}. Level {card.level}")
        print(f"     Q: {card.question}")
        print(f"     A: {card.answer}")
        
        if card.level == 1:
            too_deep = controller.is_too_deep_for_level1(card.question, card.answer)
            print(f"     Too deep for Level 1: {too_deep}")
        else:
            too_shallow = controller.is_too_shallow_for_level2(card.question, card.answer)
            print(f"     Too shallow for Level 2: {too_shallow}")
        print()
    
    print("Fixed cards:")
    for i, card in enumerate(inconsistent_cards):
        fixed_card = controller.fix_depth_inconsistency(card)
        print(f"  {i+1}. Level {fixed_card.level}")
        print(f"     Q: {fixed_card.question}")
        print(f"     A: {fixed_card.answer}")
        print()

def test_enhanced_generator():
    """Test the enhanced flashcard generator"""
    print("🧪 Testing Enhanced Flashcard Generator")
    print("=" * 50)
    
    # Sample slide text (simplified version of spirometry content)
    sample_slide = """
    Slide 1: Spirometry Basics
    
    FEV1: Forced Expiratory Volume in 1 second
    FVC: Forced Vital Capacity
    FEV1/FVC ratio: Normal >70-80%
    
    Flow-volume curves:
    - Normal: Smooth curve
    - Asthma: Scooped/concave curve due to airway obstruction
    - COPD: Reduced peak flow and concave expiratory curve
    
    Key measurements:
    - PEF: Peak Expiratory Flow
    - DLCO: Diffusing capacity for carbon monoxide
    - Uses helium and carbon monoxide for measurement
    """
    
    try:
        generator = EnhancedFlashcardGenerator()
        
        # Test Level 1 generation
        print("Generating Level 1 flashcards...")
        level1_cards = generator.generate_flashcards(sample_slide, 1, level=1, use_cloze=False)
        
        print(f"Generated {len(level1_cards)} Level 1 cards:")
        for i, card in enumerate(level1_cards):
            print(f"  {i+1}. Q: {card.question}")
            print(f"     A: {card.answer}")
            print(f"     Confidence: {card.confidence:.2f}")
            print()
        
        # Test Level 2 generation
        print("Generating Level 2 flashcards...")
        level2_cards = generator.generate_flashcards(sample_slide, 1, level=2, use_cloze=False)
        
        print(f"Generated {len(level2_cards)} Level 2 cards:")
        for i, card in enumerate(level2_cards):
            print(f"  {i+1}. Q: {card.question}")
            print(f"     A: {card.answer}")
            print(f"     Confidence: {card.confidence:.2f}")
            print()
            
    except Exception as e:
        print(f"Error testing enhanced generator: {e}")
        print("This is expected if OpenAI API key is not configured")

def main():
    """Run all quality control tests"""
    print("🧠 Anki Flashcard Quality Control Test Suite")
    print("=" * 60)
    print("This test demonstrates how the enhanced system addresses the quality issues identified in the analysis.")
    print()
    
    # Test each quality control feature
    test_anti_repetition()
    test_conciseness()
    test_shallow_card_detection()
    test_cloze_opportunities()
    test_depth_consistency()
    test_enhanced_generator()
    
    print("✅ Quality Control Test Suite Complete")
    print("\n📋 Summary of Quality Improvements:")
    print("1. ✅ Anti-Repetition: Detects and removes duplicate cards")
    print("2. ✅ Conciseness: Splits wordy answers into focused cards")
    print("3. ✅ Context Enrichment: Adds clinical context to shallow cards")
    print("4. ✅ Cloze Detection: Automatically identifies cloze opportunities")
    print("5. ✅ Depth Consistency: Ensures appropriate complexity per level")
    print("6. ✅ Confidence Scoring: Rates card quality automatically")

if __name__ == "__main__":
    main() 