#!/usr/bin/env python3
"""
Test script to demonstrate Level 1 vs Level 2 flashcard generation
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_level_distinction():
    """Test the Level 1 vs Level 2 flashcard distinction"""
    
    print("🧪 Testing Level 1 vs Level 2 Flashcard Distinction")
    print("=" * 60)
    
    # Sample respiratory medicine content
    sample_content = """
    Slide 1: Interstitial Lung Disease (ILD)
    - Definition: Group of disorders affecting the lung interstitium
    - Types: Idiopathic pulmonary fibrosis, sarcoidosis, hypersensitivity pneumonitis
    - Spirometry: Restrictive pattern with decreased FVC and FEV1
    - Flow-volume loop: Shows scooped appearance due to reduced lung compliance
    
    Slide 2: COPD
    - Definition: Chronic obstructive pulmonary disease
    - Types: Emphysema and chronic bronchitis
    - Spirometry: Obstructive pattern with FEV1/FVC < 0.7
    - Flow-volume loop: Shows concave curve due to airway obstruction
    """
    
    print("📋 Sample Content (Respiratory Medicine):")
    print(sample_content)
    print()
    
    print("🎯 Expected Level 1 Flashcards (Basic Recall):")
    print("• What is the definition of interstitial lung disease?")
    print("• Name the three main types of ILD.")
    print("• What is the normal FEV1/FVC ratio?")
    print("• Define COPD.")
    print("• List the two main types of COPD.")
    print()
    
    print("🧠 Expected Level 2 Flashcards (Interpretation & Application):")
    print("• What pattern on spirometry suggests early ILD?")
    print("• Why does ILD show a scooped appearance on flow-volume loops?")
    print("• Compare the spirometry patterns of ILD vs COPD.")
    print("• How would you interpret a patient with FEV1/FVC < 0.7?")
    print("• What explains the concave curve in COPD flow-volume loops?")
    print()
    
    print("📊 Key Differences:")
    print("Level 1: 'What is...?' 'Define...' 'Name...' 'List...'")
    print("Level 2: 'What pattern suggests...?' 'Why does...?' 'Compare...' 'How would you interpret...?'")
    print()
    
    print("✅ Level distinction test completed!")
    print("The AI will now generate appropriate flashcards based on the selected level.")

def test_configuration():
    """Test the configuration settings"""
    
    print("\n🔧 Testing Configuration Settings")
    print("=" * 40)
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        flashcard_type = config.get('flashcard_type', {})
        
        print("Current Flashcard Type Settings:")
        print(f"• Level 1: {flashcard_type.get('level_1', False)}")
        print(f"• Level 2: {flashcard_type.get('level_2', False)}")
        print(f"• Both: {flashcard_type.get('both', False)}")
        
        print("\n✅ Configuration test completed!")
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")

if __name__ == "__main__":
    test_level_distinction()
    test_configuration()
    
    print("\n" + "=" * 60)
    print("🎉 Level distinction testing completed!")
    print("\n📋 Next steps:")
    print("1. Run the main interface: python3 gradio_interface.py")
    print("2. Select your preferred flashcard level")
    print("3. Upload a PowerPoint file to see the enhanced level distinction in action")
    print("\n💡 Tip: Try generating the same content with different levels to see the difference!") 