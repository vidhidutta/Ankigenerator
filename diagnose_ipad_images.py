#!/usr/bin/env python3
"""
Diagnostic script for iPad image display issues
"""

import os
import tempfile
from PIL import Image, ImageDraw
import genanki

def create_diagnostic_deck():
    """Create a comprehensive diagnostic deck to test different image scenarios"""
    
    print("🔍 Creating iPad Image Diagnostic Deck")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test Case 1: Simple image without occlusion
        print("1. Creating simple image test...")
        simple_img = Image.new('RGB', (400, 300), color='lightgreen')
        draw = ImageDraw.Draw(simple_img)
        draw.text((50, 50), "Simple Test Image", fill='black')
        draw.text((50, 100), "This should always show", fill='black')
        simple_path = os.path.join(temp_dir, "simple_test.png")
        simple_img.save(simple_path)
        
        # Test Case 2: Basic occlusion
        print("2. Creating basic occlusion test...")
        basic_img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(basic_img)
        draw.text((50, 50), "Basic Occlusion Test", fill='black')
        draw.text((50, 100), "This text should be visible", fill='black')
        draw.rectangle([50, 150, 200, 200], outline='red', width=2)
        draw.text((60, 170), "HIDDEN TEXT", fill='red')
        
        basic_orig_path = os.path.join(temp_dir, "basic_original.png")
        basic_img.save(basic_orig_path)
        
        basic_occ_img = basic_img.copy()
        draw_occ = ImageDraw.Draw(basic_occ_img)
        draw_occ.rectangle([50, 150, 200, 200], fill='white')
        basic_occ_path = os.path.join(temp_dir, "basic_occluded.png")
        basic_occ_img.save(basic_occ_path)
        
        # Test Case 3: Medical-style image
        print("3. Creating medical-style test...")
        medical_img = Image.new('RGB', (500, 400), color='white')
        draw = ImageDraw.Draw(medical_img)
        draw.text((50, 30), "Anatomical Diagram", fill='black')
        draw.ellipse([100, 100, 200, 150], outline='black', width=2)
        draw.text((110, 120), "Heart", fill='black')
        draw.rectangle([250, 100, 350, 150], outline='blue', width=2)
        draw.text((260, 120), "Lungs", fill='blue')
        draw.line([50, 200, 450, 200], fill='gray', width=1)
        draw.text((50, 220), "Blood vessels", fill='red')
        
        medical_orig_path = os.path.join(temp_dir, "medical_original.png")
        medical_img.save(medical_orig_path)
        
        medical_occ_img = medical_img.copy()
        draw_occ = ImageDraw.Draw(medical_occ_img)
        draw_occ.rectangle([250, 100, 350, 150], fill='white')
        medical_occ_path = os.path.join(temp_dir, "medical_occluded.png")
        medical_occ_img.save(medical_occ_path)
        
        # Create different note types for testing
        
        # 1. Simple image model
        SIMPLE_MODEL = genanki.Model(
            111111111,
            'Simple Image Test',
            fields=[
                {'name': 'Question'},
                {'name': 'Answer'},
                {'name': 'Image'},
            ],
            templates=[
                {
                    'name': 'Simple Image Card',
                    'qfmt': """
<div style="text-align: center; font-size: 18px; margin-bottom: 10px;">{{Question}}</div>
<div style="text-align: center;">
  <img src="{{Image}}" alt="Test Image" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
</div>
""",
                    'afmt': """
<div style="text-align: center; font-size: 18px; margin-bottom: 10px;">{{Question}}</div>
<div style="text-align: center;">
  <img src="{{Image}}" alt="Test Image" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
</div>
<hr id="answer">
<div style="text-align: center; font-size: 16px; color: #333;">{{Answer}}</div>
"""
                },
            ],
        )
        
        # 2. Improved occlusion model
        IMPROVED_MODEL = genanki.Model(
            222222222,
            'Improved Occlusion Test',
            fields=[
                {'name': 'Header'},
                {'name': 'Question Image'},
                {'name': 'Answer Image'},
                {'name': 'Footer'},
            ],
            templates=[
                {
                    'name': 'Improved Occlusion Card',
                    'qfmt': """
<style>
#io-wrapper { position: relative; display: inline-block; }
#io-overlay { position: absolute; top: 0; left: 0; z-index: 2; }
#io-original { position: absolute; top: 0; left: 0; z-index: 1; visibility: hidden; }
#io-original img, #io-overlay img { max-width: 100%; height: auto; border: 1px solid #ccc; }
</style>
<div style="text-align: center; margin-bottom: 10px;">{{Header}}</div>
<div style="text-align: center;">
  <div id="io-wrapper">
    <div id="io-overlay">
      <img src="{{Question Image}}" alt="Question" onload="this.style.display='block';">
    </div>
    <div id="io-original">
      <img src="{{Answer Image}}" alt="Answer" onload="this.style.display='block';">
    </div>
  </div>
</div>
<div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">{{Footer}}</div>
<script>
(function() {
  var wrapper = document.getElementById('io-wrapper');
  var overlay = document.getElementById('io-overlay');
  var original = document.getElementById('io-original');
  
  if (wrapper && overlay && original) {
    wrapper.addEventListener('click', function() {
      if (overlay.style.display !== 'none') {
        overlay.style.display = 'none';
        original.style.visibility = 'visible';
      } else {
        overlay.style.display = 'block';
        original.style.visibility = 'hidden';
      }
    });
    
    wrapper.addEventListener('touchstart', function(e) {
      e.preventDefault();
      if (overlay.style.display !== 'none') {
        overlay.style.display = 'none';
        original.style.visibility = 'visible';
      } else {
        overlay.style.display = 'block';
        original.style.visibility = 'hidden';
      }
    });
  }
})();
</script>
""",
                    'afmt': """
<style>
#io-wrapper { position: relative; display: inline-block; }
#io-overlay { position: absolute; top: 0; left: 0; z-index: 2; }
#io-original { position: absolute; top: 0; left: 0; z-index: 1; visibility: visible; }
#io-original img, #io-overlay img { max-width: 100%; height: auto; border: 1px solid #ccc; }
</style>
<div style="text-align: center; margin-bottom: 10px;">{{Header}}</div>
<div style="text-align: center;">
  <div id="io-wrapper">
    <div id="io-overlay">
      <img src="{{Question Image}}" alt="Question" onload="this.style.display='block';">
    </div>
    <div id="io-original">
      <img src="{{Answer Image}}" alt="Answer" onload="this.style.display='block';">
    </div>
  </div>
</div>
<div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">{{Footer}}</div>
<script>
(function() {
  var overlay = document.getElementById('io-overlay');
  var original = document.getElementById('io-original');
  
  if (overlay && original) {
    overlay.style.display = 'block';
    original.style.visibility = 'visible';
  }
})();
</script>
"""
                },
            ],
        )
        
        # 3. Fallback model (no JavaScript)
        FALLBACK_MODEL = genanki.Model(
            333333333,
            'Fallback Test (No JS)',
            fields=[
                {'name': 'Header'},
                {'name': 'Question Image'},
                {'name': 'Answer Image'},
                {'name': 'Footer'},
            ],
            templates=[
                {
                    'name': 'Fallback Card',
                    'qfmt': """
<div style="text-align: center; margin-bottom: 10px;">{{Header}}</div>
<div style="text-align: center;">
  <img src="{{Question Image}}" alt="Question" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
</div>
<div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">{{Footer}}</div>
""",
                    'afmt': """
<div style="text-align: center; margin-bottom: 10px;">{{Header}}</div>
<div style="text-align: center;">
  <img src="{{Question Image}}" alt="Question" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
</div>
<hr id="answer">
<div style="text-align: center;">
  <img src="{{Answer Image}}" alt="Answer" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
</div>
<div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">{{Footer}}</div>
"""
                },
            ],
        )
        
        # Create deck with all test cases
        deck = genanki.Deck(2059400113, 'iPad Image Diagnostic Deck')
        
        # Add simple image test
        simple_note = genanki.Note(
            model=SIMPLE_MODEL,
            fields=['Can you see this image?', 'If you can see this, basic images work!', 'simple_test.png'],
        )
        deck.add_note(simple_note)
        
        # Add improved occlusion test
        improved_note = genanki.Note(
            model=IMPROVED_MODEL,
            fields=['Basic Occlusion Test', 'basic_occluded.png', 'basic_original.png', 'Tap to reveal hidden text'],
        )
        deck.add_note(improved_note)
        
        # Add medical occlusion test
        medical_note = genanki.Note(
            model=IMPROVED_MODEL,
            fields=['Medical Diagram Test', 'medical_occluded.png', 'medical_original.png', 'Tap to reveal lung labels'],
        )
        deck.add_note(medical_note)
        
        # Add fallback test
        fallback_note = genanki.Note(
            model=FALLBACK_MODEL,
            fields=['Fallback Test (No JavaScript)', 'basic_occluded.png', 'basic_original.png', 'Manual flip required'],
        )
        deck.add_note(fallback_note)
        
        # Export deck
        output_path = "ipad_diagnostic_deck.apkg"
        media_files = [simple_path, basic_orig_path, basic_occ_path, medical_orig_path, medical_occ_path]
        
        import shutil
        temp_media_dir = tempfile.mkdtemp()
        media_paths = []
        for f in media_files:
            if os.path.exists(f):
                dest = os.path.join(temp_media_dir, os.path.basename(f))
                shutil.copy2(f, dest)
                media_paths.append(dest)
        
        genanki.Package(deck, media_files=media_paths).write_to_file(output_path)
        shutil.rmtree(temp_media_dir)
        
        print(f"✅ Created diagnostic deck: {output_path}")
        
        # Print diagnostic instructions
        print("\n📋 Diagnostic Instructions:")
        print("=" * 30)
        print("1. Import this deck to your iPad")
        print("2. Test each card type:")
        print("   - Card 1: Simple image (should always work)")
        print("   - Card 2: Basic occlusion with JavaScript")
        print("   - Card 3: Medical diagram with JavaScript")
        print("   - Card 4: Fallback (no JavaScript)")
        print("3. Report which cards show images and which don't")
        print("4. Test tapping on occlusion cards to see if they reveal answers")
        
        return output_path

if __name__ == "__main__":
    diagnostic_output = create_diagnostic_deck()
    print(f"\n✅ Diagnostic deck created: {diagnostic_output}")
    print("💡 Import this to your iPad and test each card type!") 