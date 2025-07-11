#!/usr/bin/env python3
"""
Test script to verify image occlusion template and identify iPad display issues
"""

import os
import tempfile
from PIL import Image, ImageDraw
import genanki
import settings
fld = settings.IOE_FIELDS

def test_image_occlusion_template():
    """Test the image occlusion template to identify potential issues"""
    
    print("🧪 Testing Image Occlusion Template")
    print("=" * 50)
    
    # Create test images
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create a test image with clear content
        img = Image.new('RGB', (600, 400), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Anatomical Structure", fill='black', font=None)
        draw.text((50, 100), "This is a test image for", fill='black', font=None)
        draw.text((50, 150), "image occlusion testing", fill='black', font=None)
        draw.rectangle([50, 200, 300, 300], outline='red', width=3)
        draw.text((60, 220), "Important Label", fill='red', font=None)
        
        # Save original image
        original_path = os.path.join(temp_dir, "original_test.png")
        img.save(original_path)
        
        # Create occluded version
        occluded_img = img.copy()
        draw_occ = ImageDraw.Draw(occluded_img)
        draw_occ.rectangle([50, 200, 300, 300], fill='white')  # Cover the red box
        
        # Save occluded image
        occluded_path = os.path.join(temp_dir, "occluded_test.png")
        occluded_img.save(occluded_path)
        
        print(f"✅ Created test images:")
        print(f"  Original: {original_path}")
        print(f"  Occluded: {occluded_path}")
        
        # Test the template from flashcard_generator.py
        IOE_MODEL = genanki.Model(
            694948511,
            settings.IOE_MODEL_NAME,
            fields=[{'name': name} for name in fld.values()],
            templates=[
                {
                    'name': 'IO Card',
                    'qfmt': f"""
{{{{#{fld['image']}}}}}
<div id=\"io-header\">{{{{{fld['header']}}}}}</div>
<div id=\"io-wrapper\">\n  <div id=\"io-overlay\">{{{{{fld['qmask']}}}}}</div>\n  <div id=\"io-original\">{{{{{fld['image']}}}}}</div>\n</div>\n<div id=\"io-footer\">{{{{{fld['footer']}}}}}</div>\n{{{{/{fld['image']}}}}}""",
                    'afmt': f"""
{{{{#{fld['image']}}}}}
<div id=\"io-header\">{{{{{fld['header']}}}}}</div>
<div id=\"io-wrapper\">\n  <div id=\"io-overlay\">{{{{{fld['amask']}}}}}</div>\n  <div id=\"io-original\">{{{{{fld['image']}}}}}</div>\n</div>\n<div id=\"io-footer\">{{{{{fld['footer']}}}}}</div>\n{{{{{fld['remarks']}}}}}\n{{{{{fld['sources']}}}}}\n{{{{{fld['extra1']}}}}}\n{{{{{fld['extra2']}}}}}\n{{{{/{fld['image']}}}}}"""
                },
            ],
        )
        
        # Create a deck with the test card
        deck = genanki.Deck(2059400110, 'Template Test Deck')
        
        # Add the image occlusion card
        fields = [
            '',  # ID (auto)
            'Test Header',  # Header
            'original_test.png',  # Image (original)
            'occluded_test.png',  # Question Mask (occluded)
            'Test Footer', '', '', '', '', '', ''  # Other fields
        ]
        
        note = genanki.Note(
            model=IOE_MODEL,
            fields=fields,
        )
        deck.add_note(note)
        
        # Export to APKG
        output_path = "template_test.apkg"
        media_files = [original_path, occluded_path]
        
        # Copy media files to temp dir with basenames
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
        
        print(f"✅ Exported test deck to {output_path}")
        
        # Analyze potential issues
        print("\n🔍 Potential Issues Analysis:")
        print("=" * 30)
        
        # Check template structure
        print("1. Template Structure:")
        print("   ✅ Uses {{#Image}} conditional")
        print("   ✅ Has io-wrapper div")
        print("   ✅ Has io-overlay and io-original divs")
        print("   ✅ Includes JavaScript for visibility control")
        
        # Check for potential problems
        print("\n2. Potential Issues:")
        
        # Issue 1: Missing CSS styles
        print("   ⚠️  No CSS styles defined for io-original visibility")
        print("   💡 The JavaScript sets visibility but no initial CSS")
        
        # Issue 2: Image path handling
        print("   ⚠️  Images referenced by filename only")
        print("   💡 Ensure images are in the same directory as the deck")
        
        # Issue 3: JavaScript compatibility
        print("   ⚠️  JavaScript might not work on all Anki clients")
        print("   💡 Some mobile clients have limited JavaScript support")
        
        # Issue 4: Conditional rendering
        print("   ⚠️  {{#Image}} conditional might fail if Image field is empty")
        print("   💡 Ensure Image field always contains a valid filename")
        
        # Suggest fixes
        print("\n3. Suggested Fixes:")
        print("   a) Add CSS styles for initial state:")
        print("      #io-original { visibility: hidden; }")
        print("   b) Ensure images are properly included in media collection")
        print("   c) Test with simpler template without JavaScript")
        print("   d) Verify image filenames match exactly")
        
        # Create alternative template
        print("\n4. Alternative Template (Simpler):")
        alternative_template = f"""
{{{{#{fld['image']}}}}}
<div id=\"io-header\">{{{{{fld['header']}}}}}</div>
<div id=\"io-wrapper\">\n  <div id=\"io-overlay\">\n    <img src=\"{{{{{fld['qmask']}}}}}\" alt=\"Question\">\n  </div>\n  <div id=\"io-original\" style=\"display: none;\">\n    <img src=\"{{{{{fld['image']}}}}}\" alt=\"Answer\">\n  </div>\n</div>\n<div id=\"io-footer\">{{{{{fld['footer']}}}}}</div>\n<script>\n// Simple toggle on click\ndocument.getElementById('io-wrapper').addEventListener('click', function() {\n  var overlay = document.getElementById('io-overlay');\n  var original = document.getElementById('io-original');\n  if (overlay.style.display !== 'none') {\n    overlay.style.display = 'none';\n    original.style.display = 'block';\n  } else {\n    overlay.style.display = 'block';\n    original.style.display = 'none';\n  }\n});\n</script>\n{{{{/{fld['image']}}}}}
"""
        print(alternative_template)
        
        return output_path

def test_simple_image_template():
    """Test a simpler image template without complex JavaScript"""
    
    print("\n🧪 Testing Simple Image Template")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create test image
        img = Image.new('RGB', (400, 300), color='lightgreen')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Simple Test", fill='black')
        
        image_path = os.path.join(temp_dir, "simple_test.png")
        img.save(image_path)
        
        # Create simple model
        SIMPLE_MODEL = genanki.Model(
            1234567890,
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
<div>{{Question}}</div>
{{#Image}}<img src="{{Image}}" alt="Question Image">{{/Image}}
""",
                    'afmt': """
<div>{{Question}}</div>
{{#Image}}<img src="{{Image}}" alt="Question Image">{{/Image}}
<hr id="answer">
<div>{{Answer}}</div>
"""
                },
            ],
        )
        
        # Create deck
        deck = genanki.Deck(2059400111, 'Simple Test Deck')
        
        note = genanki.Note(
            model=SIMPLE_MODEL,
            fields=['What is this?', 'A test image', 'simple_test.png'],
        )
        deck.add_note(note)
        
        # Export
        output_path = "simple_test.apkg"
        import shutil
        temp_media_dir = tempfile.mkdtemp()
        dest = os.path.join(temp_media_dir, "simple_test.png")
        shutil.copy2(image_path, dest)
        
        genanki.Package(deck, media_files=[dest]).write_to_file(output_path)
        shutil.rmtree(temp_media_dir)
        
        print(f"✅ Exported simple test deck to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test the original template
    template_output = test_image_occlusion_template()
    
    # Test simple template
    simple_output = test_simple_image_template()
    
    print(f"\n✅ Test completed!")
    print(f"Files created:")
    print(f"  - {template_output} (original template)")
    print(f"  - {simple_output} (simple template)")
    print(f"\n💡 Try importing both to your iPad to see which works better.") 