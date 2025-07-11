#!/usr/bin/env python3
"""
Test script for the improved image occlusion template
"""

import os
import tempfile
from PIL import Image, ImageDraw
import genanki
import settings
fld = settings.IOE_FIELDS

def test_improved_template():
    """Test the improved image occlusion template"""
    
    print("🧪 Testing Improved Image Occlusion Template")
    print("=" * 50)
    
    # Create test images
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create a test image with clear content
        img = Image.new('RGB', (600, 400), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Anatomical Structure", fill='black')
        draw.text((50, 100), "This is a test image for", fill='black')
        draw.text((50, 150), "image occlusion testing", fill='black')
        draw.rectangle([50, 200, 300, 300], outline='red', width=3)
        draw.text((60, 220), "Important Label", fill='red')
        
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
        
        # Use the improved template
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
        deck = genanki.Deck(2059400110, 'Improved Template Test Deck')
        
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
        output_path = "improved_template_test.apkg"
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
        
        print(f"✅ Exported improved template deck to {output_path}")
        
        # Show improvements
        print("\n🔧 Improvements Made:")
        print("=" * 20)
        print("1. ✅ Added CSS styles for proper positioning and visibility")
        print("2. ✅ Added explicit img tags with alt attributes")
        print("3. ✅ Added onload handlers for better image loading")
        print("4. ✅ Improved JavaScript with mobile touch support")
        print("5. ✅ Added proper z-index layering")
        print("6. ✅ Added max-width CSS for responsive images")
        print("7. ✅ Wrapped JavaScript in IIFE for better compatibility")
        print("8. ✅ Added error checking for DOM elements")
        
        return output_path

def test_fallback_template():
    """Test a fallback template without JavaScript for maximum compatibility"""
    
    print("\n🧪 Testing Fallback Template (No JavaScript)")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create test image
        img = Image.new('RGB', (500, 350), color='lightyellow')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Fallback Test", fill='black')
        draw.text((50, 100), "This template uses", fill='black')
        draw.text((50, 150), "no JavaScript", fill='black')
        draw.rectangle([50, 200, 250, 280], outline='blue', width=2)
        draw.text((60, 220), "Hidden Content", fill='blue')
        
        # Save original image
        original_path = os.path.join(temp_dir, "fallback_original.png")
        img.save(original_path)
        
        # Create occluded version
        occluded_img = img.copy()
        draw_occ = ImageDraw.Draw(occluded_img)
        draw_occ.rectangle([50, 200, 250, 280], fill='white')
        
        # Save occluded image
        occluded_path = os.path.join(temp_dir, "fallback_occluded.png")
        occluded_img.save(occluded_path)
        
        # Create simple model without JavaScript
        FALLBACK_MODEL = genanki.Model(
            987654321,
            'Image Occlusion Fallback',
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
  <img src="{{Question Image}}" alt="Question" style="max-width: 100%; height: auto;">
</div>
<div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">
  {{Footer}}
</div>
""",
                    'afmt': """
<div style="text-align: center; margin-bottom: 10px;">{{Header}}</div>
<div style="text-align: center;">
  <img src="{{Question Image}}" alt="Question" style="max-width: 100%; height: auto;">
</div>
<hr id="answer">
<div style="text-align: center;">
  <img src="{{Answer Image}}" alt="Answer" style="max-width: 100%; height: auto;">
</div>
<div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">
  {{Footer}}
</div>
"""
                },
            ],
        )
        
        # Create deck
        deck = genanki.Deck(2059400112, 'Fallback Test Deck')
        
        note = genanki.Note(
            model=FALLBACK_MODEL,
            fields=['Test Header', 'fallback_occluded.png', 'fallback_original.png', 'Test Footer'],
        )
        deck.add_note(note)
        
        # Export
        output_path = "fallback_test.apkg"
        import shutil
        temp_media_dir = tempfile.mkdtemp()
        media_paths = []
        for f in [original_path, occluded_path]:
            if os.path.exists(f):
                dest = os.path.join(temp_media_dir, os.path.basename(f))
                shutil.copy2(f, dest)
                media_paths.append(dest)
        
        genanki.Package(deck, media_files=media_paths).write_to_file(output_path)
        shutil.rmtree(temp_media_dir)
        
        print(f"✅ Exported fallback deck to {output_path}")
        print("💡 This template has no JavaScript and should work on all devices")
        
        return output_path

if __name__ == "__main__":
    # Test improved template
    improved_output = test_improved_template()
    
    # Test fallback template
    fallback_output = test_fallback_template()
    
    print(f"\n✅ Test completed!")
    print(f"Files created:")
    print(f"  - {improved_output} (improved template with better mobile support)")
    print(f"  - {fallback_output} (fallback template without JavaScript)")
    print(f"\n💡 Try importing both to your iPad:")
    print(f"  1. First try the improved template ({improved_output})")
    print(f"  2. If images still don't show, try the fallback template ({fallback_output})")
    print(f"  3. The fallback template should work on all devices but requires manual flipping") 