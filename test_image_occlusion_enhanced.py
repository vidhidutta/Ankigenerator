#!/usr/bin/env python3
"""
Test script for Image Occlusion Enhanced card type with correct template and image handling.
"""
import os
import tempfile
from PIL import Image, ImageDraw
import genanki
import settings
fld = settings.IOE_FIELDS

def test_image_occlusion_enhanced():
    print("🧪 Testing Image Occlusion Enhanced Card Type")
    print("=" * 50)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create original image
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Occlusion Test", fill='black')
        draw.text((50, 100), "Original Image", fill='blue')
        original_path = os.path.join(temp_dir, "original_test.png")
        img.save(original_path)
        # Create occluded image
        occluded_img = img.copy()
        draw_occ = ImageDraw.Draw(occluded_img)
        draw_occ.rectangle([40, 90, 250, 130], fill='white')
        occluded_path = os.path.join(temp_dir, "occluded_test.png")
        occluded_img.save(occluded_path)
        # Centralized IOE model using settings
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
  <div id=\"io-mask\">{{{{{fld['qmask']}}}}}</div>
  <div id=\"io-footer\">{{{{{fld['footer']}}}}}</div>
{{{{/{fld['image']}}}}}""",
                    'afmt': f"""
{{{{#{fld['image']}}}}}
  <div id=\"io-header\">{{{{{fld['header']}}}}}</div>
  <div id=\"io-mask\">{{{{{fld['amask']}}}}}</div>
  <div id=\"io-original\">{{{{{fld['omask']}}}}}</div>
  <div id=\"io-footer\">{{{{{fld['footer']}}}}}</div>
  <hr>
  {{{{{fld['remarks']}}}}}
  {{{{{fld['sources']}}}}}
  {{{{{fld['extra1']}}}}}
  {{{{{fld['extra2']}}}}}
{{{{/{fld['image']}}}}}"""
                },
            ],
        )
        # Add a single test card
        fields = [
            '',  # ID
            'Test Header',  # Header
            'original_test.png',  # Image (original)
            'occluded_test.png',  # Question Mask (occluded)
            'Test Footer', '', '', '', '', '', ''
        ]
        note = genanki.Note(
            model=IOE_MODEL,
            fields=fields,
        )
        deck = genanki.Deck(2059400999, 'IOE Minimal Test Deck')
        deck.add_note(note)
        # Copy images to temp dir with basenames
        import shutil
        temp_media_dir = tempfile.mkdtemp()
        for f in [original_path, occluded_path]:
            dest = os.path.join(temp_media_dir, os.path.basename(f))
            shutil.copy2(f, dest)
        genanki.Package(deck, media_files=[
            os.path.join(temp_media_dir, 'original_test.png'),
            os.path.join(temp_media_dir, 'occluded_test.png')
        ]).write_to_file('ioe_minimal_test.apkg')
        shutil.rmtree(temp_media_dir)
        print("✅ Exported IOE minimal test deck to ioe_minimal_test.apkg")

if __name__ == "__main__":
    test_image_occlusion_enhanced() 