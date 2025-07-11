from utils.image_occlusion import encode_image_to_base64, make_qmask, make_omask
import tempfile
import os
from PIL import Image, ImageDraw
import genanki
import settings
from anki_models import IOE_MODEL

# Create a temporary directory for test images
with tempfile.TemporaryDirectory() as temp_dir:
    # Use an existing image from the apkg_media_check directory
    existing_img_path = 'apkg_media_check/slide11_img1.jpg'

    # Load the existing image without resizing
    img = Image.open(existing_img_path)
    img_path = os.path.join(temp_dir, 'test_image.png')
    img.save(img_path)

    # Generate red-box masks instead of white fill
    region = (50, 50, 150, 100)
    qmask_img = make_qmask(img, region)
    omask_img = make_omask(img, region)

    occluded_img_path  = os.path.join(temp_dir, 'occluded_test_image.png')  # question side
    original_mask_path = os.path.join(temp_dir, 'original_test_image.png')  # answer side
    qmask_img.save(occluded_img_path)
    omask_img.save(original_mask_path)

    # Debugging: Print image details
    print(f"Image size: {img.size}, Mode: {img.mode}")
    print(f"Q-mask Image size: {qmask_img.size}, Mode: {qmask_img.mode}")

    # Save images to verify
    img.show(title='Test Image')
    qmask_img.show(title='Q-mask Test Image')

    # Prepare flashcard entries with base64 encoded images
    flashcard_entries = [
        {
            'question_image_path': occluded_img_path,
            'answer_image_path': original_mask_path,
            'alt_text': 'What is hidden here?'
        }
    ]

    # Create a genanki deck using IOE model name
    deck = genanki.Deck(
        2059400110,
        settings.IOE_MODEL_NAME
    )

    # Add notes to the deck
    for entry in flashcard_entries:
        note = genanki.Note(
            model=IOE_MODEL,
            fields=[
                f"<img src='{os.path.basename(entry['question_image_path'])}'>",
                f"<img src='{os.path.basename(entry['answer_image_path'])}'>"
            ]
        )
        deck.add_note(note)

    # Build media list from entries
    media = []
    for entry in flashcard_entries:
        media.extend([entry['question_image_path'], entry['answer_image_path']])

    # Determine output filename (use provided variable or fallback)
    output_file = 'test_occlusion_deck.apkg'

    # 'output_file' should be defined from CLI args earlier in the script
    genanki.Package(deck, media_files=media).write_to_file(output_file)

# Define a specific directory to save images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Save images to the specific directory
img_path        = os.path.join(output_dir, 'test_image.png')
qmask_out_path  = os.path.join(output_dir, 'occluded_test_image.png')
omask_out_path  = os.path.join(output_dir, 'original_test_image.png')
img.save(img_path)
qmask_img.save(qmask_out_path)
omask_img.save(omask_out_path)

# Debugging: Print image paths
print(f"Images saved to: {output_dir}")

print("Deck generated: test_occlusion_deck.apkg") 