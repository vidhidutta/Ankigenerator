import zipfile
import json
import os
from PIL import Image

APKG_FILE = "generated_flashcards.apkg"  # Change if your file is named differently

def check_apkg_media(apkg_path):
    with zipfile.ZipFile(apkg_path, 'r') as z:
        # Extract media mapping
        if "media" not in z.namelist():
            print("❌ No media file found in .apkg!")
            return
        media_json = z.read("media").decode("utf-8")
        media_map = json.loads(media_json)
        print(f"Found {len(media_map)} media entries.")

        # Extract all media files to a temp folder
        temp_dir = "apkg_media_check"
        os.makedirs(temp_dir, exist_ok=True)
        missing = []
        invalid = []
        for num, fname in media_map.items():
            if num not in z.namelist():
                missing.append(fname)
                continue
            out_path = os.path.join(temp_dir, fname)
            with open(out_path, "wb") as f:
                f.write(z.read(num))
            # Try to open as image
            try:
                with Image.open(out_path) as img:
                    img.verify()  # Will not load image data, just check integrity
            except Exception as e:
                invalid.append(fname)
        print(f"✅ Extracted media to: {temp_dir}")
        if missing:
            print(f"❌ Missing files: {missing}")
        else:
            print("✅ All mapped files are present.")
        if invalid:
            print(f"❌ Invalid/corrupted images: {invalid}")
        else:
            print("✅ All images are valid.")

if __name__ == "__main__":
    check_apkg_media(APKG_FILE)

