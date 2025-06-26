import os
import uuid
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFilter
import pytesseract
import csv


def detect_text_regions(image: Image.Image, conf_threshold: int = 60) -> List[Tuple[int, int, int, int]]:
    """
    Detect text regions in an image using pytesseract OCR.
    Returns a list of bounding boxes (x, y, w, h) for regions with confidence >= conf_threshold.
    """
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(ocr_data['level'])
    regions = []
    for i in range(n_boxes):
        conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i].isdigit() else -1
        if conf >= conf_threshold:
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            regions.append((x, y, w, h))
    return regions


def mask_regions(image: Image.Image, regions: List[Tuple[int, int, int, int]], method: str = 'rectangle') -> Image.Image:
    """
    Mask regions on the image using the specified method.
    method: 'rectangle' (white box), 'blur', or 'erase' (white fill)
    Returns a new occluded image.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in regions:
        if method == 'rectangle' or method == 'erase':
            draw.rectangle([x, y, x + w, y + h], fill='white')
        elif method == 'blur':
            region = img.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(radius=8))
            img.paste(region, (x, y))
    return img 


def generate_occlusion_flashcard_entry(occluded_path: str, original_path: str, alt_text: str = "What is hidden here?", label: str = None) -> Dict:
    """
    Return a dictionary representing an image occlusion flashcard entry.
    Optionally include label metadata.
    """
    entry = {
        "question_image_path": occluded_path,
        "answer_image_path": original_path,
        "alt_text": alt_text
    }
    if label:
        entry["label"] = label
    return entry


def save_occlusion_pair(image: Image.Image, occluded_img: Image.Image, export_dir: str, base_name: str = None) -> Tuple[str, str]:
    """
    Save the original and occluded images to disk with unique filenames.
    Returns (occluded_path, original_path).
    """
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    if base_name is None:
        base_name = str(uuid.uuid4())
    orig_path = os.path.join(export_dir, f"original_{base_name}.png")
    occ_path = os.path.join(export_dir, f"occluded_{base_name}.png")
    image.save(orig_path)
    occluded_img.save(occ_path)
    return occ_path, orig_path 


def export_occlusion_flashcards_to_csv(flashcard_entries, csv_path):
    """
    Export a list of occlusion flashcard entries to a CSV file for Anki import.
    Each entry should have 'question_image_path' and 'answer_image_path'.
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Front", "Back"])
        for entry in flashcard_entries:
            front = f"<img src='{os.path.basename(entry['question_image_path'])}'>"
            back = f"<img src='{os.path.basename(entry['answer_image_path'])}'>"
            writer.writerow([front, back])


def batch_generate_image_occlusion_flashcards(image_paths, export_dir, csv_path, conf_threshold=60, mask_method='rectangle'):
    """
    For a list of image paths, generate occlusion flashcards using OCR-based region detection.
    Saves occluded/original image pairs and exports a CSV for Anki.
    """
    flashcard_entries = []
    for img_path in image_paths:
        image = Image.open(img_path)
        regions = detect_text_regions(image, conf_threshold=conf_threshold)
        if not regions:
            continue  # Skip images with no detected regions
        for idx, region in enumerate(regions):
            occluded_img = mask_regions(image, [region], method=mask_method)
            base_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{idx}"
            occ_path, orig_path = save_occlusion_pair(image, occluded_img, export_dir, base_name=base_name)
            entry = generate_occlusion_flashcard_entry(occ_path, orig_path)
            flashcard_entries.append(entry)
    export_occlusion_flashcards_to_csv(flashcard_entries, csv_path)
    return flashcard_entries 