import os
import uuid
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pytesseract
import base64
import yaml
import re
import settings
import openai  # NEW: OpenAI Python SDK for LLM selection
# Ensure API key is set (env var or settings fallback)
if not openai.api_key:
    openai.api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", "")


# === Mask generation helpers ===

def make_qmask(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    """Return an RGBA mask with a semi-opaque red fill over the region."""
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))  # type: ignore[arg-type]
    draw = ImageDraw.Draw(mask)
    x, y, w, h = region
    # Increase opacity for better visibility (alpha 200 out of 255)
    # Use a fully opaque red rectangle so the mask is unmistakable
    draw.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, 255))
    return mask


def make_omask(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    """Return an RGBA mask with a solid red outline around the region."""
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))  # type: ignore[arg-type]
    draw = ImageDraw.Draw(mask)
    x, y, w, h = region
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0, 255), width=3)
    return mask

# -----------------------------
# File-cleanup helper
# -----------------------------

from typing import List


def cleanup_files(file_paths: List[str]):
    """Delete each file in the list if it exists (best-effort)."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"[DEBUG] Cleaned up temporary file: {path}")
        except Exception as e:
            # Log but do not interrupt further cleanup
            print(f"[DEBUG] Could not delete {path}: {e}")

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Update detect_text_regions function
conf_threshold = config.get('conf_threshold', 75)
min_region_area = config.get('min_region_area', 150)
max_region_area_ratio = config.get('max_region_area_ratio', 0.2)
max_region_width_ratio = config.get('max_region_width_ratio', 0.7)
max_region_height_ratio = config.get('max_region_height_ratio', 0.7)
min_text_length = config.get('min_text_length', 2)
ignore_nonsemantic_chars = config.get('ignore_nonsemantic_chars', True)

# Load max_masks_per_image from config
max_masks_per_image = config.get('max_masks_per_image', 6)


def detect_text_regions(image: Image.Image, conf_threshold: int = 75) -> List[Tuple[int, int, int, int]]:
    print(f"[DEBUG] Executing detect_text_regions with conf_threshold: {conf_threshold}")
    image = preprocess_image_for_ocr(image)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    regions = []
    img_area = image.width * image.height
    max_area_ratio = 0.2

    for i in range(len(ocr_data['level'])):
        text_raw = ocr_data['text'][i]
        text = text_raw.strip() if isinstance(text_raw, str) else ""
        if not text:
            continue

        # Minimum character length filter (now uses configurable min_text_length)
        if len(re.sub(r"[^\w]", "", text.strip())) < min_text_length:
            continue

        # Improved symbol/noise filter
        if re.match(r"^[\s\.\,\-\|\_\(\)\[\]\:\/\\]*$", text.strip()):
            continue

        conf_str = str(ocr_data['conf'][i])
        conf = int(conf_str) if conf_str.isdigit() else -1
        if conf < conf_threshold:
            continue

        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        area = w * h

        # Skip overly tiny or huge regions
        if area < 100 or area > img_area * 0.25:
            continue

        if area < min_region_area:
            continue
        if area / img_area > max_area_ratio:
            continue
        if w > image.width * max_region_width_ratio and h > image.height * max_region_height_ratio:
            continue

        if any(abs(x - rx) < 10 and abs(y - ry) < 10 for rx, ry, rw, rh in regions):
            continue

        regions.append((x, y, w, h))

        # Add more comprehensive debugging information
        print(f"[DEBUG] Processed region: '{text}' | Conf: {conf} | Box: ({x}, {y}, {w}, {h})")

    # Sort regions by vertical position, then left-to-right
    regions.sort(key=lambda box: (box[1], box[0]))

    # Add padding to each region
    padding = 5
    padded_regions = []
    for x, y, w, h in regions:
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.width)
        y2 = min(y + h + padding, image.height)
        # Convert back to width/height representation for downstream code
        padded_regions.append((x1, y1, x2 - x1, y2 - y1))

    # Filter overlapping boxes
    def boxes_overlap(box1, box2, threshold=0.6):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        box1_area = w1 * h1
        box2_area = w2 * h2
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou > threshold

    filtered_regions = []
    for box in padded_regions:
        if not any(boxes_overlap(box, other_box) for other_box in filtered_regions):
            filtered_regions.append(box)

    # Sort regions by area (descending) and keep up to max_masks_per_image (configurable)
    filtered_regions.sort(key=lambda box: box[2] * box[3], reverse=True)
    filtered_regions = filtered_regions[:max_masks_per_image]

    # =========================
    # Expand bounding boxes so they fully cover each word/phrase
    # Pad each box by 20% horizontally and 20% vertically
    # =========================
    padded = []
    for x, y, w, h in filtered_regions:
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        nx = max(0, x - pad_x)
        ny = max(0, y - pad_y)
        nw = min(image.width - nx, w + 2 * pad_x)
        nh = min(image.height - ny, h + 2 * pad_y)
        padded.append((nx, ny, nw, nh))
    filtered_regions = padded

    # Merge adjacent boxes on the same line (e.g., "SPIROMETRY" + "TEST")
    filtered_regions = merge_adjacent_boxes(filtered_regions, y_tol=10, x_gap_tol=20)

    # --------------------------------------------------------
    # NEW: Ask LLM to choose the single best region to mask
    # --------------------------------------------------------
    try:
        # Extract snippet text for each region via a lightweight OCR pass
        texts: list[str] = []
        for (x, y, w, h) in filtered_regions:
            roi = image.crop((x, y, x + w, y + h))
            raw_txt = pytesseract.image_to_string(roi, config='--psm 6')
            snippet = raw_txt.strip() or raw_txt  # fallback to raw text if stripping empties it
            texts.append(snippet if snippet else "<blank>")

        if filtered_regions and texts:
            best_i = pick_best_snippet_via_llm(filtered_regions, texts, image.size)
            # Clamp index to valid range just in case
            best_i = max(0, min(best_i, len(filtered_regions) - 1))
            print(f"[DEBUG] LLM selected snippet: {texts[best_i]}")
            filtered_regions = [filtered_regions[best_i]]
    except Exception as e:
        print(f"[DEBUG] LLM selection failed: {e}")

    # Final debug statement
    print(f"[DEBUG] Regions after LLM selection: {len(filtered_regions)}")

    return filtered_regions


def mask_regions(image: Image.Image, regions: List[Tuple[int, int, int, int]], method: str = 'rectangle') -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in regions:
        print(f"[DEBUG] Masking region at ({x}, {y}, {w}, {h})")  # Debugging
        if method in ['rectangle', 'erase']:
            draw.rectangle([x, y, x + w, y + h], fill='white')
        elif method == 'blur':
            region = img.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(radius=8))
            img.paste(region, (x, y))
    return img


def generate_occlusion_flashcard_entry(occluded_path: str, original_path: str, alt_text: str = "What is hidden here?", label: str = "") -> Dict:
    occluded_base64 = encode_image_to_base64(occluded_path)
    original_base64 = encode_image_to_base64(original_path)
    entry = {
        "question_image_base64": occluded_base64,
        "answer_image_base64": original_base64,
        "alt_text": alt_text
    }
    if label:
        entry["label"] = label
    return entry


def save_occlusion_pair(image: Image.Image, occluded_img: Image.Image, base_name: str = "") -> Tuple[str, str]:
    export_dir = os.path.expanduser('~/anki-flashcard-generator/debug_images')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    if not base_name:
        base_name = str(uuid.uuid4())
    orig_path = os.path.join(export_dir, f"original_{base_name}.png")
    occ_path = os.path.join(export_dir, f"occluded_{base_name}.png")
    image.save(orig_path)
    occluded_img.save(occ_path)
    print(f"[DEBUG] Saved original image to: {orig_path}")
    print(f"[DEBUG] Saved occluded image to: {occ_path}")
    return occ_path, orig_path


def draw_debug_boxes(image: Image.Image, regions: List[Tuple[int, int, int, int]], output_path: str):
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    for x, y, w, h in regions:
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
    debug_img_dir = os.path.dirname(output_path)
    if not os.path.exists(debug_img_dir):
        os.makedirs(debug_img_dir, exist_ok=True)
    debug_img.save(output_path)
    print(f"Saving debug image to: {output_path}")



def batch_generate_image_occlusion_flashcards(image_paths, export_dir, conf_threshold=60, max_masks=10, mask_method='rectangle'):
    # Ensure the export directory exists before writing any files
    os.makedirs(export_dir, exist_ok=True)
    flashcard_entries = []
    for img_path in image_paths:
        print(f"[DEBUG] Processing image: {img_path}")
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Use higher confidence threshold (default now 50 via config)
        regions = detect_text_regions(image, conf_threshold=conf_threshold)
        if not regions:
            print(f"[DEBUG] No regions detected for image: {img_path}")
            continue
        # No need for generic 10-mask warning; regions already capped to top 2
        print(f"[DEBUG] Using {len(regions)} masks for image: {img_path}")
        for idx, region in enumerate(regions):
            area = region[2] * region[3]
            img_area = image.width * image.height
            if area / img_area < 0.005 or area / img_area > 0.3:
                continue
            files_to_cleanup = []
            try:
                # Create masks and save question/answer images
                base_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{idx}"

                alt_text_for_this_image = "What is hidden here?"

                # Generate the question and original masks
                qmask_layer = make_qmask(image, region)
                omask_layer = make_omask(image, region)

                # Composite masks onto the original image for clearer visuals
                q_overlay = Image.alpha_composite(image.convert('RGBA'), qmask_layer)
                o_overlay = Image.alpha_composite(image.convert('RGBA'), omask_layer)

                # Paths for saved images
                occ_path = os.path.join(export_dir, f"{base_name}_q.png")
                om_path  = os.path.join(export_dir, f"{base_name}_o.png")

                # Save images to disk
                q_overlay.save(occ_path)
                o_overlay.save(om_path)

                # Track for potential cleanup
                files_to_cleanup.extend([occ_path, om_path])

                # Append flashcard entry (paths, not Base64)
                flashcard_entries.append({
                    "question_image_path": occ_path,
                    "answer_image_path":   om_path,
                    "alt_text":            alt_text_for_this_image,
                })
 
                # Debug image with boxes
                debug_img_path = os.path.join(export_dir, f"{base_name}_debug.png")
                draw_debug_boxes(image, [region], debug_img_path)
                files_to_cleanup.append(debug_img_path)
                print(f"[DEBUG] Saved debug image to: {debug_img_path}")

            except Exception as e:
                cleanup_files(files_to_cleanup)
                raise
    return flashcard_entries


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(2.0)
    width, height = image.size
    image = image.resize((int(width * 1.5), int(height * 1.5)), Image.LANCZOS)
    image = image.convert('L')
    return image


# Helper to merge adjacent text boxes on same horizontal band
def merge_adjacent_boxes(boxes, y_tol=10, x_gap_tol=20):
    """Merge neighbouring boxes that likely belong to the same text line.

    Args:
        boxes: list of (x, y, w, h) tuples.
        y_tol: vertical tolerance (pixels) to consider boxes on same line.
        x_gap_tol: maximum horizontal gap between two boxes to merge.

    Returns:
        List of merged boxes in (x, y, w, h) format.
    """
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        x, y, w, h = box
        placed = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            # Same horizontal band?
            if abs(y - my) < y_tol:
                # Gap between right edge of merged box and left edge of current
                if 0 <= x - (mx + mw) < x_gap_tol or 0 <= (mx) - (x + w) < x_gap_tol:
                    nx = min(mx, x)
                    ny = min(my, y)
                    nx2 = max(mx + mw, x + w)
                    ny2 = max(my + mh, y + h)
                    merged[i] = (nx, ny, nx2 - nx, ny2 - ny)
                    placed = True
                    break
        if not placed:
            merged.append(box)
    return merged


export_dir = os.path.expanduser('~/anki-flashcard-generator/debug_images')


def pick_best_snippet_via_llm(regions, texts, img_size):
    """Use an LLM (GPT-4o-mini) to choose which snippet to mask.

    Args:
        regions (List[Tuple[int,int,int,int]]): bounding boxes for candidate snippets.
        texts   (List[str]): OCR-extracted text corresponding to each region.
        img_size (Tuple[int,int]): (width, height) of the source image.

    Returns:
        int: index of the region deemed most pedagogically valuable to hide.
    """
    H, W = img_size[1], img_size[0]
    lines = []
    for idx, (txt, (x, y, w, h)) in enumerate(zip(texts, regions)):
        h_band = "top" if y < H * 0.33 else "middle" if y < H * 0.66 else "bottom"
        v_band = "left" if x < W * 0.33 else "centre" if x < W * 0.66 else "right"
        # Escape newlines/quotes in OCR text to keep prompt compact
        snippet_txt = txt.replace("\n", " ").strip()
        lines.append(f"{idx}) \"{snippet_txt}\" at {h_band}-{v_band}")

    prompt = (
        "You are a medical educator creating an Anki flashcard.\n"
        "Slide snippets:\n" + "\n".join(lines) + "\n"
        "Which ONE snippet, if hidden, would best test a student's understanding?\n"
        "Reply with ONLY the integer index."
    )

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=3,
    )
    import re as _re
    idx_match = _re.search(r"\d+", resp.choices[0].message.content)
    if not idx_match:
        raise ValueError("LLM response did not contain an index")
    return int(idx_match.group())
