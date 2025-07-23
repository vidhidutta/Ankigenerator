import os
import uuid
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pytesseract
import base64
import yaml
import re
import settings
from openai import OpenAI  # NEW: OpenAI Python SDK v1
# Ensure environment variables from a .env file are available before accessing them
from dotenv import load_dotenv
load_dotenv()

import os
_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", "")
)

# If no key, disable LLM-based selection but allow rest of pipeline to run
if not _client.api_key:
    print(
        "[WARN] OpenAI API key not set – proceeding without LLM snippet selection."
    )
    _client = None  # signal downstream to use default behaviour


# === Mask generation helpers ===

FORCE_MASK_DEBUG = True  # Set to True to force a visible mask for debugging

# Update make_qmask to use the debug color if enabled
def make_qmask(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    """Return an RGBA mask with a semi-opaque red fill over the region, or bright green if FORCE_MASK_DEBUG."""
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))  # type: ignore[arg-type]
    draw = ImageDraw.Draw(mask)
    x, y, w, h = region
    if FORCE_MASK_DEBUG:
        # Fully opaque, bright green for debug
        draw.rectangle([x, y, x + w, y + h], fill=(0, 255, 0, 255))
    else:
        # Usual red mask
        draw.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, 255))
    print(f"[DEBUG] make_qmask: region=({x}, {y}, {w}, {h}), FORCE_MASK_DEBUG={FORCE_MASK_DEBUG}")
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

# Optional: Use LLM to pick just one best region (disabled by default)
llm_region_selection: bool = config.get('llm_region_selection', False)

# Update detect_text_regions function
conf_threshold = config.get('conf_threshold', 75)
min_region_area = config.get('min_region_area', 150)
max_region_area_ratio = config.get('max_region_area_ratio', 0.2)
max_region_width_ratio = config.get('max_region_width_ratio', 0.7)
max_region_height_ratio = config.get('max_region_height_ratio', 0.7)
min_text_length = config.get('min_text_length', 2)
ignore_nonsemantic_chars = config.get('ignore_nonsemantic_chars', True)

# Percentage padding (horizontal & vertical) to expand each OCR box before masking
region_expand_pct = config.get('region_expand_pct', 0.3)

# Load optional fine-tuning parameters for more precise masking
max_masks_per_image = config.get('max_masks_per_image', 6)
# Horizontal pixel gap under which separate OCR boxes will be merged.
# A lower value prevents large multi-cell merges in tables.
merge_x_gap_tol = config.get('merge_x_gap_tol', 20)

# Load image occlusion specific configuration if available
if 'image_occlusion' in config:
    io_config = config['image_occlusion']
    region_expand_pct = io_config.get('region_expand_pct', 0.4)
    conf_threshold = io_config.get('conf_threshold', 50)
    max_masks_per_image = io_config.get('max_masks_per_image', 6)
    min_region_area = io_config.get('min_region_area', 150)
    max_region_area_ratio = io_config.get('max_region_area_ratio', 0.2)
    max_region_width_ratio = io_config.get('max_region_width_ratio', 0.7)
    max_region_height_ratio = io_config.get('max_region_height_ratio', 0.7)
    min_text_length = io_config.get('min_text_length', 4)
    ignore_nonsemantic_chars = io_config.get('ignore_nonsemantic_chars', True)
    merge_x_gap_tol = io_config.get('merge_x_gap_tol', 20)
    prefer_small_regions = io_config.get('prefer_small_regions', True)
    llm_region_selection = io_config.get('llm_region_selection', False)
    # --- Block detection refinements ---
    morph_kernel_width = io_config.get('morph_kernel_width', 25)
    morph_kernel_height = io_config.get('morph_kernel_height', 25)
    dbscan_eps = io_config.get('dbscan_eps', 50)
    dbscan_min_samples = io_config.get('dbscan_min_samples', 1)
    min_block_area = io_config.get('min_block_area', 200)
    table_merge_band = io_config.get('table_merge_band', 20)
    centroid_merge_dist = io_config.get('centroid_merge_dist', 30)
    iou_merge_thresh = io_config.get('iou_merge_thresh', 0.2)


def detect_text_regions(image: Image.Image, conf_threshold: int = 75, use_blocks: bool = True, block_kernel: tuple = (25, 25), debug_path: str = None) -> List[Tuple[int, int, int, int]]:
    io_config = config.get('image_occlusion', {})
    enable_semantic_masking = io_config.get('enable_semantic_masking', False)
    print(f"[DEBUG] Executing detect_text_regions with conf_threshold: {conf_threshold}, use_blocks={use_blocks}, enable_semantic_masking={enable_semantic_masking}")
    image = preprocess_image_for_ocr(image)

    if enable_semantic_masking:
        # --- Experimental semantic masking pipeline ---
        print("[DEBUG] Experimental semantic masking mode enabled.")
        # TODO: Implement line-level OCR extraction, semantic chunking, and region unioning here.
        # For now, just return an empty list as a placeholder.
        return []
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    ocr_boxes = []
    img_area = image.width * image.height
    max_area_ratio = 0.2
    for i in range(len(ocr_data['level'])):
        text_raw = ocr_data['text'][i]
        text = text_raw.strip() if isinstance(text_raw, str) else ""
        if not text:
            continue
        if len(re.sub(r"[^\w]", "", text.strip())) < min_text_length:
            continue
        if re.match(r"^[\s\.,\-\|_\(\)\[\]:/\\]*$", text.strip()):
            continue
        conf_str = str(ocr_data['conf'][i])
        conf = int(conf_str) if conf_str.isdigit() else -1
        if conf < conf_threshold:
            continue
        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        area = w * h
        if area < 100 or area > img_area * 0.25:
            continue
        if area < min_region_area:
            continue
        if area / img_area > max_area_ratio:
            continue
        if w > image.width * max_region_width_ratio and h > image.height * max_region_height_ratio:
            continue
        if any(abs(x - rx) < 10 and abs(y - ry) < 10 for rx, ry, rw, rh in ocr_boxes):
            continue
        try:
            if _looks_like_header(text, y, image.height):
                continue
        except NameError:
            pass
        ocr_boxes.append((x, y, w, h))
        print(f"[DEBUG] Processed OCR region: '{text}' | Conf: {conf} | Box: ({x}, {y}, {w}, {h})")
    # Debug log
    print(f"[DEBUG] OCR words: {len(ocr_boxes)}")
    if use_blocks:
        # --- Block-level detection ---
        blocks = detect_text_blocks_cv(
            image,
            kernel_size=(morph_kernel_width, morph_kernel_height),
            min_area=min_region_area,
            use_closing=True,
            debug_path=debug_path
        )
        print(f"[DEBUG] Morphology blocks: {len(blocks)}")
        merged_blocks = merge_ocr_boxes_into_blocks(ocr_boxes, blocks)
        print(f"[DEBUG] Merged into {len(merged_blocks)} block-level regions.")
        if len(merged_blocks) == 0:
            # Fallback: cluster OCR boxes
            print("[DEBUG] Morphology produced zero blocks, falling back to OCR clustering.")
            clusters = cluster_ocr_boxes(
                ocr_boxes,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                debug_path=debug_path,
                image=image
            )
            print(f"[DEBUG] OCR clusters: {len(clusters)}")
            print("[DEBUG] Fallback used: OCR clustering")
            # Post-process clusters
            post_clusters = postprocess_blocks(clusters)
            print(f"[DEBUG] Post-processed clusters: {len(post_clusters)}")
            return post_clusters
        else:
            print("[DEBUG] Fallback not needed: morphology blocks used")
            # Post-process merged blocks
            post_blocks = postprocess_blocks(merged_blocks)
            print(f"[DEBUG] Post-processed blocks: {len(post_blocks)}")
            return post_blocks
    else:
        return ocr_boxes


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
        io_config = config.get("image_occlusion", {})
        use_blocks = io_config.get("use_blocks", True)
        regions = detect_text_regions(image, conf_threshold=conf_threshold, use_blocks=use_blocks)
        print(f"[DEBUG] Regions for {img_path}: {regions}")
        # Always save a debug overlay image, even if regions is empty
        debug_img_path = os.path.join(export_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_debug_detected_regions.png")
        draw_debug_boxes(image, regions, debug_img_path)
        if not regions:
            print(f"[DEBUG] No regions detected for image: {img_path}")
            continue
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

                # Debug: print region info before masking
                print(f"[DEBUG] Masking region {idx}: {region}")
                # Generate the question and original masks
                qmask_layer = make_qmask(image, region)
                omask_layer = make_omask(image, region)
                print(f"[DEBUG] Mask created for region {idx}")

                # Composite masks onto the original image for clearer visuals
                q_overlay = Image.alpha_composite(image.convert('RGBA'), qmask_layer)
                o_overlay = Image.alpha_composite(image.convert('RGBA'), omask_layer)
                print(f"[DEBUG] Overlay composited for region {idx}")

                # Paths for saved images
                occ_path = os.path.join(export_dir, f"{base_name}_q.png")
                om_path  = os.path.join(export_dir, f"{base_name}_o.png")

                # Save images to disk
                q_overlay.save(occ_path)
                o_overlay.save(om_path)
                print(f"[DEBUG] Saved masked images: {occ_path}, {om_path}")

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

# --- Helper: detect likely header text vs data cell ---
def _looks_like_header(txt: str, y: int, img_h: int) -> bool:
    """Return True for obvious headers (ALL-CAPS at top row, short & alpha, etc.)."""
    txt_stripped = txt.strip()

    # NEW RULE: treat any ALL-CAPS string of ≤3 words as a header (regardless of position)
    if txt_stripped.isupper() and len(txt_stripped.split()) <= 3:
        return True

    all_caps     = txt_stripped.isupper() and len(txt_stripped) <= 3 or txt_stripped in {"NORMAL","ABNORMAL"}
    top_band     = y < img_h * 0.25       # very top quarter of slide
    very_short   = len(txt_stripped.split()) <= 2
    return (all_caps and top_band) or (very_short and top_band)


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

    # --- NEW: filter out likely headers ---
    content_regions = []
    content_texts   = []
    for (txt, (x, y, w, h)) in zip(texts, regions):
        if not _looks_like_header(txt, y, H):
            content_regions.append((x, y, w, h))
            content_texts.append(txt)

    # Fall back to all regions if everything was filtered out
    if content_regions:
        regions, texts = content_regions, content_texts
    else:
        # No data cells survived the header filter – keep the full list
        pass

    # Debug: show texts that will be provided to LLM for selection
    print("[DEBUG] Texts sent to LLM:", texts)

    lines = []
    for idx, (txt, (x, y, w, h)) in enumerate(zip(texts, regions)):
        h_band = "top" if y < H * 0.33 else "middle" if y < H * 0.66 else "bottom"
        v_band = "left" if x < W * 0.33 else "centre" if x < W * 0.66 else "right"
        # Escape newlines/quotes in OCR text to keep prompt compact
        snippet_txt = txt.replace("\n", " ").strip()
        lines.append(f"{idx}) \"{snippet_txt}\" at {h_band}-{v_band}")

    prompt = (
        "You are a medical educator creating an Anki flashcard.\n"
        "Here are the merged & padded slide snippets:\n"
        + "\n".join(lines) + "\n"
        "Which ONE snippet, if hidden, would best test a student's understanding?\n"
        "Reply with ONLY the integer index."
    )

    # If no client/key, default to first region
    if _client is None:
        return 0

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=3,
    )
    txt = resp.choices[0].message.content
    import re as _re
    idx_match = _re.search(r"\d+", txt)
    if not idx_match:
        # Ask once more with a stricter system message
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Reply ONLY with a single integer 0-n that appears in the list."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=3,
        )
        txt = resp.choices[0].message.content
        idx_match = _re.search(r"\d+", txt)

    if not idx_match:
        raise ValueError(f"LLM response had no integer index: {txt}")

    best_i = int(idx_match.group())
    best_i = max(0, min(best_i, len(regions) - 1))  # clamp to valid range

    # Detailed debugging information
    print("[DEBUG] GPT raw reply:", txt)
    print("[DEBUG] Candidate list AFTER filter:", texts)
    print("[DEBUG] Index chosen:", best_i)
    print("[DEBUG] LLM chose:", texts[best_i] if texts else "<none>")

    # previous concise debug kept for compatibility (optional)
    # print(f"[DEBUG] LLM chose index {best_i} → {texts[best_i]}")

    return best_i

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import math

def detect_text_blocks_cv(image: Image.Image, kernel_size: tuple = (25, 25), min_area: int = 100, use_closing: bool = True, debug_path: str = None) -> list:
    """
    Detect visually connected text blocks using OpenCV morphology.
    Args:
        image: PIL Image
        kernel_size: (width, height) of the dilation/closing kernel
        min_area: minimum area for a block
        use_closing: whether to use closing (dilate then erode)
        debug_path: if provided, saves a debug image with block rectangles
    Returns:
        List of (x, y, w, h) bounding boxes for detected blocks
    """
    img = np.array(image.convert('L'))
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if use_closing:
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        morphed = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            blocks.append((x, y, w, h))
    # Save debug image
    if debug_path:
        import os
        debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.basename(debug_path)
        debug_save_path = os.path.join(debug_dir, debug_filename)
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in blocks:
            color = (0, 255, 0) if FORCE_MASK_DEBUG else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
        cv2.imwrite(debug_save_path, debug_img)
        print(f"[DEBUG] Block debug image saved to: {debug_save_path}")
    return blocks

def cluster_ocr_boxes(ocr_boxes, eps: int = 50, min_samples: int = 1, debug_path: str = None, image=None):
    """
    Cluster OCR boxes by proximity using DBSCAN, then merge each cluster into a block.
    """
    if not ocr_boxes:
        return []
    centers = np.array([[x + w/2, y + h/2] for x, y, w, h in ocr_boxes])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    clusters = {}
    for label, box in zip(labels, ocr_boxes):
        clusters.setdefault(label, []).append(box)
    merged = []
    for boxes in clusters.values():
        xs = [b[0] for b in boxes]
        ys = [b[1] for b in boxes]
        ws = [b[0]+b[2] for b in boxes]
        hs = [b[1]+b[3] for b in boxes]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(ws), max(hs)
        merged.append((x0, y0, x1-x0, y1-y0))
    # Debug overlay
    if debug_path and image is not None:
        import os
        debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.basename(debug_path).replace('.png', '_cluster.png')
        debug_save_path = os.path.join(debug_dir, debug_filename)
        img = np.array(image.convert('RGB'))
        for (x, y, w, h) in merged:
            color = (0, 255, 0) if FORCE_MASK_DEBUG else (255, 0, 0)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.imwrite(debug_save_path, img)
        print(f"[DEBUG] Cluster debug image saved to: {debug_save_path}")
    return merged

def merge_ocr_boxes_into_blocks(ocr_boxes, blocks):
    """
    Assign each OCR box to the block it overlaps most, then merge all boxes per block.
    Args:
        ocr_boxes: list of (x, y, w, h)
        blocks: list of (x, y, w, h)
    Returns:
        List of merged (x, y, w, h) per block
    """
    from collections import defaultdict
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    def centroid(box):
        return (box[0] + box[2]/2, box[1] + box[3]/2)
    def area(box):
        return box[2] * box[3]
    block_to_boxes = defaultdict(list)
    for ocr in ocr_boxes:
        best_block = None
        best_iou = 0
        for block in blocks:
            score = iou(ocr, block)
            if score > best_iou:
                best_iou = score
                best_block = block
        if best_block and best_iou > 0.1:
            block_to_boxes[best_block].append(ocr)
    merged = []
    for block, boxes in block_to_boxes.items():
        xs = [b[0] for b in boxes]
        ys = [b[1] for b in boxes]
        ws = [b[0]+b[2] for b in boxes]
        hs = [b[1]+b[3] for b in boxes]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(ws), max(hs)
        merged.append((x0, y0, x1-x0, y1-y0))
    return merged

def postprocess_blocks(blocks):
    """
    Post-process block list:
    - Merge any two boxes whose IoU > iou_merge_thresh or centroids within centroid_merge_dist
    - Drop any block with area < min_block_area
    - Only merge boxes in same row/col if their centers align within table_merge_band
    """
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    def centroid(box):
        return (box[0] + box[2]/2, box[1] + box[3]/2)
    def area(box):
        return box[2] * box[3]
    merged = blocks[:]
    changed = True
    while changed:
        changed = False
        n = len(merged)
        for i in range(n):
            for j in range(i+1, n):
                b1, b2 = merged[i], merged[j]
                c1, c2 = centroid(b1), centroid(b2)
                # Table heuristics: only merge if centers align in row/col within band
                row_aligned = abs(c1[1] - c2[1]) < table_merge_band
                col_aligned = abs(c1[0] - c2[0]) < table_merge_band
                iou_val = iou(b1, b2)
                cent_dist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
                if (iou_val > iou_merge_thresh or cent_dist < centroid_merge_dist) and (row_aligned or col_aligned):
                    # Merge
                    x0 = min(b1[0], b2[0])
                    y0 = min(b1[1], b2[1])
                    x1 = max(b1[0]+b1[2], b2[0]+b2[2])
                    y1 = max(b1[1]+b1[3], b2[1]+b2[3])
                    new_box = (x0, y0, x1-x0, y1-y0)
                    merged = [merged[k] for k in range(n) if k != i and k != j] + [new_box]
                    changed = True
                    print(f"[DEBUG] Merged blocks {i},{j} (IoU={iou_val:.2f}, dist={cent_dist:.1f}, row={row_aligned}, col={col_aligned})")
                    break
            if changed:
                break
    # Filter by area
    filtered = [b for b in merged if area(b) >= min_block_area]
    dropped = len(merged) - len(filtered)
    if dropped > 0:
        print(f"[DEBUG] Dropped {dropped} blocks with area < {min_block_area}")
    print(f"[DEBUG] Final block count after post-processing: {len(filtered)}")
    return filtered
