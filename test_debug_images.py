import os
# 1. Monkey-patch detection to always return one region
import utils.image_occlusion as io_utils
io_utils.detect_text_regions = lambda image, conf_threshold: [(50, 50, 100, 100)]

# 2. Import the batch function
from utils.image_occlusion import batch_generate_image_occlusion_flashcards

# 3. Prepare debug folder
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# 4. Create a dummy white image if needed
from PIL import Image
dummy_path = os.path.join(DEBUG_DIR, "test.png")
if not os.path.exists(dummy_path):
    Image.new("RGB", (200, 200), (255,255,255)).save(dummy_path)

# 5. Run the batch to generate masks
entries = batch_generate_image_occlusion_flashcards(
    image_paths=[dummy_path],
    export_dir=DEBUG_DIR,
    conf_threshold=0,
    max_masks=1,
    mask_method="rectangle",
)
print("Generated entries:", entries)

# At this point you can `ls debug_images` and expect:
# test.png, test_0_q.png, test_0_o.png 