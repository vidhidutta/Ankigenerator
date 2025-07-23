#!/usr/bin/env python3
"""
Test script to verify the new block detection refinements:
- Parameterization from config.yaml
- Post-processing with IoU/centroid merging
- Area filtering
- Table heuristics
- Debug overlays with bright green boxes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import yaml
from utils.image_occlusion import detect_text_regions, postprocess_blocks, config

def test_block_detection():
    """Test the new block detection refinements."""
    
    # Load config to verify parameters are loaded
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    io_config = config.get('image_occlusion', {})
    print("=== Block Detection Configuration ===")
    print(f"Morphology kernel: ({io_config.get('morph_kernel_width', 25)}, {io_config.get('morph_kernel_height', 25)})")
    print(f"DBSCAN eps: {io_config.get('dbscan_eps', 50)}")
    print(f"DBSCAN min_samples: {io_config.get('dbscan_min_samples', 1)}")
    print(f"Min block area: {io_config.get('min_block_area', 200)}")
    print(f"Table merge band: {io_config.get('table_merge_band', 20)}")
    print(f"Centroid merge distance: {io_config.get('centroid_merge_dist', 30)}")
    print(f"IoU merge threshold: {io_config.get('iou_merge_thresh', 0.2)}")
    
    # Test with a sample image if available
    test_images = [
        "debug_images/debug_slide11_img1_blocks.png",
        "debug_images/debug_slide13_img1_blocks.png",
        "debug_images/debug_slide14_img1_blocks.png"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n=== Testing with {img_path} ===")
            try:
                image = Image.open(img_path)
                print(f"Image size: {image.size}")
                
                # Test block detection with new parameters
                use_blocks = io_config.get("use_blocks", True)
                regions = detect_text_regions(image, conf_threshold=50, use_blocks=use_blocks)
                print(f"Detected {len(regions)} regions")
                
                # Test post-processing
                if regions:
                    processed = postprocess_blocks(regions)
                    print(f"After post-processing: {len(processed)} regions")
                    
                    # Check for bright green boxes in debug images
                    debug_path = img_path.replace('_blocks.png', '_blocks_debug.png')
                    if os.path.exists(debug_path):
                        print(f"Debug image generated: {debug_path}")
                    else:
                        print(f"No debug image found at: {debug_path}")
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        else:
            print(f"Test image not found: {img_path}")
    
    print("\n=== Block Detection Test Complete ===")
    print("Check debug_images/ for generated debug overlays with bright green boxes")

if __name__ == "__main__":
    test_block_detection() 