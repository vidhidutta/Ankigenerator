#!/usr/bin/env python3
"""
Test script for the Adaptive Configuration System
Demonstrates how AI automatically adjusts image occlusion parameters
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tempfile

def create_test_images():
    """Create test images with different characteristics"""
    
    # Test image 1: High quality, dense text (biochemical pathway)
    img1 = Image.new('RGB', (800, 600), 'white')
    draw1 = ImageDraw.Draw(img1)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Dense biochemical text
    dense_text = [
        "Glucose 6-phosphate dehydrogenase",
        "Fructose 1,6-diphosphate",
        "Hexose monophosphate shunt",
        "NADP → NADPH",
        "GSSG → G-SH",
        "ATP → ADP",
        "Glucose → Glucose 6-phosphate",
        "Fructose 6-phosphate → Fructose 1,6-diphosphate"
    ]
    
    y_pos = 50
    for text in dense_text:
        draw1.text((50, y_pos), text, fill='black', font=font)
        y_pos += 40
    
    img1.save("test_dense_text.png")
    print("✅ Created dense text test image: test_dense_text.png")
    
    # Test image 2: Low quality, blurry
    img2 = Image.new('RGB', (600, 400), 'lightgray')
    draw2 = ImageDraw.Draw(img2)
    
    # Add some blurry text
    draw2.text((100, 100), "Blurry Text", fill='darkgray', font=font)
    draw2.text((100, 150), "Poor Quality", fill='darkgray', font=font)
    
    # Apply blur effect
    img2 = img2.filter(ImageFilter.GaussianBlur(radius=2))
    img2.save("test_blurry.png")
    print("✅ Created blurry test image: test_blurry.png")
    
    # Test image 3: Wide diagram
    img3 = Image.new('RGB', (1000, 300), 'white')
    draw3 = ImageDraw.Draw(img3)
    
    # Wide horizontal layout
    wide_text = ["Start", "→", "Process A", "→", "Process B", "→", "End"]
    x_pos = 50
    for text in wide_text:
        draw3.text((x_pos, 150), text, fill='blue', font=font)
        x_pos += 120
    
    img3.save("test_wide_diagram.png")
    print("✅ Created wide diagram test image: test_wide_diagram.png")
    
    return ["test_dense_text.png", "test_blurry.png", "test_wide_diagram.png"]

def test_adaptive_configuration():
    """Test the adaptive configuration system"""
    
    try:
        from providers.adaptive_config_provider import create_adaptive_config_provider
        
        # Create provider
        adaptive_provider = create_adaptive_config_provider()
        
        # Create test images
        test_images = create_test_images()
        
        # Base configuration
        base_config = {
            'conf_threshold': 75,
            'use_blocks': True,
            'min_region_area': 150,
            'max_masks_per_image': 6,
            'merge_x_gap_tol': 20,
            'dbscan_eps': 50,
            'region_expand_pct': 0.4,
            'morph_kernel_width': 25,
            'morph_kernel_height': 25
        }
        
        print("\n🧪 Testing Adaptive Configuration System:")
        print("=" * 60)
        
        for image_path in test_images:
            print(f"\n📸 Analyzing: {image_path}")
            print("-" * 40)
            
            # Get adaptive recommendations
            recommendations = adaptive_provider.get_adaptive_recommendations(image_path)
            
            if 'error' in recommendations:
                print(f"❌ Error: {recommendations['error']}")
                continue
            
            print(f"Image Type: {recommendations['image_type']}")
            print(f"Quality Score: {recommendations['quality_score']:.2f}")
            print(f"Suggested Confidence: {recommendations['suggested_confidence']}%")
            print(f"Suggested Masks: {recommendations['suggested_masks']}")
            print(f"Strategy: {recommendations['processing_strategy']}")
            
            if recommendations['quality_notes']:
                print("Quality Notes:")
                for note in recommendations['quality_notes']:
                    print(f"  • {note}")
            
            # Get optimized configuration
            optimized_config = adaptive_provider.analyze_image_and_optimize_config(image_path, base_config)
            
            print(f"\n🔧 Optimized Configuration:")
            print(f"  Confidence Threshold: {optimized_config.get('conf_threshold', 'N/A')}")
            print(f"  Use Blocks: {optimized_config.get('use_blocks', 'N/A')}")
            print(f"  Min Region Area: {optimized_config.get('min_region_area', 'N/A')}")
            print(f"  Max Masks: {optimized_config.get('max_masks_per_image', 'N/A')}")
            print(f"  Merge Gap Tolerance: {optimized_config.get('merge_x_gap_tol', 'N/A')}")
            print(f"  Region Expansion: {optimized_config.get('region_expand_pct', 'N/A')}")
        
        print("\n🎯 Summary of Adaptive AI Behavior:")
        print("=" * 60)
        print("• Dense Text Images: Higher confidence, more masks, aggressive grouping")
        print("• Poor Quality Images: Lower confidence, expanded regions, conservative approach")
        print("• Wide Diagrams: Increased gap tolerance, horizontal merging")
        print("• Standard Medical Images: Balanced parameters for optimal results")
        
        # Clean up test images
        for image_path in test_images:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 Cleaned up: {image_path}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install required packages: pip install opencv-python")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_adaptive_configuration()
