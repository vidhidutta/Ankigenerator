#!/usr/bin/env python3
"""
Adaptive Configuration Provider
Automatically adjusts image occlusion parameters based on AI analysis of each image
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Tuple, List, Optional
import logging

class AdaptiveConfigProvider:
    """Automatically adjusts image occlusion parameters based on image analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_image_and_optimize_config(self, image_path: str, base_config: Dict) -> Dict:
        """
        Analyze an image and return optimized configuration parameters
        
        Args:
            image_path: Path to the image to analyze
            base_config: Base configuration to optimize
            
        Returns:
            Optimized configuration dictionary
        """
        try:
            # Load and analyze the image
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Analyze image characteristics
            analysis = self._analyze_image_characteristics(image_array)
            
            # Generate optimized configuration
            optimized_config = self._generate_optimized_config(analysis, base_config)
            
            self.logger.info(f"Adaptive config for {os.path.basename(image_path)}: {optimized_config}")
            return optimized_config
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze image {image_path}: {e}, using base config")
            return base_config
    
    def _analyze_image_characteristics(self, image_array: np.ndarray) -> Dict:
        """Analyze various characteristics of the image"""
        
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Image quality metrics
        quality_metrics = {
            'contrast': self._calculate_contrast(gray),
            'brightness': self._calculate_brightness(gray),
            'sharpness': self._calculate_sharpness(gray),
            'noise_level': self._calculate_noise_level(gray),
            'text_density': self._estimate_text_density(gray),
            'image_size': gray.shape,
            'aspect_ratio': gray.shape[1] / gray.shape[0]
        }
        
        # Determine image type
        image_type = self._classify_image_type(quality_metrics)
        quality_metrics['image_type'] = image_type
        
        # Calculate quality score
        quality_metrics['quality_score'] = self._calculate_quality_score(quality_metrics)
        
        return quality_metrics
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate image contrast using standard deviation"""
        return float(np.std(gray))
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate average brightness"""
        return float(np.mean(gray))
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using high-frequency components"""
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        return float(np.std(high_freq))
    
    def _estimate_text_density(self, gray: np.ndarray) -> float:
        """Estimate text density using edge detection"""
        edges = cv2.Canny(gray, 50, 150)
        text_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return float(text_ratio)
    
    def _classify_image_type(self, metrics: Dict) -> str:
        """Classify image type based on characteristics"""
        
        contrast = metrics['contrast']
        sharpness = metrics['sharpness']
        text_density = metrics['text_density']
        aspect_ratio = metrics['aspect_ratio']
        
        # High contrast, high sharpness, high text density = dense text image
        if contrast > 60 and sharpness > 500 and text_density > 0.1:
            return 'dense_text'
        
        # High contrast, medium sharpness, medium text density = standard medical image
        elif contrast > 40 and sharpness > 200 and text_density > 0.05:
            return 'standard_medical'
        
        # Low contrast, low sharpness = poor quality image
        elif contrast < 30 or sharpness < 100:
            return 'poor_quality'
        
        # Wide aspect ratio = likely a diagram or chart
        elif aspect_ratio > 1.5:
            return 'wide_diagram'
        
        # Tall aspect ratio = likely a list or flow chart
        elif aspect_ratio < 0.7:
            return 'tall_list'
        
        else:
            return 'general'
    
    def _generate_optimized_config(self, analysis: Dict, base_config: Dict) -> Dict:
        """Generate optimized configuration based on image analysis"""
        
        image_type = analysis['image_type']
        contrast = analysis['contrast']
        sharpness = analysis['sharpness']
        text_density = analysis['text_density']
        
        # Start with base config
        optimized = base_config.copy()
        
        # Adjust confidence threshold based on image quality
        if image_type == 'poor_quality':
            optimized['conf_threshold'] = max(20, base_config.get('conf_threshold', 75) - 30)
        elif image_type == 'dense_text':
            optimized['conf_threshold'] = min(90, base_config.get('conf_threshold', 75) + 15)
        elif image_type == 'standard_medical':
            optimized['conf_threshold'] = base_config.get('conf_threshold', 75)
        
        # Adjust region size parameters based on text density
        if text_density > 0.15:  # Very dense text
            optimized['min_region_area'] = max(20, base_config.get('min_region_area', 150) // 2)
            optimized['max_masks_per_image'] = min(10, base_config.get('max_masks_per_image', 6) + 2)
        elif text_density < 0.03:  # Sparse text
            optimized['min_region_area'] = base_config.get('min_region_area', 150) * 2
            optimized['max_masks_per_image'] = max(3, base_config.get('max_masks_per_image', 6) - 2)
        
        # Adjust merge parameters based on image type
        if image_type == 'dense_text':
            optimized['merge_x_gap_tol'] = max(10, base_config.get('merge_x_gap_tol', 20) // 2)
            optimized['dbscan_eps'] = max(25, base_config.get('dbscan_eps', 50) // 2)
        elif image_type == 'wide_diagram':
            optimized['merge_x_gap_tol'] = base_config.get('merge_x_gap_tol', 20) * 2
            optimized['dbscan_eps'] = base_config.get('dbscan_eps', 50) * 1.5
        
        # Adjust block detection based on image quality
        if sharpness > 400:  # Very sharp image
            optimized['use_blocks'] = True
            optimized['morph_kernel_width'] = 15
            optimized['morph_kernel_height'] = 15
        elif sharpness < 150:  # Blurry image
            optimized['use_blocks'] = False
            optimized['morph_kernel_width'] = 35
            optimized['morph_kernel_height'] = 35
        
        # Adjust region expansion based on contrast
        if contrast < 40:  # Low contrast
            optimized['region_expand_pct'] = min(0.6, base_config.get('region_expand_pct', 0.4) * 1.5)
        elif contrast > 80:  # High contrast
            optimized['region_expand_pct'] = max(0.2, base_config.get('region_expand_pct', 0.4) * 0.8)
        
        # Add adaptive flags
        optimized['adaptive_config'] = True
        optimized['image_type'] = image_type
        optimized['quality_score'] = self._calculate_quality_score(analysis)
        
        return optimized
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """Calculate overall image quality score (0.0-1.0)"""
        
        # Normalize metrics to 0-1 range
        contrast_score = min(1.0, analysis['contrast'] / 100.0)
        sharpness_score = min(1.0, analysis['sharpness'] / 1000.0)
        text_density_score = min(1.0, analysis['text_density'] / 0.2)
        
        # Weighted combination
        quality_score = (
            contrast_score * 0.4 +
            sharpness_score * 0.4 +
            text_density_score * 0.2
        )
        
        return round(quality_score, 2)
    
    def get_adaptive_recommendations(self, image_path: str) -> Dict:
        """Get adaptive recommendations for an image"""
        
        try:
            image = Image.open(image_path)
            image_array = np.array(image)
            analysis = self._analyze_image_characteristics(image_array)
            
            recommendations = {
                'image_type': analysis['image_type'],
                'quality_score': analysis['quality_score'],
                'suggested_confidence': self._suggest_confidence_threshold(analysis),
                'suggested_masks': self._suggest_mask_count(analysis),
                'processing_strategy': self._suggest_processing_strategy(analysis),
                'quality_notes': self._generate_quality_notes(analysis)
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return {'error': str(e)}
    
    def _suggest_confidence_threshold(self, analysis: Dict) -> int:
        """Suggest optimal confidence threshold"""
        if analysis['image_type'] == 'poor_quality':
            return 25
        elif analysis['image_type'] == 'dense_text':
            return 85
        elif analysis['image_type'] == 'standard_medical':
            return 70
        else:
            return 60
    
    def _suggest_mask_count(self, analysis: Dict) -> int:
        """Suggest optimal number of masks"""
        if analysis['text_density'] > 0.15:
            return 8
        elif analysis['text_density'] < 0.03:
            return 3
        else:
            return 6
    
    def _suggest_processing_strategy(self, analysis: Dict) -> str:
        """Suggest optimal processing strategy"""
        if analysis['image_type'] == 'dense_text':
            return "Aggressive text grouping with high confidence"
        elif analysis['image_type'] == 'poor_quality':
            return "Conservative approach with expanded regions"
        elif analysis['image_type'] == 'wide_diagram':
            return "Horizontal merging with increased gap tolerance"
        else:
            return "Standard processing with balanced parameters"
    
    def _generate_quality_notes(self, analysis: Dict) -> List[str]:
        """Generate human-readable quality notes"""
        notes = []
        
        if analysis['contrast'] < 40:
            notes.append("Low contrast detected - consider image enhancement")
        
        if analysis['sharpness'] < 150:
            notes.append("Image appears blurry - text detection may be challenging")
        
        if analysis['text_density'] > 0.15:
            notes.append("High text density - expect many small regions")
        
        if analysis['text_density'] < 0.03:
            notes.append("Low text density - expect few large regions")
        
        if analysis['noise_level'] > 50:
            notes.append("High noise detected - may affect text recognition")
        
        return notes

def create_adaptive_config_provider() -> AdaptiveConfigProvider:
    """Factory function to create an adaptive config provider"""
    return AdaptiveConfigProvider()
