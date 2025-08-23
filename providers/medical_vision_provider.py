#!/usr/bin/env python3
"""
Medical Vision Provider using Google Vision AI
Intelligently analyzes medical images to identify testable content
"""

import os
import json
import base64
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    from google.cloud import vision
    from google.cloud.vision_v1 import types
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

@dataclass
class MedicalTextRegion:
    """Represents a text region that's good for medical testing"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    text: str
    confidence: float
    importance_score: float  # 0.0-1.0, how important for testing
    region_type: str  # 'anatomical_label', 'measurement', 'drug_name', 'diagnosis', etc.
    rationale: str  # Why this region is good for testing

class GoogleMedicalVisionProvider:
    """Uses Google Vision AI to intelligently analyze medical images"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.client = None
        self.credentials_path = credentials_path or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if GOOGLE_VISION_AVAILABLE and self.credentials_path:
            try:
                self.client = vision.ImageAnnotatorClient()
                logging.info("✅ Google Vision AI client initialized successfully")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Google Vision AI: {e}")
                self.client = None
        else:
            logging.warning("⚠️ Google Vision AI not available - check credentials and dependencies")
    
    @staticmethod
    def available() -> bool:
        """Check if Google Vision AI is available"""
        return GOOGLE_VISION_AVAILABLE and bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    
    def analyze_medical_image(self, image_path: str) -> List[MedicalTextRegion]:
        """
        Analyze a medical image and identify testable text regions
        
        Args:
            image_path: Path to the medical image
            
        Returns:
            List of MedicalTextRegion objects with intelligent masking recommendations
        """
        if not self.client:
            logging.warning("Google Vision AI client not available")
            return []
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = types.Image(content=content)
            
            # Perform comprehensive analysis
            text_detection = self.client.text_detection(image=image)
            label_detection = self.client.label_detection(image=image)
            object_detection = self.client.object_localization(image=image)
            
            # Extract text regions with intelligent grouping
            text_regions = self._extract_and_group_text_regions(text_detection)
            
            # Analyze medical context
            medical_labels = self._extract_medical_labels(label_detection)
            medical_objects = self._extract_medical_objects(object_detection)
            
            # Score and filter regions for testing
            scored_regions = self._score_regions_for_testing(
                text_regions, medical_labels, medical_objects
            )
            
            # Return top regions for masking
            return self._select_best_testing_regions(scored_regions)
            
        except Exception as e:
            logging.error(f"Error analyzing medical image {image_path}: {e}")
            return []
    
    def _extract_and_group_text_regions(self, text_detection) -> List[Dict]:
        """Extract text regions and intelligently group related medical terms"""
        regions = []
        
        if not text_detection.text_annotations:
            return regions
        
        # Skip the first annotation (contains all text)
        text_annotations = text_detection.text_annotations[1:]
        
        # Group annotations by proximity and medical context
        grouped_regions = self._group_medical_terms(text_annotations)
        
        for group in grouped_regions:
            if group['text'].strip():
                regions.append(group)
        
        return regions
    
    def _group_medical_terms(self, text_annotations: List) -> List[Dict]:
        """Intelligently group related medical terms together"""
        if not text_annotations:
            return []
        
        # Sort annotations by position (top to bottom, left to right)
        sorted_annotations = sorted(text_annotations, key=lambda x: (x.bounding_poly.vertices[0].y, x.bounding_poly.vertices[0].x))
        
        grouped_regions = []
        current_group = None
        
        for annotation in sorted_annotations:
            text = annotation.description.strip()
            if not text:
                continue
            
            vertices = annotation.bounding_poly.vertices
            if len(vertices) < 4:
                continue
            
            # Calculate bounding box
            x_coords = [v.x for v in vertices]
            y_coords = [v.y for v in vertices]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Check if this should be grouped with the previous annotation
            if current_group and self._should_group_with_previous(text, bbox, current_group):
                # Extend current group
                current_group['text'] += ' ' + text
                current_group['bbox'] = self._merge_bboxes(current_group['bbox'], bbox)
                current_group['confidence'] = min(current_group['confidence'], getattr(annotation, 'confidence', 0.8))
            else:
                # Start new group
                if current_group:
                    grouped_regions.append(current_group)
                
                current_group = {
                    'bbox': bbox,
                    'text': text,
                    'confidence': getattr(annotation, 'confidence', 0.8)
                }
        
        # Add the last group
        if current_group:
            grouped_regions.append(current_group)
        
        return grouped_regions
    
    def _should_group_with_previous(self, current_text: str, current_bbox: Tuple, previous_group: Dict) -> bool:
        """Determine if current text should be grouped with previous text"""
        prev_bbox = previous_group['bbox']
        prev_text = previous_group['text']
        
        # Check proximity (horizontal and vertical distance)
        x1, y1, w1, h1 = prev_bbox
        x2, y2, w2, h2 = current_bbox
        
        # Calculate centers
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        # Distance thresholds - More aggressive for medical terms
        max_horizontal_distance = max(w1, w2) * 3  # Allow 3x the width for medical terms
        max_vertical_distance = max(h1, h2) * 2   # Allow 2x the height for medical terms
        
        horizontal_distance = abs(center1_x - center2_x)
        vertical_distance = abs(center1_y - center2_y)
        
        # Check if they're close enough
        if horizontal_distance > max_horizontal_distance or vertical_distance > max_vertical_distance:
            return False
        
        # Check if they form a medical term together
        combined_text = prev_text + ' ' + current_text
        if self._is_complete_medical_term(combined_text):
            return True
        
        # Check if they're part of a multi-word medical concept
        if self._is_part_of_medical_concept(prev_text, current_text):
            return True
        
        # Special case: Check for biochemical pathway terms that should always be grouped
        if self._should_always_group(prev_text, current_text):
            return True
        
        return False

    def _should_always_group(self, prev_text: str, current_text: str) -> bool:
        """Special cases where terms should always be grouped regardless of distance"""
        prev_lower = prev_text.lower().strip()
        current_lower = current_text.lower().strip()
        
        # Biochemical pathway terms that should always be grouped
        always_group_patterns = [
            # Glucose metabolism
            (r'glucose', r'6-phosphate'),
            (r'6-phosphate', r'dehydrogenase'),
            (r'glucose\s+6-phosphate', r'dehydrogenase'),
            
            # Fructose metabolism
            (r'fructose', r'6-phosphate'),
            (r'fructose\s+6-phosphate', r'1,6-diphosphate'),
            (r'1,6-diphosphate', r'fructose'),
            
            # Hexose shunt
            (r'hexose', r'monophosphate'),
            (r'monophosphate', r'shunt'),
            (r'hexose\s+monophosphate', r'shunt'),
            
            # NADP cycle
            (r'nadp', r'nadph'),
            (r'gssg', r'g-sh'),
            
            # ATP cycle
            (r'atp', r'adp'),
            
            # General phosphate patterns
            (r'\d+', r'phosphate'),
            (r'phosphate', r'\d+'),
        ]
        
        for prev_pattern, current_pattern in always_group_patterns:
            if (re.search(prev_pattern, prev_lower) and 
                re.search(current_pattern, current_lower)):
                return True
        
        return False

    def _is_complete_medical_term(self, text: str) -> bool:
        """Check if text forms a complete medical term"""
        text_lower = text.lower()
        
        # Complete enzyme names
        enzyme_patterns = [
            r'\b(glucose|fructose|galactose|mannose)\s+\d+[-\s]phosphate\s+(dehydrogenase|isomerase|kinase|phosphatase)\b',
            r'\b(creatine|creatinine)\s+(kinase|phosphokinase)\b',
            r'\b(alanine|aspartate)\s+aminotransferase\b',
            r'\b(lactate|malate)\s+dehydrogenase\b',
            r'\b(hexokinase|phosphofructokinase|pyruvate\s+kinase)\b',
            r'\b(aldolase|enolase|triosephosphate\s+isomerase)\b',
            # Additional biochemical patterns
            r'\b(glucose|fructose)\s+\d+[-\s]phosphate\b',
            r'\b(hexose|pentose)\s+(mono|di|tri)phosphate\s+(shunt|pathway)\b',
            r'\b(glucose|fructose)\s+\d+[-\s]phosphate\s+(dehydrogenase|isomerase|kinase|phosphatase)\b',
            r'\b(glucose|fructose)\s+\d+[,\s]+\d+[-\s]diphosphate\b'
        ]
        
        for pattern in enzyme_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Complete anatomical terms
        anatomical_patterns = [
            r'\b(left|right)\s+(anterior|posterior|lateral|medial)\s+(descending|circumflex|marginal)\s+artery\b',
            r'\b(superior|inferior)\s+(vena\s+cava|mesenteric\s+artery)\b',
            r'\b(common|internal|external)\s+(carotid|iliac|femoral)\s+(artery|vein)\b',
            r'\b(anterior|posterior)\s+(tibial|fibular|peroneal)\s+(artery|vein)\b'
        ]
        
        for pattern in anatomical_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Complete medical compounds
        compound_patterns = [
            r'\b(adenosine|guanosine|cytidine|uridine)\s+(tri|di|mono)phosphate\b',
            r'\b(nicotinamide|flavin)\s+(adenine\s+dinucleotide|mononucleotide)\b',
            r'\b(coenzyme\s+[aq]|cytochrome\s+[a-z0-9]+)\b',
            r'\b(heme|bilirubin|cholesterol|triglyceride)\b'
        ]
        
        for pattern in compound_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _is_part_of_medical_concept(self, prev_text: str, current_text: str) -> bool:
        """Check if two texts are part of the same medical concept"""
        prev_lower = prev_text.lower()
        current_lower = current_text.lower()
        
        # Check for enzyme name patterns
        if any(word in prev_lower for word in ['glucose', 'fructose', 'galactose', 'mannose']):
            if any(word in current_lower for word in ['dehydrogenase', 'isomerase', 'kinase', 'phosphatase', 'phosphate']):
                return True
        
        # Check for anatomical patterns
        if any(word in prev_lower for word in ['left', 'right', 'anterior', 'posterior', 'superior', 'inferior']):
            if any(word in current_lower for word in ['artery', 'vein', 'nerve', 'muscle', 'bone']):
                return True
        
        # Check for compound patterns
        if any(word in prev_lower for word in ['adenosine', 'guanosine', 'cytidine', 'uridine']):
            if any(word in current_lower for word in ['phosphate', 'triphosphate', 'diphosphate']):
                return True
        
        return False
    
    def _merge_bboxes(self, bbox1: Tuple, bbox2: Tuple) -> Tuple:
        """Merge two bounding boxes into one"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        new_x = min(x1, x2)
        new_y = min(y1, y2)
        new_w = max(x1 + w1, x2 + w2) - new_x
        new_h = max(y1 + h1, y2 + h2) - new_y
        
        return (new_x, new_y, new_w, new_h)
    
    def _extract_text_regions(self, text_detection) -> List[Dict]:
        """Extract text regions from Google Vision response (legacy method)"""
        regions = []
        
        if not text_detection.text_annotations:
            return regions
        
        # Skip the first annotation (contains all text)
        for annotation in text_detection.text_annotations[1:]:
            vertices = annotation.bounding_poly.vertices
            if len(vertices) >= 4:
                # Calculate bounding box
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                regions.append({
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'text': annotation.description,
                    'confidence': getattr(annotation, 'confidence', 0.8)
                })
        
        return regions
    
    def _extract_medical_labels(self, label_detection) -> List[str]:
        """Extract medical-relevant labels"""
        medical_labels = []
        
        if not label_detection.label_annotations:
            return medical_labels
        
        for label in label_detection.label_annotations:
            label_text = label.description.lower()
            
            # Medical keywords
            medical_keywords = [
                'medical', 'anatomy', 'physiology', 'diagnosis', 'treatment',
                'blood', 'heart', 'lung', 'brain', 'bone', 'muscle',
                'cell', 'tissue', 'organ', 'disease', 'infection',
                'medication', 'drug', 'prescription', 'symptom',
                'chart', 'diagram', 'table', 'graph', 'scan', 'x-ray'
            ]
            
            if any(keyword in label_text for keyword in medical_keywords):
                medical_labels.append(label_text)
        
        return medical_labels
    
    def _extract_medical_objects(self, object_detection) -> List[Dict]:
        """Extract medical objects and their locations"""
        medical_objects = []
        
        if not object_detection.localized_object_annotations:
            return medical_objects
        
        for obj in object_detection.localized_object_annotations:
            obj_name = obj.name.lower()
            
            # Medical object keywords
            medical_objects_keywords = [
                'anatomical', 'medical device', 'chart', 'table', 'graph',
                'diagram', 'scan', 'microscope', 'test tube', 'syringe'
            ]
            
            if any(keyword in obj_name for keyword in medical_objects_keywords):
                # Get bounding box
                vertices = obj.bounding_poly.normalized_vertices
                if len(vertices) >= 4:
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    medical_objects.append({
                        'name': obj_name,
                        'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                        'confidence': obj.score
                    })
        
        return medical_objects
    
    def _score_regions_for_testing(self, text_regions: List[Dict], 
                                 medical_labels: List[str], 
                                 medical_objects: List[Dict]) -> List[MedicalTextRegion]:
        """Score text regions based on medical testing value"""
        scored_regions = []
        
        for region in text_regions:
            text = region['text'].lower()
            bbox = region['bbox']
            confidence = region['confidence']
            
            # Initialize scoring
            importance_score = 0.0
            region_type = 'unknown'
            rationale = []
            
            # Score based on text content
            if self._is_anatomical_label(text):
                importance_score += 0.4
                region_type = 'anatomical_label'
                rationale.append("Anatomical labels are excellent for testing")
            
            if self._is_measurement_value(text):
                importance_score += 0.3
                region_type = 'measurement'
                rationale.append("Numerical values and measurements are good test targets")
            
            if self._is_drug_name(text):
                importance_score += 0.35
                region_type = 'drug_name'
                rationale.append("Drug names are important for medical testing")
            
            if self._is_diagnosis_term(text):
                importance_score += 0.3
                region_type = 'diagnosis'
                rationale.append("Diagnostic terms are valuable for testing")
            
            if self._is_medical_abbreviation(text):
                importance_score += 0.25
                region_type = 'abbreviation'
                rationale.append("Medical abbreviations are good test targets")
            
            # NEW: Score complete medical terms higher
            if self._is_complete_medical_term(text):
                importance_score += 0.5  # Bonus for complete terms
                region_type = 'complete_medical_term'
                rationale.append("Complete medical terms are ideal for testing")
            
            # Score based on position (center regions are often more important)
            center_score = self._calculate_center_score(bbox)
            importance_score += center_score * 0.2
            if center_score > 0.5:
                rationale.append("Centrally positioned text is often more important")
            
            # Score based on text length (optimal length for testing)
            length_score = self._calculate_length_score(text)
            importance_score += length_score * 0.15
            if length_score > 0.5:
                rationale.append("Text length is optimal for testing")
            
            # Apply confidence penalty
            importance_score *= confidence
            
            # Cap at 1.0
            importance_score = min(1.0, importance_score)
            
            scored_regions.append(MedicalTextRegion(
                bbox=bbox,
                text=region['text'],
                confidence=confidence,
                importance_score=importance_score,
                region_type=region_type,
                rationale='; '.join(rationale) if rationale else 'No specific rationale'
            ))
        
        return scored_regions
    
    def _is_anatomical_label(self, text: str) -> bool:
        """Check if text is an anatomical label"""
        anatomical_keywords = [
            'artery', 'vein', 'muscle', 'bone', 'nerve', 'tendon', 'ligament',
            'organ', 'gland', 'tissue', 'cell', 'membrane', 'cavity',
            'ventricle', 'atrium', 'aorta', 'vena', 'cava', 'coronary',
            'cerebral', 'spinal', 'cervical', 'thoracic', 'lumbar'
        ]
        return any(keyword in text for keyword in anatomical_keywords)
    
    def _is_measurement_value(self, text: str) -> bool:
        """Check if text is a measurement value"""
        import re
        # Look for numbers with units
        measurement_patterns = [
            r'\d+\.?\d*\s*(mm|cm|m|ml|l|mg|g|kg|mmHg|bpm|%|°C|°F)',
            r'\d+\.?\d*',
            r'[+-]?\d+\.?\d*'
        ]
        
        for pattern in measurement_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_drug_name(self, text: str) -> bool:
        """Check if text is a drug name"""
        # This is a simplified check - in practice, you'd want a comprehensive drug database
        drug_indicators = [
            'mg', 'mcg', 'tablet', 'capsule', 'injection', 'dose',
            'prescription', 'medication', 'therapy'
        ]
        return any(indicator in text for indicator in drug_indicators)
    
    def _is_diagnosis_term(self, text: str) -> bool:
        """Check if text is a diagnostic term"""
        diagnosis_keywords = [
            'syndrome', 'disease', 'disorder', 'condition', 'infection',
            'inflammation', 'tumor', 'cancer', 'lesion', 'pathology',
            'anemia', 'diabetes', 'hypertension', 'arthritis'
        ]
        return any(keyword in text for keyword in diagnosis_keywords)
    
    def _is_medical_abbreviation(self, text: str) -> bool:
        """Check if text is a medical abbreviation"""
        medical_abbreviations = [
            'bp', 'hr', 'rr', 'temp', 'o2', 'co2', 'hct', 'hgb',
            'wbc', 'rbc', 'plt', 'pt', 'ptt', 'inr', 'bun', 'cre',
            'na', 'k', 'cl', 'hco3', 'ca', 'mg', 'po4'
        ]
        return text.lower() in medical_abbreviations
    
    def _calculate_center_score(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate how central a region is (0.0 = edge, 1.0 = center)"""
        x, y, w, h = bbox
        
        # Normalize coordinates (assuming image is 1000x1000 for simplicity)
        # In practice, you'd get actual image dimensions
        center_x = 500  # Assumed image center
        center_y = 500
        
        # Calculate distance from center
        region_center_x = x + w/2
        region_center_y = y + h/2
        
        distance = ((region_center_x - center_x)**2 + (region_center_y - center_y)**2)**0.5
        max_distance = (500**2 + 500**2)**0.5
        
        # Convert to score (closer to center = higher score)
        center_score = 1.0 - (distance / max_distance)
        return max(0.0, center_score)
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate optimal text length for testing (0.0 = too short/long, 1.0 = optimal)"""
        length = len(text.strip())
        
        # Optimal length for testing: 3-25 characters (increased for medical terms)
        if 3 <= length <= 25:
            return 1.0
        elif length < 3:
            return length / 3.0
        else:
            return max(0.0, 1.0 - (length - 25) / 15.0)
    
    def _select_best_testing_regions(self, scored_regions: List[MedicalTextRegion], 
                                   max_regions: int = 6) -> List[MedicalTextRegion]:
        """Select the best regions for testing based on importance scores"""
        # Sort by importance score (descending)
        sorted_regions = sorted(scored_regions, key=lambda r: r.importance_score, reverse=True)
        
        # Filter out low-quality regions
        quality_threshold = 0.3
        quality_regions = [r for r in sorted_regions if r.importance_score >= quality_threshold]
        
        # Return top regions, but ensure diversity
        selected_regions = []
        used_types = set()
        
        for region in quality_regions[:max_regions]:
            # Prefer diverse region types
            if region.region_type not in used_types or len(selected_regions) < max_regions // 2:
                selected_regions.append(region)
                used_types.add(region.region_type)
            
            if len(selected_regions) >= max_regions:
                break
        
        # If we don't have enough quality regions, add some lower-scoring ones
        if len(selected_regions) < max_regions:
            remaining = [r for r in quality_regions if r not in selected_regions]
            selected_regions.extend(remaining[:max_regions - len(selected_regions)])
        
        return selected_regions

def create_medical_vision_provider(credentials_path: Optional[str] = None) -> GoogleMedicalVisionProvider:
    """Factory function to create a medical vision provider"""
    return GoogleMedicalVisionProvider(credentials_path)
