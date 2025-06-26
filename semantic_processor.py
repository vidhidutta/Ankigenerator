#!/usr/bin/env python3
"""
Semantic Processing Module for Enhanced Flashcard Generation
Handles semantic chunking, embeddings, and slide-level analysis
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SemanticProcessor:
    def __init__(self, model_name: str = 'tfidf'):
        """
        Initialize the semantic processor with TF-IDF vectorizer
        
        Args:
            model_name: Model type (currently only supports 'tfidf')
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        self.stop_words = set(stopwords.words('english'))
        
        # Medical-specific stop words to preserve
        medical_terms = {
            'drug', 'drugs', 'medication', 'medications', 'treatment', 'treatments',
            'disease', 'diseases', 'symptom', 'symptoms', 'diagnosis', 'diagnoses',
            'patient', 'patients', 'clinical', 'medical', 'health', 'healthcare',
            'therapy', 'therapies', 'mechanism', 'mechanisms', 'receptor', 'receptors',
            'enzyme', 'enzymes', 'protein', 'proteins', 'gene', 'genes', 'cell', 'cells',
            'tissue', 'tissues', 'organ', 'organs', 'system', 'systems', 'function',
            'functions', 'structure', 'structures', 'pathway', 'pathways', 'process',
            'processes', 'effect', 'effects', 'action', 'actions', 'response', 'responses'
        }
        self.stop_words = self.stop_words - medical_terms
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove slide headers
        text = re.sub(r'^Slide \d+:\s*', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}]', ' ', text)
        
        return text.strip()
    
    def semantic_chunk_text(self, text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into semantic chunks based on meaning and structure
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of semantic chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(cleaned_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                # Add overlap from the end of current chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    chunks.append(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Post-process chunks to ensure they're meaningful
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 20:  # Minimum meaningful chunk size
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key medical phrases from text
        
        Args:
            text: Text to extract phrases from
            
        Returns:
            List of key phrases
        """
        # Medical term patterns
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|disorder|condition)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:receptor|enzyme|protein|gene|cell|tissue|organ)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:inhibitor|agonist|antagonist|blocker|activator)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:therapy|treatment|medication|drug)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:mechanism|pathway|process|function)\b'
        ]
        
        key_phrases = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_phrases.extend(matches)
        
        return list(set(key_phrases))
    
    def compute_slide_embeddings(self, slide_texts: List[str]) -> np.ndarray:
        """
        Compute TF-IDF embeddings for all slides
        
        Args:
            slide_texts: List of slide texts
            
        Returns:
            Array of slide embeddings
        """
        self.logger.info(f"Computing TF-IDF embeddings for {len(slide_texts)} slides")
        
        # Clean and prepare texts
        cleaned_texts = [self.clean_text(text) for text in slide_texts]
        
        # Compute TF-IDF embeddings
        embeddings = self.vectorizer.fit_transform(cleaned_texts).toarray()
        
        return embeddings
    
    def find_similar_slides(self, embeddings: np.ndarray, similarity_threshold: float = 0.7) -> List[List[int]]:
        """
        Find groups of similar slides using clustering
        
        Args:
            embeddings: Slide embeddings
            similarity_threshold: Minimum similarity for grouping
            
        Returns:
            List of slide groups (each group contains slide indices)
        """
        if len(embeddings) < 2:
            return [[0]] if len(embeddings) == 1 else []
        
        # Convert similarity threshold to distance threshold
        distance_threshold = 1 - similarity_threshold
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group slides by cluster
        slide_groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in slide_groups:
                slide_groups[label] = []
            slide_groups[label].append(i)
        
        return list(slide_groups.values())
    
    def enhance_slide_context(self, slide_texts: List[str], slide_groups: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Enhance slide content with context from similar slides
        
        Args:
            slide_texts: Original slide texts
            slide_groups: Groups of similar slides
            
        Returns:
            List of enhanced slide contexts
        """
        enhanced_slides = []
        
        for i, slide_text in enumerate(slide_texts):
            # Find which group this slide belongs to
            slide_group = None
            for group in slide_groups:
                if i in group:
                    slide_group = group
                    break
            
            # Get related slides (excluding current slide)
            related_slides = []
            if slide_group and len(slide_group) > 1:
                related_slides = [slide_texts[j] for j in slide_group if j != i]
            
            # Extract key phrases from current slide
            key_phrases = self.extract_key_phrases(slide_text)
            
            # Create enhanced context
            enhanced_context = {
                'original_text': slide_text,
                'key_phrases': key_phrases,
                'related_slides': related_slides,
                'slide_index': i,
                'group_size': len(slide_group) if slide_group else 1
            }
            
            enhanced_slides.append(enhanced_context)
        
        return enhanced_slides
    
    def create_semantic_chunks(self, slide_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from slide texts with enhanced context
        
        Args:
            slide_texts: List of slide texts
            
        Returns:
            List of semantic chunks with metadata
        """
        self.logger.info("Creating semantic chunks from slides")
        
        # Compute embeddings
        embeddings = self.compute_slide_embeddings(slide_texts)
        
        # Find similar slides
        slide_groups = self.find_similar_slides(embeddings)
        
        # Enhance slide contexts
        enhanced_slides = self.enhance_slide_context(slide_texts, slide_groups)
        
        # Create semantic chunks
        semantic_chunks = []
        
        for enhanced_slide in enhanced_slides:
            original_text = enhanced_slide['original_text']
            slide_index = enhanced_slide['slide_index']
            
            # Create semantic chunks from the slide
            chunks = self.semantic_chunk_text(original_text)
            
            for chunk_index, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk,
                    'slide_index': slide_index,
                    'chunk_index': chunk_index,
                    'key_phrases': enhanced_slide['key_phrases'],
                    'related_slides': enhanced_slide['related_slides'],
                    'group_size': enhanced_slide['group_size'],
                    'embedding': embeddings[slide_index] if len(embeddings) > slide_index else None
                }
                semantic_chunks.append(chunk_data)
        
        self.logger.info(f"Created {len(semantic_chunks)} semantic chunks from {len(slide_texts)} slides")
        return semantic_chunks
    
    def build_enhanced_prompt(self, chunk_data: Dict[str, Any], base_prompt: str) -> str:
        """
        Build an enhanced prompt using semantic chunk data
        
        Args:
            chunk_data: Semantic chunk data
            base_prompt: Base prompt template
            
        Returns:
            Enhanced prompt with context
        """
        chunk_text = chunk_data['text']
        key_phrases = chunk_data['key_phrases']
        related_slides = chunk_data['related_slides']
        
        # Build context information
        context_parts = []
        
        if key_phrases:
            context_parts.append(f"Key medical terms in this content: {', '.join(key_phrases[:5])}")
        
        if related_slides:
            # Extract key information from related slides
            related_context = []
            for i, related_slide in enumerate(related_slides[:2]):  # Limit to 2 related slides
                # Extract first sentence or key phrase
                sentences = sent_tokenize(self.clean_text(related_slide))
                if sentences:
                    related_context.append(f"Related slide {i+1}: {sentences[0][:100]}...")
            
            if related_context:
                context_parts.append("Related content from similar slides:\n" + "\n".join(related_context))
        
        # Build enhanced prompt
        enhanced_prompt = base_prompt
        
        if context_parts:
            context_section = "\n\nCONTEXT INFORMATION:\n" + "\n".join(context_parts)
            enhanced_prompt += context_section
        
        enhanced_prompt += f"\n\nCURRENT CONTENT TO PROCESS:\n{chunk_text}"
        
        return enhanced_prompt
    
    def analyze_content_quality(self, semantic_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of semantic chunks
        
        Args:
            semantic_chunks: List of semantic chunks
            
        Returns:
            Analysis results
        """
        total_chunks = len(semantic_chunks)
        total_slides = len(set(chunk['slide_index'] for chunk in semantic_chunks))
        
        # Analyze chunk sizes
        chunk_sizes = [len(chunk['text']) for chunk in semantic_chunks]
        avg_chunk_size = np.mean(chunk_sizes) if chunk_sizes else 0
        
        # Analyze key phrases
        all_key_phrases = []
        for chunk in semantic_chunks:
            all_key_phrases.extend(chunk['key_phrases'])
        
        unique_key_phrases = list(set(all_key_phrases))
        
        # Analyze slide groups
        group_sizes = [chunk['group_size'] for chunk in semantic_chunks]
        avg_group_size = np.mean(group_sizes) if group_sizes else 0
        
        analysis = {
            'total_chunks': total_chunks,
            'total_slides': total_slides,
            'avg_chunk_size': avg_chunk_size,
            'unique_key_phrases': len(unique_key_phrases),
            'avg_group_size': avg_group_size,
            'key_phrases': unique_key_phrases[:10],  # Top 10 key phrases
            'chunk_size_distribution': {
                'small': len([s for s in chunk_sizes if s < 200]),
                'medium': len([s for s in chunk_sizes if 200 <= s < 500]),
                'large': len([s for s in chunk_sizes if s >= 500])
            }
        }
        
        return analysis 