#!/usr/bin/env python3
"""
Holistic Medical Analyzer - AI as Medical Expert Educator

This module transforms the AI from a simple flashcard generator into a comprehensive
medical expert that:
1. Comprehensively understands the entire lecture content
2. Identifies knowledge gaps and fills them with expert medical knowledge
3. Creates visual mind maps showing concept relationships
4. Generates comprehensive PDF notes alongside flashcards
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import yaml

@dataclass
class MedicalConcept:
    """Represents a medical concept with its relationships and context"""
    name: str
    definition: str
    category: str
    importance: str  # "core", "important", "supplementary"
    relationships: List[str]  # List of related concept names
    clinical_relevance: str
    knowledge_level: str  # "foundational", "intermediate", "advanced"
    prerequisites: List[str]  # Concepts that should be understood first

@dataclass
class MindMapNode:
    """Represents a node in the mind map"""
    concept: MedicalConcept
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0  # Relative size based on importance
    color: str = "#E74C3C"  # Medical red theme

@dataclass
class MindMapConnection:
    """Represents a connection between mind map nodes"""
    from_concept: str
    to_concept: str
    relationship_type: str  # "prerequisite", "related", "contrast", "causes"
    strength: float = 1.0  # Connection strength (0-1)

@dataclass
class HolisticAnalysis:
    """Complete holistic analysis of a medical lecture"""
    lecture_title: str
    main_topics: List[str]
    concepts: List[MedicalConcept]
    mind_maps: List[Dict[str, Any]]  # Multiple mind maps for different topics
    knowledge_gaps: List[str]
    filled_gaps: List[Dict[str, str]]  # AI-added knowledge
    learning_objectives: List[str]
    clinical_pearls: List[str]
    glossary: Dict[str, str]
    cross_references: List[Dict[str, str]]

class HolisticMedicalAnalyzer:
    """
    AI-powered medical expert that provides comprehensive lecture analysis
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml"""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config.get('holistic_analysis', {})
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
            return {}
    
    def analyze_lecture_holistically(self, lecture_content: str, lecture_title: str = "Medical Lecture") -> HolisticAnalysis:
        """
        Perform comprehensive holistic analysis of the entire lecture
        
        Args:
            lecture_content: Full text content from all slides
            lecture_title: Title of the lecture
            
        Returns:
            HolisticAnalysis object with complete analysis
        """
        print(f"🧠 [HolisticAnalyzer] Starting comprehensive analysis of: {lecture_title}")
        
        # Step 1: Extract and categorize all medical concepts
        concepts = self._extract_medical_concepts(lecture_content)
        print(f"📚 [HolisticAnalyzer] Extracted {len(concepts)} medical concepts")
        
        # Step 2: Identify knowledge gaps and fill them
        knowledge_gaps = self._identify_knowledge_gaps(concepts, lecture_content)
        filled_gaps = self._fill_knowledge_gaps(knowledge_gaps, concepts, lecture_content)
        print(f"🔍 [HolisticAnalyzer] Identified {len(knowledge_gaps)} gaps, filled {len(filled_gaps)}")
        
        # Step 3: Generate mind maps
        mind_maps = self._generate_mind_maps(concepts)
        print(f"🗺️ [HolisticAnalyzer] Generated {len(mind_maps)} mind maps")
        
        # Step 4: Create learning objectives and clinical pearls
        learning_objectives = self._generate_learning_objectives(concepts, lecture_content)
        clinical_pearls = self._extract_clinical_pearls(concepts, lecture_content)
        
        # Step 5: Build comprehensive glossary
        glossary = self._build_glossary(concepts, lecture_content)
        
        # Step 6: Identify cross-references between topics
        cross_references = self._identify_cross_references(concepts, lecture_content)
        
        return HolisticAnalysis(
            lecture_title=lecture_title,
            main_topics=self._extract_main_topics(lecture_content),
            concepts=concepts,
            mind_maps=mind_maps,
            knowledge_gaps=knowledge_gaps,
            filled_gaps=filled_gaps,
            learning_objectives=learning_objectives,
            clinical_pearls=clinical_pearls,
            glossary=glossary,
            cross_references=cross_references
        )
    
    def _extract_medical_concepts(self, content: str) -> List[MedicalConcept]:
        """Extract and categorize all medical concepts from the lecture"""
        
        prompt = f"""You are a senior medical educator with 20+ years of experience. 
        Your task is to comprehensively analyze this medical lecture and extract ALL medical concepts.

        **Your Role**: Act as if you are the professor who created this lecture. You understand:
        - The foundational knowledge students need
        - How concepts connect and build upon each other
        - What clinical context makes concepts memorable
        - The logical flow of medical reasoning

        **Analysis Instructions**:
        1. Read the ENTIRE lecture content multiple times
        2. Identify EVERY medical concept, mechanism, drug, condition, etc.
        3. Categorize each concept by importance and complexity
        4. Identify relationships between concepts
        5. Note any missing foundational knowledge that would help understanding

        **Output Format**: Return ONLY valid JSON:
        {{
            "concepts": [
                {{
                    "name": "Concept name",
                    "definition": "Clear, concise definition",
                    "category": "pharmacology|physiology|pathology|anatomy|clinical",
                    "importance": "core|important|supplementary",
                    "relationships": ["related concept 1", "related concept 2"],
                    "clinical_relevance": "Why this matters clinically",
                    "knowledge_level": "foundational|intermediate|advanced",
                    "prerequisites": ["concept that should be understood first"]
                }}
            ]
        }}

        **Lecture Content**:
        {content[:8000]}  # First 8000 chars for context

        Remember: You are the expert. Think like a professor explaining to students who need to understand the BIG PICTURE, not just memorize facts."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                concepts_data = data.get('concepts', [])
                
                concepts = []
                for concept_data in concepts_data:
                    concept = MedicalConcept(
                        name=concept_data.get('name', ''),
                        definition=concept_data.get('definition', ''),
                        category=concept_data.get('category', 'clinical'),
                        importance=concept_data.get('importance', 'important'),
                        relationships=concept_data.get('relationships', []),
                        clinical_relevance=concept_data.get('clinical_relevance', ''),
                        knowledge_level=concept_data.get('knowledge_level', 'intermediate'),
                        prerequisites=concept_data.get('prerequisites', [])
                    )
                    concepts.append(concept)
                
                return concepts
            else:
                print("⚠️ [HolisticAnalyzer] Could not extract JSON from AI response")
                return []
                
        except Exception as e:
            print(f"❌ [HolisticAnalyzer] Error extracting concepts: {e}")
            return []
    
    def _identify_knowledge_gaps(self, concepts: List[MedicalConcept], content: str) -> List[str]:
        """Identify gaps in foundational knowledge that would help understanding"""
        
        prompt = f"""As a senior medical educator, analyze this lecture content and identify 
        foundational knowledge gaps that would help students understand the material better.

        **Current Concepts Identified**:
        {[c.name for c in concepts[:10]]}  # First 10 concepts

        **Lecture Content**:
        {content[:4000]}

        **Your Task**: Identify what foundational knowledge is MISSING that would help students:
        1. Understand the mechanisms better
        2. See the clinical relevance
        3. Connect concepts together
        4. Apply knowledge in practice

        **Output**: Return ONLY a JSON list of gap descriptions:
        {{
            "gaps": [
                "Missing foundational concept 1",
                "Missing foundational concept 2",
                "Missing clinical context for concept X",
                "Missing mechanism explanation for Y"
            ]
        }}

        Think like a professor who knows what students struggle with when learning this topic."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('gaps', [])
            else:
                return []
                
        except Exception as e:
            print(f"❌ [HolisticAnalyzer] Error identifying gaps: {e}")
            return []
    
    def _fill_knowledge_gaps(self, gaps: List[str], concepts: List[MedicalConcept], content: str) -> List[Dict[str, str]]:
        """Fill identified knowledge gaps with expert medical knowledge"""
        
        if not gaps:
            return []
        
        prompt = f"""As a senior medical educator, you need to fill knowledge gaps to help students 
        understand this lecture comprehensively.

        **Lecture Context**:
        {content[:3000]}

        **Current Concepts**:
        {[c.name for c in concepts[:8]]}

        **Knowledge Gaps to Fill**:
        {gaps[:5]}  # First 5 gaps

        **Your Task**: For each gap, provide:
        1. The missing foundational knowledge
        2. How it connects to the lecture concepts
        3. Clinical relevance and examples
        4. Simple explanations that build understanding

        **Output**: Return ONLY valid JSON:
        {{
            "filled_gaps": [
                {{
                    "gap": "Original gap description",
                    "explanation": "Comprehensive explanation of missing knowledge",
                    "clinical_relevance": "Why this matters in practice",
                    "connections": "How this connects to lecture concepts"
                }}
            ]
        }}

        Remember: You are the expert professor. Explain things the way you would to a bright student 
        who needs to understand the BIG PICTURE, not just memorize facts."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('filled_gaps', [])
            else:
                return []
                
        except Exception as e:
            print(f"❌ [HolisticAnalyzer] Error filling gaps: {e}")
            return []
    
    def _generate_mind_maps(self, concepts: List[MedicalConcept]) -> List[Dict[str, Any]]:
        """Generate visual mind maps showing concept relationships"""
        
        if not concepts:
            return []
        
        # Group concepts by category for multiple mind maps
        categories = {}
        for concept in concepts:
            if concept.category not in categories:
                categories[concept.category] = []
            categories[concept.category].append(concept)
        
        mind_maps = []
        
        for category, category_concepts in categories.items():
            if len(category_concepts) > 1:  # Only create maps for categories with multiple concepts
                mind_map = self._create_single_mind_map(category_concepts, category)
                mind_maps.append(mind_map)
        
        return mind_maps
    
    def _create_single_mind_map(self, concepts: List[MedicalConcept], category: str) -> Dict[str, Any]:
        """Create a single mind map for a category of concepts"""
        
        # Create nodes
        nodes = []
        for i, concept in enumerate(concepts):
            # Simple circular layout
            angle = (2 * 3.14159 * i) / len(concepts)
            radius = 200
            x = 300 + radius * math.cos(angle)
            y = 300 + radius * math.sin(angle)
            
            # Size based on importance
            size = 1.5 if concept.importance == "core" else 1.0
            
            node = MindMapNode(
                concept=concept,
                x=x,
                y=y,
                size=size,
                color="#E74C3C" if concept.importance == "core" else "#95A5A6"
            )
            nodes.append(node)
        
        # Create connections based on relationships
        connections = []
        for concept in concepts:
            for related_name in concept.relationships:
                # Find the related concept
                related_concept = next((c for c in concepts if c.name == related_name), None)
                if related_concept:
                    connection = MindMapConnection(
                        from_concept=concept.name,
                        to_concept=related_name,
                        relationship_type="related",
                        strength=0.8
                    )
                    connections.append(connection)
        
        return {
            "category": category,
            "title": f"{category.title()} Concept Map",
            "nodes": [self._node_to_dict(node) for node in nodes],
            "connections": [self._connection_to_dict(conn) for conn in connections],
            "layout": "circular",
            "style": "medical_theme"
        }
    
    def _node_to_dict(self, node: MindMapNode) -> Dict[str, Any]:
        """Convert MindMapNode to dictionary for JSON serialization"""
        return {
            "concept_name": node.concept.name,
            "definition": node.concept.definition,
            "importance": node.concept.importance,
            "category": node.concept.category,
            "x": node.x,
            "y": node.y,
            "size": node.size,
            "color": node.color
        }
    
    def _connection_to_dict(self, connection: MindMapConnection) -> Dict[str, Any]:
        """Convert MindMapConnection to dictionary for JSON serialization"""
        return {
            "from": connection.from_concept,
            "to": connection.to_concept,
            "type": connection.relationship_type,
            "strength": connection.strength
        }
    
    def _generate_learning_objectives(self, concepts: List[MedicalConcept], content: str) -> List[str]:
        """Generate clear learning objectives for the lecture"""
        
        prompt = f"""As a senior medical educator, create clear, measurable learning objectives 
        for this lecture based on the concepts covered.

        **Concepts Covered**:
        {[c.name for c in concepts[:10]]}

        **Lecture Content**:
        {content[:2000]}

        **Your Task**: Create 5-7 learning objectives that:
        1. Are specific and measurable
        2. Cover different levels of understanding (recall, comprehension, application)
        3. Focus on what students should be able to DO after the lecture
        4. Are written in clear, student-friendly language

        **Output**: Return ONLY a JSON list:
        {{
            "objectives": [
                "By the end of this lecture, students will be able to...",
                "Students will demonstrate understanding of...",
                "After this lecture, students can apply..."
            ]
        }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('objectives', [])
            else:
                return []
                
        except Exception as e:
            print(f"❌ [HolisticAnalyzer] Error generating objectives: {e}")
            return []
    
    def _extract_clinical_pearls(self, concepts: List[MedicalConcept], content: str) -> List[str]:
        """Extract key clinical insights and pearls from the lecture"""
        
        prompt = f"""As a senior medical educator, extract the most important clinical pearls 
        and insights from this lecture that students should remember for practice.

        **Concepts Covered**:
        {[c.name for c in concepts[:8]]}

        **Lecture Content**:
        {content[:3000]}

        **Your Task**: Identify 5-8 clinical pearls that:
        1. Are immediately applicable in clinical practice
        2. Help with diagnosis, treatment, or patient management
        3. Are evidence-based and clinically relevant
        4. Would be valuable for students to remember

        **Output**: Return ONLY a JSON list:
        {{
            "pearls": [
                "Clinical pearl 1 - practical insight for practice",
                "Clinical pearl 2 - key diagnostic point",
                "Clinical pearl 3 - treatment consideration"
            ]
        }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('pearls', [])
            else:
                return []
                
        except Exception as e:
            print(f"❌ [HolisticAnalyzer] Error extracting pearls: {e}")
            return []
    
    def _build_glossary(self, concepts: List[MedicalConcept], content: str) -> Dict[str, str]:
        """Build comprehensive medical terminology glossary"""
        
        glossary = {}
        
        # Add concepts from analysis
        for concept in concepts:
            if concept.name and concept.definition:
                glossary[concept.name] = concept.definition
        
        # Extract additional terms from content
        prompt = f"""Extract medical terminology and abbreviations from this lecture content 
        that should be included in a glossary for students.

        **Content**:
        {content[:4000]}

        **Output**: Return ONLY valid JSON:
        {{
            "terms": {{
                "Term1": "Definition1",
                "Term2": "Definition2"
            }}
        }}

        Focus on terms that students might not know or need clarification on."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                additional_terms = data.get('terms', {})
                glossary.update(additional_terms)
                
        except Exception as e:
            print(f"⚠️ [HolisticAnalyzer] Error building glossary: {e}")
        
        return glossary
    
    def _identify_cross_references(self, concepts: List[MedicalConcept], content: str) -> List[Dict[str, str]]:
        """Identify connections between different topics in the lecture"""
        
        cross_refs = []
        
        # Look for relationships between concepts
        for concept in concepts:
            for related in concept.relationships:
                cross_refs.append({
                    "from_topic": concept.name,
                    "to_topic": related,
                    "relationship": "related",
                    "explanation": f"Connects to {related} concept"
                })
        
        return cross_refs
    
    def _extract_main_topics(self, content: str) -> List[str]:
        """Extract main topic areas from the lecture"""
        
        prompt = f"""Identify the main topic areas or sections in this medical lecture.

        **Content**:
        {content[:3000]}

        **Output**: Return ONLY a JSON list of main topics:
        {{
            "topics": [
                "Main topic 1",
                "Main topic 2",
                "Main topic 3"
            ]
        }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('topics', [])
            else:
                return []
                
        except Exception as e:
            print(f"⚠️ [HolisticAnalyzer] Error extracting topics: {e}")
            return []

# Import math for calculations
import math
