#!/usr/bin/env python3
"""
Medical Notes PDF Generator - Creates Comprehensive Visual Notes

This module generates beautiful PDF documents containing:
1. Visual mind maps showing concept relationships
2. Comprehensive medical explanations
3. Clinical pearls and learning objectives
4. Glossary and cross-references
5. Professional medical education layout
"""

import os
import json
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, red, blue, green, gray
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, String, Line, Circle, Rect
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import math

class MedicalNotesPDFGenerator:
    """
    Generates comprehensive PDF medical notes with visual elements
    """
    
    def __init__(self, output_path: str = "medical_notes.pdf"):
        self.output_path = output_path
        self.styles = self._create_styles()
        self.page_width, self.page_height = A4
        
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for medical notes"""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='MedicalTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#E74C3C'),  # Medical red
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=HexColor('#2C3E50'),  # Dark blue
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection header style
        styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=HexColor('#34495E'),  # Medium blue
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        styles.add(ParagraphStyle(
            name='BodyText',
            parent=styles['Normal'],
            fontSize=11,
            textColor=black,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Clinical pearl style
        styles.add(ParagraphStyle(
            name='ClinicalPearl',
            parent=styles['Normal'],
            fontSize=11,
            textColor=HexColor('#27AE60'),  # Green
            spaceAfter=8,
            spaceBefore=8,
            leftIndent=20,
            fontName='Helvetica-Bold',
            backColor=HexColor('#E8F5E8')  # Light green background
        ))
        
        # Glossary term style
        styles.add(ParagraphStyle(
            name='GlossaryTerm',
            parent=styles['Normal'],
            fontSize=11,
            textColor=HexColor('#8E44AD'),  # Purple
            spaceAfter=4,
            fontName='Helvetica-Bold'
        ))
        
        return styles
    
    def generate_comprehensive_notes(self, holistic_analysis: Any) -> str:
        """
        Generate comprehensive PDF notes from holistic analysis
        
        Args:
            holistic_analysis: HolisticAnalysis object from the analyzer
            
        Returns:
            Path to generated PDF file
        """
        print(f"📄 [PDFGenerator] Creating comprehensive medical notes...")
        
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page(holistic_analysis))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_table_of_contents(holistic_analysis))
        story.append(PageBreak())
        
        # Learning objectives
        story.extend(self._create_learning_objectives(holistic_analysis))
        story.append(PageBreak())
        
        # Main topics and concepts
        story.extend(self._create_main_content(holistic_analysis))
        
        # Mind maps
        story.extend(self._create_mind_maps(holistic_analysis))
        
        # Clinical pearls
        story.extend(self._create_clinical_pearls(holistic_analysis))
        
        # Knowledge gaps and filled content
        story.extend(self._create_knowledge_gaps_section(holistic_analysis))
        
        # Glossary
        story.extend(self._create_glossary(holistic_analysis))
        
        # Cross-references
        story.extend(self._create_cross_references(holistic_analysis))
        
        # Build PDF
        doc.build(story)
        print(f"✅ [PDFGenerator] PDF generated successfully: {self.output_path}")
        
        return self.output_path
    
    def _create_title_page(self, analysis: Any) -> List:
        """Create the title page"""
        elements = []
        
        # Main title
        title = Paragraph(f"<b>{analysis.lecture_title}</b>", self.styles['MedicalTitle'])
        elements.append(title)
        elements.append(Spacer(1, 2*inch))
        
        # Subtitle
        subtitle = Paragraph(
            "Comprehensive Medical Notes with Concept Maps",
            self.styles['SectionHeader']
        )
        subtitle.alignment = TA_CENTER
        elements.append(subtitle)
        elements.append(Spacer(1, 1*inch))
        
        # Main topics overview
        if hasattr(analysis, 'main_topics') and analysis.main_topics:
            topics_text = "<b>Main Topics Covered:</b><br/>"
            for topic in analysis.main_topics[:5]:  # First 5 topics
                topics_text += f"• {topic}<br/>"
            
            topics_para = Paragraph(topics_text, self.styles['BodyText'])
            topics_para.alignment = TA_CENTER
            elements.append(topics_para)
        
        elements.append(Spacer(1, 2*inch))
        
        # Generated by info
        generated_by = Paragraph(
            "Generated by OjaMed AI Medical Expert System<br/>"
            "Comprehensive analysis with knowledge gap filling and concept mapping",
            self.styles['BodyText']
        )
        generated_by.alignment = TA_CENTER
        elements.append(generated_by)
        
        return elements
    
    def _create_table_of_contents(self, analysis: Any) -> List:
        """Create table of contents"""
        elements = []
        
        toc_title = Paragraph("Table of Contents", self.styles['SectionHeader'])
        elements.append(toc_title)
        elements.append(Spacer(1, 0.5*inch))
        
        # TOC items
        toc_items = [
            ("Learning Objectives", "objectives"),
            ("Main Concepts & Topics", "concepts"),
            ("Concept Mind Maps", "mindmaps"),
            ("Clinical Pearls", "pearls"),
            ("Knowledge Gaps & Explanations", "gaps"),
            ("Medical Glossary", "glossary"),
            ("Cross-References", "crossrefs")
        ]
        
        for item_text, item_id in toc_items:
            item = Paragraph(f"• {item_text}", self.styles['BodyText'])
            elements.append(item)
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_learning_objectives(self, analysis: Any) -> List:
        """Create learning objectives section"""
        elements = []
        
        objectives_title = Paragraph("Learning Objectives", self.styles['SectionHeader'])
        elements.append(objectives_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'learning_objectives') and analysis.learning_objectives:
            for i, objective in enumerate(analysis.learning_objectives, 1):
                obj_text = f"<b>{i}.</b> {objective}"
                obj_para = Paragraph(obj_text, self.styles['BodyText'])
                elements.append(obj_para)
                elements.append(Spacer(1, 0.2*inch))
        else:
            no_obj = Paragraph("Learning objectives will be generated based on lecture content.", 
                             self.styles['BodyText'])
            elements.append(no_obj)
        
        return elements
    
    def _create_main_content(self, analysis: Any) -> List:
        """Create main content section with concepts"""
        elements = []
        
        content_title = Paragraph("Main Concepts & Topics", self.styles['SectionHeader'])
        elements.append(content_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'concepts') and analysis.concepts:
            # Group concepts by category
            categories = {}
            for concept in analysis.concepts:
                if concept.category not in categories:
                    categories[concept.category] = []
                categories[concept.category].append(concept)
            
            for category, concepts in categories.items():
                # Category header
                cat_header = Paragraph(f"{category.title()} Concepts", self.styles['SubsectionHeader'])
                elements.append(cat_header)
                elements.append(Spacer(1, 0.2*inch))
                
                # Concepts in this category
                for concept in concepts:
                    # Concept name and importance
                    importance_color = "#E74C3C" if concept.importance == "core" else "#95A5A6"
                    concept_header = f'<font color="{importance_color}"><b>{concept.name}</b></font>'
                    if concept.importance == "core":
                        concept_header += " <i>(Core Concept)</i>"
                    
                    concept_title = Paragraph(concept_header, self.styles['BodyText'])
                    elements.append(concept_title)
                    
                    # Definition
                    if concept.definition:
                        def_text = f"<b>Definition:</b> {concept.definition}"
                        def_para = Paragraph(def_text, self.styles['BodyText'])
                        elements.append(def_para)
                    
                    # Clinical relevance
                    if concept.clinical_relevance:
                        clin_text = f"<b>Clinical Relevance:</b> {concept.clinical_relevance}"
                        clin_para = Paragraph(clin_text, self.styles['BodyText'])
                        elements.append(clin_para)
                    
                    # Relationships
                    if concept.relationships:
                        rel_text = f"<b>Related to:</b> {', '.join(concept.relationships)}"
                        rel_para = Paragraph(rel_text, self.styles['BodyText'])
                        elements.append(rel_para)
                    
                    elements.append(Spacer(1, 0.3*inch))
        else:
            no_concepts = Paragraph("Concepts will be extracted and analyzed from the lecture content.", 
                                  self.styles['BodyText'])
            elements.append(no_concepts)
        
        return elements
    
    def _create_mind_maps(self, analysis: Any) -> List:
        """Create mind maps section"""
        elements = []
        
        maps_title = Paragraph("Concept Mind Maps", self.styles['SectionHeader'])
        elements.append(maps_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'mind_maps') and analysis.mind_maps:
            for i, mind_map in enumerate(analysis.mind_maps):
                # Map title
                map_title = Paragraph(f"{mind_map.get('title', f'Concept Map {i+1}')}", 
                                    self.styles['SubsectionHeader'])
                elements.append(map_title)
                elements.append(Spacer(1, 0.2*inch))
                
                # Create visual representation
                try:
                    map_drawing = self._create_mind_map_drawing(mind_map)
                    if map_drawing:
                        elements.append(map_drawing)
                        elements.append(Spacer(1, 0.3*inch))
                except Exception as e:
                    print(f"⚠️ [PDFGenerator] Error creating mind map drawing: {e}")
                    # Fallback to text representation
                    map_text = self._create_mind_map_text(mind_map)
                    elements.append(map_text)
                
                elements.append(PageBreak())
        else:
            no_maps = Paragraph("Mind maps will be generated based on concept relationships.", 
                              self.styles['BodyText'])
            elements.append(no_maps)
        
        return elements
    
    def _create_mind_map_drawing(self, mind_map: Dict[str, Any]) -> Optional[Drawing]:
        """Create a visual drawing of the mind map"""
        try:
            # Get map dimensions
            nodes = mind_map.get('nodes', [])
            if not nodes:
                return None
            
            # Calculate drawing dimensions
            max_x = max([node.get('x', 0) for node in nodes]) + 100
            max_y = max([node.get('y', 0) for node in nodes]) + 100
            min_x = min([node.get('x', 0) for node in nodes]) - 100
            min_y = min([node.get('y', 0) for node in nodes]) - 100
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Scale to fit on page
            scale = min(6*inch/width, 8*inch/height)
            
            drawing = Drawing(width*scale, height*scale)
            
            # Draw connections first (so they're behind nodes)
            connections = mind_map.get('connections', [])
            for conn in connections:
                from_node = next((n for n in nodes if n.get('concept_name') == conn.get('from')), None)
                to_node = next((n for n in nodes if n.get('concept_name') == conn.get('to')), None)
                
                if from_node and to_node:
                    x1 = (from_node.get('x', 0) - min_x) * scale
                    y1 = (from_node.get('y', 0) - min_y) * scale
                    x2 = (to_node.get('x', 0) - min_x) * scale
                    y2 = (to_node.get('y', 0) - min_y) * scale
                    
                    line = Line(x1, y1, x2, y2)
                    line.strokeColor = HexColor('#BDC3C7')
                    line.strokeWidth = 1
                    drawing.add(line)
            
            # Draw nodes
            for node in nodes:
                x = (node.get('x', 0) - min_x) * scale
                y = (node.get('y', 0) - min_y) * scale
                size = node.get('size', 1.0) * 20 * scale
                
                # Node circle
                circle = Circle(x, y, size)
                circle.fillColor = HexColor(node.get('color', '#E74C3C'))
                circle.strokeColor = black
                circle.strokeWidth = 2
                drawing.add(circle)
                
                # Node label (concept name)
                label = String(x, y - size - 10, node.get('concept_name', ''))
                label.fontSize = 8 * scale
                label.fontName = 'Helvetica-Bold'
                label.textAnchor = 'middle'
                drawing.add(label)
            
            return drawing
            
        except Exception as e:
            print(f"❌ [PDFGenerator] Error in mind map drawing: {e}")
            return None
    
    def _create_mind_map_text(self, mind_map: Dict[str, Any]) -> Paragraph:
        """Create text representation of mind map as fallback"""
        nodes = mind_map.get('nodes', [])
        connections = mind_map.get('connections', [])
        
        text = f"<b>{mind_map.get('title', 'Concept Map')}</b><br/><br/>"
        
        # List nodes
        text += "<b>Concepts:</b><br/>"
        for node in nodes:
            importance = " (Core)" if node.get('importance') == "core" else ""
            text += f"• {node.get('concept_name', 'Unknown')}{importance}<br/>"
        
        # List connections
        if connections:
            text += "<br/><b>Relationships:</b><br/>"
            for conn in connections:
                text += f"• {conn.get('from', 'Unknown')} → {conn.get('to', 'Unknown')}<br/>"
        
        return Paragraph(text, self.styles['BodyText'])
    
    def _create_clinical_pearls(self, analysis: Any) -> List:
        """Create clinical pearls section"""
        elements = []
        
        pearls_title = Paragraph("Clinical Pearls", self.styles['SectionHeader'])
        elements.append(pearls_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'clinical_pearls') and analysis.clinical_pearls:
            for i, pearl in enumerate(analysis.clinical_pearls, 1):
                pearl_text = f"<b>💎 Pearl {i}:</b> {pearl}"
                pearl_para = Paragraph(pearl_text, self.styles['ClinicalPearl'])
                elements.append(pearl_para)
                elements.append(Spacer(1, 0.2*inch))
        else:
            no_pearls = Paragraph("Clinical pearls will be extracted from the lecture content.", 
                                self.styles['BodyText'])
            elements.append(no_pearls)
        
        return elements
    
    def _create_knowledge_gaps_section(self, analysis: Any) -> List:
        """Create knowledge gaps and filled content section"""
        elements = []
        
        gaps_title = Paragraph("Knowledge Gaps & Expert Explanations", self.styles['SectionHeader'])
        elements.append(gaps_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'filled_gaps') and analysis.filled_gaps:
            intro_text = "The AI has identified and filled knowledge gaps to provide comprehensive understanding:"
            intro_para = Paragraph(intro_text, self.styles['BodyText'])
            elements.append(intro_para)
            elements.append(Spacer(1, 0.3*inch))
            
            for i, gap_info in enumerate(analysis.filled_gaps, 1):
                # Gap description
                gap_text = f"<b>Gap {i}:</b> {gap_info.get('gap', 'Unknown gap')}"
                gap_para = Paragraph(gap_text, self.styles['BodyText'])
                elements.append(gap_para)
                
                # Explanation
                if gap_info.get('explanation'):
                    expl_text = f"<b>Expert Explanation:</b> {gap_info.get('explanation')}"
                    expl_para = Paragraph(expl_text, self.styles['BodyText'])
                    elements.append(expl_para)
                
                # Clinical relevance
                if gap_info.get('clinical_relevance'):
                    clin_text = f"<b>Clinical Relevance:</b> {gap_info.get('clinical_relevance')}"
                    clin_para = Paragraph(clin_text, self.styles['BodyText'])
                    elements.append(clin_para)
                
                elements.append(Spacer(1, 0.4*inch))
        else:
            no_gaps = Paragraph("Knowledge gaps will be identified and filled to enhance understanding.", 
                              self.styles['BodyText'])
            elements.append(no_gaps)
        
        return elements
    
    def _create_glossary(self, analysis: Any) -> List:
        """Create medical glossary section"""
        elements = []
        
        glossary_title = Paragraph("Medical Glossary", self.styles['SectionHeader'])
        elements.append(glossary_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'glossary') and analysis.glossary:
            # Sort glossary alphabetically
            sorted_terms = sorted(analysis.glossary.items())
            
            for term, definition in sorted_terms:
                term_text = f"<b>{term}</b>: {definition}"
                term_para = Paragraph(term_text, self.styles['GlossaryTerm'])
                elements.append(term_para)
                elements.append(Spacer(1, 0.2*inch))
        else:
            no_glossary = Paragraph("Medical terminology glossary will be generated from the lecture content.", 
                                  self.styles['BodyText'])
            elements.append(no_glossary)
        
        return elements
    
    def _create_cross_references(self, analysis: Any) -> List:
        """Create cross-references section"""
        elements = []
        
        crossref_title = Paragraph("Cross-References & Connections", self.styles['SectionHeader'])
        elements.append(crossref_title)
        elements.append(Spacer(1, 0.3*inch))
        
        if hasattr(analysis, 'cross_references') and analysis.cross_references:
            intro_text = "Key connections between different topics in this lecture:"
            intro_para = Paragraph(intro_text, self.styles['BodyText'])
            elements.append(intro_para)
            elements.append(Spacer(1, 0.3*inch))
            
            for crossref in analysis.cross_references:
                ref_text = f"<b>{crossref.get('from_topic', 'Unknown')}</b> → "
                ref_text += f"<b>{crossref.get('to_topic', 'Unknown')}</b>: "
                ref_text += f"{crossref.get('explanation', 'Related concept')}"
                
                ref_para = Paragraph(ref_text, self.styles['BodyText'])
                elements.append(ref_para)
                elements.append(Spacer(1, 0.2*inch))
        else:
            no_crossref = Paragraph("Cross-references will be identified to show concept relationships.", 
                                  self.styles['BodyText'])
            elements.append(no_crossref)
        
        return elements

def generate_medical_notes_pdf(holistic_analysis: Any, output_path: str = "medical_notes.pdf") -> str:
    """
    Convenience function to generate medical notes PDF
    
    Args:
        holistic_analysis: HolisticAnalysis object
        output_path: Path for output PDF
        
    Returns:
        Path to generated PDF
    """
    generator = MedicalNotesPDFGenerator(output_path)
    return generator.generate_comprehensive_notes(holistic_analysis)
