import argparse
import os
import yaml
from dotenv import load_dotenv
from pptx import Presentation
from openai import OpenAI
from fpdf import FPDF
import base64
import requests
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import logging
import genanki

def tuple_to_dict(card, use_cloze=False):
    # Accepts a Flashcard object or tuple or dict
    # Add debug print for first 3 calls
    if not hasattr(tuple_to_dict, 'call_count'):
        tuple_to_dict.call_count = 0
    tuple_to_dict.call_count += 1
    result = None
    if isinstance(card, dict):
        result = card
    elif hasattr(card, 'is_cloze') and hasattr(card, 'cloze_text'):
        if getattr(card, 'is_cloze', False) and getattr(card, 'cloze_text', ''):
            result = {"question": card.cloze_text, "answer": "", "type": "cloze"}
        else:
            result = {"question": card.question, "answer": card.answer, "type": "basic"}
    elif isinstance(card, tuple):
        q, a = card if len(card) == 2 else (card[0], "")
        if "{{c" in q:
            result = {"question": q, "answer": "", "type": "cloze"}
        else:
            result = {"question": q, "answer": a, "type": "basic"}
    else:
        result = {"question": str(card), "answer": "", "type": "basic"}
    if tuple_to_dict.call_count <= 3:
        print(f"[DEBUG][tuple_to_dict] card: {card}, result: {result}")
    return result

# Import semantic processing
from semantic_processor import SemanticProcessor

# =====================
# Configuration Section
# =====================
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load config from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PROMPT_TEMPLATE = config.get('prompt_template')
MODEL_NAME = config.get('model_name', 'gpt-4o')
MAX_TOKENS = config.get('max_tokens', 300)
TEMPERATURE = config.get('temperature', 0.7)

# User preferences
CATEGORY = config.get('category', 'Other')
EXAM = config.get('exam', 'Other')
ORGANISATION = config.get('organisation', {})
FEATURES = config.get('features', {})
FLASHCARD_TYPE = config.get('flashcard_type', {})
ANSWER_FORMAT = config.get('answer_format', 'best')
CLOZE = config.get('cloze', 'dont_mind')

# Semantic processing configuration
SEMANTIC_CONFIG = config.get('semantic_processing', {
    'enabled': True,
    'chunk_size': 500,
    'overlap': 50,
    'similarity_threshold': 0.7,
    'embedding_model': 'tfidf'
})

# Initialize semantic processor
semantic_processor = None
if SEMANTIC_CONFIG.get('enabled', True):
    try:
        semantic_processor = SemanticProcessor(
            model_name=SEMANTIC_CONFIG.get('embedding_model', 'tfidf')
        )
        print("✅ Semantic processing initialized successfully (TF-IDF mode)")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize semantic processing: {e}")
        print("Falling back to basic processing mode")
        semantic_processor = None

# =====================
# Utility Functions
# =====================

def parse_args():
    parser = argparse.ArgumentParser(description='Convert PowerPoint slides to Anki flashcards and extra materials.')
    parser.add_argument('pptx_path', help='Path to the PowerPoint (.pptx) file')
    parser.add_argument('--output', default='flashcards.csv', help='Output CSV file for Anki import')
    parser.add_argument('--notes', default='lecture_notes.pdf', help='Output PDF file for extra materials')
    parser.add_argument('--use_cloze', action='store_true', help='Use cloze cards in generation')
    return parser.parse_args()


def extract_text_from_pptx(pptx_path):
    """Extract text and speaker notes from all slides in a PowerPoint file."""
    try:
        prs = Presentation(pptx_path)
        slides_data = []
        for i, slide in enumerate(prs.slides):
            slide_content = [f"Slide {i+1}:"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text.strip())
            slide_text = "\n".join(slide_content) if len(slide_content) > 1 else ""
            # Extract speaker notes if present
            notes_text = ""
            if hasattr(slide, "notes_slide") and slide.notes_slide:
                notes_frame = getattr(slide.notes_slide, "notes_text_frame", None)
                if notes_frame and notes_frame.text:
                    notes_text = notes_frame.text.strip()
            slides_data.append({
                "slide_text": slide_text,
                "notes_text": notes_text
            })
        print(f"Extracted text and notes from {len(slides_data)} slides")
        return slides_data
    except Exception as e:
        print(f"Error reading PowerPoint file: {e}")
        return []


def extract_images_from_pptx(pptx_path, output_dir="slide_images"):
    prs = Presentation(pptx_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    slide_images = []
    for i, slide in enumerate(prs.slides):
        images_for_slide = []
        for j, shape in enumerate(slide.shapes):
            if shape.shape_type == 13:  # 13 = PICTURE
                image = shape.image
                image_bytes = image.blob
                ext = image.ext
                image_filename = f"slide{i+1}_img{j+1}.{ext}"
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                images_for_slide.append(image_path)
        slide_images.append(images_for_slide)
    print(f"Extracted images for {len(slide_images)} slides (some may be empty if no images on slide)")
    return slide_images


def stringify_dict(d):
    """Convert a dictionary of options to a readable string for the prompt."""
    if not isinstance(d, dict):
        return str(d)
    return ', '.join([k.replace('_', ' ').capitalize() for k, v in d.items() if v]) or 'None'


def build_extra_materials_prompt(slide_texts, features):
    sections = []
    if features.get('topic_map'):
        sections.append("a topic map (outline of main lecture themes)")
    if features.get('index'):
        sections.append("an index (list of all flashcards)")
    if features.get('glossary'):
        sections.append("a glossary of key terms (with brief definitions)")
    if features.get('summary_review_sheet'):
        sections.append("a summary review sheet (compressed revision version of the whole lecture)")
    if not sections:
        return None  # No extra materials requested
    prompt = (
        "You are an expert medical educator.\n"
        f"Category: {CATEGORY}\nExam: {EXAM}\n"
        "For the following slides, generate:\n- " + "\n- ".join(sections) +
        "\nFormat each section with a clear heading.\n" + "\n\n".join(slide_texts)
    )
    return prompt


def call_api_for_extra_materials(prompt, api_key, model, max_tokens, temperature):
    """Call OpenAI API for extra materials (glossary, etc.)"""
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code != 200:
            return f"__API_ERROR__Status {response.status_code}: {response.text}"
        response_json = response.json()
        if "choices" not in response_json or not response_json["choices"]:
            return "__API_ERROR__Invalid response format"
        return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__API_ERROR__{e}"


def parse_flashcards(ai_response, use_cloze=False, slide_number=0, level=1, quality_controller=None):
    flashcards = []
    if ai_response.startswith("__API_ERROR__"):
        return flashcards, ai_response[len("__API_ERROR__"):]
    if quality_controller is None:
        from flashcard_generator import QualityController
        quality_controller = QualityController()
    print(f"DEBUG: Parsing AI response of length {len(ai_response)}")
    print(f"DEBUG: First 200 chars: {ai_response[:200]}")
    # Patterns as before...
    pattern_strict = re.compile(r"Question:\s*(.*?)\s*\|\s*Answer:\s*(.*)")
    pattern_qa = re.compile(r"Question:?\s*(.*?)\s*\nAnswer:?\s*(.*?)(?:\n|$)", re.IGNORECASE)
    pattern_md = re.compile(r"\*\*Question:?\*\*\s*(.*?)\s*\n\*\*Answer:?\*\*\s*(.*?)(?:\n|$)", re.IGNORECASE)
    pattern_numbered = re.compile(r"\d+\.\s*Question:\s*(.*?)\s*\n\s*Answer:\s*(.*?)(?=\n\d+\.|$)", re.DOTALL | re.IGNORECASE)
    pattern_simple_numbered = re.compile(r"\d+\.\s*(.*?)\s*\n\s*(.*?)(?=\n\d+\.|$)", re.DOTALL)
    pattern_oneline = re.compile(r"Question:\s*(.*?)\s*\|\s*Answer:\s*(.*?)(?=\nQuestion:|$)", re.DOTALL)
    matches = list(pattern_strict.finditer(ai_response))
    for match in matches:
        question, answer = match.groups()
        if question.strip() and answer.strip():
            flashcards.append((question.strip(), answer.strip()))
    if not flashcards:
        matches = list(pattern_oneline.finditer(ai_response))
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip():
                flashcards.append((question.strip(), answer.strip()))
    if not flashcards:
        matches = list(pattern_numbered.finditer(ai_response))
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip():
                flashcards.append((question.strip(), answer.strip()))
    if not flashcards:
        matches = list(pattern_simple_numbered.finditer(ai_response))
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip() and is_medical_content(question, answer):
                flashcards.append((question.strip(), answer.strip()))
    if not flashcards:
        matches = list(pattern_md.finditer(ai_response))
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip() and is_medical_content(question, answer):
                flashcards.append((question.strip(), answer.strip()))
    if not flashcards:
        matches = list(pattern_qa.finditer(ai_response))
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip() and is_medical_content(question, answer):
                flashcards.append((question.strip(), answer.strip()))
    print(f"DEBUG: Total flashcards parsed: {len(flashcards)}")
    # Convert to Flashcard objects and apply cloze if needed
    flashcard_objs = []
    for q, a in flashcards:
        is_cloze = False
        cloze_text = ""
        if use_cloze:
            is_cloze, cloze_text = quality_controller.detect_cloze_opportunities(q, a)
        flashcard_objs.append(Flashcard(
            question=q,
            answer=a,
            level=level,
            slide_number=slide_number,
            is_cloze=is_cloze and use_cloze,
            cloze_text=cloze_text if is_cloze and use_cloze else ""
        ))
    return flashcard_objs, None


def is_medical_content(question, answer):
    """
    Filter out non-medical content like "setting the scene" slides.
    Returns True if the content appears to be medical/exam-relevant.
    """
    # Convert to lowercase for easier matching
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Skip obvious non-medical content
    skip_patterns = [
        # Generic lecture framework questions
        r"what are the key questions",
        r"what does the.*question focus on",
        r"how does the.*question contribute",
        r"when should the.*question be applied",
        r"what is the purpose of asking",
        
        # Generic analysis frameworks
        r"what\? how\? why\? when\?",
        r"key questions to consider",
        r"medical analysis",
        r"medical decision-making",
        
        # Lecture navigation content
        r"title slide",
        r"contents slide",
        r"lecture outline",
        r"learning objectives",
        r"introduction",
        
        # Too vague or generic
        r"what is.*important",
        r"why is.*important",
        r"how does.*work",
        r"what happens when",
        
        # Lecturer-specific content
        r"lecturer",
        r"professor",
        r"dr\.",
        r"dr ",
        
        # Context-dependent language (references to slides, diagrams, etc.)
        r"mentioned in.*slide",
        r"described in.*slide",
        r"listed on.*slide",
        r"as shown in",
        r"as seen in",
        r"in the slide",
        r"from the slide",
        r"on the slide",
        r"the slide",
        r"this slide",
        r"that slide",
        r"the diagram",
        r"this diagram",
        r"that diagram",
        r"the chart",
        r"this chart",
        r"that chart",
        r"the graph",
        r"this graph",
        r"that graph",
        r"the image",
        r"this image",
        r"that image",
        r"the figure",
        r"this figure",
        r"that figure",
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, q_lower) or re.search(pattern, a_lower):
            return False
    
    # Must contain medical/clinical terms to be considered relevant
    medical_terms = [
        r"drug", r"medication", r"therapy", r"treatment", r"diagnosis", r"symptom",
        r"disease", r"condition", r"syndrome", r"mechanism", r"receptor", r"enzyme",
        r"protein", r"cell", r"tissue", r"organ", r"system", r"function", r"action",
        r"effect", r"side effect", r"adverse", r"contraindication", r"indication",
        r"dose", r"dosage", r"administration", r"metabolism", r"excretion",
        r"pharmacokinetics", r"pharmacodynamics", r"interaction", r"toxicity",
        r"efficacy", r"potency", r"selectivity", r"specificity", r"affinity",
        r"agonist", r"antagonist", r"inhibitor", r"activator", r"modulator",
        r"pathway", r"cascade", r"signal", r"transduction", r"regulation",
        r"homeostasis", r"balance", r"equilibrium", r"threshold", r"baseline",
        r"normal", r"abnormal", r"pathological", r"physiological", r"clinical",
        r"patient", r"case", r"presentation", r"history", r"examination",
        r"investigation", r"test", r"result", r"finding", r"observation",
        r"assessment", r"evaluation", r"management", r"care", r"monitoring",
        r"follow-up", r"outcome", r"prognosis", r"complication", r"risk",
        r"factor", r"etiology", r"pathogenesis", r"pathophysiology", r"anatomy",
        r"physiology", r"biochemistry", r"molecular", r"genetic", r"immunology",
        r"microbiology", r"infection", r"bacteria", r"virus", r"fungus",
        r"parasite", r"antibiotic", r"antiviral", r"antifungal", r"vaccine",
        r"immunization", r"allergy", r"hypersensitivity", r"autoimmune",
        r"inflammation", r"injury", r"trauma", r"surgery", r"procedure",
        r"technique", r"method", r"approach", r"strategy", r"protocol",
        r"guideline", r"recommendation", r"standard", r"practice", r"policy"
    ]
    
    # Check if question or answer contains medical terms
    for term in medical_terms:
        if re.search(term, q_lower) or re.search(term, a_lower):
            return True
    
    # If no medical terms found, it's likely not medical content
    return False


def export_flashcards_to_apkg(flashcards, output_path='flashcards.apkg'):
    BASIC_MODEL = genanki.Model(
        1607392319,
        'Basic (My Generated Deck)',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ],
    )

    CLOZE_MODEL = genanki.Model(
        998877661,
        'Cloze (My Generated Deck)',
        fields=[
            {'name': 'Text'},
        ],
        templates=[
            {
                'name': 'Cloze Card',
                'qfmt': '{{cloze:Text}}',
                'afmt': '{{cloze:Text}}',
            },
        ],
        model_type=genanki.Model.CLOZE,
    )

    deck = genanki.Deck(2059400110, 'My Generated Deck')

    for card in flashcards:
        # Accept both Flashcard objects and dicts
        if hasattr(card, 'is_cloze') and hasattr(card, 'cloze_text'):
            is_cloze = getattr(card, 'is_cloze', False)
            cloze_text = getattr(card, 'cloze_text', '')
            question = getattr(card, 'question', '')
            answer = getattr(card, 'answer', '')
        elif isinstance(card, dict):
            is_cloze = card.get('type', 'basic') == 'cloze'
            cloze_text = card.get('question', '')
            question = card.get('question', '')
            answer = card.get('answer', '')
        else:
            # fallback: treat as basic
            is_cloze = False
            cloze_text = ''
            question = str(card)
            answer = ''

        if is_cloze and cloze_text:
            note = genanki.Note(
                model=CLOZE_MODEL,
                fields=[cloze_text],
            )
        else:
            note = genanki.Note(
                model=BASIC_MODEL,
                fields=[question, answer],
            )
        deck.add_note(note)

    genanki.Package(deck).write_to_file(output_path)


def save_text_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, line, ln=True)
    pdf.output(filename)


def generate_multimodal_flashcards_http(slide_texts, slide_images, api_key, model, max_tokens, temperature):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    all_flashcards = []
    for i, (slide_text, image_paths) in enumerate(zip(slide_texts, slide_images)):
        content_blocks = [
            {"type": "text", "text": PROMPT_TEMPLATE.format(
                category=CATEGORY,
                exam=EXAM,
                organisation=stringify_dict(ORGANISATION),
                features=stringify_dict(FEATURES),
                flashcard_type=stringify_dict(FLASHCARD_TYPE),
                answer_format=ANSWER_FORMAT,
                cloze=CLOZE,
                batch_start=i+1,
                batch_end=i+1,
                batch_text=slide_text
            )},
            {"type": "text", "text": slide_text}
        ]
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                ext = os.path.splitext(image_path)[1][1:]  # e.g., 'png', 'jpg'
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{base64_image}"}
                })
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": content_blocks}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code} for slide {i+1}")
                print("Response text:", response.text)
                continue
            try:
                response_json = response.json()
            except Exception as e:
                print(f"Error decoding JSON for slide {i+1}: {e}")
                print("Raw response:", response.text)
                continue
            if "choices" not in response_json or not response_json["choices"]:
                print(f"Error: 'choices' key missing or empty in API response for slide {i+1}")
                print("Full response:", response_json)
                continue
            ai_content = response_json["choices"][0]["message"]["content"]
            print(f"\n--- AI RAW RESPONSE FOR SLIDE {i+1} ---\n", ai_content, "\n----------------------\n")
            # Parse flashcards from the response
            flashcards, _ = parse_flashcards(ai_content)
            print(f"Generated {len(flashcards)} flashcards from slide {i+1}")
            all_flashcards.extend(flashcards)
        except Exception as e:
            print(f"Error generating flashcards for slide {i+1}: {e}")
    return all_flashcards


def filter_slides(slide_texts):
    """
    Filter out slides that are likely to be empty or contain only navigation content.
    Returns a list of slide texts that should be processed.
    """
    filtered_slides = []
    for i, slide_text in enumerate(slide_texts):
        # Skip slides that are likely to be empty or navigation
        if should_skip_slide(slide_text):
            print(f"Skipping slide {i+1} (likely empty or navigation content)")
            continue
        filtered_slides.append(slide_text)
    return filtered_slides


def should_skip_slide(slide_text):
    """
    Determine if a slide should be skipped based on its content.
    Returns True if the slide should be skipped.
    """
    text_lower = slide_text.lower()
    
    # Skip patterns for slides that should be ignored
    skip_patterns = [
        # Empty or nearly empty slides
        r"^slide \d+:\s*$",
        r"^slide \d+:\s*\n\s*$",
        
        # Title slides
        r"^slide \d+:\s*[a-z\s]+\s*$",  # Just a title
        
        # Contents/navigation slides
        r"contents",
        r"outline",
        r"agenda",
        r"learning objectives",
        r"objectives",
        r"introduction",
        r"overview",
        r"summary",
        r"conclusion",
        r"references",
        r"bibliography",
        r"further reading",
        r"questions",
        r"discussion",
        
        # Generic framework slides
        r"what\? how\? why\? when\?",
        r"key questions",
        r"framework",
        r"approach",
        r"methodology",
        
        # Lecturer information
        r"lecturer",
        r"professor",
        r"dr\.",
        r"dr ",
        r"presented by",
        r"by ",
        
        # Too short to be meaningful
        r"^slide \d+:\s*\w+\s*$",  # Just one word
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # If slide has very little content (less than 50 characters excluding "Slide X:")
    slide_match = re.match(r"^slide \d+:", slide_text)
    slide_header_length = len(slide_match.group(0)) if slide_match else 0
    content_length = len(slide_text) - slide_header_length
    if content_length < 50:
        return True
    
    return False

# =====================
# Enhanced Flashcard Generation with Semantic Processing
# =====================

def generate_flashcards_from_semantic_chunks(semantic_chunks, slide_images, api_key, model, max_tokens, temperature, progress=None, use_cloze=False):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    all_flashcards = []
    total_chunks = len(semantic_chunks)
    quality_controller = QualityController()
    for i, chunk_data in enumerate(semantic_chunks):
        if progress:
            progress_percent = 0.4 + (0.45 * (i / total_chunks))
            progress(progress_percent, desc=f"Processing semantic chunk {i+1}/{total_chunks}...")
        if semantic_processor is not None:
            enhanced_prompt = semantic_processor.build_enhanced_prompt(chunk_data, PROMPT_TEMPLATE)
        else:
            enhanced_prompt = PROMPT_TEMPLATE.format(
                category=CATEGORY,
                exam=EXAM,
                organisation=stringify_dict(ORGANISATION),
                features=stringify_dict(FEATURES),
                flashcard_type=stringify_dict(FLASHCARD_TYPE),
                answer_format=ANSWER_FORMAT,
                cloze=CLOZE,
                batch_start=chunk_data.get('slide_index', 0) + 1,
                batch_end=chunk_data.get('slide_index', 0) + 1,
                batch_text=chunk_data['text']
            )
        content_blocks = [
            {"type": "text", "text": enhanced_prompt}
        ]
        slide_index = chunk_data['slide_index']
        if slide_index < len(slide_images):
            image_paths = slide_images[slide_index]
            for image_path in image_paths:
                try:
                    with open(image_path, "rb") as image_file:
                        ext = os.path.splitext(image_path)[1][1:]
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{base64_image}"}
                        })
                except Exception as e:
                    print(f"Warning: Could not process image {image_path}: {e}")
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": content_blocks}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code} for chunk {i+1}")
                print("Response text:", response.text)
                continue
            try:
                response_json = response.json()
            except Exception as e:
                print(f"Error decoding JSON for chunk {i+1}: {e}")
                print("Raw response:", response.text)
                continue
            if "choices" not in response_json or not response_json["choices"]:
                print(f"Error: 'choices' key missing or empty in API response for chunk {i+1}")
                print("Full response:", response_json)
                continue
            ai_content = response_json["choices"][0]["message"]["content"]
            print(f"\n--- AI RAW RESPONSE FOR SEMANTIC CHUNK {i+1} ---\n", ai_content, "\n----------------------\n")
            # Parse flashcards from the response as Flashcard objects
            flashcard_objs, _ = parse_flashcards(
                ai_content,
                use_cloze=use_cloze,
                slide_number=slide_index,
                level=1,  # Default to level 1; can be improved if level info is available
                quality_controller=quality_controller
            )
            print(f"Generated {len(flashcard_objs)} flashcards from semantic chunk {i+1}")
            all_flashcards.extend(flashcard_objs)
            if progress:
                total_generated = len(all_flashcards)
                progress(progress_percent, desc=f"Chunk {i+1}/{total_chunks}: Generated {len(flashcard_objs)} cards (Total: {total_generated})")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error generating flashcards for semantic chunk {i+1}: {e}")
    return all_flashcards

def generate_enhanced_flashcards_with_progress(slide_texts, slide_images, api_key, model, max_tokens, temperature, progress=None, use_cloze=False):
    """
    Generate flashcards using semantic processing with progress tracking
    
    Args:
        slide_texts: List of slide texts
        slide_images: List of image paths for each slide
        api_key: OpenAI API key
        model: Model name
        max_tokens: Max tokens for generation
        temperature: Temperature for generation
        progress: Progress callback function
        use_cloze: Boolean indicating whether to use cloze cards
        
    Returns:
        Tuple of (flashcards, analysis_data)
    """
    if semantic_processor is None:
        print("⚠️ Semantic processing not available, falling back to basic processing")
        return generate_multimodal_flashcards_http(slide_texts, slide_images, api_key, model, max_tokens, temperature), None
    
    try:
        # Step 1: Create semantic chunks
        if progress:
            progress(0.4, desc="Creating semantic chunks and analyzing content...")
        
        semantic_chunks = semantic_processor.create_semantic_chunks(slide_texts)
        
        # Step 2: Analyze content quality
        analysis_data = semantic_processor.analyze_content_quality(semantic_chunks)
        
        print(f"📊 Content Analysis:")
        print(f"   • Total chunks: {analysis_data['total_chunks']}")
        print(f"   • Total slides: {analysis_data['total_slides']}")
        print(f"   • Average chunk size: {analysis_data['avg_chunk_size']:.1f} characters")
        print(f"   • Unique key phrases: {analysis_data['unique_key_phrases']}")
        print(f"   • Average group size: {analysis_data['avg_group_size']:.1f}")
        print(f"   • Key phrases: {', '.join(analysis_data['key_phrases'][:5])}")
        
        # Step 3: Generate flashcards from semantic chunks
        all_flashcards = generate_flashcards_from_semantic_chunks(
            semantic_chunks, slide_images, api_key, model, max_tokens, temperature, progress, use_cloze
        )
        
        print("\nSample of generated flashcards (before export):")
        for card in all_flashcards[:10]:
            print(card)
        
        return all_flashcards, analysis_data
        
    except Exception as e:
        print(f"Error in enhanced flashcard generation: {e}")
        print("Falling back to basic processing")
        return generate_multimodal_flashcards_http(slide_texts, slide_images, api_key, model, max_tokens, temperature), None

def remove_duplicate_flashcards(flashcards, similarity_threshold=0.8):
    """
    Remove duplicate or very similar flashcards using semantic similarity
    
    Args:
        flashcards: List of (question, answer) tuples
        similarity_threshold: Threshold for considering flashcards similar
        
    Returns:
        List of deduplicated flashcards
    """
    if not flashcards or semantic_processor is None:
        return flashcards
    
    try:
        # Extract questions and answers
        questions = [q for q, a in flashcards]
        answers = [a for q, a in flashcards]
        
        # Compute TF-IDF embeddings for questions
        question_embeddings = semantic_processor.vectorizer.fit_transform(questions).toarray()
        
        # Find similar questions
        similarity_matrix = cosine_similarity(question_embeddings)
        
        # Find duplicates
        duplicates = set()
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                if similarity_matrix[i][j] > similarity_threshold:
                    # Keep the one with longer answer (more detailed)
                    if len(answers[i]) < len(answers[j]):
                        duplicates.add(i)
                    else:
                        duplicates.add(j)
        
        # Remove duplicates
        deduplicated = [flashcards[i] for i in range(len(flashcards)) if i not in duplicates]
        
        print(f"🔄 Removed {len(flashcards) - len(deduplicated)} duplicate flashcards")
        return deduplicated
        
    except Exception as e:
        print(f"Warning: Could not remove duplicates: {e}")
        return flashcards

# =====================
# Main Orchestration
# =====================

def main():
    args = parse_args()
    if not OPENAI_API_KEY:
        print('Error: OPENAI_API_KEY not set in .env file')
        return
    if not os.path.exists(args.pptx_path):
        print(f'Error: PowerPoint file not found: {args.pptx_path}')
        return
    print(f"Processing PowerPoint file: {args.pptx_path}")
    slide_data = extract_text_from_pptx(args.pptx_path)
    # Prepend notes to slide text if present
    slide_texts = []
    for entry in slide_data:
        notes = entry.get('notes_text', '').strip()
        slide_text = entry.get('slide_text', '').strip()
        if notes:
            combined = f"[NOTES]\n{notes}\n[SLIDE]\n{slide_text}" if slide_text else f"[NOTES]\n{notes}"
        else:
            combined = slide_text
        slide_texts.append(combined)
    slide_images = extract_images_from_pptx(args.pptx_path)  # Extract images for each slide
    if not slide_texts:
        print("No text found in PowerPoint file or error occurred during extraction")
        return
    
    # Filter out slides that should be skipped
    print("Filtering slides to remove navigation and empty content...")
    filtered_slide_texts = filter_slides(slide_texts)
    print(f"Processing {len(filtered_slide_texts)} slides (filtered from {len(slide_texts)} total slides)")
    
    # Filter slide_images to match filtered_slide_texts
    # We need to keep track of which slides were kept
    kept_slide_indices = []
    for i, slide_text in enumerate(slide_texts):
        if not should_skip_slide(slide_text):
            kept_slide_indices.append(i)
    
    filtered_slide_images = [slide_images[i] for i in kept_slide_indices]
    
    # Use enhanced flashcard generation with semantic processing
    print("Generating enhanced flashcards with semantic processing...")
    # Add use_cloze flag from config or CLI if available
    use_cloze = getattr(args, 'use_cloze', False)
    all_flashcards, analysis_data = generate_enhanced_flashcards_with_progress(
        filtered_slide_texts, 
        filtered_slide_images, 
        OPENAI_API_KEY, 
        MODEL_NAME, 
        MAX_TOKENS, 
        TEMPERATURE,
        None,
        use_cloze=use_cloze
    )
    
    if analysis_data:
        print(f"\n📊 Enhanced Processing Results:")
        print(f"   • Semantic chunks created: {analysis_data['total_chunks']}")
        print(f"   • Content quality score: {analysis_data['unique_key_phrases']} unique medical terms")
        print(f"   • Average chunk size: {analysis_data['avg_chunk_size']:.1f} characters")
    
    # Bypass deduplication and tuple conversion for cloze export
    print(f"Total generated: {len(all_flashcards)} flashcards")
    all_flashcards = [tuple_to_dict(card, use_cloze=use_cloze) for card in all_flashcards]
    export_flashcards_to_apkg(all_flashcards)
    print(f'Success! {len(all_flashcards)} flashcards exported to flashcards.apkg')
    print(f'You can now import this file into Anki using File > Import')

    # Generate extra materials if requested
    extra_prompt = build_extra_materials_prompt(filtered_slide_texts, FEATURES)
    if extra_prompt:
        print("Generating extra materials (glossary, index, etc.) with AI...")
        extra_response = call_api_for_extra_materials(extra_prompt, OPENAI_API_KEY, MODEL_NAME, 1000, TEMPERATURE)
        if extra_response.startswith("__API_ERROR__"):
            print("Error generating extra materials:", extra_response)
        else:
            save_text_to_pdf(extra_response, args.notes)
            print(f'Success! Extra materials exported to {args.notes}')
    else:
        print("No extra materials requested by user preferences.")

if __name__ == '__main__':
    main() 

# =====================
# Quality Control Classes
# =====================

@dataclass
class Flashcard:
    question: str
    answer: str
    level: int
    slide_number: int
    confidence: float = 0.0
    is_cloze: bool = False
    cloze_text: str = ""

class QualityController:
    """Handles flashcard quality control and optimization"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key medical terms from text"""
        # Remove common words and extract medical terminology
        words = word_tokenize(text.lower())
        key_terms = [word for word in words if word not in self.stop_words and len(word) > 2]
        return key_terms
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            # Use TF-IDF for similarity calculation
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback to sequence matcher
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def detect_repetition(self, cards: List[Flashcard], threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Detect repetitive flashcards"""
        duplicates = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                # Check question similarity
                q_similarity = self.calculate_similarity(cards[i].question, cards[j].question)
                # Check answer similarity
                a_similarity = self.calculate_similarity(cards[i].answer, cards[j].answer)
                
                # If both question and answer are similar, mark as duplicate
                if q_similarity > threshold and a_similarity > threshold:
                    duplicates.append((i, j))
        
        return duplicates
    
    def is_too_wordy(self, text: str, max_sentences: int = 2, max_words: int = 25) -> bool:
        """Check if text is too wordy for a flashcard"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return len(sentences) > max_sentences or len(words) > max_words
    
    def split_wordy_answer(self, question: str, answer: str) -> List[Tuple[str, str]]:
        """Split a wordy answer into multiple focused cards"""
        sentences = sent_tokenize(answer)
        if len(sentences) <= 2:
            return [(question, answer)]
        
        # Split into multiple cards
        cards = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Only create cards for substantial sentences
                new_question = f"{question} (Part {i+1})"
                cards.append((new_question, sentence.strip()))
        
        return cards
    
    def is_shallow_card(self, question: str, answer: str, level: int) -> bool:
        """Detect if a card is too shallow for its level"""
        # Level 1 can be basic, but Level 2 should have reasoning
        if level == 1:
            # Check if it's just a definition without context
            definition_indicators = ['stands for', 'means', 'is defined as', 'refers to']
            has_context = any(indicator in question.lower() for indicator in ['why', 'how', 'what explains', 'what pattern'])
            return not has_context and any(indicator in question.lower() for indicator in definition_indicators)
        
        elif level == 2:
            # Level 2 should have reasoning words
            reasoning_indicators = ['why', 'how', 'explain', 'compare', 'interpret', 'pattern', 'suggests', 'indicates']
            return not any(indicator in question.lower() for indicator in reasoning_indicators)
        
        return False
    
    def enrich_shallow_card(self, question: str, answer: str, level: int) -> Tuple[str, str]:
        """Enrich a shallow card with more context"""
        if level == 1:
            # Add minimal context for Level 1
            if 'what does' in question.lower() and 'stand for' in question.lower():
                # Convert "What does X stand for?" to "What does X measure/represent?"
                question = question.replace('stand for', 'measure in clinical practice')
            elif 'what is' in question.lower() and len(answer.split()) < 5:
                # Add context for short definitions
                question = question.replace('What is', 'What is the clinical significance of')
        
        elif level == 2:
            # Add reasoning for Level 2
            if 'what is' in question.lower():
                question = question.replace('What is', 'What explains why')
            elif 'what does' in question.lower():
                question = question.replace('What does', 'What pattern does')
        
        return question, answer
    
    def detect_cloze_opportunities(self, question: str, answer: str) -> Tuple[bool, str]:
        """Detect if a card would be better as a cloze deletion"""
        # Patterns that work well as cloze
        cloze_patterns = [
            r'(\d+(?:\.\d+)?%?)',  # Numbers and percentages
            r'([A-Z]{2,}(?:\d+)?)',  # Acronyms like FEV1, FVC
            r'(\w+ (?:and|or) \w+)',  # Lists
            r'(\w+ (?:is|are) \w+)',  # Definitions
        ]
        
        for pattern in cloze_patterns:
            matches = re.findall(pattern, answer)
            if matches and len(matches) <= 3:  # Don't make clozes with too many blanks
                return True, self.create_cloze_text(answer, matches)
        
        return False, ""
    
    def create_cloze_text(self, answer: str, key_terms: List[str]) -> str:
        """Create cloze deletion text"""
        cloze_text = answer
        for i, term in enumerate(key_terms, 1):
            cloze_text = cloze_text.replace(term, f"{{{{c{i}::{term}}}}}", 1)
        return cloze_text
    
    def assess_depth_consistency(self, cards: List[Flashcard]) -> List[int]:
        """Assess if cards are at the right depth for their level"""
        inconsistent_cards = []
        
        for i, card in enumerate(cards):
            if card.level == 1 and self.is_too_deep_for_level1(card.question, card.answer):
                inconsistent_cards.append(i)
            elif card.level == 2 and self.is_too_shallow_for_level2(card.question, card.answer):
                inconsistent_cards.append(i)
        
        return inconsistent_cards
    
    def is_too_deep_for_level1(self, question: str, answer: str) -> bool:
        """Check if Level 1 card is too deep"""
        deep_indicators = ['pattern', 'interpret', 'compare', 'explain why', 'clinical reasoning', 'differential']
        return any(indicator in question.lower() for indicator in deep_indicators)
    
    def is_too_shallow_for_level2(self, question: str, answer: str) -> bool:
        """Check if Level 2 card is too shallow"""
        shallow_indicators = ['what does', 'stand for', 'define', 'name', 'list']
        return any(indicator in question.lower() for indicator in shallow_indicators)
    
    def fix_depth_inconsistency(self, card: Flashcard) -> Flashcard:
        """Fix depth inconsistency by adjusting level or content"""
        if card.level == 1 and self.is_too_deep_for_level1(card.question, card.answer):
            # Move to Level 2
            card.level = 2
        elif card.level == 2 and self.is_too_shallow_for_level2(card.question, card.answer):
            # Either move to Level 1 or enrich
            if len(card.answer.split()) < 10:
                card.level = 1
            else:
                # Enrich the question
                card.question, card.answer = self.enrich_shallow_card(card.question, card.answer, 2)
        
        return card

class EnhancedFlashcardGenerator:
    """Enhanced flashcard generator with quality control"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.quality_controller = QualityController()
        self.openai_client = OpenAI(api_key=self.config.get('openai_api_key'))
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return {}
    
    def generate_flashcards(self, slide_text: str, slide_number: int, level: int = 1, 
                          use_cloze: bool = False, progress_callback=None) -> List[Flashcard]:
        """Generate flashcards with quality control"""
        
        # Generate initial flashcards
        raw_cards = self._generate_raw_flashcards(slide_text, slide_number, level, use_cloze)
        
        if progress_callback:
            progress_callback(0.3, "Generated initial flashcards, applying quality control...")
        
        # Apply quality control
        quality_cards = self._apply_quality_control(raw_cards, use_cloze)
        
        if progress_callback:
            progress_callback(0.7, "Quality control applied, finalizing cards...")
        
        # Final optimization
        final_cards = self._finalize_cards(quality_cards)
        
        if progress_callback:
            progress_callback(1.0, f"Generated {len(final_cards)} quality-controlled flashcards")
        
        return final_cards
    
    def _generate_raw_flashcards(self, slide_text: str, slide_number: int, level: int, use_cloze: bool) -> List[Flashcard]:
        """Generate raw flashcards using OpenAI"""
        prompt = self.config.get('prompt_template', '').format(batch_text=slide_text)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return self._parse_flashcards(content, slide_number, level, use_cloze)
            
        except Exception as e:
            logging.error(f"Error generating flashcards: {e}")
            return []
    
    def _parse_flashcards(self, content: str, slide_number: int, level: int, use_cloze: bool) -> List[Flashcard]:
        """Parse flashcards from OpenAI response"""
        cards = []
        lines = content.split('\n')
        
        for line in lines:
            if '|' in line and 'Question:' in line and 'Answer:' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    question = parts[0].replace('Question:', '').strip()
                    answer = parts[1].replace('Answer:', '').strip()
                    
                    if question and answer:
                        # Check for cloze opportunities
                        is_cloze, cloze_text = self.quality_controller.detect_cloze_opportunities(question, answer)
                        
                        card = Flashcard(
                            question=question,
                            answer=answer,
                            level=level,
                            slide_number=slide_number,
                            is_cloze=is_cloze and use_cloze,
                            cloze_text=cloze_text if is_cloze and use_cloze else ""
                        )
                        cards.append(card)
        
        return cards
    
    def _apply_quality_control(self, cards: List[Flashcard], use_cloze: bool) -> List[Flashcard]:
        """Apply comprehensive quality control"""
        if not cards:
            return cards
        
        # 1. Remove duplicates
        duplicates = self.quality_controller.detect_repetition(cards)
        unique_cards = []
        duplicate_indices = set()
        
        for i, j in duplicates:
            duplicate_indices.add(j)
        
        for i, card in enumerate(cards):
            if i not in duplicate_indices:
                unique_cards.append(card)
        
        # 2. Fix wordy answers
        expanded_cards = []
        for card in unique_cards:
            if self.quality_controller.is_too_wordy(card.answer):
                split_cards = self.quality_controller.split_wordy_answer(card.question, card.answer)
                for q, a in split_cards:
                    new_card = Flashcard(
                        question=q,
                        answer=a,
                        level=card.level,
                        slide_number=card.slide_number,
                        is_cloze=card.is_cloze,
                        cloze_text=card.cloze_text
                    )
                    expanded_cards.append(new_card)
            else:
                expanded_cards.append(card)
        
        # 3. Fix shallow cards
        for card in expanded_cards:
            if self.quality_controller.is_shallow_card(card.question, card.answer, card.level):
                card.question, card.answer = self.quality_controller.enrich_shallow_card(
                    card.question, card.answer, card.level
                )
        
        # 4. Fix depth inconsistency
        for card in expanded_cards:
            card = self.quality_controller.fix_depth_inconsistency(card)
        
        # 5. Apply cloze if requested
        if use_cloze:
            for card in expanded_cards:
                # TEMPORARY: Force all cards to be cloze for testing
                card.is_cloze = True
                # Wrap the entire answer as a cloze deletion
                card.cloze_text = f"{{{{c1::{card.answer}}}}}"
        
        return expanded_cards
    
    def _finalize_cards(self, cards: List[Flashcard]) -> List[Flashcard]:
        """Final optimization and validation"""
        final_cards = []
        
        for card in cards:
            # Final validation
            if len(card.question.strip()) > 5 and len(card.answer.strip()) > 2:
                # Calculate confidence based on content quality
                card.confidence = self._calculate_confidence(card)
                final_cards.append(card)
        
        return final_cards
    
    def _calculate_confidence(self, card: Flashcard) -> float:
        """Calculate confidence score for a card"""
        confidence = 0.5  # Base confidence
        
        # Bonus for good question structure
        if any(word in card.question.lower() for word in ['what', 'why', 'how', 'compare', 'explain']):
            confidence += 0.2
        
        # Bonus for appropriate answer length
        if 5 <= len(card.answer.split()) <= 20:
            confidence += 0.2
        
        # Bonus for level appropriateness
        if card.level == 1 and not self.quality_controller.is_too_deep_for_level1(card.question, card.answer):
            confidence += 0.1
        elif card.level == 2 and not self.quality_controller.is_too_shallow_for_level2(card.question, card.answer):
            confidence += 0.1
        
        return min(confidence, 1.0) 