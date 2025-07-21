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
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import logging
import genanki
import urllib.request
import shutil
import csv
from utils.image_occlusion import batch_generate_image_occlusion_flashcards
from utils.image_occlusion import cleanup_files
import settings
from anki_models import IOE_MODEL

def summarize_card(card):
    if isinstance(card, dict):
        return f"[Card type: {card.get('type', 'unknown')}]"
    else:
        return f"[Card class: {type(card).__name__}, Question: {getattr(card, 'question', '')[:50]}...]"

# Import semantic processing
from semantic_processor import SemanticProcessor

# =====================
# Cloze Utility Helper
# =====================


def clean_cloze_text(text: str) -> str:
    # 1) Iteratively remove any outer {{cN::…}} wrapper:
    outer = re.compile(r'^{{c\d+::(.+)}}$')
    while True:
        m = outer.match(text)
        if not m:
            break
        text = m.group(1)
    # 2) Convert all remaining {{cN::…}} to [text]
    return re.sub(r'{{c\d+::(.*?)}}', r'[\1]', text)

# =====================
# Configuration Section
# =====================
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load config from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PROMPT_TEMPLATE = config.get('prompt_template', """You are an expert medical educator creating Anki flashcards for medical students.

Create concise, focused flashcards that test specific knowledge points. Follow these guidelines:

**Question Style:**
- Keep questions short and direct (1-2 lines max)
- Focus on one specific concept per card
- Use clear, medical terminology
- Avoid verbose explanations in questions

**Answer Style:**
- Provide direct, concise answers (1-3 lines max)
- Include key medical terms and definitions
- For lists, use bullet points or numbered items
- Avoid lengthy explanations unless absolutely necessary

**Examples of Good Questions:**
- "What condition is known for reduced DLCO due to Kco?"
- "What are two concerns for using DLCO for ILD patients?"
- "What stage of ILD is DLCO most important for?"

**Examples of Good Answers:**
- "Interstitial Lung Disease (ILD)"
- "• Falsely reduced in individuals who fail to inspire to TLC\n• Significant variation"
- "Early stage"

**Content Focus:**
- Test key medical concepts, mechanisms, and clinical applications
- Include important clinical signs, symptoms, and diagnostic criteria
- Cover drug mechanisms, side effects, and contraindications
- Test understanding of physiological processes and pathological changes

Generate {flashcard_type} flashcards from the following content:

{batch_text}

Format each flashcard as:
Question: [concise question]
Answer: [direct answer]

Create concise answers that are {cloze}.""")
MODEL_NAME = config.get('model_name', 'gpt-4o')
MAX_TOKENS = config.get('max_tokens', 2000)
TEMPERATURE = config.get('temperature', 0.3)

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
    """Extract all pictures from a PowerPoint file.

    The function now returns a list that has **exactly one entry per slide** so that
    downstream logic can safely `zip(slide_texts, slide_images)` without running
    out-of-sync.  Each entry is a list (possibly empty) of image paths extracted
    from that slide.
    """

    prs = Presentation(pptx_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    slide_images: List[List[str]] = []

    for i, slide in enumerate(prs.slides):
        images_for_slide: List[str] = []

        for j, shp in enumerate(slide.shapes):
            # Only process shapes that *have* an image attribute to avoid
            # AttributeError on non-picture shapes (e.g. GraphicFrame)
            if getattr(shp, "shape_type", None) == 13 and hasattr(shp, "image"):
                try:
                    pic = shp.image  # type: ignore[attr-defined]
                    image_bytes = pic.blob
                    ext = pic.ext
                    img_name = f"slide{i+1}_img{j+1}.{ext}"
                    img_path = os.path.join(output_dir, img_name)

                    with open(img_path, "wb") as f:
                        f.write(image_bytes)

                    images_for_slide.append(img_path)
                except Exception as e:
                    # Log and continue on any unexpected problem with this picture
                    print(f"⚠️  Skipped image on slide {i+1}: {e}")

        # Always append – even if the list is empty – so indices stay aligned
        slide_images.append(images_for_slide)

    print(
        f"Extracted images for {len(slide_images)} slides "
        f"(total images: {sum(len(lst) for lst in slide_images)})"
    )

    return slide_images


def stringify_dict(obj: Any, use_cloze: bool = False) -> str:
    """
    Convert a dict—or any object with attributes—to a readable string.
    If use_cloze is True, wraps each value in {{c1::…}} Anki cloze syntax.
    """
    # If obj isn't dict-like, fall back to its attributes
    if not hasattr(obj, "items"):
        try:
            obj = obj.__dict__
        except AttributeError:
            obj = vars(obj)

    items = []
    for key, value in obj.items():
        text = str(value)
        if use_cloze:
            text = f"{{{{c1::{text}}}}}"
        items.append(f"{key}: {text}")
    return "\n".join(items)


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
        print(f"⚠️ OpenAI API error: {ai_response[len('__API_ERROR__'):].strip()}")
        return [], None
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
            preview_text = clean_cloze_text(cloze_text)
        else:
            preview_text = q
        flashcard_objs.append(Flashcard(
            question=q,
            answer=a,
            level=level,
            slide_number=slide_number,
            is_cloze=is_cloze and use_cloze,
            cloze_text=cloze_text if is_cloze and use_cloze else "",
            preview_text=preview_text,
            raw_cloze_text=cloze_text  # store raw cloze text
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


def export_flashcards_to_apkg(flashcards, output_path='flashcards.apkg', pptx_filename=None, config_topic=None):
    import base64
    import os

    # Detect topic from filename or config
    if config_topic:
        topic = config_topic
    elif pptx_filename:
        topic = os.path.splitext(os.path.basename(pptx_filename))[0].replace('_', ' ').replace('-', ' ').title()
    else:
        topic = "Lecture"

    # Define Anki note models
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

    # Use a stable but fairly unique deck ID – hash the topic string
    deck_id = abs(hash(topic)) % 2147483647  # Anki requires 32-bit signed int
    deck = genanki.Deck(deck_id, topic)

    print("[DEBUG] export_flashcards_to_apkg() called")
    print(f"[DEBUG] Number of flashcards to export: {len(flashcards)}")
    print(f"[DEBUG] Flashcard types and counts: {[(type(card), card.get('type', 'unknown') if isinstance(card, dict) else 'N/A') for card in flashcards]}")

    print("\n===== Exporting Flashcards =====")
    media = []  # Collect image paths for the package

    # Lazy import of Flashcard class to avoid cyclic issues
    try:
        from flashcard_generator import Flashcard  # type: ignore
    except Exception:
        Flashcard = None  # Fallback – isinstance checks will fail gracefully

    for entry in flashcards:
        # -----------------------------
        # IMAGE-OCCLUSION DICT HANDLING
        # -----------------------------
        if isinstance(entry, dict):
            if 'question_image_path' in entry and 'answer_image_path' in entry:
                qfile = os.path.basename(entry['question_image_path'])
                afile = os.path.basename(entry['answer_image_path'])

                # Track media files
                media.extend([entry['question_image_path'], entry['answer_image_path']])

                note = genanki.Note(
                    model=IOE_MODEL,
                    fields=[f"<img src='{qfile}'>", f"<img src='{afile}'>"],
                )
                deck.add_note(note)
            else:
                print(f"[WARN] Skipping dict entry missing expected keys: {entry}")
            continue

        # -----------------------------
        # TEXT FLASHCARD OBJECTS
        # -----------------------------
        if Flashcard is not None and isinstance(entry, Flashcard):
            # Determine note model based on cloze status
            if getattr(entry, 'is_cloze', False):
                text_field = entry.cloze_text or f"{entry.question} – {entry.answer}"
                note = genanki.Note(model=CLOZE_MODEL, fields=[text_field])
            else:
                note = genanki.Note(model=BASIC_MODEL, fields=[entry.question, entry.answer])

            # If the Flashcard references an image_path, embed it for context
            img_path = getattr(entry, 'image_path', None)
            if img_path and os.path.exists(img_path):
                fname = os.path.basename(img_path)
                media.append(img_path)

                # Append image below the answer/cloze content.
                # For Basic model (2+ fields) -> use Answer field (index 1)
                if len(note.fields) >= 2:
                    note.fields[1] += f"<br><img src='{fname}'>"
                else:
                    # Cloze card – single field; append image to same field
                    note.fields[0] += f"<br><img src='{fname}'>"

            deck.add_note(note)
            continue

        # -----------------------------
        # UNKNOWN ENTRY TYPE
        # -----------------------------
        print(f"[WARN] Unrecognized flashcard entry – skipped: {entry}")

    files_to_cleanup: list[str] = []
    try:
        genanki.Package(deck, media_files=media).write_to_file(output_path)
    except Exception:
        cleanup_files(files_to_cleanup)
        raise
    print("[DEBUG] Finished adding flashcards, writing to file...")
    print(f"[DEBUG] Successfully wrote APKG file to: {output_path}")
    print(f"[SUCCESS] APKG exported to {output_path}")



def save_text_to_pdf(text, filename):
    # Use a Unicode font for full character support
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    font_path = os.path.join(font_dir, "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"DejaVuSans.ttf not found in {font_dir}. Please download it and place it in the fonts/ directory."
        )
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, line, ln=True)
    pdf.output(filename)


def generate_multimodal_flashcards_http(slide_texts, slide_images, api_key, model, max_tokens, temperature):
    import os, base64
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    all_flashcards = []

    for slide_index in range(len(slide_images)):
        # Per-slide initializations
        image_paths = slide_images[slide_index] if slide_index < len(slide_images) else []
        last_img_path = None
        question = ""
        answer = ""
        level = 1
        is_cloze = False
        cloze_text = ""
        content_blocks = [{"type": "text", "text": slide_texts[slide_index]}]

        # Embed each image and remember only the last path
        for img_path in image_paths:
            last_img_path = img_path
            try:
                with open(img_path, "rb") as f:
                    ext = os.path.splitext(img_path)[1][1:]
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                })
            except Exception as e:
                print(f"Warning: Could not process image {img_path}: {e}")

        # Build a single Flashcard per slide, using last_img_path (or None)
        flashcard = Flashcard(
            question=question,
            answer=answer,
            level=level,
            slide_number=slide_index,
            is_cloze=is_cloze,
            cloze_text=cloze_text,
            image_path=last_img_path
        )
        all_flashcards.append(flashcard)

    return all_flashcards




def filter_slides(slide_texts, slide_images):
    """
    Filter out slides that are likely to be empty or contain only navigation content.
    Returns a list of slide texts that should be processed and their indices.
    """
    filtered_slides = []
    kept_indices = []
    for i, (slide_text, images) in enumerate(zip(slide_texts, slide_images)):
        # Skip slides that are likely to be empty or navigation
        if should_skip_slide(slide_text, images):
            print(f"[DEBUG] Skipping slide {i+1}: no text, no images")
            continue
        print(f"[DEBUG] Keeping slide {i+1}: has text or images")
        filtered_slides.append(slide_text)
        kept_indices.append(i)
    return filtered_slides, kept_indices


def should_skip_slide(slide_text, images):
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
    
    # If slide has very little content (less than 50 characters excluding "Slide X:") and no images
    slide_match = re.match(r"^slide \d+:", slide_text)
    slide_header_length = len(slide_match.group(0)) if slide_match else 0
    content_length = len(slide_text) - slide_header_length
    if content_length < 50 and not images:
        return True
    
    return False

# =====================
# Enhanced Flashcard Generation with Semantic Processing
# =====================

def generate_flashcards_from_semantic_chunks(semantic_chunks, slide_images, api_key, model, max_tokens, temperature, progress=None, use_cloze=False, question_style="Word for word"):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    all_flashcards = []
    total_chunks = len(semantic_chunks)
    quality_controller = QualityController()
    for i, chunk_data in enumerate(semantic_chunks):
        image_paths = []
        last_img_path = None  # Track last image path for this chunk
        if progress:
            progress_percent = 0.4 + (0.45 * (i / total_chunks))
            progress(progress_percent, desc=f"Processing semantic chunk {i+1}/{total_chunks}...")
        if semantic_processor is not None:
            enhanced_prompt = semantic_processor.build_enhanced_prompt(chunk_data, PROMPT_TEMPLATE)
        else:
            # Add question style instructions to the prompt
            style_instructions = ""
            if question_style == "Word for word":
                style_instructions = """
 **Question Style: Word for word**
 - Keep answers in the exact same format as presented in the slide
 - Use the precise terminology and phrasing from the original content
 - Maintain the original structure and order of information
 - For Level 2 questions: Use reasoning questions but keep answers in exact slide format
 """
            elif question_style == "Elaborated":
                style_instructions = """
 **Question Style: Elaborated**
 - Keep answers in the same format as the slide content
 - Add explanations below the original answer to help clarify concepts
 - Provide additional context and clinical significance
 - Format: [Original answer] + [Explanation/context]
 - For Level 1: Keep explanations brief and focused on basic understanding
 - For Level 2: Add deeper clinical reasoning and implications
 """
            elif question_style == "Simplified":
                style_instructions = """
 **Question Style: Simplified**
 - Take complex explanations and convert them to concise, simple phrases
 - Use clear, straightforward language that's easy to understand
 - Break down complex concepts into digestible parts
 - Maintain accuracy while improving clarity
 - For Level 1: Focus on basic definitions and facts
 - For Level 2: Simplify complex reasoning and patterns
 """

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
            ) + style_instructions
        content_blocks = [
            {"type": "text", "text": enhanced_prompt}
        ]
        slide_index = chunk_data['slide_index']
        if 0 <= slide_index < len(slide_images):
            image_paths = slide_images[slide_index]
            for img_path in image_paths:
                last_img_path = img_path
                try:
                    with open(img_path, "rb") as image_file:
                        ext = os.path.splitext(img_path)[1][1:]
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{base64_image}"}
                        })
                except Exception as e:
                    print(f"Warning: could not read {img_path}: {e}")
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
            # Attach image_path information to each generated Flashcard
            for fc in flashcard_objs:
                fc.image_path = last_img_path
            print(f"Generated {len(flashcard_objs)} flashcards from semantic chunk {i+1}")
            all_flashcards.extend(flashcard_objs)
            if progress:
                total_generated = len(all_flashcards)
                progress(progress_percent, desc=f"Chunk {i+1}/{total_chunks}: Generated {len(flashcard_objs)} cards (Total: {total_generated})")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error generating flashcards for semantic chunk {i+1}: {e}")
    return all_flashcards

def generate_enhanced_flashcards_with_progress(slide_texts, slide_images, api_key, model, max_tokens, temperature, progress=None, use_cloze=False, question_style="Word for word"):
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
        question_style: Style of questions ("Word for word", "Elaborated", "Simplified")
        
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
            semantic_chunks, slide_images, api_key, model, max_tokens, temperature, progress, use_cloze, question_style
        )
        
        print("\nSample of generated flashcards (before export):")
        for card in all_flashcards[:10]:
            print(summarize_card(card))

        
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
# Test Deck Creation
# =====================

def create_test_apkg_deck(output_path='test_deck.apkg'):
    # Sample flashcards with image occlusion
    sample_flashcards = [
        {
            'type': 'image_occlusion',
            'question_img': 'apkg_media_check/occluded_slide10_img1.png',
            'answer_img': 'apkg_media_check/slide11_img1.jpg',
        },
        {
            'type': 'image_occlusion',
            'question_img': 'apkg_media_check/occluded_slide13_img1.png',
            'answer_img': 'apkg_media_check/slide12_img2.png',
        },
    ]
    
    # Export the sample flashcards to an APKG file
    export_flashcards_to_apkg(sample_flashcards, output_path=output_path)
    print(f"Test deck created at {output_path}")

# =====================
# Main Orchestration
# =====================

if __name__ == "__main__":
    # Directly create a test APKG deck
    create_test_apkg_deck()

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
    image_path: Optional[str] = None  # Path to last related image (if any)
    preview_text: str = ""  # Cleaned cloze text for UI preview
    raw_cloze_text: str = ""  # Original cloze text as received or generated
    card_type: str = ""  # e.g., 'occlusion' for masked image cards

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

def is_image_relevant_for_occlusion(image_path: str, slide_text: str, api_key: str, model: str) -> bool:
    """
    Determine if an image is relevant for image occlusion flashcards.
    Filters out decorative, scene-setting, or non-medical images.
    
    Args:
        image_path: Path to the image file
        slide_text: Associated slide text for context
        api_key: OpenAI API key
        model: Model name to use for analysis
        
    Returns:
        True if the image contains relevant medical/clinical content for occlusion
    """
    try:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            ext = os.path.splitext(image_path)[1][1:]
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create the analysis prompt
        analysis_prompt = f"""
        Analyze this image and determine if it contains content suitable for medical flashcard creation via image occlusion.
        
        CONTEXT: This image is from a slide with the following text:
        "{slide_text[:500]}..."  # Truncated for brevity
        
        CRITICAL FILTERING RULES:
        REJECT the image if it contains:
        - Decorative elements (logos, decorative graphics, stock photos)
        - Scene-setting images (hospitals, doctors, patients in general)
        - Generic medical imagery (stethoscopes, medical symbols)
        - Navigation elements (arrows, buttons, icons)
        - Pure text slides (better handled as text flashcards)
        - Charts/graphs with no medical data
        - Generic illustrations without specific medical content
        
        ACCEPT the image if it contains:
        - Anatomical diagrams with labeled structures
        - Medical charts/graphs with clinical data
        - Drug mechanism diagrams
        - Pathological specimens or histological images
        - ECG traces, X-rays, or other medical imaging
        - Flowcharts of medical processes or decision trees
        - Tables with medical data, lab values, or drug information
        - Biochemical pathways or molecular diagrams
        - Clinical algorithms or protocols
        
        Respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief explanation.
        """
        
        # Call the API for analysis
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # Parse the response
        if result.startswith("RELEVANT"):
            print(f"✅ Image {os.path.basename(image_path)} deemed relevant for occlusion")
            return True
        else:
            print(f"❌ Image {os.path.basename(image_path)} filtered out as not relevant")
            return False
            
    except Exception as e:
        print(f"⚠️ Error analyzing image relevance: {e}")
        # Default to False if analysis fails
        return False


def filter_relevant_images_for_occlusion(slide_images: List[List[str]], slide_texts: List[str], api_key: str, model: str) -> List[List[str]]:
    """
    Filter images to only include those relevant for image occlusion flashcards.
    
    Args:
        slide_images: List of image paths for each slide
        slide_texts: List of slide texts
        api_key: OpenAI API key
        model: Model name to use for analysis
        
    Returns:
        Filtered list of relevant images for each slide
    """
    if not slide_images or not slide_texts:
        return slide_images
    
    filtered_images = []
    
    for slide_idx, (images, slide_text) in enumerate(zip(slide_images, slide_texts)):
        relevant_images = []
        
        for image_path in images:
            if is_image_relevant_for_occlusion(image_path, slide_text, api_key, model):
                relevant_images.append(image_path)
        
        filtered_images.append(relevant_images)
        
        if images and not relevant_images:
            print(f"📝 Slide {slide_idx + 1}: All {len(images)} images filtered out as not relevant")
        elif images:
            print(f"📝 Slide {slide_idx + 1}: {len(relevant_images)}/{len(images)} images deemed relevant")
    
    return filtered_images

def export_occlusion_flashcards_to_csv(flashcard_entries, csv_path):
    """
    Export a list of occlusion flashcard entries to a CSV file for Anki import.
    Each entry should have 'question_image_base64' and 'answer_image_base64'.
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Front", "Back"])
        for entry in flashcard_entries:
            front = f"<img src='{os.path.basename(entry['question_image_path'])}'>"
            back  = f"<img src='{os.path.basename(entry['answer_image_path'])}'>"
            writer.writerow([front, back])

# Example usage of batch_generate_image_occlusion_flashcards
# image_paths = ['path/to/image1.png', 'path/to/image2.png']  # Replace with actual image paths
# export_dir = 'path/to/export_dir'  # Replace with actual export directory
# flashcard_entries = batch_generate_image_occlusion_flashcards(image_paths, export_dir)

# Process flashcard_entries as needed
# for entry in flashcard_entries:
#     print(f"Generated flashcard entry: {entry}")

# Use config values in batch_generate_image_occlusion_flashcards
# batch_generate_image_occlusion_flashcards(image_paths, export_dir, conf_threshold=config['conf_threshold'], mask_method='rectangle')

# batch_generate_image_occlusion_flashcards(image_paths, export_dir, conf_threshold=config['conf_threshold'], mask_method='rectangle') 