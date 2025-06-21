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

# =====================
# Utility Functions
# =====================

def parse_args():
    parser = argparse.ArgumentParser(description='Convert PowerPoint slides to Anki flashcards and extra materials.')
    parser.add_argument('pptx_path', help='Path to the PowerPoint (.pptx) file')
    parser.add_argument('--output', default='flashcards.csv', help='Output CSV file for Anki import')
    parser.add_argument('--notes', default='lecture_notes.pdf', help='Output PDF file for extra materials')
    return parser.parse_args()


def extract_text_from_pptx(pptx_path):
    """Extract text from all slides in a PowerPoint file."""
    try:
        prs = Presentation(pptx_path)
        slide_texts = []
        for i, slide in enumerate(prs.slides):
            slide_content = [f"Slide {i+1}:"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text.strip())
            if len(slide_content) > 1:
                slide_texts.append("\n".join(slide_content))
        print(f"Extracted text from {len(slide_texts)} slides")
        return slide_texts
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


def parse_flashcards(ai_response):
    flashcards = []
    if ai_response.startswith("__API_ERROR__"):
        return flashcards, ai_response[len("__API_ERROR__"):]
    
    print(f"DEBUG: Parsing AI response of length {len(ai_response)}")
    print(f"DEBUG: First 200 chars: {ai_response[:200]}")
    
    # Pattern 1: Strict format (Question: ... | Answer: ...)
    pattern_strict = re.compile(r"Question:\s*(.*?)\s*\|\s*Answer:\s*(.*)")
    
    # Pattern 2: Markdown or plain text (Question: ...\nAnswer: ...)
    pattern_qa = re.compile(r"Question:?\s*(.*?)\s*\nAnswer:?\s*(.*?)(?:\n|$)", re.IGNORECASE)
    
    # Pattern 3: Markdown bold (**Question:** ...\n**Answer:** ...)
    pattern_md = re.compile(r"\*\*Question:?\*\*\s*(.*?)\s*\n\*\*Answer:?\*\*\s*(.*?)(?:\n|$)", re.IGNORECASE)
    
    # Pattern 4: Numbered list format (1. Question: ...\n   Answer: ...)
    pattern_numbered = re.compile(r"\d+\.\s*Question:\s*(.*?)\s*\n\s*Answer:\s*(.*?)(?=\n\d+\.|$)", re.DOTALL | re.IGNORECASE)
    
    # Pattern 5: Simple numbered list (1. Question: ...\n   Answer: ...) without "Question:" and "Answer:" labels
    pattern_simple_numbered = re.compile(r"\d+\.\s*(.*?)\s*\n\s*(.*?)(?=\n\d+\.|$)", re.DOTALL)
    
    # Pattern 6: Handle the format where everything is on one line with pipe separator
    # This handles cases like: "Question: What is X? | Answer: Y is Z"
    pattern_oneline = re.compile(r"Question:\s*(.*?)\s*\|\s*Answer:\s*(.*?)(?=\nQuestion:|$)", re.DOTALL)

    # Try strict format first
    matches = list(pattern_strict.finditer(ai_response))
    print(f"DEBUG: Found {len(matches)} matches with pattern_strict")
    for match in matches:
        question, answer = match.groups()
        if question.strip() and answer.strip():  # Make sure both fields have content
            flashcards.append((question.strip(), answer.strip()))
            print(f"DEBUG: Added flashcard - Q: {question.strip()[:50]}... A: {answer.strip()[:50]}...")
    
    # If none found, try the oneline format
    if not flashcards:
        matches = list(pattern_oneline.finditer(ai_response))
        print(f"DEBUG: Found {len(matches)} matches with pattern_oneline")
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip():  # Make sure both fields have content
                flashcards.append((question.strip(), answer.strip()))
                print(f"DEBUG: Added flashcard - Q: {question.strip()[:50]}... A: {answer.strip()[:50]}...")
    
    # If still none, try numbered list format
    if not flashcards:
        matches = list(pattern_numbered.finditer(ai_response))
        print(f"DEBUG: Found {len(matches)} matches with pattern_numbered")
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip():  # Make sure both fields have content
                flashcards.append((question.strip(), answer.strip()))
                print(f"DEBUG: Added flashcard - Q: {question.strip()[:50]}... A: {answer.strip()[:50]}...")
    
    # If still none, try simple numbered list
    if not flashcards:
        matches = list(pattern_simple_numbered.finditer(ai_response))
        print(f"DEBUG: Found {len(matches)} matches with pattern_simple_numbered")
        for match in matches:
            question, answer = match.groups()
            # Filter out obvious non-medical content
            if question.strip() and answer.strip() and is_medical_content(question, answer):
                flashcards.append((question.strip(), answer.strip()))
                print(f"DEBUG: Added flashcard - Q: {question.strip()[:50]}... A: {answer.strip()[:50]}...")
    
    # If still none, try Markdown bold
    if not flashcards:
        matches = list(pattern_md.finditer(ai_response))
        print(f"DEBUG: Found {len(matches)} matches with pattern_md")
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip() and is_medical_content(question, answer):
                flashcards.append((question.strip(), answer.strip()))
                print(f"DEBUG: Added flashcard - Q: {question.strip()[:50]}... A: {answer.strip()[:50]}...")
    
    # If still none, try plain Q/A
    if not flashcards:
        matches = list(pattern_qa.finditer(ai_response))
        print(f"DEBUG: Found {len(matches)} matches with pattern_qa")
        for match in matches:
            question, answer = match.groups()
            if question.strip() and answer.strip() and is_medical_content(question, answer):
                flashcards.append((question.strip(), answer.strip()))
                print(f"DEBUG: Added flashcard - Q: {question.strip()[:50]}... A: {answer.strip()[:50]}...")
    
    print(f"DEBUG: Total flashcards parsed: {len(flashcards)}")
    return flashcards, None


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


def export_flashcards_to_csv(flashcards, output_path):
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "Answer"])
        for q, a in flashcards:
            writer.writerow([q, a])


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
    slide_texts = extract_text_from_pptx(args.pptx_path)
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
    
    print("Generating multimodal flashcards with AI (text + images) via HTTP API...")
    all_flashcards = generate_multimodal_flashcards_http(filtered_slide_texts, filtered_slide_images, OPENAI_API_KEY, MODEL_NAME, MAX_TOKENS, TEMPERATURE)
    print(f"Total generated: {len(all_flashcards)} flashcards")
    export_flashcards_to_csv(all_flashcards, args.output)
    print(f'Success! {len(all_flashcards)} flashcards exported to {args.output}')
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