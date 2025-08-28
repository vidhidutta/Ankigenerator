from __future__ import annotations

"""
Adapter for slide-to-cards conversion.

Exposes a single function:
  extract_cards_from_ppt(input_path: str) -> list[tuple[str, str]]

This calls the existing converter in `flashcard_generator.py`, then normalizes
results to a list of (question, answer) tuples.

Environment variables used by the underlying converter (detected from the codebase):
- OPENAI_API_KEY
- GOOGLE_APPLICATION_CREDENTIALS (used by image/vision providers; not required for basic text cards)

If the converter fails or returns nothing, this function returns an empty list
so the caller can provide a fallback card.
"""

from typing import Any, Iterable, List, Tuple
import os
import tempfile
import traceback
import json
from pptx import Presentation
from openai import OpenAI


def _flatten(items: Iterable[Any]) -> list[Any]:
    flat: list[Any] = []
    for it in items or []:
        if isinstance(it, list):
            flat.extend(_flatten(it))
        else:
            flat.append(it)
    return flat


def _to_pairs(obj: Any) -> list[Tuple[str, str]]:
    pairs: list[Tuple[str, str]] = []

    # Already a list of pairs
    if isinstance(obj, list) and obj and isinstance(obj[0], tuple) and len(obj[0]) == 2:
        return [(str(q), str(a)) for (q, a) in obj]

    # DataFrame-like (duck typing; avoid hard pandas dependency)
    try:
        cols = getattr(obj, "columns", None)
        if cols is not None:
            col_names = [str(c).lower() for c in list(cols)]
            qcol = "q" if "q" in col_names else ("question" if "question" in col_names else None)
            acol = "a" if "a" in col_names else ("answer" if "answer" in col_names else None)
            if qcol and acol:
                # Iterate rows without importing pandas explicitly
                for idx in range(len(getattr(obj, "index", []))):
                    row = obj.iloc[idx]  # type: ignore[attr-defined]
                    q = str(row[qcol]) if qcol in row else ""
                    a = str(row[acol]) if acol in row else ""
                    if q.strip() and a.strip():
                        pairs.append((q.strip(), a.strip()))
                return pairs
    except Exception:
        pass

    # List of dicts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for d in obj:
            q = d.get("q") if "q" in d else d.get("question")
            a = d.get("a") if "a" in d else d.get("answer")
            if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                pairs.append((q.strip(), a.strip()))
        return pairs

    # List of objects with .question/.answer (e.g., Flashcard)
    if isinstance(obj, list) and obj and hasattr(obj[0], "question") and hasattr(obj[0], "answer"):
        for fc in obj:
            try:
                q = str(getattr(fc, "question", ""))
                a = str(getattr(fc, "answer", ""))
                if q.strip() and a.strip():
                    pairs.append((q.strip(), a.strip()))
            except Exception:
                continue
        return pairs

    # Unknown shape
    return []


def _pptx_text_blocks(path: str) -> list[str]:
    """Extract text blocks from PPTX using python-pptx (lightweight)."""
    texts = []
    prs = Presentation(path)
    for slide in prs.slides:
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                parts.append(shape.text)
        if parts:
            texts.append("\n".join(parts))
    return texts


def _openai_cards_from_texts(blocks: list[str], max_cards: int = 30) -> list[tuple[str,str]]:
    """Generate flashcards from text blocks using OpenAI directly (lightweight)."""
    if not blocks:
        return []
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        "You convert lecture notes into Anki Basic flashcards.\n"
        "Return ONLY valid JSON: a list of objects {\"q\":..., \"a\":...}.\n"
        f"Max {max_cards} cards. Use concise, clinically relevant Q/A. No markdown.\n"
    )
    joined = "\n\n--- SLIDES ---\n\n" + "\n\n---\n\n".join(blocks[:15])  # limit context
    r = client.chat.completions.create(
        model=os.getenv("OJAMED_OPENAI_MODEL","gpt-4o-mini"),
        messages=[{"role":"system","content":prompt},
                  {"role":"user","content":joined}],
        temperature=float(os.getenv("OJAMED_TEMPERATURE","0.2"))
    )
    text = r.choices[0].message.content.strip()
    try:
        data = json.loads(text)
    except Exception:
        # attempt to extract JSON substring
        import re
        m = re.search(r'\[.*\]', text, flags=re.S)
        data = json.loads(m.group(0)) if m else []
    cards = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                q = it.get("q") or it.get("question")
                a = it.get("a") or it.get("answer")
                if q and a:
                    cards.append((str(q).strip(), str(a).strip()))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                cards.append((str(it[0]).strip(), str(it[1]).strip()))
    return cards[:max_cards]


def extract_cards_from_ppt(input_path: str) -> List[Tuple[str, str]]:
    """
    Extract flashcards from a PPT/PDF path and normalize to (question, answer) tuples.

    Returns an empty list on failure so the caller can decide a fallback.
    """
    # DEMO BYPASS
    if os.getenv("OJAMED_FORCE_DEMO") == "1":
        print("[OjaMed][ADAPTER] DEMO MODE -> returning 3 static cards")
        return [
            ("What drug class is furosemide?", "Loop diuretic"),
            ("Main adverse effect?", "Hypokalemia"),
            ("Contraindicated with?", "Sulfa allergy (relative)"),
        ]

    # If light mode, skip importing flashcard_generator/sklearn entirely
    if os.getenv("OJAMED_DISABLE_SKLEARN") == "1":
        try:
            blocks = _pptx_text_blocks(input_path)
            cards = _openai_cards_from_texts(blocks, max_cards=int(os.getenv("OJAMED_MAX_CARDS","30")))
            print(f"[OjaMed][ADAPTER][LIGHT] -> {len(cards)} cards")
            return cards
        except Exception as e:
            print("[OjaMed][ADAPTER][LIGHT] failed:", repr(e))
            traceback.print_exc()
            return []

    # else use the existing heavy path (unchanged)
    try:
        # Lazy import to avoid import-time errors
        from flashcard_generator import (
            extract_text_from_pptx,
            extract_images_from_pptx,
            generate_enhanced_flashcards_with_progress,
        )
        
        import inspect
        print("[OjaMed][ADAPTER] signature(generate_enhanced_flashcards_with_progress) =",
              inspect.signature(generate_enhanced_flashcards_with_progress))
    except Exception as e:
        print("[OjaMed][ADAPTER] import failed:", repr(e))
        traceback.print_exc()
        if os.getenv("OJAMED_DEBUG") == "1":
            raise
        return []

    try:
        # Extract texts and images
        with tempfile.TemporaryDirectory() as temp_dir:
            texts = extract_text_from_pptx(input_path)
            try:
                images = extract_images_from_pptx(input_path, temp_dir)
            except Exception:
                images = []
            # Call with sensible defaults; map envs to arguments
            cards_obj = generate_enhanced_flashcards_with_progress(
                texts,
                images,
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OJAMED_OPENAI_MODEL", "gpt-4o-mini"),
                max_tokens=int(os.getenv("OJAMED_MAX_TOKENS", "500")),
                temperature=float(os.getenv("OJAMED_TEMPERATURE", "0.2")),
                progress=None,
                use_cloze=False,
                question_style="Word for word",
                audio_bundle=None,
            )

        # Some paths return (cards, analysis), others just cards
        cards_obj = cards_obj[0] if (isinstance(cards_obj, tuple) and cards_obj) else cards_obj

        # Flatten if nested
        if isinstance(cards_obj, list) and any(isinstance(x, list) for x in cards_obj):
            cards_obj = _flatten(cards_obj)

        cards = _to_pairs(cards_obj)
        print(f"[OjaMed][ADAPTER] -> {len(cards)} cards")
        return cards
    except Exception as e:
        print("[OjaMed][ADAPTER] converter raised:", repr(e))
        traceback.print_exc()
        if os.getenv("OJAMED_DEBUG") == "1":
            raise
        return []


