from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import spacy

from .types import CandidateTerms, OCRWord

_word_re = re.compile(r"[\w\-/]+")
_punct_table = str.maketrans("", "", string.punctuation)


def _normalize_text(s: str) -> str:
    return s.strip().lower().translate(_punct_table)


def _tokenize(text: str) -> List[str]:
    return [t.group(0) for t in _word_re.finditer(text.lower())]


class CandidateTermGenerator:
    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        # Lazy load to avoid large import on module import
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            # Fall back to a blank english with simple sentencizer if model is missing
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def generate(
        self,
        ocr_words: Sequence[OCRWord],
        slide_text_bullets: Sequence[str],
        transcript_window_texts: Sequence[str],
        top_k: int = 30,
    ) -> CandidateTerms:
        # Combine sources
        corpus_docs: List[str] = []
        slide_all_text = "\n".join(slide_text_bullets)
        transcript_all_text = "\n".join(transcript_window_texts)
        ocr_text = " ".join([w.text for w in ocr_words])
        combined = "\n".join([slide_all_text, transcript_all_text, ocr_text])

        # Extract noun phrases
        phrases: List[str] = []
        doc = self.nlp(combined)
        if hasattr(doc, "noun_chunks"):
            try:
                for chunk in doc.noun_chunks:
                    phrases.append(_normalize_text(chunk.text))
            except Exception:
                pass
        # Fallback: use simple bigram/trigram noun-like heuristics if model lacks parser
        if not phrases:
            tokens = [t.text.lower() for t in doc if t.is_alpha]
            for i in range(len(tokens)):
                for n in (2, 3):
                    if i + n <= len(tokens):
                        phrases.append(" ".join(tokens[i : i + n]))

        # Keep unique phrases and also include unigrams of OCR words
        unigram_terms = [_normalize_text(w.text) for w in ocr_words]
        all_terms = set([p for p in phrases if p] + [t for t in unigram_terms if t])

        # Build a lightweight TF-IDF across available slide deck context
        # For this step, treat slide_text_bullets as the document set
        docs = [t for t in slide_text_bullets if t.strip()] or [combined]
        term_to_docfreq = Counter()
        doc_term_counts: List[Counter] = []
        for doc_text in docs:
            tokens = set(_tokenize(doc_text))
            doc_term_counts.append(Counter(_tokenize(doc_text)))
            for tok in tokens:
                term_to_docfreq[tok] += 1

        num_docs = max(1, len(docs))

        def tfidf(term: str) -> float:
            tf = sum(cnt[term] for cnt in doc_term_counts)
            df = term_to_docfreq.get(term, 0)
            idf = math.log((num_docs + 1) / (1 + df)) + 1.0
            return tf * idf

        # Score terms by max TF-IDF of any of their words; multi-word phrases use avg word score
        scored: List[tuple[str, float]] = []
        for term in all_terms:
            words = _tokenize(term)
            if not words:
                continue
            score = sum(tfidf(w) for w in words) / len(words)
            scored.append((term, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_terms = [t for t, _ in scored[:top_k]]
        return CandidateTerms(terms=top_terms) 