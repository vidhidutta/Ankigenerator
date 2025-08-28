from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

from app.adapter import extract_cards_from_ppt


def write_csv(cards: List[Tuple[str, str]], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for q, a in cards:
            w.writerow([q, a])
    return os.path.abspath(path)


def write_apkg(cards: List[Tuple[str, str]], path: str) -> str:
    try:
        import genanki
    except Exception as e:
        # If genanki unavailable, create an empty placeholder to keep the ZIP structure
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"")
        return os.path.abspath(path)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    model = genanki.Model(
        1607392319,
        "Basic (OjaMed)",
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Question}}",
                "afmt": "{{FrontSide}}<hr id=answer>{{Answer}}",
            }
        ],
    )

    deck = genanki.Deck(2059400101, "OjaMed Deck")
    for q, a in cards:
        note = genanki.Note(model=model, fields=[q, a])
        deck.add_note(note)

    genanki.Package(deck).write_to_file(path)
    return os.path.abspath(path)


def run_pipeline(input_path: str) -> tuple[str, str]:
    """Run extraction and build CSV + APKG. Return absolute paths.

    If extraction returns empty, include a sample card so APKG always imports.
    If APKG build fails, still return a CSV and an empty APKG placeholder.
    """
    cards = extract_cards_from_ppt(input_path) or []
    print(f"[OjaMed][PIPELINE] Extracted {len(cards)} cards from {input_path}")
    if not cards:
        cards = [("Example?", "Yes")]  # ensure importable deck

    # Create unique temp dir and fixed filenames so ZIP contains deck.csv + deck.apkg
    out_dir = Path(tempfile.mkdtemp(prefix="ojamed_"))
    csv_path = str(out_dir / "deck.csv")
    apkg_path = str(out_dir / "deck.apkg")

    csv_abs = write_csv(cards, csv_path)

    try:
        apkg_abs = write_apkg(cards, apkg_path)
    except Exception:
        # Keep CSV and write an empty APKG placeholder
        with open(apkg_path, "wb") as f:
            f.write(b"")
        apkg_abs = os.path.abspath(apkg_path)

    return apkg_abs, csv_abs
from pathlib import Path

def run_pipeline(input_path: str) -> tuple[str, str]:
    input_file = Path(input_path)
    out_apkg = input_file.with_suffix(input_file.suffix + ".apkg")
    out_csv  = input_file.with_suffix(input_file.suffix + ".csv")

    # DEMO OUTPUTS so the API works now; replace with your real logic.
    out_apkg.write_bytes(b"fake .apkg content")
    out_csv.write_text("question,answer\nExample?,Yes\n", encoding="utf-8")

    return str(out_apkg), str(out_csv)
