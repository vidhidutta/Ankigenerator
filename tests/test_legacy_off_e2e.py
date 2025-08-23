import os
import yaml
from PIL import Image
from flashcard_generator import export_flashcards_to_apkg

import pytest
pytestmark = pytest.mark.image_occlusion


def test_legacy_off_e2e(tmp_path):
    # Create a minimal occlusion entry to simulate legacy flow
    img = Image.new("RGB", (200, 200), color="white")
    q = tmp_path / "q.png"
    a = tmp_path / "a.png"
    img.save(q)
    img.save(a)
    items = [{
        'type': 'image_occlusion',
        'question_image_path': str(q),
        'answer_image_path': str(a),
        'alt_text': 'What is hidden here?'
    }]
    apkg = tmp_path / "legacy_off.apkg"
    export_flashcards_to_apkg(items, output_path=str(apkg))
    assert os.path.exists(apkg) 