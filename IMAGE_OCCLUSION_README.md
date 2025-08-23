# Image Occlusion (Detection → Segmentation → Ranking)

This feature generates image-occlusion cards from lecture images/diagrams by finding testable regions, masking them on the front, and revealing the answer on the back.

## How it works

- OCR (PaddleOCR)
  - Preprocess (grayscale → contrast → adaptive threshold) if enabled
  - Extracts words and bounding boxes to seed candidate terms
- Candidate terms (spaCy + TF‑IDF)
  - Combines OCR words, slide text, and transcript snippet
  - Extracts noun phrases and keeps top‑K rare/important terms
- Detection (GroundingDINO)
  - Detects regions for the candidate terms
  - Threshold + NMS applied; capped at ~20
  - Fallback: if none, ask the VLM to propose labels, or map terms to OCR boxes
- Segmentation (SAM/SAM2)
  - Turns boxes into tight masks; filters specks by min area
  - Merges highly overlapping masks (IoU > 0.8)
- Ranking (VLM)
  - Ranks regions by “what is testable”; returns index + importance + short label + rationale
- Export
  - Masks the region (solid fill or blur), draws outline on back
  - Builds Anki Image Occlusion Enhanced+ notes; optionally shows “Why masked?” rationale on the back

## Configuration (config.yaml)

```yaml
image_understanding:
  enabled: true
  vlm: "qwen2-vl"         # options: qwen2-vl, llava-onevision, cloud
  max_masks_per_image: 6
  min_mask_area_px: 900
  detection_threshold: 0.25
  nms_iou_threshold: 0.5
  use_ocr_preprocess: true
  keep_image_only_slides: true
  candidate_terms_topk: 30
  show_rationale_on_back: true
```

- enabled: Master switch for the pipeline
- vlm: Which ranking provider to use
- max_masks_per_image: Final top‑K masks/regions kept for export
- min_mask_area_px: Filter tiny masks
- detection_threshold: GroundingDINO confidence threshold
- nms_iou_threshold: Non‑max suppression IoU
- use_ocr_preprocess: Improve OCR robustness
- keep_image_only_slides: If a slide has only images, keep it regardless of relevance filtering
- candidate_terms_topk: Number of terms considered by TF‑IDF
- show_rationale_on_back: Append the VLM’s short rationale under the back image

## Providers

- OCR: PaddleOCR
  - Install: `paddleocr>=2.7`; requires `paddlepaddle` (see official wheels)
- Detection: GroundingDINO
  - Installed via GitHub; uses `groundingdino_pytorch` wrapper
- Segmentation: SAM/SAM2
  - SAM2 preferred; otherwise Segment‑Anything with a checkpoint
  - Set `SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth`
- VLM ranking (choose one):
  - Local Qwen2‑VL (set `ALLOW_LOCAL_VLM=1`)
  - Local LLaVA OneVision (set `ALLOW_LOCAL_VLM=1`)
  - Cloud (OpenAI or OpenRouter): set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`

## UI (preview and edits)

In the Gradio UI:
- After generation, open the “Image Occlusion Review” panel
- Select an image, view overlays (#1, #2 …), and edit the table:
  - keep, label, rationale, importance, area, x1, y1, x2, y2
- Preview masked vs original with adjustable mask opacity
- Click “Export selected occlusions” to create only the checked items

## Performance & Stability

- Models are lazy‑loaded once per session
- Caching by image hash for OCR, detection, and segmentation
- Images resized to a max long side of 1600 px for processing (originals used for export)
- Timeouts return partial results with warnings; fallbacks kick in (e.g., OCR boxes + VLM labels)

## Switching the VLM

- Local Qwen2‑VL: `ALLOW_LOCAL_VLM=1` (models download via `transformers`)
- Local LLaVA‑OneVision: `ALLOW_LOCAL_VLM=1`
- Cloud: set `OPENAI_API_KEY` (default `gpt-4o-mini`) or `OPENROUTER_API_KEY`
- In `config.yaml`, set `image_understanding.vlm` to `qwen2-vl`, `llava-onevision`, or `cloud`.

## Troubleshooting

- No masks:
  - Ensure SAM/SAM2 is available; set `SAM_CHECKPOINT`
  - Increase `detection_threshold` down (e.g., 0.2) and ensure candidate terms are reasonable
  - Check `min_mask_area_px` isn’t too high
- Low OCR quality:
  - Set `use_ocr_preprocess: true`
  - Verify PaddlePaddle install and try CPU vs CUDA wheels per platform
- Timeouts or missing providers:
  - The pipeline will return partial results with fallbacks; check logs for “timeout” or “provider unavailable”
- VLM ranking returns nothing:
  - Ensure the chosen VLM is enabled (`ALLOW_LOCAL_VLM=1` or API keys set)

## Examples

- Quick test (exports an APKG with masked cards):
```bash
python -m pytest tests/test_occlusion_pipeline.py -q
```

- End‑to‑end (skips unavailable providers):
```bash
pytest -m image_occlusion -q
``` 