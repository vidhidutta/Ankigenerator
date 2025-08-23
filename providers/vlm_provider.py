from __future__ import annotations

import base64
import io
import os
import json
from dataclasses import dataclass
from typing import Optional, List, Sequence
import importlib.util

from PIL import Image

from .types import Region


@dataclass
class VLMResponse:
    text: str


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


_SUGGEST_PROMPT = (
    "List up to K key anatomical or diagram labels you see. "
    "Respond as a comma-separated list only, no extra text. K={K}."
)

_RANK_SYSTEM = (
    "You are a medical education assistant. Pick diagram regions that are most testable for exams. "
    "Favor named structures, key steps, axes/values in charts, and distinctive anomalies. Avoid trivial background areas."
)

_RANK_USER_TEMPLATE = (
    "SLIDE TEXT:\n{slide_text}\n\n"
    "LECTURE TRANSCRIPT EXCERPT:\n{transcript_text}\n\n"
    "DETECTED REGIONS (index, term, score):\n{region_list_table}\n\n"
    "TASK:\n"
    "Return a JSON array of up to {top_k} objects. Each object:\n"
    "{{\n"
    "  \"region_index\": <int>,\n"
    "  \"importance_score\": <float 0..1>,\n"
    "  \"short_label\": \"<2-5 words>\",\n"
    "  \"rationale\": \"<<=20 words>\"\n"
    "}}\n"
    "Only return JSON. No prose."
)


def _format_region_table(regions: Sequence[Region]) -> str:
    lines = []
    for idx, r in enumerate(regions):
        lines.append(f"{idx}\t{r.term}\t{r.score:.2f}")
    return "\n".join(lines)


def _apply_rank_output(regions: List[Region], json_text: str) -> List[Region]:
    try:
        data = json.loads(json_text)
        if not isinstance(data, list):
            return regions
        for item in data:
            try:
                idx = int(item.get("region_index"))
            except Exception:
                continue
            if 0 <= idx < len(regions):
                regions[idx].importance_score = float(item.get("importance_score", 0.0))
                regions[idx].short_label = str(item.get("short_label") or "").strip() or None
                regions[idx].rationale = str(item.get("rationale") or "").strip() or None
        regions_sorted = sorted(
            regions,
            key=lambda r: (r.importance_score if r.importance_score is not None else -1.0),
            reverse=True,
        )
        return regions_sorted
    except Exception:
        return regions


class LocalQwen2VLProvider:
    """Local Qwen2-VL provider."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    @staticmethod
    def available() -> bool:
        allow = os.getenv("ALLOW_LOCAL_VLM", "0").lower() in {"1", "true", "yes"}
        has_transformers = importlib.util.find_spec("transformers") is not None
        return bool(allow and has_transformers)

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
            import torch

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                device_map="auto" if self.device.startswith("cuda") else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._pipeline = (processor, model)

    def generate(self, image: Image.Image, question: str, max_new_tokens: int = 128) -> VLMResponse:
        self._ensure_loaded()
        assert self._pipeline is not None
        processor, model = self._pipeline

        import torch

        inputs = processor(text=question, images=image, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return VLMResponse(text=output_text.strip())

    def suggest_key_structures(self, image: Image.Image, k: int = 8) -> List[str]:
        prompt = _SUGGEST_PROMPT.replace("{K}", str(k))
        text = self.generate(image, prompt, max_new_tokens=64).text
        return [t.strip() for t in text.split(",") if t.strip()]

    def rank_regions(
        self,
        image: Image.Image,
        regions: List[Region],
        slide_text: str,
        transcript_text: str,
        top_k: int,
    ) -> List[Region]:
        k = max(1, min(top_k, len(regions)))
        region_table = _format_region_table(regions)
        user = _RANK_USER_TEMPLATE.format(
            slide_text=slide_text,
            transcript_text=transcript_text,
            region_list_table=region_table,
            top_k=k,
        )
        prompt = f"System: {_RANK_SYSTEM}\n\nUser:\n{user}"
        text = self.generate(image, prompt, max_new_tokens=256).text
        return _apply_rank_output(regions, text)

    def clear(self) -> None:
        self._pipeline = None


class LocalLLaVAOneVisionProvider:
    """Local LLaVA OneVision provider (fallback)."""

    def __init__(self, model_name: str = "llava-hf/llava-onevision-qwen2-0.5b-hf", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    @staticmethod
    def available() -> bool:
        allow = os.getenv("ALLOW_LOCAL_VLM", "0").lower() in {"1", "true", "yes"}
        has_transformers = importlib.util.find_spec("transformers") is not None
        return bool(allow and has_transformers)

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
            import torch

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                device_map="auto" if self.device.startswith("cuda") else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._pipeline = (processor, model)

    def generate(self, image: Image.Image, question: str, max_new_tokens: int = 128) -> VLMResponse:
        self._ensure_loaded()
        assert self._pipeline is not None
        processor, model = self._pipeline

        import torch

        inputs = processor(text=question, images=image, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return VLMResponse(text=output_text.strip())

    def suggest_key_structures(self, image: Image.Image, k: int = 8) -> List[str]:
        prompt = _SUGGEST_PROMPT.replace("{K}", str(k))
        text = self.generate(image, prompt, max_new_tokens=64).text
        return [t.strip() for t in text.split(",") if t.strip()]

    def rank_regions(
        self,
        image: Image.Image,
        regions: List[Region],
        slide_text: str,
        transcript_text: str,
        top_k: int,
    ) -> List[Region]:
        k = max(1, min(top_k, len(regions)))
        region_table = _format_region_table(regions)
        user = _RANK_USER_TEMPLATE.format(
            slide_text=slide_text,
            transcript_text=transcript_text,
            region_list_table=region_table,
            top_k=k,
        )
        prompt = f"System: {_RANK_SYSTEM}\n\nUser:\n{user}"
        text = self.generate(image, prompt, max_new_tokens=256).text
        return _apply_rank_output(regions, text)

    def clear(self) -> None:
        self._pipeline = None


class CloudVLMProvider:
    """Cloud VLM provider using OpenAI or OpenRouter if configured.

    Reads `OPENAI_API_KEY` or `OPENROUTER_API_KEY` from environment.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")
        self._client = None

    @staticmethod
    def available() -> bool:
        has_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        return bool(has_key)

    def _ensure_loaded(self) -> None:
        if self._client is None:
            if os.getenv("OPENROUTER_API_KEY"):
                import requests  # simple REST fallback for OpenRouter
                self._client = ("openrouter", requests)
            else:
                from openai import OpenAI
                self._client = ("openai", OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def generate(self, image: Image.Image, question: str, max_new_tokens: int = 128) -> VLMResponse:
        self._ensure_loaded()
        assert self._client is not None
        kind, client = self._client
        image_b64 = _pil_to_b64(image)

        if kind == "openai":
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": image_b64}},
                        ],
                    }
                ],
                max_tokens=max_new_tokens,
            )
            text = resp.choices[0].message.content  # type: ignore[attr-defined]
            return VLMResponse(text=text.strip())

        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
        }
        import requests

        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return VLMResponse(text=text.strip())

    def suggest_key_structures(self, image: Image.Image, k: int = 8) -> List[str]:
        prompt = _SUGGEST_PROMPT.replace("{K}", str(k))
        text = self.generate(image, prompt, max_new_tokens=64).text
        return [t.strip() for t in text.split(",") if t.strip()]

    def rank_regions(
        self,
        image: Image.Image,
        regions: List[Region],
        slide_text: str,
        transcript_text: str,
        top_k: int,
    ) -> List[Region]:
        k = max(1, min(top_k, len(regions)))
        region_table = _format_region_table(regions)
        user = _RANK_USER_TEMPLATE.format(
            slide_text=slide_text,
            transcript_text=transcript_text,
            region_list_table=region_table,
            top_k=k,
        )
        prompt = f"System: {_RANK_SYSTEM}\n\nUser:\n{user}"
        text = self.generate(image, prompt, max_new_tokens=256).text
        return _apply_rank_output(regions, text)

    def clear(self) -> None:
        self._client = None 