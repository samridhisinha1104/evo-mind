"""Multi-provider LLM client with support for Groq, HuggingFace, and Anthropic.

Reads EVOMIND_PROVIDER from the environment to decide which backend to use.
All clients implement the same LLMClient protocol so the rest of the codebase
is provider-agnostic.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()


import json
import os
from typing import Any, Protocol


class LLMClient(Protocol):
    """Minimal contract every LLM backend must satisfy."""

    def complete_json(self, system: str, prompt: str) -> dict[str, Any]:
        """Return a parsed JSON object from the model's response."""
        ...


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

_JSON_SUFFIX = "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble."


def _strip_fences(text: str) -> str:
    """Remove optional ```json ... ``` wrappers that models sometimes emit."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class GroqLLMClient:
    """Uses the Groq cloud API (free tier available)."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model: str | None = None):
        from groq import Groq

        self.model = model or os.environ.get("EVOMIND_MODEL", self.DEFAULT_MODEL)
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def complete_json(self, system: str, prompt: str) -> dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system + _JSON_SUFFIX},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(_strip_fences(text))


class HuggingFaceLLMClient:
    """Uses HuggingFace Inference API (free tier with HF_TOKEN)."""

    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

    def __init__(self, model: str | None = None):
        from huggingface_hub import InferenceClient

        self.model = model or os.environ.get("EVOMIND_MODEL", self.DEFAULT_MODEL)
        self._client = InferenceClient(
            model=self.model,
            token=os.environ.get("HF_TOKEN"),
        )

    def complete_json(self, system: str, prompt: str) -> dict[str, Any]:
        resp = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": system + _JSON_SUFFIX},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=2048,
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(_strip_fences(text))


class AnthropicLLMClient:
    """Anthropic Claude API (paid)."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or os.environ.get("EVOMIND_MODEL", self.DEFAULT_MODEL)
        self._client = anthropic.Anthropic()

    def complete_json(self, system: str, prompt: str) -> dict[str, Any]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system + _JSON_SUFFIX,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return json.loads(_strip_fences(text))


class FallbackLLMClient:
    """Tries Groq first, falls back to HuggingFace on rate limits or errors."""

    def __init__(self, model: str | None = None):
        self.primary = GroqLLMClient(model=model)
        # HF has its own default model, so we don't force the Groq model name on it
        # unless EVOMIND_MODEL is explicitly set in the environment.
        self.fallback = HuggingFaceLLMClient()

    def complete_json(self, system: str, prompt: str) -> dict[str, Any]:
        try:
            return self.primary.complete_json(system, prompt)
        except Exception as e:
            print(f"[FallbackLLMClient] Groq failed ({type(e).__name__}: {e}). Falling back to HuggingFace...")
            return self.fallback.complete_json(system, prompt)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type] = {
    "groq": GroqLLMClient,
    "huggingface": HuggingFaceLLMClient,
    "anthropic": AnthropicLLMClient,
    "auto": FallbackLLMClient,
}


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Build an LLM client based on EVOMIND_PROVIDER (or explicit override)."""
    provider = (provider or os.environ.get("EVOMIND_PROVIDER", "groq")).lower().strip()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown EVOMIND_PROVIDER={provider!r}. Supported: {supported}"
        )
    return cls(model=model)


_embedding_model = None


def get_task_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a task description using sentence-transformers."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model.encode(text).tolist()
