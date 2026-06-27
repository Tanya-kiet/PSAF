"""
providers/gemini_provider.py — Google Gemini implementation of LLMProvider.

Phase 2: Adds Gemini support alongside existing Groq provider.
Groq remains the default; this class is only used when
get_provider("gemini") is called explicitly.

Output format is identical to GroqProvider so the PSI pipeline
receives the same data structure regardless of which backend is used.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import google.generativeai as genai

import config
from providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

# ── Constants (mirror GroqProvider values) ────────────────────────────────────
_MAX_RETRIES = 5

# Gemini model — flash is fast and cheap; swappable via env var
_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Reuse the same system prompts as GroqProvider so output is comparable
_VARIATION_SYSTEM = (
    "You are a prompt rewriting expert. "
    "Rewrite the given question in different ways while preserving its exact meaning. "
    "Each rewrite must differ in wording and structure but ask the same thing. "
    "Return ONLY a valid JSON array of strings — no markdown, no explanation."
)

_RESPONSE_SYSTEM = (
    "You are a knowledgeable and concise assistant. "
    "Answer the user's question clearly and accurately."
)


def _gemini_client() -> genai.GenerativeModel:
    """Return an authenticated Gemini GenerativeModel instance."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Export it in your shell:\n"
            "  export GEMINI_API_KEY='AIza...'"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=_GEMINI_MODEL,
        system_instruction=_RESPONSE_SYSTEM,
    )


class GeminiProvider(LLMProvider):
    """
    Google Gemini-backed LLM provider.

    Implements the same interface as GroqProvider so it is a drop-in
    replacement inside the PSI pipeline.  Output format (plain string)
    is identical; only the underlying HTTP calls differ.
    """

    def __init__(self) -> None:
        self._client: genai.GenerativeModel = _gemini_client()
        # Separate model for variation generation (uses a different
        # system instruction that instructs JSON-only output)
        api_key = os.getenv("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        self._variation_client: genai.GenerativeModel = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            system_instruction=_VARIATION_SYSTEM,
        )

    # ── public interface ──────────────────────────────────────────────────────

    def generate_response(self, prompt: str) -> str:
        """
        Send a single user prompt to Gemini and return the response text.

        Uses the same exponential-backoff retry pattern as GroqProvider.
        """
        return self._call_with_retry(self._client, prompt, temperature=config.TEMPERATURE)

    def generate_variations(self, prompt: str) -> list[str]:
        """
        Generate paraphrased variations of *prompt* via Gemini.

        Applies the same JSON-parsing and fallback logic as GroqProvider
        so the PSI pipeline receives identically shaped output.
        """
        n = config.MAX_VARIATIONS
        user_msg = f"Rewrite this question in {n} different ways:\n\n{prompt}"

        raw = self._call_with_retry(
            self._variation_client,
            user_msg,
            temperature=config.PARAPHRASE_TEMPERATURE,
        )

        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                variations = [str(v) for v in parsed if str(v).strip()][:n]
                if len(variations) < n:
                    variations += [prompt] * (n - len(variations))
                return variations
        except Exception:
            pass

        return [prompt] * n

    # ── internal helpers ──────────────────────────────────────────────────────

    def _call_with_retry(
        self,
        model: genai.GenerativeModel,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Single Gemini call with exponential-backoff retry.

        Mirrors the retry pattern in GroqProvider._call_with_retry
        so failure behaviour is consistent across providers.
        """
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=config.MAX_TOKENS,
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                return response.text.strip()
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Gemini attempt %d/%d failed (%s) — waiting %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        return ""
