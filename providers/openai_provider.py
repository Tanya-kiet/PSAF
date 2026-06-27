"""
providers/openai_provider.py — OpenAI implementation of LLMProvider.

Phase 2 (revised): Replaces GeminiProvider with OpenAI as the second LLM backend.
Groq remains the default; this class is only used when get_provider("openai")
is called explicitly.

Output format is identical to GroqProvider so the PSI pipeline receives the
same data structure regardless of which backend is used.
"""

from __future__ import annotations

import json
import logging
import re
import time

import streamlit as st
from openai import OpenAI

import config
from providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

# ── Retry / call settings ─────────────────────────────────────────────────────
_MAX_RETRIES = 5

# Default model: gpt-4o-mini (fast + cost-efficient)
_OPENAI_MODEL = "gpt-4o-mini"

# ── Variation system prompt (mirrors GroqProvider) ────────────────────────────
_VARIATION_SYSTEM = (
    "You are an expert at rewriting questions in diverse ways. "
    "Given a question, produce EXACTLY the requested number of paraphrases. "
    "RULES — every paraphrase MUST:\n"
    "  1. Preserve the original meaning completely.\n"
    "  2. Use DIFFERENT sentence structure from the original and from each other.\n"
    "  3. Use DIFFERENT vocabulary where possible (synonyms, alternative phrasings).\n"
    "  4. NOT simply swap one or two words — substantially reword each version.\n"
    "  5. NOT repeat the original question verbatim.\n"
    "Output ONLY a valid JSON array of strings, with no markdown fences, "
    "no numbering, no extra text, and no explanation."
)

_RESPONSE_SYSTEM = (
    "You are a knowledgeable and concise assistant. "
    "Answer the user's question clearly and accurately."
)

# Safe non-empty fallback so compute_psi never sees an empty-response list
_FALLBACK_RESPONSE = "[No response — OpenAI provider call failed]"


def _openai_client() -> OpenAI:
    """Return an authenticated OpenAI client using the Streamlit secret."""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = ""

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found in st.secrets. "
            "Add it to .streamlit/secrets.toml:\n"
            '  OPENAI_API_KEY = "sk-..."'
        )
    return OpenAI(api_key=api_key)


class OpenAIProvider(LLMProvider):
    """
    OpenAI-backed LLM provider.

    Implements the same interface as GroqProvider so it is a drop-in
    replacement inside the PSI pipeline. Uses gpt-4o-mini for speed
    and cost efficiency.
    """

    def __init__(self) -> None:
        self._client: OpenAI = _openai_client()

    # ── public interface ──────────────────────────────────────────────────────

    def generate_response(self, prompt: str) -> str:
        """
        Send a single user prompt to OpenAI and return the response text.

        Returns _FALLBACK_RESPONSE (never empty string) so compute_psi
        always receives at least a non-empty token sequence.
        """
        messages = [
            {"role": "system", "content": _RESPONSE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = self._call_with_retry(messages, temperature=config.TEMPERATURE)
        if not result.strip():
            logger.warning("generate_response: empty reply from OpenAI, using fallback.")
            return _FALLBACK_RESPONSE
        return result

    def generate_variations(self, prompt: str) -> list[str]:
        """
        Generate paraphrased variations of *prompt* via OpenAI.

        Mirrors the improved GroqProvider.generate_variations logic:
          • Requests N+2 candidates and discards near-duplicates.
          • Retries with higher temperature if all candidates match the seed.
          • Debug-logs each variation for diagnosis.
        """
        n = config.MAX_VARIATIONS
        n_request = n + 2

        variations = self._fetch_variations(prompt, n_request, attempt=1)

        # ── Safety check: retry if all match the seed ─────────────────────────
        unique = [v for v in variations if v.strip().lower() != prompt.strip().lower()]
        if not unique:
            logger.warning(
                "generate_variations (OpenAI): all %d candidates matched seed — "
                "retrying with higher temperature.", len(variations)
            )
            variations = self._fetch_variations(prompt, n_request, attempt=2, boost_temp=True)
            unique = [v for v in variations if v.strip().lower() != prompt.strip().lower()]

        if not unique:
            logger.error(
                "generate_variations (OpenAI): still no diverse variations after retry. "
                "PSI will reflect identical-response behaviour."
            )
            unique = variations

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for v in unique:
            key = v.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(v)

        result = deduped[:n]
        if len(result) < n:
            result += [prompt] * (n - len(result))

        # ── Debug logging ──────────────────────────────────────────────────────
        logger.debug("generate_variations (OpenAI) — seed: %r", prompt)
        for i, v in enumerate(result):
            logger.debug("  variation[%d]: %r", i, v)
        unique_count = len({v.strip().lower() for v in result})
        logger.debug("  → %d variations returned, %d unique", len(result), unique_count)

        return result

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fetch_variations(
        self,
        prompt: str,
        n: int,
        attempt: int = 1,
        boost_temp: bool = False,
    ) -> list[str]:
        """Make one OpenAI call and return parsed variation list."""
        temp = min(config.PARAPHRASE_TEMPERATURE + (0.15 if boost_temp else 0.0), 1.0)
        user_msg = (
            f"Rewrite the following question in EXACTLY {n} different ways. "
            f"Each version must use a noticeably different sentence structure and vocabulary.\n\n"
            f"Question: {prompt}\n\n"
            f"Return ONLY a JSON array of {n} strings."
        )
        raw = self._call_with_retry(
            [
                {"role": "system", "content": _VARIATION_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=temp,
        )

        logger.debug("_fetch_variations (OpenAI) attempt %d raw reply: %r", attempt, raw[:300])

        if not raw.strip():
            logger.warning("_fetch_variations (OpenAI): empty reply on attempt %d", attempt)
            return []

        # Strip markdown fences if the model added them despite instructions
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Extract JSON array even if extra text surrounds it
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                variations = [str(v).strip() for v in parsed if str(v).strip()]
                logger.debug(
                    "_fetch_variations (OpenAI) attempt %d: parsed %d items", attempt, len(variations)
                )
                return variations
        except json.JSONDecodeError as exc:
            logger.warning(
                "_fetch_variations (OpenAI) attempt %d: JSON parse failed (%s). raw=%r",
                attempt, exc, raw[:200],
            )

        return []

    def _call_with_retry(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """
        Single OpenAI chat-completions call with exponential-backoff retry.

        Mirrors the retry logic in GroqProvider._call_with_retry exactly.
        Returns empty string only on total failure (caller converts to fallback).
        """
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=_OPENAI_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=config.MAX_TOKENS,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI attempt %d/%d failed (%s) — waiting %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        logger.error("OpenAI: all %d retries exhausted — returning empty string.", _MAX_RETRIES)
        return ""
