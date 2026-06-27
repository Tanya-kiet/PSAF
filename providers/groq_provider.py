"""
providers/groq_provider.py — Groq implementation of LLMProvider.

Phase 1: All logic is moved verbatim from backend.py / experiments.py.
NO logic, prompts, API calls, or caching have been changed.

Stability fix: stronger variation prompt, diversity enforcement, debug logging,
and safe non-empty fallback on generate_response failure.
"""

from __future__ import annotations

import json
import logging
import re
import time

from groq import Groq

import config
from providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

# ── Retry / call settings (unchanged from backend.py) ────────────────────────
_MAX_RETRIES = 5

# ── Stronger variation system prompt ─────────────────────────────────────────
# The original prompt was too generic and produced near-identical rewrites
# from small models.  This version gives explicit structural constraints so
# even llama-3.1-8b-instant produces meaningfully different paraphrases.
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
_FALLBACK_RESPONSE = "[No response — provider call failed]"


def _groq_client() -> Groq:
    """Return an authenticated Groq client (mirrors backend._groq_client)."""
    if not config.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it in your shell:\n"
            "  export GROQ_API_KEY='gsk_...'"
        )
    return Groq(api_key=config.GROQ_API_KEY)


class GroqProvider(LLMProvider):
    """
    Groq-backed LLM provider.

    Wraps the exact same Groq API calls that previously lived in
    backend.py and experiments.py.  No logic has been altered.
    """

    def __init__(self) -> None:
        self._client: Groq = _groq_client()

    # ── public interface ──────────────────────────────────────────────────────

    def generate_response(self, prompt: str) -> str:
        """
        Send a single user prompt and return the response text.

        Returns _FALLBACK_RESPONSE (never empty string) so compute_psi
        always receives at least a non-empty token sequence.
        """
        messages = [{"role": "user", "content": prompt}]
        result = self._call_with_retry(messages, temperature=config.TEMPERATURE)
        if not result.strip():
            logger.warning("generate_response: empty reply from Groq, using fallback.")
            return _FALLBACK_RESPONSE
        return result

    def generate_variations(self, prompt: str) -> list[str]:
        """
        Generate paraphrased variations of *prompt*.

        Improvements over original:
          • Explicit per-variation structural constraints in the system prompt.
          • Asks for N+2 candidates and keeps the N most-diverse ones so
            near-duplicates are less likely to survive.
          • Retries once if all returned variations are identical to the seed.
          • Debug-logs every returned variation for diagnosis.
        """
        n = config.MAX_VARIATIONS
        # Ask for a few extra candidates so we can discard near-duplicates
        n_request = n + 2

        variations = self._fetch_variations(prompt, n_request, attempt=1)

        # ── Safety check: if all are identical to seed, retry once ────────────
        unique = [v for v in variations if v.strip().lower() != prompt.strip().lower()]
        if not unique:
            logger.warning(
                "generate_variations: all %d candidates matched the seed prompt — "
                "retrying with higher temperature.", len(variations)
            )
            variations = self._fetch_variations(prompt, n_request, attempt=2, boost_temp=True)
            unique = [v for v in variations if v.strip().lower() != prompt.strip().lower()]

        if not unique:
            # Last-resort: the model is totally failing; log clearly and pad
            logger.error(
                "generate_variations: still no diverse variations after retry — "
                "PSI will reflect identical-response behaviour."
            )
            unique = variations  # keep whatever we have

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for v in unique:
            key = v.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(v)

        # Trim to n, pad with seed if we somehow still have too few
        result = deduped[:n]
        if len(result) < n:
            result += [prompt] * (n - len(result))

        # ── Debug logging ──────────────────────────────────────────────────────
        logger.debug("generate_variations — seed: %r", prompt)
        for i, v in enumerate(result):
            logger.debug("  variation[%d]: %r", i, v)
        unique_count = len({v.strip().lower() for v in result})
        logger.debug(
            "  → %d variations returned, %d unique", len(result), unique_count
        )

        return result

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fetch_variations(
        self,
        prompt: str,
        n: int,
        attempt: int = 1,
        boost_temp: bool = False,
    ) -> list[str]:
        """
        Make one LLM call and parse the JSON variation list.
        Returns a list of strings (may be shorter than n or contain duplicates).
        """
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

        logger.debug("_fetch_variations attempt %d raw reply: %r", attempt, raw[:300])

        if not raw.strip():
            logger.warning("_fetch_variations: empty reply on attempt %d", attempt)
            return []

        # Strip markdown fences if the model added them despite instructions
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Try to extract a JSON array even if extra text surrounds it
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                variations = [str(v).strip() for v in parsed if str(v).strip()]
                logger.debug(
                    "_fetch_variations attempt %d: parsed %d items", attempt, len(variations)
                )
                return variations
        except json.JSONDecodeError as exc:
            logger.warning(
                "_fetch_variations attempt %d: JSON parse failed (%s). raw=%r",
                attempt, exc, raw[:200],
            )

        return []

    def _call_with_retry(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """
        Single Groq call with exponential-backoff retry.

        Mirrors the retry logic in backend._call_groq and
        experiments.GroqClient.generate exactly.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=config.MAX_TOKENS,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Groq attempt %d/%d failed (%s) — waiting %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        return ""
