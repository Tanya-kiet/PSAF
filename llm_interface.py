"""
llm_interface.py — LLM response collection for Phase 1 of PSAF (Groq version)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from groq import Groq

import config
from prompt_generator import PromptRecord

logger = logging.getLogger(__name__)

# ── Retry settings ────────────────────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_DELAY_S: float = 2.0


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ResponseRecord:
    prompt_id: str
    category: str
    original: str
    variation: str
    variation_idx: int
    response_text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "original": self.original,
            "variation": self.variation,
            "variation_idx": self.variation_idx,
            "response_text": self.response_text,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


# ── LLM Interface ────────────────────────────────────────────────────────────

class LLMInterface:
    _SYSTEM_PROMPT = (
        "You are a knowledgeable and concise assistant. "
        "Answer the user's question clearly and accurately."
    )

    def __init__(self, client: Groq) -> None:
        self._client = client

    def collect_responses(self, prompt_records: list[PromptRecord]) -> list[ResponseRecord]:
        results: list[ResponseRecord] = []

        for i, record in enumerate(prompt_records, start=1):
            logger.info(
                "[%d/%d] Querying LLM | prompt_id=%s",
                i, len(prompt_records), record.prompt_id
            )
            results.append(self._query_with_retry(record))

        return results

    # ── retry wrapper ─────────────────────────────────────────────────────────

    def _query_with_retry(self, record: PromptRecord) -> ResponseRecord:
        delay = RETRY_DELAY_S

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call_api(record)

            except Exception as exc:
                logger.warning(
                    "Attempt %d failed for %s: %s",
                    attempt, record.prompt_id, str(exc)
                )

                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return self._error_record(record, str(exc))

        return self._error_record(record, "Exceeded retry limit")

    # ── actual Groq API call ──────────────────────────────────────────────────

    def _call_api(self, record: PromptRecord) -> ResponseRecord:
        start = time.monotonic()

        resp = self._client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": record.variation},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

        latency_ms = (time.monotonic() - start) * 1000

        text = resp.choices[0].message.content

        return ResponseRecord(
            prompt_id=record.prompt_id,
            category=record.category,
            original=record.original,
            variation=record.variation,
            variation_idx=record.variation_idx,
            response_text=text,
            model=config.LLM_MODEL,
            input_tokens=getattr(resp, "usage", {}).get("prompt_tokens", 0)
            if hasattr(resp, "usage") else 0,
            output_tokens=getattr(resp, "usage", {}).get("completion_tokens", 0)
            if hasattr(resp, "usage") else 0,
            latency_ms=latency_ms,
            error="",
        )

    @staticmethod
    def _error_record(record: PromptRecord, error_msg: str) -> ResponseRecord:
        return ResponseRecord(
            prompt_id=record.prompt_id,
            category=record.category,
            original=record.original,
            variation=record.variation,
            variation_idx=record.variation_idx,
            response_text="",
            model=config.LLM_MODEL,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            error=error_msg,
        )