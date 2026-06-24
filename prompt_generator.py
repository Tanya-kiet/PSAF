"""
prompt_generator.py — Prompt variation generation (Groq version)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from groq import Groq

import config

logger = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class PromptRecord:
    prompt_id: str
    category: str
    original: str
    variation: str
    variation_idx: int
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "original": self.original,
            "variation": self.variation,
            "variation_idx": self.variation_idx,
            "tags": "|".join(self.tags),
        }


# ── Category Manager ─────────────────────────────────────────────────────────

class PromptCategoryManager:
    def __init__(self) -> None:
        self._categories = config.PROMPT_CATEGORIES

    def all_prompts(self):
        return list(
            (cat, seed)
            for cat, seeds in self._categories.items()
            for seed in seeds
        )

    def prompts_for(self, category: str):
        return self._categories.get(category, [])

    def category_abbreviation(self, category: str) -> str:
        return "".join(w[0] for w in category.split()[:3]).upper()


# ── Variation Generator (GROQ FIXED) ────────────────────────────────────────

class PromptVariationGenerator:
    _SYSTEM_PROMPT = (
        "You are a prompt engineering expert. "
        "Rewrite the given question in multiple different ways while preserving meaning. "
        "Each rewrite must be meaningfully different in structure and wording. "
        "Return ONLY a valid JSON array of strings. "
        "No markdown, no explanation, no extra text."
    )

    def __init__(self, client: Groq) -> None:
        self._client = client
        self._n = config.VARIATIONS_PER_PROMPT

    def generate_all(self, category_manager: PromptCategoryManager) -> list[PromptRecord]:
        records: list[PromptRecord] = []

        for idx, (category, seed) in enumerate(category_manager.all_prompts()):
            abbrev = category_manager.category_abbreviation(category)

            base_id = f"{abbrev}_{idx}"

            # ── original ─────────────────────────────────────────────
            records.append(
                PromptRecord(
                    prompt_id=f"{base_id}_var0",
                    category=category,
                    original=seed,
                    variation=seed,
                    variation_idx=0,
                    tags=["original"],
                )
            )

            # ── paraphrases ──────────────────────────────────────────
            variations = self._fetch_variations(seed)

            for i, v in enumerate(variations, start=1):
                records.append(
                    PromptRecord(
                        prompt_id=f"{base_id}_var{i}",
                        category=category,
                        original=seed,
                        variation=v,
                        variation_idx=i,
                        tags=["paraphrase"],
                    )
                )

        return records

    # ── GROQ API CALL (FIXED) ───────────────────────────────────────────────

    def _fetch_variations(self, seed: str) -> list[str]:
        user_msg = (
            f"Rewrite this question in {self._n} different ways:\n\n{seed}\n\n"
            f"Return ONLY a JSON array of strings."
        )

        try:
            resp = self._client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=config.PARAPHRASE_TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )

            raw = resp.choices[0].message.content.strip()

            # ── FIX: clean markdown fences if model adds them ──
            cleaned = re.sub(r"```json|```", "", raw).strip()

            variations = json.loads(cleaned)

            # ── safety: ensure correct length ───────────────────
            if len(variations) < self._n:
                variations += [seed] * (self._n - len(variations))

            return variations[: self._n]

        except Exception as e:
            logger.error("Variation generation failed for '%s': %s", seed, e)
            return [seed for _ in range(self._n)]