"""
main.py — Orchestrator for the Prompt Stability Analysis Framework (PSAF).

Pipeline:
  1. Initialise output directories and logging.
  2. Load prompt categories (PromptCategoryManager).
  3. Generate paraphrased variations (PromptVariationGenerator).
  4. Save all prompts to prompts.csv.
  5. Collect LLM responses (LLMInterface).
  6. Save all responses to responses.csv.
  7. (Phase 2) Compute the Prompt Stability Index (PSI) for every seed
     prompt and save the results to psi_results.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

from groq import Groq

import config
import psi
from llm_interface import LLMInterface, ResponseRecord
from prompt_generator import (
    PromptCategoryManager,
    PromptRecord,
    PromptVariationGenerator,
)


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"psaf_{int(time.time())}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)   # silence HTTP noise


logger = logging.getLogger(__name__)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_prompts_csv(records: list[PromptRecord], path: Path) -> None:
    """Write all PromptRecord objects to a CSV file."""
    if not records:
        logger.warning("No prompt records to save.")
        return

    fieldnames = list(records[0].to_dict().keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())

    logger.info("Prompts saved → %s  (%d rows)", path, len(records))


def _save_responses_csv(records: list[ResponseRecord], path: Path) -> None:
    """Write all ResponseRecord objects to a CSV file."""
    if not records:
        logger.warning("No response records to save.")
        return

    fieldnames = list(records[0].to_dict().keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())

    logger.info("Responses saved → %s  (%d rows)", path, len(records))


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(
    prompt_records: list[PromptRecord],
    response_records: list[ResponseRecord],
    psi_results: list[psi.PSIResult] | None = None,
) -> None:
    categories = {r.category for r in prompt_records}
    originals  = [r for r in prompt_records if r.variation_idx == 0]
    variants   = [r for r in prompt_records if r.variation_idx > 0]
    errors     = [r for r in response_records if r.error]

    logger.info("=" * 60)
    logger.info("PSAF — Run Summary")
    logger.info("=" * 60)
    logger.info("Categories processed : %d", len(categories))
    logger.info("Seed prompts         : %d", len(originals))
    logger.info("Paraphrased variants : %d", len(variants))
    logger.info("Total prompts        : %d", len(prompt_records))
    logger.info("Responses collected  : %d", len(response_records))
    logger.info("Errors               : %d", len(errors))
    if errors:
        for e in errors:
            logger.warning("  ✗  %s — %s", e.prompt_id, e.error)
    if psi_results:
        avg_psi = sum(r.psi_score for r in psi_results) / len(psi_results)
        logger.info("PSI groups scored    : %d", len(psi_results))
        logger.info("Average PSI score    : %.2f / 100", avg_psi)
    logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _setup_logging()
    logger.info("PSAF starting …")

    # ── Validate API key ──────────────────────────────────────────────────────
    if not config.GROQ_API_KEY:
        logger.error(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running: export ANTHROPIC_API_KEY=sk-..."
        )
        sys.exit(1)

    # ── Initialise Anthropic client ───────────────────────────────────────────
    client = Groq(
    api_key=config.GROQ_API_KEY
)

    # ── Step 1: Load categories ───────────────────────────────────────────────
    logger.info("Step 1/5 — Loading prompt categories …")
    category_manager = PromptCategoryManager()
    logger.info(
        "Loaded %d categories: %s",
        len(category_manager.categories),
        ", ".join(category_manager.categories),
    )

    # ── Step 2: Generate variations ───────────────────────────────────────────
    logger.info(
        "Step 2/5 — Generating %d paraphrased variations per prompt …",
        config.VARIATIONS_PER_PROMPT,
    )
    variation_generator = PromptVariationGenerator(client)
    prompt_records = variation_generator.generate_all(category_manager)
    logger.info("Total prompt records generated: %d", len(prompt_records))

    # ── Step 3: Save prompts CSV ──────────────────────────────────────────────
    logger.info("Step 3/5 — Saving prompts to CSV …")
    prompts_path = config.OUTPUT_DIR / config.PROMPTS_CSV_FILENAME
    _save_prompts_csv(prompt_records, prompts_path)

    # ── Step 4: Collect LLM responses ────────────────────────────────────────
    logger.info("Step 4/5 — Collecting LLM responses …")
    llm = LLMInterface(client)
    response_records = llm.collect_responses(prompt_records)

    responses_path = config.OUTPUT_DIR / config.RESPONSES_CSV_FILENAME
    _save_responses_csv(response_records, responses_path)

    # ── Step 5: Compute Prompt Stability Index (Phase 2) ─────────────────────
    logger.info("Step 5/5 — Computing Prompt Stability Index (PSI) …")
    psi_path = config.OUTPUT_DIR / config.PSI_RESULTS_CSV_FILENAME
    psi_results = psi.run(responses_path, psi_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(prompt_records, response_records, psi_results)
    logger.info("PSAF run complete. Output files are in: %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
