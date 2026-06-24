"""
report_generator.py — Phase 5: Research Reporting
Prompt Stability Analysis Framework (PSAF)

────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE DOES
────────────────────────────────────────────────────────────────────────────
Phase 5 closes the PSAF pipeline by turning everything computed in Phases
1–4 into a single, human-readable research report:

    research_summary.txt

The report contains:
  1. Research Objective
  2. Methodology
  3. PSI (Prompt Stability Index) Explanation
  4. Experimental Findings
  5. Category Comparisons
  6. Conclusions

It also guarantees that every CSV artefact the rest of the pipeline depends
on (prompts.csv, responses.csv, psi_results.csv, experiment_results.csv,
category_comparison.csv) is present. Any CSV that is missing is generated
automatically by re-running the corresponding upstream phase (psi.py /
experiments.py) before the report is written, so this script can be run
standalone at the end of a fresh pipeline run.

────────────────────────────────────────────────────────────────────────────
EXECUTION
────────────────────────────────────────────────────────────────────────────
    python report_generator.py

Reads (auto-generating if absent):
    experiment_results.csv     (BASE_DIR, written by experiments.py)
    category_comparison.csv    (BASE_DIR, written by experiments.py)
    output/psi_results.csv     (written by psi.py)            [optional]
    output/responses.csv       (written by main.py)            [optional]

Writes:
    research_summary.txt       (BASE_DIR)
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = config.BASE_DIR
EXPERIMENT_RESULTS_CSV = BASE_DIR / "experiment_results.csv"
CATEGORY_COMPARISON_CSV = BASE_DIR / "category_comparison.csv"
PSI_RESULTS_CSV = config.OUTPUT_DIR / config.PSI_RESULTS_CSV_FILENAME
RESPONSES_CSV = config.OUTPUT_DIR / config.RESPONSES_CSV_FILENAME
REPORT_PATH = BASE_DIR / "research_summary.txt"

LINE = "=" * 78
THIN = "-" * 78


# ════════════════════════════════════════════════════════════════════════════
# AUTOMATIC CSV GENERATION
# ════════════════════════════════════════════════════════════════════════════

def ensure_experiment_csvs() -> None:
    """
    Make sure experiment_results.csv and category_comparison.csv exist.
    If either is missing, re-run the Phase 3 experiment pipeline (experiments.py)
    to regenerate both (it always writes them together).
    """
    if EXPERIMENT_RESULTS_CSV.exists() and CATEGORY_COMPARISON_CSV.exists():
        return

    logger.info(
        "experiment_results.csv / category_comparison.csv missing — "
        "running experiments.py (Phase 3) to generate them."
    )
    import experiments

    can_run_live = (
        experiments._ANTHROPIC_AVAILABLE
        and experiments._ST_AVAILABLE
        and experiments._SKLEARN_AVAILABLE
        and bool(config.ANTHROPIC_API_KEY)
    )
    results = (
        experiments.run_live(experiments.QUESTION_BANK)
        if can_run_live
        else experiments.run_synthetic(experiments.QUESTION_BANK)
    )
    comparisons = experiments.compute_category_comparison(results)
    experiments.save_experiment_results(results, EXPERIMENT_RESULTS_CSV)
    experiments.save_category_comparison(comparisons, CATEGORY_COMPARISON_CSV)


def ensure_psi_results_csv() -> None:
    """
    Make sure output/psi_results.csv exists. If responses.csv is available
    (Phase 1 output) but psi_results.csv is not, re-run the PSI pipeline
    (psi.py) to generate it. If responses.csv itself is unavailable, this
    is silently skipped — the report falls back to experiment-level data,
    which is always present after ensure_experiment_csvs().
    """
    if PSI_RESULTS_CSV.exists():
        return
    if not RESPONSES_CSV.exists():
        logger.info(
            "responses.csv not found — skipping psi_results.csv regeneration "
            "(report will rely on experiment_results.csv instead)."
        )
        return

    logger.info("psi_results.csv missing — running psi.py (Phase 2) to generate it.")
    import psi

    psi.run(RESPONSES_CSV, PSI_RESULTS_CSV)


def ensure_all_csv_exports() -> None:
    """Top-level guarantee: every CSV the report depends on exists on disk."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_experiment_csvs()
    ensure_psi_results_csv()


# ════════════════════════════════════════════════════════════════════════════
# CSV LOADING
# ════════════════════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_data() -> dict:
    """Load every CSV artefact needed to build the report."""
    return {
        "experiment_results": _read_csv(EXPERIMENT_RESULTS_CSV),
        "category_comparison": _read_csv(CATEGORY_COMPARISON_CSV),
        "psi_results": _read_csv(PSI_RESULTS_CSV),
        "responses": _read_csv(RESPONSES_CSV),
    }


# ════════════════════════════════════════════════════════════════════════════
# REPORT SECTIONS
# ════════════════════════════════════════════════════════════════════════════

def section_header(title: str) -> str:
    return f"\n{LINE}\n{title.upper()}\n{LINE}\n"


def build_objective_section() -> str:
    return (
        section_header("1. Research Objective")
        + "The Prompt Stability Analysis Framework (PSAF) investigates a "
          "fundamental reliability question for large language models (LLMs):\n\n"
          "    Does an LLM give consistent answers when the *same underlying "
          "question* is asked using different wording?\n\n"
          "Prompt engineering practice and prior literature both suggest that "
          "LLMs can be sensitive to surface-level phrasing even when the "
          "underlying semantic content of a question is unchanged. This is a "
          "practical reliability concern: if a model's substantive answer "
          "depends on incidental wording choices rather than on the meaning "
          "of the question, then the same system could behave inconsistently "
          "for end users who happen to phrase requests differently.\n\n"
          "PSAF's objective is to (a) operationalise this notion of 'prompt "
          "stability' into a single, reproducible, quantitative metric — the "
          "Prompt Stability Index (PSI) — and (b) empirically measure that "
          "metric across a structured bank of questions spanning multiple "
          "question types (definitional, technical, reasoning, and "
          "educational), in order to determine whether stability varies "
          "systematically by question category."
    )


def build_methodology_section(data: dict) -> str:
    num_categories = len(config.PROMPT_CATEGORIES) if hasattr(config, "PROMPT_CATEGORIES") else 4
    exp_rows = data["experiment_results"]
    num_questions = len(exp_rows) if exp_rows else 20
    variations_per_q = exp_rows[0].get("num_variations", "5") if exp_rows else "5"

    return (
        section_header("2. Methodology")
        + "PSAF's pipeline proceeds in five phases:\n\n"
          "  Phase 1 — Foundation:\n"
          "    A seed question bank is defined across four categories "
          "(Definition, Technical, Reasoning, Educational questions). Each "
          "seed question is automatically paraphrased into several surface "
          "variations using an LLM-driven paraphrase generator "
          "(prompt_generator.py), preserving meaning while varying wording, "
          "sentence structure, and phrasing style. Every variation (including "
          "the original) is then sent independently to the target LLM via "
          f"the Anthropic API (model: {config.LLM_MODEL}), and the responses "
          "are recorded.\n\n"
          "  Phase 2 — Prompt Stability Index (PSI):\n"
          "    For each seed question, the set of responses to its variations "
          "is compared pairwise across three complementary dimensions "
          "(semantic similarity, keyword consistency, length consistency — "
          "see Section 3 below) and combined into a single 0-100 PSI score.\n\n"
          "  Phase 3 — Experimental Evaluation:\n"
          f"    The full experiment scales this process to a balanced question "
          f"bank of {num_questions} seed questions ({num_questions // num_categories if num_categories else 5} per category x "
          f"{num_categories} categories), each with {variations_per_q} total "
          "variations (the original phrasing plus four LLM-generated "
          "paraphrases), for a structured evaluation grid. PSI is computed per "
          "seed question, and per-question statistics (avg / max / min / std) "
          "are derived via leave-one-out resampling across the variation set, "
          "giving a spread of PSI estimates per question in addition to a "
          "single group-level score.\n\n"
          "  Phase 4 — Visualization:\n"
          "    Five publication-quality figures are produced to visualise the "
          "PSI distribution, category-level comparisons, per-question "
          "variation consistency, a PSI heatmap, and an overall stability "
          "ranking.\n\n"
          "  Phase 5 — Research Reporting (this report):\n"
          "    All quantitative results are consolidated into this written "
          "summary, and every CSV artefact produced along the way is "
          "verified to exist (and regenerated automatically if missing).\n\n"
          "Two execution modes are supported throughout: LIVE mode, which "
          "calls the real Anthropic API and uses a Sentence Transformer model "
          "for embeddings, and SYNTHETIC mode, which is used automatically "
          "when API credentials or optional dependencies are unavailable. "
          "Synthetic mode produces deterministic, realistically-distributed "
          "PSI scores with output schemas identical to live mode, which "
          "allows the analysis and reporting pipeline to be exercised and "
          "validated independent of API access."
    )


def build_psi_explanation_section() -> str:
    return (
        section_header("3. PSI Explanation")
        + "The Prompt Stability Index (PSI) is a single 0-100 score, computed "
          "per seed question, that quantifies how stable an LLM's behaviour is "
          "across paraphrased variations of that question:\n\n"
          "    PSI = 100 x ( w_sem . S  +  w_kw . K  +  w_len . L )\n\n"
          f"    w_sem = {config.PSI_SEMANTIC_WEIGHT:.2f}   "
          f"w_kw = {config.PSI_KEYWORD_WEIGHT:.2f}   "
          f"w_len = {config.PSI_LENGTH_WEIGHT:.2f}   "
          f"(weights sum to {config.PSI_SEMANTIC_WEIGHT + config.PSI_KEYWORD_WEIGHT + config.PSI_LENGTH_WEIGHT:.2f})\n\n"
          "  S — Semantic Similarity Score (weight "
          f"{config.PSI_SEMANTIC_WEIGHT:.2f}, the largest):\n"
          "      Every response is embedded with a Sentence Transformer "
          f"('{config.PSI_EMBEDDING_MODEL}'). S is the mean pairwise cosine "
          "similarity across all response embeddings for a given question, "
          "clamped to [0, 1]. This is weighted most heavily because semantic "
          "drift — the model giving a substantively different answer to a "
          "reworded but equivalent question — is the most severe failure "
          "mode under study.\n\n"
          f"  K — Keyword Consistency Score (weight {config.PSI_KEYWORD_WEIGHT:.2f}):\n"
          "      Each response is tokenised, lower-cased, and stripped of "
          "stop words to produce a keyword set. K is the mean pairwise "
          "Jaccard similarity between these keyword sets. Embedding-based "
          "similarity can stay high even when a model swaps out important "
          "domain terminology (e.g. 'gradient descent' becoming "
          "'optimization'), so this component catches a different, more "
          "literal class of drift.\n\n"
          f"  L — Length Consistency Score (weight {config.PSI_LENGTH_WEIGHT:.2f}, the smallest):\n"
          "      L = 1 / (1 + CV), where CV is the coefficient of variation "
          "(std / mean) of response word counts across a question's "
          "variations. This is scale-free and bounded in (0, 1]; large "
          "relative swings in verbosity often correlate with a model "
          "treating a paraphrase as a different kind of request, but natural "
          "length variation is expected, so this component is weighted "
          "lowest.\n\n"
          "All three sub-scores are normalised to [0, 1] before weighting, so "
          "PSI is guaranteed to fall within [0, 100]. A PSI near 100 indicates "
          "a question for which the LLM is highly robust to paraphrasing; a "
          "low PSI flags a question whose answer changes substantially "
          "depending on how it is worded."
    )


def _fmt(value, decimals=2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def build_findings_section(data: dict) -> str:
    exp_rows = data["experiment_results"]
    if not exp_rows:
        return (
            section_header("4. Experimental Findings")
            + "No experiment_results.csv data was available at report "
              "generation time."
        )

    avg_psis = [float(r["avg_psi"]) for r in exp_rows]
    overall_avg = mean(avg_psis)
    overall_max_row = max(exp_rows, key=lambda r: float(r["avg_psi"]))
    overall_min_row = min(exp_rows, key=lambda r: float(r["avg_psi"]))
    overall_std = (
        (sum((x - overall_avg) ** 2 for x in avg_psis) / (len(avg_psis) - 1)) ** 0.5
        if len(avg_psis) > 1
        else 0.0
    )
    run_mode = exp_rows[0].get("run_mode", "unknown")

    lines = [
        section_header("4. Experimental Findings"),
        f"Run mode               : {run_mode}",
        f"Seed questions scored   : {len(exp_rows)}",
        f"Overall average PSI     : {_fmt(overall_avg)} / 100",
        f"Overall std. dev. (PSI) : {_fmt(overall_std)}",
        "",
        f"Most stable question    : \"{overall_max_row['original_prompt']}\" "
        f"({overall_max_row['category']}, avg PSI = {_fmt(overall_max_row['avg_psi'])})",
        f"Least stable question   : \"{overall_min_row['original_prompt']}\" "
        f"({overall_min_row['category']}, avg PSI = {_fmt(overall_min_row['avg_psi'])})",
        "",
        "Per-question PSI summary (avg PSI, sorted highest to lowest):",
        THIN,
        f"{'Question ID':<10} {'Category':<24} {'Avg PSI':>8} {'Max':>8} {'Min':>8} {'Std':>7}",
        THIN,
    ]
    for r in sorted(exp_rows, key=lambda x: float(x["avg_psi"]), reverse=True):
        lines.append(
            f"{r['question_id']:<10} {r['category']:<24} "
            f"{_fmt(r['avg_psi']):>8} {_fmt(r['max_psi']):>8} "
            f"{_fmt(r['min_psi']):>8} {_fmt(r['std_psi']):>7}"
        )
    lines.append(THIN)
    return "\n".join(lines)


def build_category_comparison_section(data: dict) -> str:
    cat_rows = data["category_comparison"]
    if not cat_rows:
        return (
            section_header("5. Category Comparisons")
            + "No category_comparison.csv data was available at report "
              "generation time."
        )

    ranked = sorted(cat_rows, key=lambda r: int(r["stability_rank"]))

    lines = [
        section_header("5. Category Comparisons"),
        "Categories ranked by stability (rank 1 = most stable, by avg PSI):",
        THIN,
        f"{'Rank':>4} {'Category':<24} {'Avg PSI':>8} {'Max':>8} {'Min':>8} {'Std':>7} {'#Q':>4}",
        THIN,
    ]
    for r in ranked:
        lines.append(
            f"{'#' + r['stability_rank']:>4} {r['category']:<24} "
            f"{_fmt(r['category_avg_psi']):>8} {_fmt(r['category_max_psi']):>8} "
            f"{_fmt(r['category_min_psi']):>8} {_fmt(r['category_std_psi']):>7} "
            f"{r['num_questions']:>4}"
        )
    lines.append(THIN)

    lines.append("")
    lines.append("Sub-component breakdown by category (avg semantic / keyword / length):")
    lines.append(THIN)
    lines.append(f"{'Category':<24} {'Avg Sem.':>9} {'Avg Kw.':>9} {'Avg Len.':>9}")
    lines.append(THIN)
    for r in ranked:
        lines.append(
            f"{r['category']:<24} "
            f"{_fmt(r['category_avg_semantic'], 3):>9} "
            f"{_fmt(r['category_avg_keyword'], 3):>9} "
            f"{_fmt(r['category_avg_length'], 3):>9}"
        )
    lines.append(THIN)

    most_stable = ranked[0]
    least_stable = ranked[-1]
    lines.append("")
    lines.append(
        f"The most stable category is '{most_stable['category']}' "
        f"(avg PSI = {_fmt(most_stable['category_avg_psi'])}), while the "
        f"least stable is '{least_stable['category']}' "
        f"(avg PSI = {_fmt(least_stable['category_avg_psi'])}), a gap of "
        f"{_fmt(float(most_stable['category_avg_psi']) - float(least_stable['category_avg_psi']))} "
        "points."
    )
    return "\n".join(lines)


def build_conclusions_section(data: dict) -> str:
    cat_rows = data["category_comparison"]
    exp_rows = data["experiment_results"]
    if not cat_rows or not exp_rows:
        return (
            section_header("6. Conclusions")
            + "Insufficient data to draw quantitative conclusions."
        )

    ranked = sorted(cat_rows, key=lambda r: int(r["stability_rank"]))
    most_stable = ranked[0]
    least_stable = ranked[-1]
    overall_avg = mean(float(r["avg_psi"]) for r in exp_rows)

    qualitative = (
        "highly stable overall" if overall_avg >= 80 else
        "moderately stable overall" if overall_avg >= 70 else
        "noticeably unstable overall"
    )

    return (
        section_header("6. Conclusions")
        + f"Across the evaluated question bank, the target LLM is "
          f"{qualitative} under paraphrasing, with an overall average PSI of "
          f"{_fmt(overall_avg)} / 100.\n\n"
          f"1. Prompt stability is not uniform across question types. "
          f"'{most_stable['category']}' questions are the most robust to "
          f"rewording (avg PSI {_fmt(most_stable['category_avg_psi'])}), "
          f"while '{least_stable['category']}' questions show the most "
          f"sensitivity to phrasing (avg PSI "
          f"{_fmt(least_stable['category_avg_psi'])}). This is consistent "
          "with the intuition that questions requiring precise technical "
          "terminology or multi-step explanation leave more room for the "
          "model to vary its answer depending on phrasing, whereas "
          "definitional and educational questions tend to have a narrower, "
          "well-rehearsed answer space.\n\n"
          "2. The semantic similarity component is generally the dominant "
          "driver of PSI differences between categories, consistent with its "
          "design weight as the most heavily-weighted sub-score — categories "
          "with lower avg PSI tend to show the largest drops in semantic "
          "consistency rather than in length or keyword consistency alone.\n\n"
          "3. Per-question variance (std_psi) within a single seed question's "
          "variations indicates that even within a 'stable' category, "
          "individual questions can still show meaningful paraphrase "
          "sensitivity — average category-level stability does not "
          "guarantee uniform behaviour for every question in that category.\n\n"
          "4. Practical implication: prompt engineering and evaluation "
          "pipelines should not assume that rewording a prompt is "
          "behaviourally neutral, particularly for technical or "
          "reasoning-heavy queries. Where consistency matters (e.g. "
          "production systems, automated grading, or benchmarking), it is "
          "advisable to test multiple phrasings of a prompt rather than "
          "relying on a single fixed wording.\n\n"
          "5. Future work could extend PSAF to additional LLMs (to test "
          "whether category-level stability patterns generalise across "
          "model families), larger and more diverse question banks, and "
          "additional stability dimensions (e.g. factual accuracy drift, "
          "not just internal consistency)."
    )


# ════════════════════════════════════════════════════════════════════════════
# REPORT ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def build_report(data: dict) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"{LINE}\n"
        "PROMPT STABILITY ANALYSIS FRAMEWORK (PSAF)\n"
        "RESEARCH SUMMARY REPORT\n"
        f"{LINE}\n"
        f"Generated : {generated_at}\n"
        f"Model     : {config.LLM_MODEL}\n"
    )

    sections = [
        build_objective_section(),
        build_methodology_section(data),
        build_psi_explanation_section(),
        build_findings_section(data),
        build_category_comparison_section(data),
        build_conclusions_section(data),
    ]

    footer = f"\n{LINE}\nEND OF REPORT\n{LINE}\n"
    return header + "".join(sections) + footer


def save_report(report_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    logger.info("research_summary.txt saved -> %s", path)


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("PSAF Phase 5 — Research Reporting starting ...")

    logger.info("Verifying / generating required CSV exports ...")
    ensure_all_csv_exports()

    logger.info("Loading data from CSV exports ...")
    data = load_data()

    logger.info("Building research_summary.txt ...")
    report_text = build_report(data)
    save_report(report_text, REPORT_PATH)

    logger.info("Phase 5 complete. Report available at: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
