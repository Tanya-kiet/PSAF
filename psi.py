"""
psi.py — Prompt Stability Index (PSI): the Phase 2 research contribution of the
Prompt Stability Analysis Framework (PSAF).

──────────────────────────────────────────────────────────────────────────────
WHAT PSI MEASURES
──────────────────────────────────────────────────────────────────────────────
Phase 1 generated paraphrased *variations* of each seed prompt and recorded the
LLM's response to every variation. If an LLM is truly robust to surface-level
rewording, then all responses belonging to the same seed prompt (the original
plus its paraphrases) should be:

  1. Semantically equivalent          → say the same thing, even if worded
                                          differently.
  2. Lexically consistent             → reuse the same core technical
                                          vocabulary / key terms.
  3. Similar in scale / verbosity     → not wildly different in length,
                                          which is often a symptom of the
                                          model "noticing" a phrasing change
                                          and treating it as a different
                                          kind of question.

The Prompt Stability Index (PSI) is a single 0–100 score, computed per seed
prompt, that quantifies how stable an LLM's behaviour is across paraphrases
of the same underlying question.

──────────────────────────────────────────────────────────────────────────────
WHY THESE THREE COMPONENTS, AND WHY THESE WEIGHTS
──────────────────────────────────────────────────────────────────────────────
PSI = 100 × ( w_sem · S  +  w_kw · K  +  w_len · L )

with  w_sem = 0.50, w_kw = 0.30, w_len = 0.20   (w_sem + w_kw + w_len = 1.0)

  • S (Semantic Similarity Score, weight 0.50): measures whether the *meaning*
    of the responses is preserved. This is the most important signal: a
    prompt could be perfectly "stable" in wording/length but still receive a
    factually different answer, which is the real failure mode we care about.
    It is given the largest weight because semantic drift is the most severe
    form of instability.

  • K (Keyword Consistency Score, weight 0.30): semantic similarity from a
    sentence embedding can be high even if a model swaps out important
    domain terminology (e.g. "gradient descent" → "optimization"), because
    embeddings capture *gist*, not precise terminology. Keyword overlap is a
    complementary, more literal check that the model is grounding its
    answers in the same concepts. It is weighted second-highest because it
    catches a different class of drift than S.

  • L (Length Consistency Score, weight 0.20): captures whether the model's
    *verbosity/scope* stays stable. This is the weakest signal of true
    instability (some natural length variation is expected and harmless), so
    it receives the smallest weight, but large length swings often
    correlate with a model treating a paraphrase as a meaningfully different
    request (e.g. answering one phrasing in one sentence and another with a
    five-paragraph essay), so it still contributes.

All three sub-scores are independently normalised to the [0, 1] range before
weighting, so PSI is guaranteed to fall in [0, 100]. A PSI near 100 indicates
a seed prompt for which the LLM is highly robust to paraphrasing; a low PSI
flags a prompt whose answer changes substantially depending on how it is
worded.

──────────────────────────────────────────────────────────────────────────────
HOW EACH COMPONENT IS COMPUTED
──────────────────────────────────────────────────────────────────────────────
For a seed prompt with responses r_1 … r_n (the original's response plus all
of its paraphrases' responses):

  S — Semantic Similarity Score
      1. Encode every r_i with a Sentence Transformer
         ("all-MiniLM-L6-v2") to get dense embeddings.
      2. Compute the pairwise cosine similarity between every pair (r_i, r_j),
         i ≠ j.
      3. S = mean of all pairwise cosine similarities, clamped to [0, 1]
         (cosine similarity can be slightly negative in theory; in practice
         for semantically related natural-language answers it is virtually
         always positive, but we clamp defensively so PSI never leaves its
         documented range).

  K — Keyword Consistency Score
      1. Tokenise each r_i, lower-case, strip punctuation, and remove
         English stop words (scikit-learn's standard stop-word list) to get
         a keyword set KW_i.
      2. Compute the pairwise Jaccard similarity |KW_i ∩ KW_j| / |KW_i ∪ KW_j|
         for every pair.
      3. K = mean of all pairwise Jaccard similarities (already in [0, 1]
         by construction).

  L — Length Consistency Score
      1. Compute the word count len_i = number of whitespace-delimited
         tokens in r_i.
      2. Compute the coefficient of variation CV = std(len) / mean(len),
         a scale-free measure of relative dispersion (it would be unfair to
         use raw standard deviation, since a 50-word swing means something
         different for a 20-word answer than for a 500-word answer).
      3. L = 1 / (1 + CV)  → CV = 0 (identical lengths) gives L = 1.0;
         as the relative spread grows, L smoothly decays toward 0, asymptotic
         and bounded in (0, 1] by construction.

──────────────────────────────────────────────────────────────────────────────
REQUIRED LIBRARIES
──────────────────────────────────────────────────────────────────────────────
  • sentence-transformers  → Sentence Transformer embeddings (S)
  • scikit-learn           → cosine_similarity() and the English stop-word
                              list used for keyword extraction
  • numpy                  → vectorised pairwise statistics
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)

# ── Tunable parameters (component weights live in config.py) ────────────────
_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
_MIN_KEYWORD_LEN = 3  # discard very short tokens (a, an, ml-noise, etc.)

# Sentence-transformer model is loaded lazily and cached so that repeated
# calls to compute_psi_for_groups() within one process only pay the model
# load cost once.
_model_cache: "SentenceTransformer | None" = None  # noqa: F821 (forward ref)


def _get_embedding_model():
    """Lazily load (and cache) the Sentence Transformer embedding model."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading Sentence Transformer model '%s' …", config.PSI_EMBEDDING_MODEL)
        _model_cache = SentenceTransformer(config.PSI_EMBEDDING_MODEL)
    return _model_cache


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PSIResult:
    """PSI score and its three sub-components for one seed-prompt group."""
    prompt_group_id:       str    # e.g. "DEF_0"  (category abbrev + seed idx)
    category:               str
    original_prompt:        str
    num_variations_scored:  int    # how many non-error responses were compared
    semantic_similarity:    float  # S, in [0, 1]
    keyword_consistency:    float  # K, in [0, 1]
    length_consistency:     float  # L, in [0, 1]
    psi_score:               float  # final PSI, in [0, 100]

    def to_dict(self) -> dict:
        return {
            "prompt_group_id":      self.prompt_group_id,
            "category":             self.category,
            "original_prompt":      self.original_prompt,
            "num_variations_scored": self.num_variations_scored,
            "semantic_similarity":   round(self.semantic_similarity, 4),
            "keyword_consistency":   round(self.keyword_consistency, 4),
            "length_consistency":    round(self.length_consistency, 4),
            "psi_score":             round(self.psi_score, 2),
        }


# ── CSV loading & grouping ───────────────────────────────────────────────────

def load_responses(path: Path) -> list[dict]:
    """Read responses.csv (produced by Phase 1 / main.py) into a list of dicts."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_by_prompt(rows: list[dict]) -> dict[str, list[dict]]:
    """
    Group response rows by seed prompt.

    Each prompt_id has the form '<CAT_ABBR>_<seed_idx>_var<N>'
    (e.g. 'DEF_0_var0', 'DEF_0_var1', ...). Stripping the trailing
    '_var<N>' segment recovers the group key shared by the original prompt
    and all of its paraphrases.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group_id = re.sub(r"_var\d+$", "", row["prompt_id"])
        groups.setdefault(group_id, []).append(row)
    return groups


# ── Component computations ───────────────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    """Lower-case, tokenise, and remove stop words / short tokens."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return {
        t for t in tokens
        if len(t) >= _MIN_KEYWORD_LEN and t not in ENGLISH_STOP_WORDS
    }


def _pairwise_mean(values_matrix: np.ndarray) -> float:
    """Mean of the strictly-upper-triangular part of a square similarity matrix."""
    n = values_matrix.shape[0]
    if n < 2:
        return 1.0  # a single response is trivially "consistent" with itself
    iu = np.triu_indices(n, k=1)
    return float(np.mean(values_matrix[iu]))


def compute_semantic_similarity(responses: list[str]) -> float:
    """
    S — mean pairwise cosine similarity between Sentence Transformer
    embeddings of the responses, clamped to [0, 1].
    """
    if len(responses) < 2:
        return 1.0

    model = _get_embedding_model()
    embeddings = model.encode(responses, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    raw_score = _pairwise_mean(sim_matrix)
    return max(0.0, min(1.0, raw_score))  # defensive clamp


def compute_keyword_consistency(responses: list[str]) -> float:
    """
    K — mean pairwise Jaccard similarity between each response's keyword set.
    """
    if len(responses) < 2:
        return 1.0

    keyword_sets = [_extract_keywords(r) for r in responses]
    n = len(keyword_sets)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = keyword_sets[i], keyword_sets[j]
            if not a and not b:
                sims.append(1.0)  # both empty → trivially identical
            else:
                sims.append(len(a & b) / len(a | b))
    return float(mean(sims)) if sims else 1.0


def compute_length_consistency(responses: list[str]) -> float:
    """
    L — 1 / (1 + CV), where CV is the coefficient of variation of word
    counts across the responses. Scale-free, bounded in (0, 1].
    """
    if len(responses) < 2:
        return 1.0

    lengths = [len(r.split()) for r in responses]
    avg_len = mean(lengths)
    if avg_len == 0:
        return 0.0  # every response was empty — treat as fully unstable

    std_len = pstdev(lengths)
    cv = std_len / avg_len
    return 1.0 / (1.0 + cv)


# ── PSI aggregation ───────────────────────────────────────────────────────────

def compute_psi_for_group(group_id: str, rows: list[dict]) -> PSIResult | None:
    """
    Compute the PSI score for a single seed-prompt group.
    Rows with a non-empty 'error' field (failed API calls) are excluded
    since they carry no response text to compare.
    """
    valid_rows = [r for r in rows if not r.get("error") and r.get("response_text", "").strip()]

    if len(valid_rows) < 2:
        logger.warning(
            "Group '%s' has fewer than 2 valid responses (%d) — skipping PSI.",
            group_id, len(valid_rows),
        )
        return None

    responses = [r["response_text"] for r in valid_rows]
    category = valid_rows[0]["category"]
    original_prompt = valid_rows[0]["original"]

    s = compute_semantic_similarity(responses)
    k = compute_keyword_consistency(responses)
    l = compute_length_consistency(responses)

    psi_score = 100.0 * (
        config.PSI_SEMANTIC_WEIGHT * s
        + config.PSI_KEYWORD_WEIGHT * k
        + config.PSI_LENGTH_WEIGHT * l
    )
    # Defensive clamp: guarantees PSI ∈ [0, 100] even under floating-point edge cases.
    psi_score = max(0.0, min(100.0, psi_score))

    return PSIResult(
        prompt_group_id=group_id,
        category=category,
        original_prompt=original_prompt,
        num_variations_scored=len(valid_rows),
        semantic_similarity=s,
        keyword_consistency=k,
        length_consistency=l,
        psi_score=psi_score,
    )


def compute_psi_for_all_groups(rows: list[dict]) -> list[PSIResult]:
    """Compute PSI for every seed-prompt group found in `rows`."""
    groups = group_by_prompt(rows)
    results: list[PSIResult] = []

    for group_id, group_rows in groups.items():
        result = compute_psi_for_group(group_id, group_rows)
        if result is not None:
            results.append(result)

    # Sort by category then prompt_group_id for a stable, readable output order.
    results.sort(key=lambda r: (r.category, r.prompt_group_id))
    return results


# ── CSV output ────────────────────────────────────────────────────────────────

def save_psi_csv(results: list[PSIResult], path: Path) -> None:
    """Write PSIResult objects to psi_results.csv."""
    if not results:
        logger.warning("No PSI results to save.")
        return

    fieldnames = list(results[0].to_dict().keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    logger.info("PSI results saved → %s  (%d rows)", path, len(results))


# ── Orchestration entry point (used by main.py and standalone runs) ─────────

def run(responses_path: Path, output_path: Path) -> list[PSIResult]:
    """
    Full PSI pipeline: load responses.csv → group by seed prompt →
    compute PSI per group → save psi_results.csv.
    Returns the computed results list for any in-process use (e.g. logging
    a summary in main.py).
    """
    logger.info("Loading responses from %s …", responses_path)
    rows = load_responses(responses_path)
    logger.info("Loaded %d response rows.", len(rows))

    logger.info("Computing Prompt Stability Index (PSI) per seed prompt …")
    results = compute_psi_for_all_groups(rows)
    logger.info("Computed PSI for %d seed-prompt groups.", len(results))

    save_psi_csv(results, output_path)
    return results


# ── Standalone CLI usage ──────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    responses_path = config.OUTPUT_DIR / config.RESPONSES_CSV_FILENAME
    output_path = config.OUTPUT_DIR / config.PSI_RESULTS_CSV_FILENAME

    if not responses_path.exists():
        logger.error(
            "responses.csv not found at %s — run main.py (Phase 1) first.",
            responses_path,
        )
        return

    run(responses_path, output_path)


if __name__ == "__main__":
    main()
