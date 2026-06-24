"""
backend.py — Optimized PSAF experiment engine.

Design goals:
  • ONE batch of Groq calls per "Run" click — no repeated calls on tab switch.
  • Results cached to disk (JSON) so re-runs are instant if nothing changed.
  • SentenceTransformer loaded once per process (module-level singleton).
  • Max 5 variations per question (configurable in config.py).
  • Progress reported via a callback so the Streamlit UI can update a bar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Optional

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)

# ── Singleton embedding model ─────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading SentenceTransformer '%s'…", config.PSI_EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(config.PSI_EMBEDDING_MODEL)
    return _embed_model


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VariationResult:
    variation: str
    response: str


@dataclass
class QuestionResult:
    category: str
    question_id: str
    original_prompt: str
    variations: list[VariationResult]
    # PSI components
    psi_score: float
    semantic_similarity: float
    keyword_consistency: float
    length_consistency: float
    # similarity matrix (list of lists for JSON serialisability)
    similarity_matrix: list[list[float]]


@dataclass
class ExperimentResult:
    questions: list[QuestionResult]
    category_stats: dict[str, dict]  # category → {avg, max, min, rank}
    groq_calls_made: int
    total_time_s: float


# ── Groq helpers ──────────────────────────────────────────────────────────────

def _groq_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it in your shell:\n"
            "  export GROQ_API_KEY='gsk_...'"
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _call_groq(client: Groq, messages: list[dict], temperature: float = 0.7) -> str:
    """Single Groq call with exponential-backoff retry."""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=config.MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("Groq attempt %d failed (%s) — waiting %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    return ""


# ── Variation generation ──────────────────────────────────────────────────────

_VARIATION_SYSTEM = (
    "You are a prompt rewriting expert. "
    "Rewrite the given question in different ways while preserving its exact meaning. "
    "Each rewrite must differ in wording and structure but ask the same thing. "
    "Return ONLY a valid JSON array of strings — no markdown, no explanation."
)


def generate_variations(client: Groq, seed: str, n: int = config.MAX_VARIATIONS) -> list[str]:
    """Generate up to `n` paraphrases of `seed` using one Groq call."""
    user_msg = f"Rewrite this question in {n} different ways:\n\n{seed}"
    raw = _call_groq(
        client,
        [{"role": "system", "content": _VARIATION_SYSTEM},
         {"role": "user", "content": user_msg}],
        temperature=config.PARAPHRASE_TEMPERATURE,
    )
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            variations = [str(v) for v in parsed if str(v).strip()][:n]
            if len(variations) < n:
                variations += [seed] * (n - len(variations))
            return variations
    except Exception:
        pass
    return [seed] * n


# ── PSI computation ───────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _keywords(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) > 2 and t.lower() not in ENGLISH_STOP_WORDS
    }


def compute_psi(responses: list[str]) -> tuple[float, float, float, float, list[list[float]]]:
    """
    Returns (psi, S, K, L, similarity_matrix).
    All components in [0,1] except psi which is in [0,100].
    """
    valid = [r for r in responses if r.strip()]
    if len(valid) < 2:
        n = max(len(valid), 1)
        return 0.0, 0.0, 0.0, 0.0, [[1.0] * n] * n

    model = get_embed_model()
    embs = model.encode(valid, show_progress_bar=False)
    sim_matrix = cosine_similarity(embs)

    idx = np.triu_indices_from(sim_matrix, k=1)
    S = float(np.clip(np.mean(sim_matrix[idx]), 0, 1))

    sets = [_keywords(r) for r in valid]
    K_scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            K_scores.append(len(a & b) / len(a | b) if (a or b) else 1.0)
    K = float(mean(K_scores)) if K_scores else 1.0

    lengths = [len(r.split()) for r in valid]
    mu = mean(lengths)
    L = 1.0 / (1.0 + (pstdev(lengths) / mu)) if mu > 0 else 0.0

    psi = float(np.clip(
        100 * (config.PSI_SEMANTIC_WEIGHT * S
               + config.PSI_KEYWORD_WEIGHT * K
               + config.PSI_LENGTH_WEIGHT * L),
        0, 100,
    ))

    return psi, S, K, L, sim_matrix.tolist()


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(category: str, prompt: str, n_variations: int) -> str:
    raw = f"{category}|{prompt}|{n_variations}|{config.LLM_MODEL}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(key: str, data: dict) -> None:
    _cache_path(key).write_text(json.dumps(data, indent=2))


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    category: str,
    prompt: str,
    n_variations: int = config.MAX_VARIATIONS,
    force_rerun: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> QuestionResult:
    """
    Run a single question experiment.

    Uses disk cache unless `force_rerun=True`.
    Calls `progress_cb(message)` at each stage if provided.
    """
    def emit(msg: str):
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    cache_key = _cache_key(category, prompt, n_variations)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if not force_rerun:
        cached = _load_cache(cache_key)
        if cached:
            emit(f"✅ Cache hit for: {prompt[:50]}…")
            qr = QuestionResult(**cached)
            qr.variations = [VariationResult(**v) for v in cached["variations"]]
            return qr

    # ── Cache miss: call Groq ─────────────────────────────────────────────────
    client = _groq_client()
    groq_calls = 0

    emit(f"🔀 Generating {n_variations} variations…")
    variations = generate_variations(client, prompt, n_variations)
    groq_calls += 1
    time.sleep(0.3)  # brief pause to avoid rate-limit burst

    all_prompts = [prompt] + variations  # original + paraphrases
    responses: list[str] = []

    for i, var in enumerate(all_prompts):
        emit(f"🤖 Getting LLM response {i + 1}/{len(all_prompts)}…")
        resp = _call_groq(client, [{"role": "user", "content": var}])
        responses.append(resp)
        groq_calls += 1
        if i < len(all_prompts) - 1:
            time.sleep(0.5)  # rate-limit breathing room

    emit("📐 Computing PSI score…")
    psi, S, K, L, sim_matrix = compute_psi(responses)

    variation_results = [
        VariationResult(variation=v, response=r)
        for v, r in zip(all_prompts, responses)
    ]

    qid = re.sub(r"\W+", "_", f"{category}_{prompt}")[:40]
    result = QuestionResult(
        category=category,
        question_id=qid,
        original_prompt=prompt,
        variations=variation_results,
        psi_score=psi,
        semantic_similarity=S,
        keyword_consistency=K,
        length_consistency=L,
        similarity_matrix=sim_matrix,
    )

    # serialise for cache
    data = asdict(result)
    _save_cache(cache_key, data)
    emit(f"💾 Result cached ({groq_calls} Groq calls made)")

    return result


def run_category_experiment(
    category: str,
    n_variations: int = config.MAX_VARIATIONS,
    force_rerun: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[QuestionResult]:
    """Run all questions in a category."""
    prompts = config.PROMPT_CATEGORIES.get(category, [])
    results = []
    for i, prompt in enumerate(prompts):
        if progress_cb:
            progress_cb(f"Question {i+1}/{len(prompts)}: {prompt[:50]}…")
        result = run_experiment(category, prompt, n_variations, force_rerun, progress_cb)
        results.append(result)
    return results


def compute_category_stats(results: list[QuestionResult]) -> dict[str, dict]:
    """Aggregate PSI stats per category."""
    groups: dict[str, list[float]] = {}
    for r in results:
        groups.setdefault(r.category, []).append(r.psi_score)

    stats = {}
    for cat, scores in groups.items():
        stats[cat] = {
            "avg": round(mean(scores), 2),
            "max": round(max(scores), 2),
            "min": round(min(scores), 2),
            "count": len(scores),
        }

    # Rank by avg PSI descending
    ranked = sorted(stats.items(), key=lambda x: x[1]["avg"], reverse=True)
    for rank, (cat, _) in enumerate(ranked, 1):
        stats[cat]["rank"] = rank

    return stats
