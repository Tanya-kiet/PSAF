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
from providers import get_provider

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

# Safe non-empty fallback so compute_psi never sees an empty response
_FALLBACK_RESPONSE = "[No response — provider call failed]"


def generate_variations(client: Groq, seed: str, n: int = config.MAX_VARIATIONS) -> list[str]:
    """Generate up to `n` paraphrases of `seed` using one Groq call.

    Legacy entry-point used by experiments.py.  Now uses the stronger
    variation prompt and includes the diversity safety check.
    """
    n_request = n + 2
    user_msg = (
        f"Rewrite the following question in EXACTLY {n_request} different ways. "
        f"Each version must use a noticeably different sentence structure and vocabulary.\n\n"
        f"Question: {seed}\n\n"
        f"Return ONLY a JSON array of {n_request} strings."
    )
    raw = _call_groq(
        client,
        [{"role": "system", "content": _VARIATION_SYSTEM},
         {"role": "user", "content": user_msg}],
        temperature=config.PARAPHRASE_TEMPERATURE,
    )

    logger.debug("generate_variations (legacy) raw reply: %r", raw[:300])

    variations: list[str] = []
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            variations = [str(v).strip() for v in parsed if str(v).strip()]
    except Exception as exc:
        logger.warning("generate_variations: JSON parse failed (%s)", exc)

    # Filter out copies of the seed
    unique = [v for v in variations if v.strip().lower() != seed.strip().lower()]
    if not unique:
        logger.warning("generate_variations: no diverse variations produced; using seed copies.")
        unique = variations or [seed]

    # Deduplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for v in unique:
        k = v.strip().lower()
        if k not in seen:
            seen.add(k)
            deduped.append(v)

    result = deduped[:n]
    if len(result) < n:
        result += [seed] * (n - len(result))

    logger.debug("generate_variations: returning %d variations, %d unique",
                 len(result), len({v.lower() for v in result}))
    return result


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
    # Replace empty strings with the fallback token so they survive the valid filter
    sanitised = [r if r.strip() else _FALLBACK_RESPONSE for r in responses]
    valid = [r for r in sanitised if r.strip()]

    # ── Debug logging ──────────────────────────────────────────────────────────
    logger.debug("compute_psi: %d responses passed in, %d non-empty", len(responses), len(valid))
    for i, r in enumerate(valid):
        logger.debug("  response[%d] (%d words): %r…", i, len(r.split()), r[:80])

    if len(valid) < 2:
        logger.warning(
            "compute_psi: fewer than 2 valid responses (%d) — returning 0.0. "
            "Check that the LLM is actually returning text.", len(valid)
        )
        n = max(len(valid), 1)
        return 0.0, 0.0, 0.0, 0.0, [[1.0] * n] * n

    model = get_embed_model()
    embs = model.encode(valid, show_progress_bar=False)
    sim_matrix = cosine_similarity(embs)

    # ── Debug: print similarity matrix ────────────────────────────────────────
    logger.debug("compute_psi: embedding shape = %s", embs.shape)
    logger.debug("compute_psi: similarity matrix =\n%s",
                 "\n".join("  " + " ".join(f"{v:.3f}" for v in row) for row in sim_matrix))

    idx = np.triu_indices_from(sim_matrix, k=1)
    _vals = sim_matrix[idx]
    S = float(np.clip(np.mean(_vals), 0, 1)) if _vals.size > 0 else 1.0

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

    logger.debug("compute_psi: S=%.4f  K=%.4f  L=%.4f  PSI=%.2f", S, K, L, psi)

    return psi, S, K, L, sim_matrix.tolist()


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(category: str, prompt: str, n_variations: int, provider_name: str = "groq") -> str:
    # Include provider_name so Groq and OpenAI results never share a cache entry
    raw = f"{category}|{prompt}|{n_variations}|{config.LLM_MODEL}|{provider_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # ── Stale-cache guard ─────────────────────────────────────────────
            # Old runs with broken variation generation stored PSI=0.0 with
            # all-identical variation texts.  Detect and evict those entries so
            # they don't replay incorrect results after the fix.
            variations = data.get("variations", [])
            if variations:
                var_texts = {str(v.get("variation", "")).strip().lower() for v in variations}
                if len(var_texts) <= 1 and data.get("psi_score", -1) == 0.0:
                    logger.warning(
                        "_load_cache: evicting stale zero-PSI cache entry %s "
                        "(all-identical variations detected).", key
                    )
                    p.unlink(missing_ok=True)
                    return None
            return data
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
    provider_name: str = "groq",
) -> QuestionResult:
    """
    Run a single question experiment using the specified LLM provider.

    Uses disk cache unless `force_rerun=True`.
    Calls `progress_cb(message)` at each stage if provided.

    Args:
        category:      Prompt category label.
        prompt:        The seed prompt to evaluate.
        n_variations:  Number of paraphrases to generate (default from config).
        force_rerun:   Bypass cache and re-run all LLM calls when True.
        progress_cb:   Optional callback for progress messages.
        provider_name: LLM backend to use — "groq" (default) or "openai".
                       Defaults to "groq" so all existing call-sites that
                       omit this argument behave exactly as before.
    """
    def emit(msg: str):
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    cache_key = _cache_key(category, prompt, n_variations, provider_name)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if not force_rerun:
        cached = _load_cache(cache_key)
        if cached:
            emit(f"✅ Cache hit for: {prompt[:50]}…")
            qr = QuestionResult(**cached)
            qr.variations = [VariationResult(**v) for v in cached["variations"]]
            return qr

    # ── Cache miss: call LLM via provider ────────────────────────────────────
    provider = get_provider(provider_name)
    llm_calls = 0

    emit(f"🔀 Generating {n_variations} variations…")
    variations = provider.generate_variations(prompt)[:n_variations]
    # Pad to requested count if provider returned fewer
    if len(variations) < n_variations:
        variations += [prompt] * (n_variations - len(variations))
    llm_calls += 1

    # ── Debug: log what was actually generated ────────────────────────────────
    logger.debug("run_experiment: %d variations for prompt %r", len(variations), prompt[:60])
    for i, v in enumerate(variations):
        logger.debug("  var[%d]: %r", i, v[:100])
    n_unique = len({v.strip().lower() for v in variations})
    if n_unique == 1:
        logger.warning(
            "run_experiment: ALL %d variations are identical — "
            "PSI will be artificially high (all-same responses). "
            "Check generate_variations() and the GROQ_API_KEY.", len(variations)
        )
    logger.debug("run_experiment: %d unique variations out of %d", n_unique, len(variations))

    time.sleep(0.3)  # brief pause to avoid rate-limit burst

    all_prompts = [prompt] + variations  # original + paraphrases
    responses: list[str] = []

    for i, var in enumerate(all_prompts):
        emit(f"🤖 Getting LLM response {i + 1}/{len(all_prompts)}…")
        resp = provider.generate_response(var)
        # Safety: never let an empty string into the PSI pipeline
        if not resp.strip():
            logger.warning(
                "run_experiment: empty response for prompt[%d] %r — using fallback.", i, var[:60]
            )
            resp = _FALLBACK_RESPONSE
        responses.append(resp)
        llm_calls += 1
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
    emit(f"💾 Result cached ({llm_calls} LLM calls made via {provider_name})")

    return result


def run_category_experiment(
    category: str,
    n_variations: int = config.MAX_VARIATIONS,
    force_rerun: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    provider_name: str = "groq",
) -> list[QuestionResult]:
    """
    Run all questions in a category using the specified LLM provider.

    Args:
        category:      Prompt category to evaluate.
        n_variations:  Number of paraphrases per prompt.
        force_rerun:   Bypass cache when True.
        progress_cb:   Optional progress callback.
        provider_name: LLM backend — "groq" (default) or "openai".
    """
    prompts = config.PROMPT_CATEGORIES.get(category, [])
    results = []
    for i, prompt in enumerate(prompts):
        if progress_cb:
            progress_cb(f"Question {i+1}/{len(prompts)}: {prompt[:50]}…")
        result = run_experiment(
            category, prompt, n_variations, force_rerun, progress_cb, provider_name
        )
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


# ── Multi-provider result container ───────────────────────────────────────────

@dataclass
class MultiProviderResult:
    """
    Holds the outcome of an "all models" run for a single prompt.

    Structure mirrors the requirement:
        {
          "prompt": "...",
          "results": {
            "groq":   QuestionResult,
            "openai": QuestionResult,
          }
        }

    Single-provider QuestionResult objects are stored verbatim under
    their provider key — no PSI values are merged or averaged.
    """
    prompt: str
    category: str
    results: dict[str, QuestionResult]   # provider_name → QuestionResult

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "results": {k: asdict(v) for k, v in self.results.items()},
        }


# ── All-providers experiment runner ──────────────────────────────────────────

def run_all_providers(
    category: str,
    prompt: str,
    n_variations: int = config.MAX_VARIATIONS,
    force_rerun: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> MultiProviderResult:
    """
    Run the same experiment sequentially across every registered provider.

    Each provider gets its own independent call to run_experiment(), so:
      • PSI is computed separately per provider (never merged).
      • Cache keys are provider-scoped (no cross-contamination).
      • Single-provider run_experiment() is reused unchanged.

    Args:
        category:     Prompt category label.
        prompt:       The seed prompt to evaluate.
        n_variations: Paraphrase count (passed through to each provider).
        force_rerun:  Bypass cache for all providers when True.
        progress_cb:  Optional progress callback (prefixed with provider name).

    Returns:
        MultiProviderResult containing one QuestionResult per provider.
    """
    from providers import PROVIDER_REGISTRY   # import here to avoid circular at module load

    # Groq-first ordering: run the fast baseline provider before slower ones
    # (e.g. Gemini) so the UI receives early feedback quickly.
    ordered = ["groq"] + [p for p in PROVIDER_REGISTRY if p != "groq"]

    provider_results: dict[str, QuestionResult] = {}

    for provider_name in ordered:
        def _scoped_cb(msg: str, pname: str = provider_name) -> None:
            if progress_cb:
                progress_cb(f"[{pname.upper()}] {msg}")

        logger.info("run_all_providers — starting provider: %s", provider_name)
        qr = run_experiment(
            category=category,
            prompt=prompt,
            n_variations=n_variations,
            force_rerun=force_rerun,
            progress_cb=_scoped_cb,
            provider_name=provider_name,
        )
        provider_results[provider_name] = qr
        logger.info(
            "run_all_providers — %s complete | PSI=%.2f",
            provider_name, qr.psi_score,
        )

    return MultiProviderResult(
        prompt=prompt,
        category=category,
        results=provider_results,
    )
