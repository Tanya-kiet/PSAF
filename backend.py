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


@dataclass
class FailedProviderResult:
    """
    Sentinel returned by run_all_providers() for a provider that raised an
    exception.  Lets the rest of the pipeline continue and the UI to show a
    per-provider error without aborting the whole experiment.
    """
    provider_name: str
    error: str          # human-readable error message
    psi_score: float = 0.0   # sentinel — never used for real PSI computation


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

# Map provider name → actual model string so cache keys are fully unambiguous.
# Using config.LLM_MODEL for every provider was wrong: it is the Groq model name
# (llama-3.1-8b-instant) and would produce the same model-component in the key
# for both Groq and OpenAI, relying SOLELY on provider_name for differentiation.
# Adding the real model name makes each cache entry self-documenting and safe
# even if provider_name handling ever changes.
_PROVIDER_MODEL_MAP: dict[str, str] = {
    "groq":   config.LLM_MODEL,   # llama-3.1-8b-instant
    "openai": "gpt-4o-mini",
}


def _cache_key(category: str, prompt: str, n_variations: int, provider_name: str = "groq") -> str:
    # Use the ACTUAL model name for each provider, not the global LLM_MODEL,
    # so Groq and OpenAI cache entries are distinguished by both model AND provider.
    model_name = _PROVIDER_MODEL_MAP.get(provider_name, f"unknown-{provider_name}")
    raw = f"{category}|{prompt}|{n_variations}|{model_name}|{provider_name}"
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
    mode: str = "fast",
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
        mode:          Execution mode — "fast" (Groq-only, default) or
                       "research" (Groq + OpenAI, returns first provider's
                       result; use run_all_providers() for full comparison).
                       In "fast" mode, provider_name is forced to "groq"
                       regardless of the value passed in.
    """
    # ── Mode gate: Fast mode is always Groq-only ──────────────────────────────
    if mode == "fast":
        provider_name = "groq"  # enforce: no OpenAI calls in fast mode
    # research mode uses whatever provider_name was passed (or groq by default)
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
    try:
        variations = provider.generate_variations(prompt)[:n_variations]
    except Exception as exc:
        logger.error(
            "run_experiment[%s]: generate_variations raised %s: %s — "
            "using seed prompt copies as fallback variations.",
            provider_name, type(exc).__name__, exc,
        )
        variations = []
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
        try:
            resp = provider.generate_response(var)
        except Exception as exc:
            # Log per-response failure but DO NOT raise — one bad response
            # should not abort the entire provider run.  Use the fallback
            # token so compute_psi() still receives a valid (non-empty) string.
            # This is intentional: a single timeout/rate-limit shouldn't kill
            # the whole experiment; PSI will naturally reflect the degraded data.
            logger.error(
                "run_experiment[%s]: provider.generate_response raised %s: %s — "
                "using fallback response for prompt[%d].",
                provider_name, type(exc).__name__, exc, i,
            )
            resp = _FALLBACK_RESPONSE

        # ── Debug print: verify WHICH provider returned WHAT ─────────────────
        print(f"[DEBUG] PROVIDER: {provider_name!r}  |  prompt[{i}]: {var[:60]!r}")
        print(f"[DEBUG] RESPONSE[:200]: {resp[:200]!r}")

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
    mode: str = "fast",
) -> list[QuestionResult]:
    """
    Run all questions in a category using the specified LLM provider.

    Args:
        category:      Prompt category to evaluate.
        n_variations:  Number of paraphrases per prompt.
        force_rerun:   Bypass cache when True.
        progress_cb:   Optional progress callback.
        provider_name: LLM backend — "groq" (default) or "openai".
        mode:          Execution mode — "fast" (Groq-only) or "research"
                       (Groq + OpenAI).
    """
    prompts = config.PROMPT_CATEGORIES.get(category, [])
    results = []
    for i, prompt in enumerate(prompts):
        if progress_cb:
            progress_cb(f"Question {i+1}/{len(prompts)}: {prompt[:50]}…")
        result = run_experiment(
            category, prompt, n_variations, force_rerun, progress_cb, provider_name, mode
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
    Holds the outcome of a Research Mode run for a single prompt.

    `results` maps provider_name → QuestionResult (success) OR
    FailedProviderResult (failure).  The UI inspects isinstance() to decide
    how to render each provider slot.

    At least one entry in `results` is always a QuestionResult
    (run_all_providers raises RuntimeError if every provider fails).
    """
    prompt: str
    category: str
    results: dict  # provider_name → QuestionResult | FailedProviderResult

    @property
    def successful_results(self) -> dict:
        """Only the providers that succeeded."""
        return {k: v for k, v in self.results.items() if isinstance(v, QuestionResult)}

    @property
    def failed_results(self) -> dict:
        """Only the providers that failed."""
        return {k: v for k, v in self.results.items() if isinstance(v, FailedProviderResult)}

    def to_dict(self) -> dict:
        out: dict = {"prompt": self.prompt, "category": self.category, "results": {}}
        for k, v in self.results.items():
            if isinstance(v, QuestionResult):
                out["results"][k] = asdict(v)
            else:
                out["results"][k] = {"status": "failed", "error": v.error}
        return out


# ── All-providers experiment runner ──────────────────────────────────────────

def run_all_providers(
    category: str,
    prompt: str,
    n_variations: int = config.MAX_VARIATIONS,
    force_rerun: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    mode: str = "research",
) -> MultiProviderResult:
    """
    Run the same experiment sequentially across providers selected by mode.

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
        mode:         Execution mode.
                      "fast"     → Groq only (instant, no OpenAI calls).
                      "research" → Groq + OpenAI (scientifically valid
                                   cross-provider PSI comparison).

    Returns:
        MultiProviderResult containing one QuestionResult per provider.
        PSI scores are computed independently per provider — never merged.
        Example structure:
            {
              "groq":   QuestionResult(psi_score=0.87, ...),
              "openai": QuestionResult(psi_score=0.81, ...),
            }
    """
    # ── Mode-controlled provider selection ───────────────────────────────────
    # FAST MODE:     Groq only — no OpenAI calls, maximum speed.
    # RESEARCH MODE: Groq + OpenAI — independent PSI per provider for
    #                scientifically valid comparison dashboard.
    if mode == "fast":
        ordered = ["groq"]
    else:  # "research"
        ordered = ["groq", "openai"]

    # provider_name -> QuestionResult (success) or FailedProviderResult (failure)
    provider_results: dict = {}
    provider_errors:  dict[str, str] = {}

    for provider_name in ordered:
        def _scoped_cb(msg: str, pname: str = provider_name) -> None:
            if progress_cb:
                progress_cb(f"[{pname.upper()}] {msg}")

        logger.info("run_all_providers — starting provider: %s (mode=%s)", provider_name, mode)
        try:
            qr = run_experiment(
                category=category,
                prompt=prompt,
                n_variations=n_variations,
                force_rerun=force_rerun,
                progress_cb=_scoped_cb,
                provider_name=provider_name,
                mode=mode,
            )
            provider_results[provider_name] = qr
            logger.info(
                "run_all_providers — %s complete | PSI=%.2f",
                provider_name, qr.psi_score,
            )
        except Exception as exc:
            # Provider failed — record it but NEVER abort other providers.
            # This is the core "best-effort multi-model" contract:
            # one failure must not prevent partial results from the rest.
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "run_all_providers — %s FAILED: %s — continuing with remaining providers.",
                provider_name, err_msg,
            )
            provider_errors[provider_name] = err_msg
            provider_results[provider_name] = FailedProviderResult(
                provider_name=provider_name,
                error=err_msg,
            )
            if progress_cb:
                progress_cb(f"[{provider_name.upper()}] ❌ Provider failed: {err_msg}")
            # Continue loop — do NOT raise, do NOT break

    # ── Abort only if ALL providers failed (nothing to show the user) ─────────
    successful = [
        pname for pname, r in provider_results.items()
        if isinstance(r, QuestionResult)
    ]
    if not successful:
        all_errors = "; ".join(f"{p}: {e}" for p, e in provider_errors.items())
        raise RuntimeError(
            f"All providers failed — no results available. Errors: {all_errors}"
        )

    # ── Cross-provider divergence check (non-fatal, log-only) ─────────────────────
    # No assert — divergence logging must never kill a partial-success run.
    groq_qr   = provider_results.get("groq")
    openai_qr = provider_results.get("openai")
    if isinstance(groq_qr, QuestionResult) and isinstance(openai_qr, QuestionResult):
        gvars = groq_qr.variations
        ovars = openai_qr.variations
        if gvars and ovars:
            gr  = gvars[0].response
            or_ = ovars[0].response
            _is_fallback = lambda r: r.startswith("[No response")
            if not _is_fallback(gr) and not _is_fallback(or_):
                if gr == or_:
                    logger.error(
                        "run_all_providers: Groq and OpenAI returned IDENTICAL responses — "
                        "one provider may be secretly calling the other's API. "
                        "(Non-fatal — results still stored.)"
                    )
                else:
                    logger.info(
                        "run_all_providers: response divergence confirmed ✓ "
                        "GROQ[:80]=%r  OPENAI[:80]=%r", gr[:80], or_[:80],
                    )
                    print("[DEBUG] PSI divergence check:")
                    print(f"  GROQ   PSI = {groq_qr.psi_score:.2f}")
                    print(f"  OPENAI PSI = {openai_qr.psi_score:.2f}")

    return MultiProviderResult(
        prompt=prompt,
        category=category,
        results=provider_results,
    )
