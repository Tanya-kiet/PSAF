"""
experiments.py — Phase 3: Experimental Evaluation for PSAF (Groq-only version)
"""

from __future__ import annotations

import csv
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev, pstdev

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import config

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
VARIATIONS_PER_QUESTION = 3

CATEGORY_ABBREV = {
    "Definition Questions": "DEF",
    "Technical Questions": "TEC",
    "Reasoning Questions": "REA",
    "Educational Questions": "EDU",
}

QUESTION_BANK = {
    "Definition Questions": [
        "What is machine learning?",
        "What is a neural network?",
    ],
    "Technical Questions": [
        "How does gradient descent work?",
        "What is learning rate?",
    ],
    "Reasoning Questions": [
        "Why does overfitting happen?",
        "Why is preprocessing important?",
    ],
    "Educational Questions": [
        "Explain regularization simply",
        "What is an epoch?",
    ],
}

# ── Groq Client Wrapper ─────────────────────────────────────────────────────
import time

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def generate(self, messages, temperature=0.7, max_retries=5):
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=config.MAX_TOKENS,
                )
                return resp.choices[0].message.content

            except Exception as e:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s, 8s...
                print(f"[Groq Retry {attempt+1}] Error: {e} | waiting {wait}s")
                time.sleep(wait)

        return ""  # fail-safe fallback


# ── Variation Generator ─────────────────────────────────────────────────────
class VariationGenerator:
    SYSTEM = (
        "You are a prompt rewriting system. "
        "Return ONLY a valid JSON array of strings. "
        "No explanation, no markdown, no numbering."
    )

    def __init__(self, client: GroqClient):
        self.client = client

    def generate(self, seed: str):
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Rewrite this question in 4 different ways:\n{seed}"}
        ]

        try:
            raw = self.client.generate(messages, temperature=0.7)
            import json
            return json.loads(raw)
        except Exception:
            return [seed] * (VARIATIONS_PER_QUESTION - 1)


# ── Response Collector ──────────────────────────────────────────────────────

class ResponseCollector:

    def __init__(self, client: GroqClient):
        self.client = client

    def collect(self, variations):
        outputs = []
        for v in variations:
            try:
                outputs.append(
                    self.client.generate(
                        [{"role": "user", "content": v}]
                    )
                )
                time.sleep(0.5)  # 👈 ADD THIS LINE HERE

            except Exception:
                outputs.append("")
        return outputs


# ── PSI Engine ──────────────────────────────────────────────────────────────
class PSI:
    def __init__(self):
        self.model = SentenceTransformer(config.PSI_EMBEDDING_MODEL)
        self.token_re = re.compile(r"[a-zA-Z]+")

    def keywords(self, text):
        return {
            t.lower()
            for t in self.token_re.findall(text)
            if t.lower() not in ENGLISH_STOP_WORDS and len(t) > 2
        }

    def compute(self, responses):
        valid = [r for r in responses if r.strip()]
        if len(valid) < 2:
            return 0, 0, 0, 0

        emb = self.model.encode(valid)
        sim = cosine_similarity(emb)

        S = np.mean(sim[np.triu_indices_from(sim, k=1)])

        sets = [self.keywords(r) for r in valid]
        K_scores = []

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                if not a and not b:
                    K_scores.append(1)
                else:
                    K_scores.append(len(a & b) / len(a | b))

        K = np.mean(K_scores)

        lengths = [len(r.split()) for r in valid]
        L = 1 / (1 + pstdev(lengths) / (mean(lengths) + 1e-9))

        psi = 100 * (0.5 * S + 0.3 * K + 0.2 * L)

        return psi, S, K, L


# ── Main Experiment ─────────────────────────────────────────────────────────
def run():
    client = GroqClient()
    gen = VariationGenerator(client)
    collector = ResponseCollector(client)
    psi_engine = PSI()

    results = []

    for cat, questions in QUESTION_BANK.items():
        for i, q in enumerate(questions):
            qid = f"{CATEGORY_ABBREV[cat]}_{i}"

            variations = gen.generate(q)
            responses = collector.collect(variations)

            psi, S, K, L = psi_engine.compute(responses)

            results.append(
                ExperimentResult(
                    question_id=qid,
                    category=cat,
                    original_prompt=q,
                    num_variations=len(responses),
                    avg_psi=psi,
                    max_psi=psi,
                    min_psi=psi,
                    std_psi=0.0,
                    avg_semantic=S,
                    avg_keyword=K,
                    avg_length=L,
                    run_mode="groq",
                )
            )

    return results


# ── Data Models ─────────────────────────────────────────────────────────────
@dataclass
class ExperimentResult:
    question_id: str
    category: str
    original_prompt: str
    num_variations: int
    avg_psi: float
    max_psi: float
    min_psi: float
    std_psi: float
    avg_semantic: float
    avg_keyword: float
    avg_length: float
    run_mode: str

    def to_dict(self):
        return self.__dict__


# ── Aggregation ─────────────────────────────────────────────────────────────
def aggregate(results):
    groups = {}

    for r in results:
        groups.setdefault(r.category, []).append(r)

    out = []

    for cat, vals in groups.items():
        psis = [v.avg_psi for v in vals]

        out.append({
            "category": cat,
            "num_questions": len(vals),
            "category_avg_psi": mean(psis),
            "category_max_psi": max(psis),
            "category_min_psi": min(psis),
            "category_std_psi": stdev(psis) if len(psis) > 1 else 0,
            "stability_rank": 0
        })

    out.sort(key=lambda x: x["category_avg_psi"], reverse=True)

    for i, o in enumerate(out, 1):
        o["stability_rank"] = i

    return out


# ── Save CSV ────────────────────────────────────────────────────────────────
def save(results, comparisons):
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)

    with open(config.OUTPUT_DIR / "experiment_results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    with open(config.OUTPUT_DIR / "category_comparison.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
        writer.writeheader()
        for c in comparisons:
            writer.writerow(c)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    logger.info("Running Groq-only PSAF experiment...")

    results = run()
    comparisons = aggregate(results)
    save(results, comparisons)

    logger.info("Done. Results saved.")


if __name__ == "__main__":
    main()