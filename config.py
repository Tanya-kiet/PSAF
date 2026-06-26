"""
config.py — Central configuration for PSAF (Prompt Stability Analysis Framework).
"""

import os
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
CACHE_DIR  = BASE_DIR / ".psaf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Groq API ──────────────────────────────────────────────────────────────────
# NEVER hardcode your key. Set it in your shell:
#   export GROQ_API_KEY="gsk_..."
# Or in a .env file (add .env to .gitignore — NEVER commit it).
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM settings
LLM_MODEL          = "llama-3.1-8b-instant"
MAX_TOKENS: int    = 512      # kept short to reduce latency
TEMPERATURE: float = 0.7

# Paraphrase generation temperature (slightly higher for diversity)
PARAPHRASE_TEMPERATURE: float = 0.85

# ── Experiment settings ───────────────────────────────────────────────────────
MAX_VARIATIONS: int = 5   # max paraphrases per seed prompt (hard cap)


# ── PSI Embedding ─────────────────────────────────────────────────────────────
PSI_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# PSI component weights — must sum to 1.0
PSI_SEMANTIC_WEIGHT: float = 0.50
PSI_KEYWORD_WEIGHT:  float = 0.30
PSI_LENGTH_WEIGHT:   float = 0.20

assert abs(PSI_SEMANTIC_WEIGHT + PSI_KEYWORD_WEIGHT + PSI_LENGTH_WEIGHT - 1.0) < 1e-9

# ── Prompt categories ─────────────────────────────────────────────────────────
PROMPT_CATEGORIES: dict[str, list[str]] = {
    "Definition Questions": [
        "What is machine learning?",
        "What is a neural network?",
        "What is the difference between AI and machine learning?",
    ],
    "Technical Questions": [
        "How does gradient descent work?",
        "What is the purpose of a learning rate in training neural networks?",
        "How does backpropagation update model weights?",
    ],
    "Reasoning Questions": [
        "Why might a machine learning model overfit on training data?",
        "Why is data preprocessing important before training a model?",
        "Why do transformers outperform RNNs on long-sequence tasks?",
    ],
    "Educational Questions": [
        "Can you explain regularization in simple terms?",
        "Can you explain what an epoch is in machine learning?",
        "Can you explain the bias-variance tradeoff to a beginner?",
    ],
}
