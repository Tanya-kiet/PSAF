"""
streamlit_app.py — PSAF Professional Dashboard (Refined)
Prompt Stability Analysis Framework

Architecture:
  • All Groq calls happen ONLY when user clicks "Run Experiment"
  • Results cached in session_state and on disk (.psaf_cache/)
  • SentenceTransformer loaded once at module level
  • Tab switching never triggers new API calls
"""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from backend import (
    QuestionResult,
    FailedProviderResult,
    compute_category_stats,
    run_experiment,
    run_all_providers,
    MultiProviderResult,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSAF · Prompt Stability Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #090d13; color: #cdd9e5; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0d1117;
  border-right: 1px solid #1c2128;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Sidebar card sections ── */
.sb-card {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 12px;
  padding: 1rem;
  margin: 0 0.75rem 0.75rem;
}
.sb-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #484f58;
  margin-bottom: 0.75rem;
  display: block;
}

/* ── Status badge ── */
.status-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 500;
}
.status-ok   { background: #0f2a1a; border: 1px solid #238636; color: #3fb950; }
.status-err  { background: #2d0f0f; border: 1px solid #b91c1c; color: #f85149; }
.status-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-ok  { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
.dot-err { background: #f85149; box-shadow: 0 0 6px #f85149; }

/* ── Sidebar logo block ── */
.sb-logo {
  padding: 1.25rem 1rem 0.75rem;
  border-bottom: 1px solid #1c2128;
  margin-bottom: 0.75rem;
}
.sb-logo-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #e6edf3;
  letter-spacing: -0.01em;
}
.sb-logo-sub {
  font-size: 0.7rem;
  color: #484f58;
  margin-top: 0.2rem;
}

/* ── Run button ── */
.stButton > button {
  background: linear-gradient(135deg, #1a5fcc 0%, #2979ff 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 999px !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 0.7rem 1.5rem !important;
  width: 100% !important;
  height: 2.8rem !important;
  letter-spacing: 0.01em !important;
  transition: box-shadow 0.2s, transform 0.15s !important;
  box-shadow: 0 0 0 0 rgba(41,121,255,0) !important;
}
.stButton > button:hover {
  box-shadow: 0 0 18px 4px rgba(41,121,255,0.35) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: scale(0.98) translateY(0) !important; }

/* ── Secondary buttons ── */
.stButton.secondary > button {
  background: transparent !important;
  border: 1px solid #30363d !important;
  color: #8b949e !important;
  border-radius: 8px !important;
}

/* ── Form controls ── */
.stSelectbox > div > div,
.stSlider > div > div {
  background: #0d1117 !important;
  border-color: #21262d !important;
  border-radius: 8px !important;
  color: #cdd9e5 !important;
}
/* Slider accent color */
[data-testid="stSlider"] [role="slider"] {
  background: #2979ff !important;
  border-color: #2979ff !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"] {
  background: #2979ff !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent;
  border-bottom: 1px solid #1c2128;
  gap: 0;
  padding: 0;
}
.stTabs [data-baseweb="tab"] {
  color: #8b949e;
  font-size: 0.8rem;
  font-weight: 500;
  padding: 0.55rem 1.1rem;
  border-radius: 0;
  letter-spacing: 0.01em;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  color: #e6edf3 !important;
  background: transparent !important;
  border-bottom: 2px solid #2979ff !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
  background: #0d1117;
  border: 1px solid #1c2128;
  border-radius: 12px;
  padding: 1rem 1.25rem;
}
div[data-testid="metric-container"] label {
  color: #484f58 !important;
  font-size: 0.67rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: #e6edf3 !important;
  font-weight: 600 !important;
  font-size: 1.65rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, #1a5fcc, #2979ff) !important;
  border-radius: 999px !important;
}
.stProgress > div > div {
  background: #1c2128 !important;
  border-radius: 999px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: #0d1117 !important;
  border: 1px solid #1c2128 !important;
  border-radius: 8px !important;
  color: #cdd9e5 !important;
  font-size: 0.85rem !important;
}
.streamlit-expanderContent {
  background: #0d1117 !important;
  border: 1px solid #1c2128 !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
}

/* ── Divider ── */
hr { border-color: #1c2128 !important; margin: 1rem 0 !important; }

/* ── Data frame ── */
.stDataFrame { background: #0d1117 !important; border: 1px solid #1c2128 !important; border-radius: 10px !important; }
.stDataFrame th { background: #161b22 !important; color: #8b949e !important; }

/* ── Sticky header ── */
.psaf-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(9, 13, 19, 0.92);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid #1c2128;
  padding: 0.6rem 0;
  margin-bottom: 1.25rem;
}
.psaf-header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
.psaf-header-left {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.psaf-logo-icon {
  width: 30px; height: 30px;
  background: linear-gradient(135deg, #1a5fcc, #2979ff);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
}
.psaf-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #e6edf3;
  letter-spacing: -0.01em;
}
.psaf-subtitle {
  font-size: 0.68rem;
  color: #484f58;
}
.psaf-badges { display: flex; gap: 0.4rem; align-items: center; }
.psaf-badge {
  font-size: 0.65rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.badge-blue  { background: #0a1e3a; border: 1px solid #1f6feb; color: #79c0ff; }
.badge-green { background: #0f2a1a; border: 1px solid #238636; color: #3fb950; }
.badge-psi   { background: #1a1a3e; border: 1px solid #534ab7; color: #b7b0f0; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }

/* ── PSI score ── */
.psi-score-large {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
}
.psi-high   { color: #3fb950; }
.psi-medium { color: #d29922; }
.psi-low    { color: #f85149; }

/* ── Section label ── */
.section-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #484f58;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid #1c2128;
  margin-bottom: 1rem;
  display: block;
}

/* ── Stage progress ── */
.stage-row {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.45rem 0;
  font-size: 0.8rem;
  color: #8b949e;
  border-bottom: 1px solid #1c2128;
}
.stage-row:last-child { border-bottom: none; }
.stage-row.done   { color: #3fb950; }
.stage-row.active { color: #e6edf3; }
.stage-icon { width: 18px; text-align: center; font-size: 0.8rem; flex-shrink: 0; }

/* ── Info pill ── */
.info-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: #0a1e3a;
  border: 1px solid #1f3a6b;
  border-radius: 999px;
  padding: 0.3rem 0.8rem;
  font-size: 0.75rem;
  color: #79c0ff;
  margin-bottom: 1rem;
}

/* ── Variation card ── */
.var-card {
  background: #0d1117;
  border: 1px solid #1c2128;
  border-radius: 10px;
  padding: 0.875rem 1rem;
  margin-bottom: 0.5rem;
  transition: border-color 0.15s;
}
.var-card:hover { border-color: #30363d; }
.var-card.original { border-color: #1f3a6b; background: #09192e; }
.var-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.tag-orig { color: #2979ff; }
.tag-var  { color: #484f58; }
.var-text { color: #cdd9e5; font-size: 0.88rem; line-height: 1.55; }

/* ── Response card ── */
.resp-pair { display: flex; gap: 0.75rem; margin-bottom: 0.75rem; }
.resp-prompt {
  flex: 0 0 220px;
  background: #09192e;
  border: 1px solid #1f3a6b;
  border-radius: 8px;
  padding: 0.75rem;
  font-size: 0.8rem;
  color: #79c0ff;
  line-height: 1.5;
}
.resp-response {
  flex: 1;
  background: #0d1117;
  border: 1px solid #1c2128;
  border-radius: 8px;
  padding: 0.75rem;
  font-size: 0.82rem;
  color: #cdd9e5;
  line-height: 1.6;
  white-space: pre-wrap;
}
.resp-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #484f58;
  margin-bottom: 0.4rem;
}

/* ── Empty state ── */
.empty-hero {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center;
  padding: 3rem 2rem 2rem;
  min-height: 60vh;
}
.empty-visual {
  width: 120px; height: 120px;
  background: linear-gradient(135deg, #09192e 0%, #0d1117 100%);
  border: 1px solid #1f3a6b;
  border-radius: 24px;
  display: flex; align-items: center; justify-content: center;
  font-size: 3rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 0 40px rgba(41,121,255,0.08);
}
.empty-title {
  font-size: 1.4rem; font-weight: 700; color: #e6edf3;
  letter-spacing: -0.02em; margin-bottom: 0.5rem;
}
.empty-sub {
  font-size: 0.88rem; color: #8b949e; max-width: 420px;
  line-height: 1.6; margin-bottom: 2rem;
}
.steps-row {
  display: flex; gap: 0; margin-bottom: 2rem;
  border: 1px solid #1c2128; border-radius: 12px; overflow: hidden;
}
.step-item {
  flex: 1; padding: 1rem;
  border-right: 1px solid #1c2128;
  background: #0d1117;
}
.step-item:last-child { border-right: none; }
.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; color: #2979ff; font-weight: 700;
  letter-spacing: 0.05em; margin-bottom: 0.35rem;
}
.step-label { font-size: 0.78rem; color: #cdd9e5; font-weight: 500; }
.step-desc  { font-size: 0.7rem; color: #484f58; margin-top: 0.2rem; }

/* ── Category comparison card ── */
.cat-stat-card {
  background: #0d1117;
  border: 1px solid #1c2128;
  border-radius: 10px;
  padding: 0.875rem 1rem;
}
.cat-stat-name { font-size: 0.7rem; color: #484f58; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.4rem; }
.cat-stat-score { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; line-height: 1; margin-bottom: 0.3rem; }
.cat-stat-meta { font-size: 0.72rem; color: #484f58; }

/* ── Rank badge ── */
.rank-pill {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem; font-weight: 700;
  padding: 0.15rem 0.5rem; border-radius: 999px;
  margin-left: 0.35rem;
  background: #1a1a3e; color: #b7b0f0; border: 1px solid #3c3489;
}

/* ── Success toast ── */
.success-toast {
  background: #0f2a1a;
  border: 1px solid #238636;
  border-radius: 10px;
  padding: 0.75rem 1rem;
  display: flex; align-items: center; gap: 0.6rem;
  font-size: 0.85rem; color: #3fb950;
  animation: fadeSlide 0.3s ease;
}
@keyframes fadeSlide {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Skeleton loader ── */
.skeleton-block {
  background: linear-gradient(90deg, #1c2128 25%, #21262d 50%, #1c2128 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 6px;
  margin-bottom: 0.5rem;
}
@keyframes shimmer {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "result": None,
    "all_results": {},
    "last_run_time": None,
    "last_run_calls": 0,
    "running": False,
    "execution_mode": "fast",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
def psi_class(score: float) -> str:
    return "psi-high" if score >= 75 else "psi-medium" if score >= 50 else "psi-low"

def psi_label(score: float) -> str:
    if score >= 75: return "Highly Stable"
    if score >= 50: return "Moderately Stable"
    return "Unstable"

def bar_color(score: float) -> str:
    if score >= 75: return "#3fb950"
    if score >= 50: return "#d29922"
    return "#f85149"

def cat_color(score: float) -> str:
    return bar_color(score)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-title">🧬 PSAF</div>
      <div class="sb-logo-sub">Prompt Stability Analysis Framework</div>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status ──
    api_ok = bool(config.GROQ_API_KEY)

    # Detect OpenAI key via env var or st.secrets
    import os as _os
    _openai_key_ok = bool(_os.getenv("OPENAI_API_KEY", ""))
    if not _openai_key_ok:
        try:
            _openai_key_ok = bool(st.secrets.get("OPENAI_API_KEY", ""))
        except Exception:
            pass

    if api_ok:
        st.markdown("""
        <div class="sb-card" style="padding: 0.65rem 1rem;">
          <div class="status-badge status-ok">
            <div class="status-dot dot-ok"></div>
            Groq Connected
          </div>
          <div style="font-size:0.67rem;color:#484f58;margin-top:0.5rem;padding-left:0.25rem;">
            Model: llama-3.1-8b-instant
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-card" style="padding: 0.65rem 1rem;">
          <div class="status-badge status-err">
            <div class="status-dot dot-err"></div>
            API Key Missing
          </div>
        </div>
        """, unsafe_allow_html=True)

    # OpenAI key status — informs user whether Research Mode will work
    if _openai_key_ok:
        st.markdown("""
        <div class="sb-card" style="padding: 0.65rem 1rem; margin-top: 0.25rem;">
          <div class="status-badge status-ok">
            <div class="status-dot dot-ok"></div>
            OpenAI Connected
          </div>
          <div style="font-size:0.67rem;color:#484f58;margin-top:0.5rem;padding-left:0.25rem;">
            Model: gpt-4o-mini &nbsp;·&nbsp; Research Mode ready
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-card" style="padding: 0.65rem 1rem; margin-top: 0.25rem;">
          <div class="status-badge status-err">
            <div class="status-dot dot-err"></div>
            OpenAI Key Missing
          </div>
          <div style="font-size:0.67rem;color:#484f58;margin-top:0.5rem;padding-left:0.25rem;">
            Research Mode unavailable
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # ── Configuration ──
    with st.container():
        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        # ── Execution Mode ────────────────────────────────────────────────────
        execution_mode = st.selectbox(
            "Execution Mode",
            options=["fast", "research"],
            format_func=lambda x: {
                "fast":     "⚡ Fast Mode (Groq only)",
                "research": "🧠 Research Mode (Groq + OpenAI)",
            }.get(x, x),
            index=["fast", "research"].index(st.session_state["execution_mode"]),
            help=(
                "⚡ Fast Mode: Groq only — instant real-time PSI.\n"
                "🧠 Research Mode: Groq + OpenAI — slower but enables "
                "scientifically valid cross-provider PSI comparison."
            ),
        )
        st.session_state["execution_mode"] = execution_mode

        categories = list(config.PROMPT_CATEGORIES.keys())
        selected_category = st.selectbox("Category", categories, label_visibility="visible")
        prompts = config.PROMPT_CATEGORIES[selected_category]
        selected_prompt = st.selectbox("Question", prompts, label_visibility="visible")
        n_variations = st.slider("Variations", 2, config.MAX_VARIATIONS, 3,
                                 help="Paraphrased versions to generate. More = more API calls.")
        # Provider is determined INTERNALLY by Execution Mode — no user dropdown.
        # fast → groq only | research → groq + openai (via run_all_providers)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

    # ── Experiment Controls ──
    with st.container():
        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        force_rerun = st.checkbox("Force re-run (bypass cache)",
                                  help="Ignore cached results and re-call Groq.")
        run_clicked = st.button("▶  Run Experiment", disabled=not api_ok)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

    # ── Session Summary ──
    n_cached = len(st.session_state.all_results)
    if n_cached > 0:
        st.markdown('<span class="sb-label">Session</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-card">
          <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.5rem;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;
            font-weight:700;color:#e6edf3;">{n_cached}</span>
            &nbsp;question{"s" if n_cached != 1 else ""} cached
          </div>
          {"<div style='font-size:0.7rem;color:#484f58;'>Last run: " + st.session_state.last_run_time + "</div>" if st.session_state.last_run_time else ""}
        </div>
        """, unsafe_allow_html=True)
        if st.button("Clear session", key="clear_session"):
            st.session_state.all_results = {}
            st.session_state.result = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
STAGES = [
    ("Generating variations", "🔀"),
    ("Collecting LLM responses", "🤖"),
    ("Computing embeddings", "📐"),
    ("Calculating PSI score", "🎯"),
    ("Saving to cache", "💾"),
]

if run_clicked:
    status_placeholder = st.empty()
    with status_placeholder.container():
        st.markdown('<span class="section-label">Running experiment</span>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        stage_html_placeholder = st.empty()
        log_placeholder = st.empty()

    stage_states = ["pending"] * len(STAGES)
    current_stage = [0]

    def render_stages():
        html = '<div style="background:#0d1117;border:1px solid #1c2128;border-radius:10px;padding:0.75rem 1rem;">'
        for i, (label, icon) in enumerate(STAGES):
            state = stage_states[i]
            if state == "done":
                cls = "done"; ic = "✓"
            elif state == "active":
                cls = "active"; ic = "⟳"
            else:
                cls = ""; ic = icon
            html += f'<div class="stage-row {cls}"><span class="stage-icon">{ic}</span>{label}</div>'
        html += "</div>"
        return html

    messages_log = []

    def progress_cb(msg: str):
        messages_log.append(msg)
        pct = min(int(len(messages_log) / (n_variations + 4) * 100), 95)
        progress_bar.progress(pct)

        # advance stage display heuristically
        if "Generating" in msg and stage_states[0] == "pending":
            stage_states[0] = "active"
        elif "response" in msg.lower() and stage_states[0] == "active":
            stage_states[0] = "done"; stage_states[1] = "active"
        elif "response" in msg.lower() and stage_states[1] == "active":
            pass
        elif "PSI" in msg or "Comput" in msg:
            stage_states[1] = "done"; stage_states[2] = "active"
        elif "cached" in msg.lower() or "cache" in msg.lower():
            stage_states[2] = "done"; stage_states[3] = "done"; stage_states[4] = "active"

        stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)
        log_placeholder.markdown(
            f"<div style='font-size:0.72rem;color:#484f58;margin-top:0.4rem;'>"
            f"› {messages_log[-1]}</div>",
            unsafe_allow_html=True
        )

    t0 = time.time()

    # ── Execution Mode is the SOLE controller of provider selection ───────────
    # fast     → Groq only (single provider, instant)
    # research → Groq + OpenAI (multi-provider comparison via run_all_providers)
    _execution_mode = st.session_state["execution_mode"]

    if _execution_mode == "research":
        # ── RESEARCH MODE: best-effort multi-model execution ──────────────────
        # run_all_providers() only raises if EVERY provider fails.
        # Partial success (e.g. Groq OK, OpenAI timeout) returns a
        # MultiProviderResult with FailedProviderResult entries for failures.
        try:
            multi_result = run_all_providers(
                category=selected_category,
                prompt=selected_prompt,
                n_variations=n_variations,
                force_rerun=force_rerun,
                progress_cb=progress_cb,
                mode="research",
            )
        except Exception as _exc:
            # Only reached when ALL providers failed simultaneously.
            progress_bar.progress(0)
            status_placeholder.error(
                f"❌ All providers failed — no results available.\n\n{_exc}\n\n"
                "Check both GROQ_API_KEY and OPENAI_API_KEY."
            )
            st.stop()
        elapsed = time.time() - t0

        for i in range(len(STAGES)):
            stage_states[i] = "done"
        stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)
        progress_bar.progress(100)
        log_placeholder.empty()

        key = f"{selected_category}||{selected_prompt}"
        st.session_state.all_results[key] = multi_result

        # Surface first SUCCESSFUL provider's QuestionResult for single-result panels
        first_provider = next(iter(multi_result.successful_results))
        st.session_state.result = multi_result.successful_results[first_provider]
        st.session_state.last_run_time = time.strftime("%H:%M:%S")
        st.session_state.last_run_calls = (n_variations + 2) * len(multi_result.successful_results)

        # Build status summary — show SUCCESS / FAILED per provider
        success_parts = [
            f"<span style='color:#3fb950;'>✓ {p.upper()}: {r.psi_score:.1f}</span>"
            for p, r in multi_result.successful_results.items()
        ]
        failed_parts = [
            f"<span style='color:#f85149;'>✗ {p.upper()}: failed</span>"
            for p in multi_result.failed_results
        ]
        status_parts = "  &nbsp;|&nbsp;  ".join(success_parts + failed_parts)

        # Show partial-success warning if any provider failed
        any_failed = bool(multi_result.failed_results)
        toast_bg    = "#1a2a1a" if not any_failed else "#1a1a00"
        toast_border = "#238636" if not any_failed else "#856404"
        toast_icon   = "✓" if not any_failed else "⚠"

        time.sleep(0.8)
        status_placeholder.markdown(f"""
        <div style="background:{toast_bg};border:1px solid {toast_border};
        border-radius:10px;padding:0.75rem 1rem;font-size:0.85rem;">
          <span style="color:{'#3fb950' if not any_failed else '#d29922'};">
            {toast_icon} &nbsp;Research Mode complete in {elapsed:.1f}s
          </span>
          <br/><span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;">
            {status_parts}
          </span>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── FAST MODE: Groq only — provider_name is always "groq" ────────────
        # No user override possible. mode="fast" in run_experiment also enforces this.
        try:
            result = run_experiment(
                category=selected_category,
                prompt=selected_prompt,
                n_variations=n_variations,
                force_rerun=force_rerun,
                progress_cb=progress_cb,
                provider_name="groq",   # always groq in fast mode
                mode="fast",
            )
        except Exception as _exc:
            progress_bar.progress(0)
            status_placeholder.error(
                f"❌ Experiment failed: {_exc}\n\n"
                "Check your GROQ_API_KEY. No silent fallback occurs — "
                "errors surface here immediately."
            )
            st.stop()
        elapsed = time.time() - t0

        for i in range(len(STAGES)):
            stage_states[i] = "done"
        stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)
        progress_bar.progress(100)
        log_placeholder.empty()

        key = f"{selected_category}||{selected_prompt}"
        st.session_state.all_results[key] = result
        st.session_state.result = result
        st.session_state.last_run_time = time.strftime("%H:%M:%S")
        st.session_state.last_run_calls = n_variations + 2

        time.sleep(0.8)
        status_placeholder.markdown(f"""
        <div class="success-toast">
          ✓ &nbsp;Fast Mode complete in {elapsed:.1f}s —
          PSI Score: <strong style="font-family:'JetBrains Mono',monospace;">{result.psi_score:.1f}</strong>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1.2)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STICKY HEADER
# ══════════════════════════════════════════════════════════════════════════════
result: QuestionResult | None = st.session_state.result

psi_badge_html = ""
if result:
    psi_badge_html = (
        f'<span class="psaf-badge badge-psi">'
        f'PSI {result.psi_score:.1f}</span>'
    )

st.markdown(f"""
<div class="psaf-header">
  <div class="psaf-header-inner">
    <div class="psaf-header-left">
      <div class="psaf-logo-icon">🧬</div>
      <div>
        <div class="psaf-title">Prompt Stability Analysis Framework</div>
        <div class="psaf-subtitle">Research Dashboard</div>
      </div>
    </div>
    <div class="psaf-badges">
      <span class="psaf-badge badge-blue">Groq</span>
      <span class="psaf-badge badge-green">llama-3.1-8b</span>
      {psi_badge_html}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if not result:
    st.markdown("""
    <div class="empty-hero">
      <div class="empty-visual">🧬</div>
      <div class="empty-title">Prompt Stability Analysis Framework</div>
      <div class="empty-sub">
        Evaluate how LLM outputs change when prompt wording changes
        while meaning stays constant — quantified as a PSI score.
      </div>

      <div class="steps-row">
        <div class="step-item">
          <div class="step-num">01</div>
          <div class="step-label">Choose a category</div>
          <div class="step-desc">Definition, Technical, Reasoning, or Educational</div>
        </div>
        <div class="step-item">
          <div class="step-num">02</div>
          <div class="step-label">Select a question</div>
          <div class="step-desc">Pick the seed prompt to test</div>
        </div>
        <div class="step-item">
          <div class="step-num">03</div>
          <div class="step-label">Run experiment</div>
          <div class="step-desc">One click — Groq generates variations & responses</div>
        </div>
        <div class="step-item">
          <div class="step-num">04</div>
          <div class="step-label">Analyze PSI score</div>
          <div class="step-desc">Explore similarity, stability, and breakdown</div>
        </div>
      </div>

      <div style="font-size:0.78rem;color:#484f58;">
        Results are cached — re-selecting a question costs zero API calls.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📝  Prompt Variations",
    "💬  LLM Responses",
    "📊  Similarity Analysis",
    "🎯  PSI Breakdown",
    "📋  Category Comparison",
    "🔬  Comparison Dashboard",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Prompt Variations
# ───────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<span class="section-label">Prompt Variations</span>', unsafe_allow_html=True)

    col_info, col_q = st.columns([2, 3])
    with col_info:
        st.markdown(f"""
        <div class="info-pill">
          🔀 &nbsp;{len(result.variations) - 1} paraphrase{"s" if len(result.variations) != 2 else ""} generated
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.82rem;color:#8b949e;line-height:1.6;">
          Each variation rephrases the original question with different wording
          while preserving its meaning. The LLM receives each independently.
        </div>
        """, unsafe_allow_html=True)
    with col_q:
        st.markdown(f"""
        <div style="background:#09192e;border:1px solid #1f3a6b;border-radius:10px;
        padding:0.875rem 1rem;">
          <div style="font-size:0.62rem;font-weight:700;letter-spacing:.07em;
          text-transform:uppercase;color:#2979ff;margin-bottom:.4rem;
          font-family:'JetBrains Mono',monospace;">Category</div>
          <div style="font-size:0.82rem;color:#79c0ff;">{result.category}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    for i, vr in enumerate(result.variations):
        is_orig = i == 0
        tag_cls = "tag-orig" if is_orig else "tag-var"
        card_cls = "var-card original" if is_orig else "var-card"
        label = "ORIGINAL" if is_orig else f"VARIATION {i}"
        st.markdown(f"""
        <div class="{card_cls}">
          <div class="var-tag {tag_cls}">{label}</div>
          <div class="var-text">{vr.variation}</div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — LLM Responses
# ───────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<span class="section-label">LLM Responses</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
      💬 &nbsp;Each variation was sent to Groq independently
    </div>
    """, unsafe_allow_html=True)

    for i, vr in enumerate(result.variations):
        label = "ORIGINAL" if i == 0 else f"VARIATION {i}"
        short = vr.variation[:70] + ("…" if len(vr.variation) > 70 else "")
        with st.expander(f"{label} — {short}", expanded=(i == 0)):
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
              <div class="resp-label">Prompt sent to LLM</div>
              <div style="background:#09192e;border:1px solid #1f3a6b;border-radius:8px;
              padding:0.75rem;font-size:0.83rem;color:#79c0ff;line-height:1.55;">
                {vr.variation}
              </div>
            </div>
            <div>
              <div class="resp-label">LLM Response</div>
              <div style="background:#0d1117;border:1px solid #1c2128;border-radius:8px;
              padding:0.875rem;font-size:0.83rem;color:#cdd9e5;line-height:1.7;white-space:pre-wrap;">
                {vr.response if vr.response else "*(no response received)*"}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Similarity Analysis
# ───────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<span class="section-label">Semantic Similarity Analysis</span>', unsafe_allow_html=True)

    mat = np.array(result.similarity_matrix)
    n = mat.shape[0]
    labels = ["Original"] + [f"Var {i}" for i in range(1, n)]

    col_heat, col_bar = st.columns([1, 1])
    with col_heat:
        st.markdown("""
        <div class="info-pill" style="margin-bottom:0.75rem;">
          📊 &nbsp;Pairwise cosine similarity between responses
        </div>
        """, unsafe_allow_html=True)
        fig_heat = go.Figure(data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale=[
                [0.0,  "#2d0f0f"],
                [0.45, "#3d2a05"],
                [0.75, "#0a1e3a"],
                [1.0,  "#0f2a1a"],
            ],
            zmin=0, zmax=1,
            text=[[f"{mat[i, j]:.3f}" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
            textfont={"size": 10, "family": "JetBrains Mono"},
            showscale=True,
            colorbar=dict(
                tickcolor="#484f58", tickfont=dict(color="#8b949e", size=10),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            ),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="Inter"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        fig_heat.update_xaxes(tickfont=dict(size=10, color="#8b949e"), gridcolor="#1c2128")
        fig_heat.update_yaxes(tickfont=dict(size=10, color="#8b949e"), gridcolor="#1c2128")
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_bar:
        if n > 1:
            sim_to_orig = [mat[0, j] for j in range(1, n)]
            bar_labels = [f"Variation {i}" for i in range(1, n)]
            st.markdown("""
            <div class="info-pill" style="margin-bottom:0.75rem;">
              🎯 &nbsp;Similarity to original response
            </div>
            """, unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=bar_labels,
                y=sim_to_orig,
                marker_color=[bar_color(s * 100) for s in sim_to_orig],
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{s:.3f}" for s in sim_to_orig],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color="#8b949e"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter"),
                yaxis=dict(
                  range=[0, 1.18],
                  gridcolor="#1c2128",
                  tickfont=dict(size=10),
                  title=dict(
                    text="Cosine similarity",
                    font=dict(size=11, color="#484f58")
                  )
                ),
                xaxis=dict(gridcolor="#1c2128", tickfont=dict(size=10)),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Mean similarity metric — guarded against empty upper-triangle (single-response / partial data)
    idx_upper = np.triu_indices_from(mat, k=1)
    vals = mat[idx_upper]
    if vals.size == 0:
        mean_sim = 0.0
        min_sim  = 0.0
        max_sim  = 0.0
    else:
        mean_sim = float(np.mean(vals))
        min_sim  = float(np.min(vals))
        max_sim  = float(np.max(vals))
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Pairwise Similarity", f"{mean_sim:.3f}")
    m2.metric("Min Pairwise Similarity",  f"{min_sim:.3f}")
    m3.metric("Max Pairwise Similarity",  f"{max_sim:.3f}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — PSI Breakdown
# ───────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<span class="section-label">PSI Score Breakdown</span>', unsafe_allow_html=True)

    # Top score card
    score_col, info_col = st.columns([1, 2])
    with score_col:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1c2128;border-radius:14px;
        padding:1.5rem;text-align:center;">
          <div style="font-size:0.65rem;font-weight:700;letter-spacing:.1em;
          text-transform:uppercase;color:#484f58;margin-bottom:0.75rem;">
            Prompt Stability Index
          </div>
          <div class="psi-score-large {psi_class(result.psi_score)}">
            {result.psi_score:.1f}
          </div>
          <div style="font-size:0.72rem;color:#484f58;margin-top:.25rem;">out of 100</div>
          <div style="margin-top:0.75rem;font-size:0.78rem;font-weight:500;
          color:{'#3fb950' if result.psi_score >= 75 else '#d29922' if result.psi_score >= 50 else '#f85149'};">
            {psi_label(result.psi_score)}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1c2128;border-radius:14px;
        padding:1.25rem;font-size:0.83rem;color:#8b949e;line-height:1.75;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
          background:#161b22;border:1px solid #21262d;border-radius:6px;
          padding:0.5rem 0.75rem;color:#79c0ff;margin-bottom:0.75rem;">
            PSI = 100 × (0.50·S + 0.30·K + 0.20·L)
          </div>
          <table style="width:100%;font-size:0.79rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #1c2128;">
              <td style="padding:0.4rem 0;color:#484f58;width:24px;">S</td>
              <td style="padding:0.4rem 0.5rem;color:#cdd9e5;font-weight:500;">Semantic similarity</td>
              <td style="padding:0.4rem 0;color:#2979ff;font-family:'JetBrains Mono',monospace;text-align:right;">50%</td>
              <td style="padding:0.4rem 0 0.4rem 1rem;color:#e6edf3;font-family:'JetBrains Mono',monospace;text-align:right;">{result.semantic_similarity:.3f}</td>
            </tr>
            <tr style="border-bottom:1px solid #1c2128;">
              <td style="padding:0.4rem 0;color:#484f58;">K</td>
              <td style="padding:0.4rem 0.5rem;color:#cdd9e5;font-weight:500;">Keyword consistency</td>
              <td style="padding:0.4rem 0;color:#2979ff;font-family:'JetBrains Mono',monospace;text-align:right;">30%</td>
              <td style="padding:0.4rem 0 0.4rem 1rem;color:#e6edf3;font-family:'JetBrains Mono',monospace;text-align:right;">{result.keyword_consistency:.3f}</td>
            </tr>
            <tr>
              <td style="padding:0.4rem 0;color:#484f58;">L</td>
              <td style="padding:0.4rem 0.5rem;color:#cdd9e5;font-weight:500;">Length consistency</td>
              <td style="padding:0.4rem 0;color:#2979ff;font-family:'JetBrains Mono',monospace;text-align:right;">20%</td>
              <td style="padding:0.4rem 0 0.4rem 1rem;color:#e6edf3;font-family:'JetBrains Mono',monospace;text-align:right;">{result.length_consistency:.3f}</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Component chart
    st.markdown('<span class="section-label">Component Scores vs Weighted Contribution</span>',
                unsafe_allow_html=True)

    comps = {
        "Semantic (S)": (result.semantic_similarity, 0.50),
        "Keyword (K)": (result.keyword_consistency, 0.30),
        "Length (L)": (result.length_consistency, 0.20),
    }
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Raw score",
        x=list(comps.keys()),
        y=[v[0] for v in comps.values()],
        marker_color="#1f3a6b",
        marker_line_color="#2979ff",
        marker_line_width=1,
        text=[f"{v[0]:.3f}" for v in comps.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#8b949e"),
    ))
    fig_comp.add_trace(go.Bar(
        name="Weighted contribution",
        x=list(comps.keys()),
        y=[v[0] * v[1] for v in comps.values()],
        marker_color="#0f2a1a",
        marker_line_color="#3fb950",
        marker_line_width=1,
        text=[f"{v[0]*v[1]:.3f}" for v in comps.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#8b949e"),
    ))
    fig_comp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Inter"),
        yaxis=dict(
          range=[0, 1.2],
          gridcolor="#1c2128",
          tickfont=dict(size=10),
          title=dict(
          text="Score (0–1)",
          font=dict(size=11, color="#484f58")
          )
        ),
        xaxis=dict(gridcolor="#1c2128", tickfont=dict(size=11, color="#cdd9e5")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#8b949e")),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Stability scale
    st.markdown('<span class="section-label">Stability Scale</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;gap:0;border:1px solid #1c2128;border-radius:10px;overflow:hidden;">
      <div style="flex:1;padding:0.875rem 1rem;
        background:{'#0f2a1a' if result.psi_score >= 75 else '#0d1117'};
        border-right:1px solid #1c2128;">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:.08em;
        text-transform:uppercase;color:#3fb950;margin-bottom:.3rem;">Highly Stable</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#3fb950;">75 – 100</div>
        <div style="font-size:0.72rem;color:#484f58;margin-top:.3rem;">Wording barely affects answer</div>
      </div>
      <div style="flex:1;padding:0.875rem 1rem;
        background:{'#3d2a05' if 50 <= result.psi_score < 75 else '#0d1117'};
        border-right:1px solid #1c2128;">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:.08em;
        text-transform:uppercase;color:#d29922;margin-bottom:.3rem;">Moderately Stable</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#d29922;">50 – 74</div>
        <div style="font-size:0.72rem;color:#484f58;margin-top:.3rem;">Some drift across paraphrases</div>
      </div>
      <div style="flex:1;padding:0.875rem 1rem;
        background:{'#2d0f0f' if result.psi_score < 50 else '#0d1117'};">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:.08em;
        text-transform:uppercase;color:#f85149;margin-bottom:.3rem;">Unstable</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#f85149;">0 – 49</div>
        <div style="font-size:0.72rem;color:#484f58;margin-top:.3rem;">Wording significantly changes answer</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — Category Comparison
# ───────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<span class="section-label">Category Comparison</span>', unsafe_allow_html=True)

    all_results = st.session_state.all_results

    if len(all_results) < 2:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:#484f58;">
          <div style="font-size:2rem;margin-bottom:0.75rem;">📋</div>
          <div style="font-size:0.88rem;color:#8b949e;margin-bottom:.5rem;">
            Run experiments across multiple questions to compare categories here.
          </div>
          <div style="font-size:0.75rem;">
            Currently showing 1 result. Run at least 2 to enable comparison.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Table — flatten MultiProviderResult → QuestionResult so both fast and
        # research mode results are displayed correctly.
        rows = []
        for key, r in all_results.items():
            if isinstance(r, MultiProviderResult):
                # Research mode: show one row per provider
                for prov, qr in r.successful_results.items():
                    rows.append({
                        "Category": qr.category,
                        "Question": qr.original_prompt[:55] + ("…" if len(qr.original_prompt) > 55 else ""),
                        "Provider": prov.capitalize(),
                        "PSI Score": round(qr.psi_score, 1),
                        "Semantic": round(qr.semantic_similarity, 3),
                        "Keyword": round(qr.keyword_consistency, 3),
                        "Length": round(qr.length_consistency, 3),
                        "Stability": psi_label(qr.psi_score),
                        "Variations": len(qr.variations) - 1,
                    })
            else:
                rows.append({
                    "Category": r.category,
                    "Question": r.original_prompt[:55] + ("…" if len(r.original_prompt) > 55 else ""),
                    "Provider": "Groq",
                    "PSI Score": round(r.psi_score, 1),
                    "Semantic": round(r.semantic_similarity, 3),
                    "Keyword": round(r.keyword_consistency, 3),
                    "Length": round(r.length_consistency, 3),
                    "Stability": psi_label(r.psi_score),
                    "Variations": len(r.variations) - 1,
                })
        rows.sort(key=lambda x: x["PSI Score"], reverse=True)

        st.dataframe(
            rows,
            use_container_width=True,
            column_config={
                "PSI Score": st.column_config.ProgressColumn(
                    "PSI Score", min_value=0, max_value=100, format="%.1f"
                ),
                "Semantic": st.column_config.NumberColumn(format="%.3f"),
                "Keyword":  st.column_config.NumberColumn(format="%.3f"),
                "Length":   st.column_config.NumberColumn(format="%.3f"),
            },
            hide_index=True,
        )

        # Category stats — flatten MultiProviderResult to QuestionResult list.
        # compute_category_stats expects QuestionResult objects; passing
        # MultiProviderResult (which has no .psi_score or .category) crashes.
        _flat_for_stats: list[QuestionResult] = []
        for _v in all_results.values():
            if isinstance(_v, MultiProviderResult):
                _flat_for_stats.extend(_v.successful_results.values())
            else:
                _flat_for_stats.append(_v)
        cat_stats = compute_category_stats(_flat_for_stats)

        if len(cat_stats) >= 2:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<span class="section-label">Per-Category Averages</span>', unsafe_allow_html=True)

            cats_sorted = sorted(cat_stats.items(), key=lambda x: x[1]["avg"], reverse=True)
            stat_cols = st.columns(len(cats_sorted))
            for col, (cat, stats) in zip(stat_cols, cats_sorted):
                with col:
                    color = cat_color(stats["avg"])
                    st.markdown(f"""
                    <div class="cat-stat-card">
                      <div class="cat-stat-name">{cat.split()[0]}</div>
                      <div class="cat-stat-score" style="color:{color};">{stats['avg']:.1f}</div>
                      <span class="rank-pill">#{stats['rank']}</span>
                      <div class="cat-stat-meta" style="margin-top:0.5rem;">
                        {stats['count']} question{"s" if stats["count"] != 1 else ""} ·
                        Max {stats['max']:.1f} · Min {stats['min']:.1f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Chart
            avgs = [s["avg"] for _, s in cats_sorted]
            cat_names = [c.split()[0] for c, _ in cats_sorted]
            fig_cat = go.Figure(go.Bar(
                x=cat_names,
                y=avgs,
                text=[f"{a:.1f}" for a in avgs],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#8b949e"),
                marker_color=[bar_color(a) for a in avgs],
                marker_line_color="rgba(0,0,0,0)",
            ))
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter"),
                yaxis=dict(range=[0, 115], gridcolor="#1c2128",
                           tickfont=dict(size=10), title="Avg PSI",
                           titlefont=dict(size=11, color="#484f58")),
                xaxis=dict(gridcolor="#1c2128", tickfont=dict(size=11, color="#cdd9e5")),
                margin=dict(l=10, r=10, t=30, b=10),
                height=260,
            )
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(fig_cat, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 6 — Comparison Dashboard (Phase 6) + Export System (Phase 7)
# ───────────────────────────────────────────────────────────────────────────
with tabs[5]:

    # ── Helper: collect all MultiProviderResult entries from session ─────────
    def _get_multi_results() -> list[MultiProviderResult]:
        """Return all MultiProviderResult objects stored in session_state.all_results."""
        out = []
        for v in st.session_state.all_results.values():
            if isinstance(v, MultiProviderResult):
                out.append(v)
        return out

    def _avg_response_words(res) -> float:
        """Safely compute average response-word count from a QuestionResult.
        VariationResult has .variation (prompt text) and .response (LLM reply).
        Falls back gracefully for plain strings (backward-compatible).
        """
        if not res.variations:
            return 0.0
        def _words(v) -> int:
            if isinstance(v, str):
                return len(v.split())
            if hasattr(v, "response"):
                return len(v.response.split())
            if hasattr(v, "variation"):
                return len(v.variation.split())
            return len(str(v).split())
        return sum(_words(v) for v in res.variations) / len(res.variations)

    multi_runs = _get_multi_results()

    # ── Empty state ──────────────────────────────────────────────────────────
    if not multi_runs:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#484f58;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">🔬</div>
          <div style="font-size:1rem;font-weight:600;color:#8b949e;margin-bottom:0.5rem;">
            No comparison data yet
          </div>
          <div style="font-size:0.82rem;color:#484f58;max-width:360px;margin:0 auto;">
            Switch to <strong style="color:#cdd9e5;">🧠 Research Mode</strong> in the sidebar
            and click <strong style="color:#cdd9e5;">▶ Run Experiment</strong>
            to generate multi-provider PSI comparison results.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Graceful disabled export buttons ────────────────────────────────
        st.markdown("---")
        st.markdown('<span class="section-label">Export Results</span>', unsafe_allow_html=True)
        st.info("No data available for export — run an All Models experiment first.")
        exp_cols = st.columns(3)
        with exp_cols[0]:
            st.download_button("⬇  Download CSV", data="", file_name="psaf_comparison.csv",
                               mime="text/csv", disabled=True, use_container_width=True)
        with exp_cols[1]:
            st.download_button("⬇  Download JSON", data="", file_name="psaf_comparison.json",
                               mime="application/json", disabled=True, use_container_width=True)
        with exp_cols[2]:
            st.download_button("⬇  Download Report (TXT)", data="",
                               file_name="psaf_report.txt", mime="text/plain",
                               disabled=True, use_container_width=True)

    else:
        # Use the most-recent multi run as the "active" comparison
        active = multi_runs[-1]

        # ── Prompt header ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                    padding:1rem 1.25rem;margin-bottom:1rem;">
          <div style="font-size:0.65rem;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;color:#484f58;margin-bottom:0.35rem;">
            Compared Prompt
          </div>
          <div style="font-size:0.92rem;color:#e6edf3;font-weight:500;">
            {active.prompt}
          </div>
          <div style="font-size:0.72rem;color:#484f58;margin-top:0.35rem;">
            Category: {active.category}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Build ranked list ─────────────────────────────────────────────────
        ranked = sorted(
            active.successful_results.items(),
            key=lambda kv: kv[1].psi_score,
            reverse=True,
        )

        def _stability_label(score: float) -> str:
            if score >= 75:
                return "Highly Stable"
            if score >= 50:
                return "Moderately Stable"
            return "Unstable"

        def _medal(rank: int) -> str:
            return ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"

        # ── Show failed provider alerts (partial-success mode) ────────────────
        failed_providers = active.failed_results
        if failed_providers:
            for fp_name, fp in failed_providers.items():
                st.markdown(f"""
                <div style="background:#2d0f0f;border:1px solid #b91c1c;border-radius:10px;
                            padding:0.75rem 1rem;margin-bottom:0.5rem;font-size:0.83rem;">
                  <span style="color:#f85149;font-weight:600;">
                    ✗ {fp_name.upper()} — Provider failed (partial results shown)
                  </span><br/>
                  <span style="color:#8b949e;font-size:0.75rem;font-family:'JetBrains Mono',monospace;">
                    {fp.error}
                  </span>
                </div>
                """, unsafe_allow_html=True)

        # ── Medal ranking row ─────────────────────────────────────────────────
        st.markdown('<span class="section-label">Provider Ranking</span>', unsafe_allow_html=True)
        medal_cols = st.columns(len(ranked))
        for idx, (prov, res) in enumerate(ranked):
            with medal_cols[idx]:
                stability = _stability_label(res.psi_score)
                color = "#3fb950" if res.psi_score >= 75 else ("#d29922" if res.psi_score >= 50 else "#f85149")
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                            padding:1.1rem;text-align:center;">
                  <div style="font-size:2rem;margin-bottom:0.4rem;">{_medal(idx)}</div>
                  <div style="font-size:0.85rem;font-weight:600;color:#e6edf3;
                              text-transform:capitalize;">{prov}</div>
                  <div style="font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono',monospace;
                              color:{color};margin:0.4rem 0;">{res.psi_score:.1f}</div>
                  <div style="font-size:0.72rem;color:{color};">{stability}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Comparison table ──────────────────────────────────────────────────
        st.markdown('<span class="section-label">Provider Comparison Table</span>', unsafe_allow_html=True)
        table_rows = []
        for idx, (prov, res) in enumerate(ranked):
            avg_words = (
                _avg_response_words(res)
                if res.variations else 0
            )
            table_rows.append({
                "Rank":     f"#{idx + 1}",
                "Provider": prov.capitalize(),
                "PSI Score": round(res.psi_score, 1),
                "Semantic":  round(res.semantic_similarity, 3),
                "Keyword":   round(res.keyword_consistency, 3),
                "Length":    round(res.length_consistency, 3),
                "Avg Words": round(avg_words, 1),
                "Stability": _stability_label(res.psi_score),
            })

        st.dataframe(
            table_rows,
            use_container_width=True,
            column_config={
                "PSI Score": st.column_config.ProgressColumn(
                    "PSI Score", min_value=0, max_value=100, format="%.1f"
                ),
                "Semantic": st.column_config.NumberColumn(format="%.3f"),
                "Keyword":  st.column_config.NumberColumn(format="%.3f"),
                "Length":   st.column_config.NumberColumn(format="%.3f"),
            },
            hide_index=True,
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown('<span class="section-label">Visual Comparison</span>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        provider_labels = [p.capitalize() for p, _ in ranked]
        psi_values      = [r.psi_score for _, r in ranked]
        bar_colors_psi  = [
            "#3fb950" if v >= 75 else ("#d29922" if v >= 50 else "#f85149")
            for v in psi_values
        ]

        with ch1:
            fig_psi = go.Figure(go.Bar(
                x=provider_labels,
                y=psi_values,
                marker_color=bar_colors_psi,
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{v:.1f}" for v in psi_values],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#8b949e"),
            ))
            fig_psi.update_layout(
                title=dict(text="PSI Score by Provider", font=dict(size=12, color="#8b949e")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter"),
                yaxis=dict(range=[0, 115], gridcolor="#1c2128",
                           tickfont=dict(size=10), title="PSI Score",
                           titlefont=dict(size=11, color="#484f58")),
                xaxis=dict(gridcolor="#1c2128", tickfont=dict(size=11, color="#cdd9e5")),
                margin=dict(l=10, r=10, t=35, b=10),
                height=280,
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        with ch2:
            avg_words_list = []
            for _, res in ranked:
                avg_words_list.append(
                    _avg_response_words(res)
                    if res.variations else 0
                )
            fig_words = go.Figure(go.Bar(
                x=provider_labels,
                y=avg_words_list,
                marker_color="#1f3a6b",
                marker_line_color="#2979ff",
                marker_line_width=1,
                text=[f"{w:.0f}" for w in avg_words_list],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#8b949e"),
            ))
            fig_words.update_layout(
                title=dict(text="Avg Response Length (words)", font=dict(size=12, color="#8b949e")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter"),
                yaxis=dict(gridcolor="#1c2128", tickfont=dict(size=10),
                           title="Avg Words", titlefont=dict(size=11, color="#484f58")),
                xaxis=dict(gridcolor="#1c2128", tickfont=dict(size=11, color="#cdd9e5")),
                margin=dict(l=10, r=10, t=35, b=10),
                height=280,
            )
            st.plotly_chart(fig_words, use_container_width=True)

        # ── PSI component breakdown chart ─────────────────────────────────────
        st.markdown('<span class="section-label">PSI Component Breakdown</span>', unsafe_allow_html=True)
        fig_comp = go.Figure()
        comp_defs = [
            ("Semantic Similarity", [r.semantic_similarity for _, r in ranked], "#1f3a6b", "#2979ff"),
            ("Keyword Consistency", [r.keyword_consistency for _, r in ranked], "#1a3a1a", "#3fb950"),
            ("Length Consistency",  [r.length_consistency  for _, r in ranked], "#3d2a05", "#d29922"),
        ]
        for comp_name, vals, fill, line in comp_defs:
            fig_comp.add_trace(go.Bar(
                name=comp_name,
                x=provider_labels,
                y=vals,
                marker_color=fill,
                marker_line_color=line,
                marker_line_width=1,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=9, color="#8b949e"),
            ))
        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="Inter"),
            yaxis=dict(range=[0, 1.25], gridcolor="#1c2128", tickfont=dict(size=10),
                       title="Score (0–1)", titlefont=dict(size=11, color="#484f58")),
            xaxis=dict(gridcolor="#1c2128", tickfont=dict(size=11, color="#cdd9e5")),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#8b949e")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── All compared prompts (when ≥2 multi-runs exist) ───────────────────
        if len(multi_runs) >= 2:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<span class="section-label">All Compared Prompts</span>', unsafe_allow_html=True)
            hist_rows = []
            for mr in multi_runs:
                mr_ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
                best_prov, best_res = mr_ranked[0]
                hist_rows.append({
                    "Prompt":     mr.prompt[:60] + ("…" if len(mr.prompt) > 60 else ""),
                    "Category":   mr.category,
                    "Best Model": best_prov.capitalize(),
                    "Best PSI":   round(best_res.psi_score, 1),
                    **{p.capitalize() + " PSI": round(r.psi_score, 1) for p, r in mr_ranked},
                })
            hist_rows.sort(key=lambda x: x["Best PSI"], reverse=True)
            st.dataframe(hist_rows, use_container_width=True, hide_index=True)

        # ── PHASE 7 — Export Section ──────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<span class="section-label">Export Results</span>', unsafe_allow_html=True)

        # ── Build export payloads ────────────────────────────────────────────

        # 1. CSV — comparison table for the active multi run
        def _build_csv(mr: MultiProviderResult) -> str:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["Rank", "Provider", "PSI Score", "Semantic Similarity",
                             "Keyword Consistency", "Length Consistency",
                             "Avg Response Words", "Stability", "Prompt", "Category"])
            _ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
            for rank_idx, (prov, res) in enumerate(_ranked):
                avg_w = (
                    _avg_response_words(res)
                    if res.variations else 0
                )
                writer.writerow([
                    rank_idx + 1,
                    prov.capitalize(),
                    round(res.psi_score, 2),
                    round(res.semantic_similarity, 4),
                    round(res.keyword_consistency, 4),
                    round(res.length_consistency, 4),
                    round(avg_w, 1),
                    _stability_label(res.psi_score),
                    mr.prompt,
                    mr.category,
                ])
            return buf.getvalue()

        # 2. JSON — full structured results
        def _build_json(mr: MultiProviderResult) -> str:
            _ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
            payload = {
                "prompt":    mr.prompt,
                "category":  mr.category,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "results": {
                    prov: {
                        "psi_score":           round(res.psi_score, 4),
                        "semantic_similarity":  round(res.semantic_similarity, 4),
                        "keyword_consistency":  round(res.keyword_consistency, 4),
                        "length_consistency":   round(res.length_consistency, 4),
                        "stability":            _stability_label(res.psi_score),
                        "num_variations":       len(res.variations),
                        "response_length_avg_words": round(
                            _avg_response_words(res)
                            if res.variations else 0, 1
                        ),
                    }
                    for prov, res in mr.successful_results.items()
                },
                "ranking": [
                    {"rank": i + 1, "provider": prov, "psi_score": round(res.psi_score, 4)}
                    for i, (prov, res) in enumerate(_ranked)
                ],
            }
            return json.dumps(payload, indent=2)

        # 3. TXT — human-readable ranking report
        def _build_txt(mr: MultiProviderResult) -> str:
            _ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
            lines = [
                "=" * 52,
                "  PSAF Comparison Report",
                "  Prompt Stability Analysis Framework",
                "=" * 52,
                "",
                f"Prompt   : {mr.prompt}",
                f"Category : {mr.category}",
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                "",
                "-" * 52,
                "Ranking",
                "-" * 52,
            ]
            medals = ["🥇 Best Model", "🥈 Second", "🥉 Third"]
            for i, (prov, res) in enumerate(_ranked):
                label = medals[i] if i < len(medals) else f"#{i+1}"
                lines.append(f"  {label}: {prov.capitalize()}")
            lines += ["", "-" * 52, "Scores", "-" * 52]
            for prov, res in _ranked:
                lines.append(f"  {prov.capitalize():<12} PSI: {res.psi_score:6.2f}  "
                             f"({_stability_label(res.psi_score)})")
            lines += ["", "-" * 52, "Component Detail", "-" * 52]
            for prov, res in _ranked:
                lines.append(f"  {prov.capitalize()}")
                lines.append(f"    Semantic Similarity : {res.semantic_similarity:.4f}  (weight 50%)")
                lines.append(f"    Keyword Consistency : {res.keyword_consistency:.4f}  (weight 30%)")
                lines.append(f"    Length Consistency  : {res.length_consistency:.4f}  (weight 20%)")
            lines += ["", "=" * 52, "  End of Report", "=" * 52, ""]
            return "\n".join(lines)

        # ── Render three download buttons ────────────────────────────────────
        csv_data  = _build_csv(active)
        json_data = _build_json(active)
        txt_data  = _build_txt(active)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        exp_c1, exp_c2, exp_c3 = st.columns(3)

        with exp_c1:
            st.download_button(
                label="⬇  Download CSV",
                data=csv_data,
                file_name=f"psaf_comparison_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Comparison table with PSI scores, components, and ranking.",
            )

        with exp_c2:
            st.download_button(
                label="⬇  Download JSON",
                data=json_data,
                file_name=f"psaf_comparison_{ts}.json",
                mime="application/json",
                use_container_width=True,
                help="Full structured results including all PSI components and ranking.",
            )

        with exp_c3:
            st.download_button(
                label="⬇  Download Report (TXT)",
                data=txt_data,
                file_name=f"psaf_report_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Human-readable ranking report with scores and stability labels.",
            )

        st.markdown(
            "<div style='font-size:0.72rem;color:#484f58;margin-top:0.4rem;'>"
            "Exports reflect the most recent All Models run shown above. "
            "Re-run an experiment to refresh.</div>",
            unsafe_allow_html=True,
        )