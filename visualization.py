"""
visualization.py — Phase 4: Publication-Quality Visualizations
Prompt Stability Analysis Framework (PSAF)

Generates five publication-quality figures saved to plots/:
  1. PSI Distribution Histogram
  2. Category Comparison Bar Chart
  3. Prompt Variation Consistency Chart
  4. PSI Heatmap
  5. Stability Ranking Chart
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
PLOTS_DIR   = BASE_DIR / "plots"
RESULTS_CSV = BASE_DIR / "experiment_results.csv"
CATEGORY_CSV = BASE_DIR / "category_comparison.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
PALETTE = {
    "Definition Questions":  "#4C72B0",
    "Educational Questions": "#55A868",
    "Reasoning Questions":   "#C44E52",
    "Technical Questions":   "#8172B2",
}
ACCENT       = "#2C3E50"
GRID_COLOR   = "#E8E8E8"
BG_COLOR     = "#FAFAFA"
TEXT_COLOR   = "#2C3E50"
FIGURE_DPI   = 180

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10.5,
    "axes.labelweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.linewidth":     0.7,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "figure.facecolor":   BG_COLOR,
    "axes.facecolor":     BG_COLOR,
    "savefig.facecolor":  "white",
    "savefig.dpi":        FIGURE_DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

CATEGORY_ORDER = [
    "Definition Questions",
    "Educational Questions",
    "Reasoning Questions",
    "Technical Questions",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df   = pd.read_csv(RESULTS_CSV)
    cat  = pd.read_csv(CATEGORY_CSV)
    df["category"]   = pd.Categorical(df["category"],   categories=CATEGORY_ORDER, ordered=True)
    cat["category"]  = pd.Categorical(cat["category"],  categories=CATEGORY_ORDER, ordered=True)
    return df.sort_values("category"), cat.sort_values("category")


# ── Helper: category colours list ────────────────────────────────────────────
def cat_colors(categories: list[str]) -> list[str]:
    return [PALETTE[c] for c in categories]


# ─────────────────────────────────────────────────────────────────────────────
# 1. PSI Distribution Histogram
# ─────────────────────────────────────────────────────────────────────────────
def plot_psi_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    psi_vals = df["avg_psi"].values
    n_bins   = 10
    counts, bin_edges = np.histogram(psi_vals, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Shade bars by PSI zone
    zone_colors = []
    for center in bin_centers:
        if center >= 80:
            zone_colors.append("#27AE60")   # stable — green
        elif center >= 65:
            zone_colors.append("#F39C12")   # moderate — amber
        else:
            zone_colors.append("#E74C3C")   # unstable — red

    ax.bar(
        bin_edges[:-1], counts,
        width=np.diff(bin_edges),
        color=zone_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.87,
        align="edge",
        zorder=3,
    )

    # KDE overlay
    from scipy.stats import gaussian_kde  # lightweight local import
    kde = gaussian_kde(psi_vals, bw_method=0.4)
    x_fine = np.linspace(psi_vals.min() - 2, psi_vals.max() + 2, 300)
    kde_scale = len(psi_vals) * (bin_edges[1] - bin_edges[0])
    ax.plot(x_fine, kde(x_fine) * kde_scale,
            color=ACCENT, linewidth=2, label="KDE")

    # Reference lines
    mean_psi   = psi_vals.mean()
    median_psi = np.median(psi_vals)
    ax.axvline(mean_psi,   color="#2980B9", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_psi:.1f}")
    ax.axvline(median_psi, color="#8E44AD", linestyle=":",  linewidth=1.5,
               label=f"Median = {median_psi:.1f}")

    # Zone legend patches
    from matplotlib.patches import Patch
    zone_patches = [
        Patch(facecolor="#27AE60", alpha=0.85, label="Stable  (PSI ≥ 80)"),
        Patch(facecolor="#F39C12", alpha=0.85, label="Moderate (65–79)"),
        Patch(facecolor="#E74C3C", alpha=0.85, label="Unstable (< 65)"),
    ]
    leg1 = ax.legend(handles=zone_patches, loc="upper left",
                     title="Stability Zones", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", title="Statistics", title_fontsize=8)

    ax.set_xlabel("Average PSI Score")
    ax.set_ylabel("Number of Prompts")
    ax.set_title("PSI Distribution Across All Prompts")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(psi_vals.min() - 3, psi_vals.max() + 3)

    # Annotation: n
    ax.text(0.98, 0.96, f"n = {len(psi_vals)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT_COLOR)

    out = PLOTS_DIR / "1_psi_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Category Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_category_comparison(cat: pd.DataFrame) -> Path:
    metrics = {
        "Avg PSI":            "category_avg_psi",
        "Avg Semantic Sim.":  "category_avg_semantic",
        "Avg Keyword Cons.":  "category_avg_keyword",
        "Avg Length Cons.":   "category_avg_length",
    }

    cats     = list(cat.sort_values("stability_rank")["category"])
    n_cats   = len(cats)
    n_met    = len(metrics)
    x        = np.arange(n_cats)
    bar_w    = 0.18
    offsets  = np.linspace(-(n_met - 1) / 2, (n_met - 1) / 2, n_met) * bar_w

    fig, ax = plt.subplots(figsize=(10, 5.5))

    metric_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for idx, ((label, col), offset, color) in enumerate(
        zip(metrics.items(), offsets, metric_colors)
    ):
        scale = 1 if col == "category_avg_psi" else 100   # normalise sims to PSI scale
        vals  = cat.set_index("category").loc[cats, col].values * scale
        bars  = ax.bar(x + offset, vals, bar_w * 0.92,
                       label=label, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.6, zorder=3)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=12, ha="right")
    ax.set_ylabel("Score (PSI 0–100 scale)")
    ax.set_title("Category Performance Comparison\n(Ranked by Stability)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", ncol=2)

    # Rank badges
    rank_labels = cat.set_index("category")["stability_rank"]
    for xi, c in zip(x, cats):
        ax.text(xi, 102, f"Rank {int(rank_labels[c])}",
                ha="center", va="bottom", fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=PALETTE[c], edgecolor="none", alpha=0.9))

    out = PLOTS_DIR / "2_category_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt Variation Consistency Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_variation_consistency(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    # ── Left: error-bar chart (avg ± std per prompt) ───────────────────────
    ax = axes[0]
    df_sorted = df.sort_values(["category", "avg_psi"], ascending=[True, False])
    labels    = [
        f"{row.question_id}\n({row.avg_psi:.0f})"
        for _, row in df_sorted.iterrows()
    ]
    y        = np.arange(len(df_sorted))
    colors   = [PALETTE[c] for c in df_sorted["category"]]

    ax.barh(y, df_sorted["avg_psi"], xerr=df_sorted["std_psi"],
            color=colors, alpha=0.80, edgecolor="white", linewidth=0.5,
            error_kw=dict(ecolor="#555555", capsize=3, linewidth=1.1),
            zorder=3, height=0.65)

    # Min / max range markers
    for yi, (_, row) in zip(y, df_sorted.iterrows()):
        ax.plot([row.min_psi, row.max_psi], [yi, yi],
                color="#888888", linewidth=0.8, zorder=2)
        ax.scatter([row.min_psi, row.max_psi], [yi, yi],
                   color="#888888", s=12, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_xlabel("PSI Score")
    ax.set_title("Per-Prompt Variation\n(bar = mean ± std; whiskers = min/max)")
    ax.axvline(80, color="#27AE60", linestyle="--", linewidth=1,
               alpha=0.7, label="Stable threshold (80)")
    ax.axvline(65, color="#F39C12", linestyle=":",  linewidth=1,
               alpha=0.7, label="Moderate threshold (65)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(50, 100)

    # Category legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=v, label=k, alpha=0.85) for k, v in PALETTE.items()]
    ax.legend(handles=patches + [
        mpl.lines.Line2D([0], [0], color="#27AE60", linestyle="--",
                         linewidth=1.2, label="Stable ≥ 80"),
        mpl.lines.Line2D([0], [0], color="#F39C12", linestyle=":",
                         linewidth=1.2, label="Moderate ≥ 65"),
    ], fontsize=7.5, loc="lower right")

    # ── Right: scatter std vs avg PSI ──────────────────────────────────────
    ax2 = axes[1]
    for cat, grp in df_sorted.groupby("category", observed=True):
        ax2.scatter(grp["avg_psi"], grp["std_psi"],
                    color=PALETTE[cat], s=70, alpha=0.85,
                    edgecolors="white", linewidths=0.5,
                    label=cat, zorder=3)
        for _, row in grp.iterrows():
            ax2.annotate(row.question_id,
                         (row.avg_psi, row.std_psi),
                         textcoords="offset points", xytext=(5, 2),
                         fontsize=6.5, color=TEXT_COLOR)

    # Trendline
    coeffs = np.polyfit(df_sorted["avg_psi"], df_sorted["std_psi"], 1)
    xr     = np.linspace(df_sorted["avg_psi"].min(), df_sorted["avg_psi"].max(), 100)
    ax2.plot(xr, np.polyval(coeffs, xr),
             color=ACCENT, linewidth=1.4, linestyle="--", alpha=0.6,
             label="Trend")

    ax2.set_xlabel("Average PSI Score")
    ax2.set_ylabel("PSI Standard Deviation (across variations)")
    ax2.set_title("Consistency vs. Stability\n(lower std = more consistent across prompts)")
    ax2.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Prompt Variation Consistency Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    out = PLOTS_DIR / "3_variation_consistency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. PSI Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_psi_heatmap(df: pd.DataFrame) -> Path:
    metrics = {
        "Avg PSI":         "avg_psi",
        "Max PSI":         "max_psi",
        "Min PSI":         "min_psi",
        "Std PSI":         "std_psi",
        "Semantic Sim.":   "avg_semantic",
        "Keyword Cons.":   "avg_keyword",
        "Length Cons.":    "avg_length",
    }

    # Build pivot: rows = prompts (sorted by category then avg_psi desc)
    df_s = df.sort_values(["category", "avg_psi"], ascending=[True, False]).copy()

    # Normalise each column to [0, 1] for uniform colour scale
    mat = pd.DataFrame(index=df_s["question_id"])
    for label, col in metrics.items():
        vals = df_s[col].values.astype(float)
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        mat[label] = norm

    # Category row-colours
    row_colors = pd.Series(
        [PALETTE[c] for c in df_s["category"]], index=df_s["question_id"]
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot=df_s[list(metrics.values())].set_index(df_s["question_id"]).rename(
            columns={v: k for k, v in metrics.items()}
        ).round(2),
        fmt=".2f",
        annot_kws={"size": 7.5},
        cbar_kws={"label": "Normalised Score (within metric)",
                  "shrink": 0.7, "pad": 0.02},
    )

    # Category colour strip on the left
    strip_w = 0.018
    for yi, qid in enumerate(mat.index):
        cat = df_s.loc[df_s["question_id"] == qid, "category"].values[0]
        rect = mpl.patches.FancyBboxPatch(
            (-strip_w * len(metrics) * 0.08, yi),
            strip_w * 0.8, 1,
            boxstyle="square,pad=0",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            facecolor=PALETTE[cat],
            edgecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xlabel("Metric", labelpad=6)
    ax.set_ylabel("Prompt ID", labelpad=6)
    ax.set_title("PSI Multi-Metric Heatmap\n(colour scale normalised per metric; green = high)",
                 pad=12)
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=20)

    # Category legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=v, label=k, alpha=0.9) for k, v in PALETTE.items()]
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(1.18, 1.0), title="Category",
              title_fontsize=8, fontsize=8, framealpha=0.85)

    fig.tight_layout()

    out = PLOTS_DIR / "4_psi_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stability Ranking Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_stability_ranking(df: pd.DataFrame, cat: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(13, 6.5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.38)

    ax_main = fig.add_subplot(gs[0])
    ax_cat  = fig.add_subplot(gs[1])

    # ── Left: ranked lollipop for individual prompts ───────────────────────
    df_ranked = df.sort_values("avg_psi", ascending=False).reset_index(drop=True)
    df_ranked["rank"] = df_ranked.index + 1
    y = df_ranked["rank"].values

    for _, row in df_ranked.iterrows():
        rank   = int(row["rank"])
        color  = PALETTE[row["category"]]
        # Horizontal stem
        ax_main.hlines(rank, 0, row["avg_psi"], color="#CCCCCC",
                       linewidth=1.2, zorder=2)
        # Dot
        ax_main.scatter(row["avg_psi"], rank, color=color, s=120,
                        zorder=4, edgecolors="white", linewidths=0.8)
        # Label
        ax_main.text(row["avg_psi"] + 0.4, rank,
                     f"{row.question_id}  ({row.avg_psi:.1f})",
                     va="center", fontsize=7.8, color=TEXT_COLOR)

    # Stability zone backgrounds
    ax_main.axvspan(80, 100, alpha=0.06, color="#27AE60", zorder=1)
    ax_main.axvspan(65,  80, alpha=0.06, color="#F39C12", zorder=1)
    ax_main.axvspan(0,   65, alpha=0.06, color="#E74C3C", zorder=1)

    ax_main.axvline(80, color="#27AE60", linewidth=1, linestyle="--", alpha=0.6)
    ax_main.axvline(65, color="#F39C12", linewidth=1, linestyle=":",  alpha=0.6)

    # Zone labels at top
    for x_pos, label, col in [
        (90, "Stable", "#27AE60"),
        (72.5, "Moderate", "#F39C12"),
        (57, "Unstable", "#E74C3C"),
    ]:
        ax_main.text(x_pos, 0.35, label, ha="center", va="bottom",
                     fontsize=8, color=col, fontweight="bold",
                     transform=ax_main.get_xaxis_transform())

    ax_main.set_yticks(y)
    ax_main.set_yticklabels([f"#{r}" for r in y], fontsize=8)
    ax_main.invert_yaxis()
    ax_main.set_xlabel("Average PSI Score")
    ax_main.set_ylabel("Rank")
    ax_main.set_title("Individual Prompt Stability Ranking")
    ax_main.set_xlim(50, 100)

    # Category legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=v, label=k, alpha=0.85) for k, v in PALETTE.items()]
    ax_main.legend(handles=patches, fontsize=7.5, loc="lower right")

    # ── Right: category podium bar chart ──────────────────────────────────
    cat_s = cat.sort_values("stability_rank")
    bars  = ax_cat.barh(
        range(len(cat_s)),
        cat_s["category_avg_psi"],
        color=[PALETTE[c] for c in cat_s["category"]],
        edgecolor="white", linewidth=0.7,
        alpha=0.85, height=0.5, zorder=3,
    )

    # Std error caps
    ax_cat.errorbar(
        cat_s["category_avg_psi"],
        range(len(cat_s)),
        xerr=cat_s["category_std_psi"],
        fmt="none",
        ecolor="#555555",
        capsize=4,
        linewidth=1.2,
        zorder=4,
    )

    ax_cat.set_yticks(range(len(cat_s)))
    ax_cat.set_yticklabels(
        [f"#{r}  {c.split()[0]}" for r, c in
         zip(cat_s["stability_rank"], cat_s["category"])],
        fontsize=8.5,
    )
    ax_cat.invert_yaxis()

    for bar, (_, row) in zip(bars, cat_s.iterrows()):
        ax_cat.text(row.category_avg_psi + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{row.category_avg_psi:.1f}",
                    va="center", fontsize=8.5, color=TEXT_COLOR)

    ax_cat.set_xlabel("Category Avg PSI")
    ax_cat.set_title("Category Rankings\n(mean ± std)")
    ax_cat.set_xlim(60, 95)
    ax_cat.axvline(80, color="#27AE60", linestyle="--", linewidth=1, alpha=0.6)

    fig.suptitle("PSAF Stability Ranking — Phase 4 Summary",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = PLOTS_DIR / "5_stability_ranking.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n══════════════════════════════════════════════════")
    print("  PSAF Phase 4 — Publication-Quality Visualizations")
    print("══════════════════════════════════════════════════\n")

    print("Loading data …")
    df, cat = load_data()
    print(f"  Experiment results : {len(df)} prompts across "
          f"{df['category'].nunique()} categories")
    print(f"  Category summary   : {len(cat)} category rows\n")

    print("Generating figures …")
    plot_psi_distribution(df)
    plot_category_comparison(cat)
    plot_variation_consistency(df)
    plot_psi_heatmap(df)
    plot_stability_ranking(df, cat)

    print(f"\nAll figures saved to: {PLOTS_DIR}/")
    saved = sorted(PLOTS_DIR.glob("*.png"))
    for p in saved:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<35s}  {size_kb:6.1f} KB")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
