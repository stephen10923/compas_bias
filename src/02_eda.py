"""
COMPAS Bias Audit — Phase 2: Exploratory Data Analysis
=======================================================
This script generates visualisations that reveal the racial
disparity in COMPAS score distributions BEFORE any modelling.

Outputs (saved to output/charts/):
  - score_distribution.png   — decile score histograms by race
  - recidivism_rate.png      — actual recidivism rate by race
  - score_label_by_race.png  — High/Medium/Low label breakdown
  - score_boxplot.png        — box plot of scores by race
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

CLEAN_PATH = "data/compas_clean.csv"
CHART_DIR  = "output/charts"
os.makedirs(CHART_DIR, exist_ok=True)

COLORS = {
    "African-American": "#E24B4A",
    "Caucasian":        "#378ADD",
    "Hispanic":         "#1D9E75",
    "Other":            "#EF9F27",
    "Native American":  "#9B59B6",
    "Asian":            "#95A5A6",
}


def load():
    df = pd.read_csv(CLEAN_PATH)
    print(f"[OK] Loaded {len(df):,} records")
    return df


# ── Chart 1: Score distribution histograms ──────────────────
def plot_score_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "COMPAS decile score distribution by race\n"
        "(1 = low risk, 10 = high risk)",
        fontsize=13, fontweight="bold"
    )

    for ax, race, color in zip(
        axes,
        ["African-American", "Caucasian"],
        ["#E24B4A", "#378ADD"]
    ):
        subset = df[df["race"] == race]["decile_score"]
        ax.hist(subset, bins=10, range=(1, 11),
                color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{race}\n(n={len(subset):,}, mean={subset.mean():.1f})",
                     fontsize=11)
        ax.set_xlabel("Decile score")
        ax.set_ylabel("Count")
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        mean_val = subset.mean()
        ax.axvline(mean_val, color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.legend()

    plt.tight_layout()
    out = f"{CHART_DIR}/score_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Chart 2: Actual recidivism rate by race ─────────────────
def plot_recidivism_rate(df):
    recid = (df.groupby("race")["label"].mean() * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [COLORS.get(r, "#95A5A6") for r in recid.index]
    bars    = ax.bar(recid.index, recid.values, color=colors,
                     edgecolor="white", linewidth=0.5)

    ax.set_title("Actual two-year recidivism rate by race\n"
                 "(ground truth — not COMPAS score)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Recidivism rate (%)")
    ax.set_ylim(0, 65)
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, recid.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = f"{CHART_DIR}/recidivism_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Chart 3: Score label breakdown by race ──────────────────
def plot_score_label_breakdown(df):
    label_order  = ["Low", "Medium", "High"]
    label_colors = {"Low": "#1D9E75", "Medium": "#EF9F27", "High": "#E24B4A"}

    races = ["African-American", "Caucasian", "Hispanic", "Other"]
    x     = np.arange(len(races))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, lbl in enumerate(label_order):
        vals = []
        for race in races:
            sub = df[df["race"] == race]
            pct = (sub["score_text"] == lbl).mean() * 100
            vals.append(pct)
        bars = ax.bar(x + i * width, vals, width,
                      label=lbl, color=label_colors[lbl],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title("COMPAS score label breakdown by race\n"
                 "(% of defendants in each racial group per label)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(races)
    ax.set_ylabel("Percentage of group (%)")
    ax.set_ylim(0, 75)
    ax.legend(title="COMPAS label")
    plt.tight_layout()

    out = f"{CHART_DIR}/score_label_by_race.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Chart 4: Box plot of scores ─────────────────────────────
def plot_score_boxplot(df):
    races = ["African-American", "Caucasian", "Hispanic", "Other"]
    data  = [df[df["race"] == r]["decile_score"].values for r in races]
    colors = [COLORS.get(r, "#95A5A6") for r in races]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(races) + 1))
    ax.set_xticklabels(races, rotation=15)
    ax.set_ylabel("COMPAS decile score")
    ax.set_title("COMPAS score distribution — box plot by race\n"
                 "(median, IQR, and outliers)",
                 fontsize=12, fontweight="bold")
    ax.set_yticks(range(1, 11))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"{CHART_DIR}/score_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Print key EDA stats ─────────────────────────────────────
def print_eda_stats(df):
    print("\n" + "=" * 55)
    print("  EDA KEY STATISTICS")
    print("=" * 55)

    for race in ["African-American", "Caucasian"]:
        sub = df[df["race"] == race]
        print(f"\n  {race} (n={len(sub):,}):")
        print(f"    Mean COMPAS score : {sub['decile_score'].mean():.2f}")
        print(f"    High-risk label   : {(sub['score_text']=='High').mean()*100:.1f}%")
        print(f"    Actual recidivism : {sub['label'].mean()*100:.1f}%")

    aa = df[df["race"] == "African-American"]
    ca = df[df["race"] == "Caucasian"]
    print(f"\n  Score disparity     : {aa['decile_score'].mean() - ca['decile_score'].mean():.2f} pts")
    print(f"  High-risk disparity : {(aa['score_text']=='High').mean()*100 - (ca['score_text']=='High').mean()*100:.1f} pp")
    print("=" * 55)


if __name__ == "__main__":
    df = load()
    print_eda_stats(df)
    plot_score_distribution(df)
    plot_recidivism_rate(df)
    plot_score_label_breakdown(df)
    plot_score_boxplot(df)
    print("\n[OK] All EDA charts saved to output/charts/")
    print("[NEXT] Run: python src/03_model_training.py")

