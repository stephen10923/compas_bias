"""
COMPAS Bias Audit — Phase 4: Bias Detection
============================================
Measures racial bias using:
  1. Per-group confusion matrices (FPR, FNR, TPR, accuracy)
  2. Standard fairness metrics (disparate impact, equal opportunity, etc.)
  3. Visual comparisons across racial groups

This is the CORE of the audit — detecting the unfairness.

Outputs:
  - output/charts/per_group_fpr.png       — FPR by race
  - output/charts/confusion_matrices.png  — side-by-side matrices
  - output/charts/fairness_metrics.png    — all metrics at once
  - output/metrics/bias_metrics.csv       — numeric results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PRED_PATH   = "data/test_predictions.csv"
METRICS_DIR = "output/metrics"
CHART_DIR   = "output/charts"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHART_DIR,   exist_ok=True)

COLORS = {
    "African-American": "#E24B4A",
    "Caucasian":        "#378ADD",
    "Hispanic":         "#1D9E75",
    "Other":            "#EF9F27",
}

# Fairness thresholds (industry standard)
FAIR_THRESHOLDS = {
    "disparate_impact":  (0.80, 1.25),   # 80% rule
    "equal_opp_diff":    (-0.10, 0.10),
    "avg_odds_diff":     (-0.10, 0.10),
    "stat_parity_diff":  (-0.10, 0.10),
}


# ── Per-group metrics ────────────────────────────────────────
def compute_group_metrics(df, pred_col="pred_logistic_regression",
                           true_col="true_label", group_col="race"):
    rows = []
    for group in df[group_col].unique():
        sub = df[df[group_col] == group]
        if len(sub) < 30:
            continue
        tn, fp, fn, tp = confusion_matrix(
            sub[true_col], sub[pred_col]
        ).ravel()
        n = len(sub)
        rows.append({
            "group":    group,
            "n":        n,
            "TP":       int(tp),
            "FP":       int(fp),
            "TN":       int(tn),
            "FN":       int(fn),
            "accuracy": round((tp + tn) / n, 3),
            "FPR":      round(fp / (fp + tn), 3),   # False Pos Rate
            "FNR":      round(fn / (fn + tp), 3),   # False Neg Rate
            "TPR":      round(tp / (tp + fn), 3),   # Recall / Sensitivity
            "PPV":      round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ── Fairness metrics (manual — no AIF360 required) ──────────
def compute_fairness_metrics(group_df,
                              privileged="Caucasian",
                              unprivileged="African-American"):
    priv   = group_df[group_df["group"] == privileged].iloc[0]
    unpriv = group_df[group_df["group"] == unprivileged].iloc[0]

    # Positive prediction rate = (TP+FP)/N
    priv_ppr   = (priv["TP"]   + priv["FP"])   / priv["n"]
    unpriv_ppr = (unpriv["TP"] + unpriv["FP"]) / unpriv["n"]

    metrics = {
        "disparate_impact":   round(unpriv_ppr / priv_ppr, 3) if priv_ppr > 0 else None,
        "stat_parity_diff":   round(unpriv_ppr - priv_ppr, 3),
        "equal_opp_diff":     round(unpriv["TPR"] - priv["TPR"], 3),
        "avg_odds_diff":      round(
            ((unpriv["TPR"] - priv["TPR"]) +
             (unpriv["FPR"] - priv["FPR"])) / 2, 3),
        "fpr_ratio":          round(unpriv["FPR"] / priv["FPR"], 3) if priv["FPR"] > 0 else None,
        "aa_fpr":             unpriv["FPR"],
        "ca_fpr":             priv["FPR"],
        "fpr_diff_pp":        round((unpriv["FPR"] - priv["FPR"]) * 100, 1),
    }
    return metrics


# ── Print bias report ────────────────────────────────────────
def print_bias_report(group_df, fairness_metrics):
    print("\n" + "=" * 60)
    print("  BIAS DETECTION RESULTS")
    print("=" * 60)

    print("\n  Per-group metrics:")
    print(f"  {'Group':<22} {'n':>5}  {'Acc':>5}  "
          f"{'FPR':>5}  {'FNR':>5}  {'TPR':>5}")
    print("  " + "-" * 55)
    for _, row in group_df.iterrows():
        flag = " ← HIGH BIAS" if row["group"] == "African-American" else ""
        print(f"  {row['group']:<22} {row['n']:>5}  "
              f"{row['accuracy']:>5.3f}  {row['FPR']:>5.3f}  "
              f"{row['FNR']:>5.3f}  {row['TPR']:>5.3f}{flag}")

    print("\n  Fairness metrics (AA vs Caucasian):")
    fm = fairness_metrics
    checks = {
        "Disparate impact":         (fm["disparate_impact"],   "(ideal = 1.0, >0.80 = fair)", fm["disparate_impact"] and 0.80 <= fm["disparate_impact"] <= 1.25),
        "Statistical parity diff":  (fm["stat_parity_diff"],   "(ideal = 0.0, |x|<0.10 = fair)", abs(fm["stat_parity_diff"]) < 0.10),
        "Equal opportunity diff":   (fm["equal_opp_diff"],     "(ideal = 0.0, |x|<0.10 = fair)", abs(fm["equal_opp_diff"]) < 0.10),
        "Average odds difference":  (fm["avg_odds_diff"],      "(ideal = 0.0, |x|<0.10 = fair)", abs(fm["avg_odds_diff"]) < 0.10),
        "FPR ratio (AA/Caucasian)": (fm["fpr_ratio"],          "(ideal = 1.0)",                  False),
    }
    print(f"\n  {'Metric':<30} {'Value':>8}  {'Status':<12}  Note")
    print("  " + "-" * 75)
    for name, (val, note, fair) in checks.items():
        status = "FAIR   " if fair else "BIASED "
        icon   = "[OK]" if fair else "[!!]"
        print(f"  {name:<30} {str(val):>8}  {icon} {status}  {note}")

    print(f"\n  KEY FINDING:")
    print(f"  Black defendants: FPR = {fm['aa_fpr']*100:.1f}%")
    print(f"  White defendants: FPR = {fm['ca_fpr']*100:.1f}%")
    print(f"  → Black defendants are {fm['fpr_ratio']:.1f}x more likely to be")
    print(f"    falsely labelled high-risk (+{fm['fpr_diff_pp']:.1f} percentage points)")
    print("=" * 60)


# ── Chart: FPR by race ───────────────────────────────────────
def plot_fpr_by_race(group_df):
    races  = group_df["group"].tolist()
    fprs   = group_df["FPR"].tolist()
    colors = [COLORS.get(r, "#95A5A6") for r in races]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(races, [f * 100 for f in fprs],
                  color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=fprs[0] * 100, color="#E24B4A", linestyle="--",
               linewidth=1, alpha=0.5, label="AA baseline")
    ax.set_title("False Positive Rate by race\n"
                 "(% of non-recidivists incorrectly labelled high-risk)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_ylim(0, 50)
    ax.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Mark the disparity
    aa_fpr = group_df[group_df["group"] == "African-American"]["FPR"].iloc[0]
    ca_fpr = group_df[group_df["group"] == "Caucasian"]["FPR"].iloc[0]
    ax.annotate(
        f"+{(aa_fpr - ca_fpr)*100:.1f}pp disparity",
        xy=(0, aa_fpr * 100), xytext=(1, aa_fpr * 100 + 8),
        arrowprops=dict(arrowstyle="->", color="#E24B4A"),
        fontsize=10, color="#E24B4A"
    )

    plt.tight_layout()
    out = f"{CHART_DIR}/per_group_fpr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Chart: Side-by-side confusion matrices ───────────────────
def plot_confusion_matrices(df, pred_col="pred_logistic_regression"):
    races = ["African-American", "Caucasian"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion matrices by race\n"
                 "(rows = actual, columns = predicted)",
                 fontsize=13, fontweight="bold")

    for ax, race in zip(axes, races):
        sub = df[df["race"] == race]
        cm  = confusion_matrix(sub["true_label"], sub[pred_col])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No recid", "Recid"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        ax.set_title(f"{race}\nFPR={fpr*100:.1f}%  FNR={fnr*100:.1f}%",
                     fontsize=11)

    plt.tight_layout()
    out = f"{CHART_DIR}/confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Chart: All fairness metrics ──────────────────────────────
def plot_fairness_metrics(group_df):
    races   = group_df["group"].tolist()
    colors  = [COLORS.get(r, "#95A5A6") for r in races]
    metrics = ["FPR", "FNR", "accuracy"]
    titles  = ["False positive rate", "False negative rate", "Accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Per-group fairness metrics — full comparison",
                 fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = group_df[metric].tolist()
        bars = ax.bar(races, [v * 100 for v in vals],
                      color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 85)
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = f"{CHART_DIR}/fairness_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    df = pd.read_csv(PRED_PATH)
    print(f"[OK] Loaded {len(df):,} predictions")

    group_df = compute_group_metrics(df)
    fm       = compute_fairness_metrics(group_df)

    print_bias_report(group_df, fm)

    # Save metrics CSV
    group_df.to_csv(f"{METRICS_DIR}/per_group_metrics.csv", index=False)
    pd.DataFrame([fm]).to_csv(f"{METRICS_DIR}/fairness_metrics.csv", index=False)
    print(f"\n[OK] Metrics saved to {METRICS_DIR}/")

    # Generate charts
    plot_fpr_by_race(group_df)
    plot_confusion_matrices(df)
    plot_fairness_metrics(group_df)

    print("\n[OK] Phase 4 complete.")
    print("[NEXT] Run: python src/05_mitigation.py")

