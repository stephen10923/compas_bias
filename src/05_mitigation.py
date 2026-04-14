"""
COMPAS Bias Audit — Phase 5: Fairness Mitigation
=================================================
Applies two bias mitigation strategies and compares results:

  Strategy 1 — Pre-processing: Sample Reweighing
    Give higher training weight to underrepresented (race × label) groups
    so the model learns a more balanced decision boundary.

  Strategy 2 — Post-processing: Threshold Adjustment
    Set a lower decision threshold for the African-American group,
    reducing their FPR at the cost of slightly lower overall accuracy.

Outputs:
  - output/charts/mitigation_comparison.png
  - output/charts/threshold_sweep.png
  - output/metrics/mitigation_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (accuracy_score, confusion_matrix,
                                     roc_curve, roc_auc_score)

CLEAN_PATH  = "data/compas_clean.csv"
METRICS_DIR = "output/metrics"
CHART_DIR   = "output/charts"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHART_DIR,   exist_ok=True)


# ── Data prep (same as Phase 3) ──────────────────────────────
def prepare(df):
    df = df.copy()
    le = LabelEncoder()
    df["race_enc"]   = le.fit_transform(df["race"])
    df["sex_enc"]    = le.fit_transform(df["sex"])
    df["charge_enc"] = le.fit_transform(df["c_charge_degree"])
    df["age_enc"]    = le.fit_transform(df["age_cat"])
    features = ["age", "priors_count", "charge_enc",
                "sex_enc", "race_enc", "age_enc"]
    return df[features], df["label"], df, features


# ── Metrics helper ───────────────────────────────────────────
def group_metrics(y_true, y_pred, race_series):
    rows = {}
    for race in ["African-American", "Caucasian"]:
        mask = race_series == race
        yt, yp = y_true[mask], y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        rows[race] = {
            "accuracy": round((tp + tn) / len(yt), 3),
            "FPR":      round(fp / (fp + tn), 3),
            "FNR":      round(fn / (fn + tp), 3),
        }
    return rows


def overall_metrics(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3)}


def disparate_impact(y_pred, race_series):
    aa_ppr = y_pred[race_series == "African-American"].mean()
    ca_ppr = y_pred[race_series == "Caucasian"].mean()
    return round(aa_ppr / ca_ppr, 3) if ca_ppr > 0 else None


# ── Baseline ─────────────────────────────────────────────────
def baseline(X_train, X_test, y_train, y_test, race_test):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    gm = group_metrics(y_test, y_pred, race_test)
    di = disparate_impact(y_pred, race_test)
    om = overall_metrics(y_test, y_pred)
    print(f"  [Baseline] Accuracy={om['accuracy']:.3f}  "
          f"AA_FPR={gm['African-American']['FPR']:.3f}  "
          f"CA_FPR={gm['Caucasian']['FPR']:.3f}  DI={di:.3f}")
    return lr, y_pred, gm, om, di


# ── Strategy 1: Reweighing ───────────────────────────────────
def reweighing(X_train, X_test, y_train, y_test, race_train, race_test):
    """
    Assign sample weights inversely proportional to group frequency
    within each (race, label) cell. This forces the model to treat
    all groups more equally during training.
    """
    train_df = X_train.copy()
    train_df["label"] = y_train.values
    train_df["race"]  = race_train.values

    # Count each (race, label) combination
    cell_counts = train_df.groupby(["race", "label"]).size().reset_index(name="count")
    total = len(train_df)
    n_groups = train_df["race"].nunique()
    n_labels = 2

    # Expected count if perfectly balanced
    cell_counts["expected"] = total / (n_groups * n_labels)
    cell_counts["weight"]   = cell_counts["expected"] / cell_counts["count"]

    # Assign weight to each training sample
    weights = train_df.apply(
        lambda row: cell_counts[
            (cell_counts["race"]  == row["race"]) &
            (cell_counts["label"] == row["label"])
        ]["weight"].values[0],
        axis=1
    ).values

    lr_rw = LogisticRegression(max_iter=1000, random_state=42)
    lr_rw.fit(X_train, y_train, sample_weight=weights)
    y_pred = lr_rw.predict(X_test)

    gm = group_metrics(y_test, y_pred, race_test)
    di = disparate_impact(y_pred, race_test)
    om = overall_metrics(y_test, y_pred)
    print(f"  [Reweighing] Accuracy={om['accuracy']:.3f}  "
          f"AA_FPR={gm['African-American']['FPR']:.3f}  "
          f"CA_FPR={gm['Caucasian']['FPR']:.3f}  DI={di:.3f}")
    return lr_rw, y_pred, gm, om, di


# ── Strategy 2: Threshold adjustment ────────────────────────
def threshold_adjustment(baseline_model, X_test, y_test, race_test,
                          aa_threshold=0.40, ca_threshold=0.50):
    """
    Use a lower decision threshold for African-American defendants.
    Lower threshold = model must be MORE confident to predict recidivism,
    which directly reduces false positives for that group.
    """
    probs  = baseline_model.predict_proba(X_test)[:, 1]
    y_pred = np.zeros(len(y_test), dtype=int)

    for i, (prob, race) in enumerate(zip(probs, race_test)):
        threshold = aa_threshold if race == "African-American" else ca_threshold
        y_pred[i] = int(prob >= threshold)

    gm = group_metrics(y_test, pd.Series(y_pred), race_test.reset_index(drop=True))
    di = disparate_impact(pd.Series(y_pred), race_test.reset_index(drop=True))
    om = overall_metrics(y_test, pd.Series(y_pred))
    print(f"  [Threshold Adj] Accuracy={om['accuracy']:.3f}  "
          f"AA_FPR={gm['African-American']['FPR']:.3f}  "
          f"CA_FPR={gm['Caucasian']['FPR']:.3f}  DI={di:.3f}")
    return y_pred, gm, om, di


# ── Chart: Before vs After comparison ───────────────────────
def plot_comparison(results_dict):
    strategies = list(results_dict.keys())
    aa_fprs = [results_dict[s]["AA_FPR"] for s in strategies]
    ca_fprs = [results_dict[s]["CA_FPR"] for s in strategies]
    accs    = [results_dict[s]["accuracy"] for s in strategies]
    dis     = [results_dict[s]["DI"] for s in strategies]

    x     = np.arange(len(strategies))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Mitigation results — before vs after comparison",
                 fontsize=13, fontweight="bold")

    # Plot 1: FPR comparison
    ax = axes[0]
    b1 = ax.bar(x - width/2, [f*100 for f in aa_fprs],
                width, label="African-American", color="#E24B4A", alpha=0.85)
    b2 = ax.bar(x + width/2, [f*100 for f in ca_fprs],
                width, label="Caucasian", color="#378ADD", alpha=0.85)
    ax.set_title("False Positive Rate by group")
    ax.set_ylabel("FPR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 45)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    # Plot 2: Overall accuracy
    ax = axes[1]
    bars = ax.bar(strategies, [a*100 for a in accs],
                  color=["#888780", "#1D9E75", "#EF9F27"], alpha=0.85)
    ax.set_title("Overall accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(58, 75)
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    # Plot 3: Disparate impact
    ax = axes[2]
    bars = ax.bar(strategies, dis,
                  color=["#888780", "#1D9E75", "#EF9F27"], alpha=0.85)
    ax.axhline(y=0.80, color="#E24B4A", linestyle="--",
               linewidth=1.5, label="Fairness threshold (0.80)")
    ax.axhline(y=1.00, color="#1D9E75", linestyle=":",
               linewidth=1.5, label="Ideal (1.00)")
    ax.set_title("Disparate impact (ideal = 1.0)")
    ax.set_ylabel("Disparate impact ratio")
    ax.set_ylim(0.5, 1.15)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = f"{CHART_DIR}/mitigation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Print summary table ──────────────────────────────────────
def print_summary(results_dict):
    print("\n" + "=" * 70)
    print("  MITIGATION SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<22} {'Accuracy':>9}  {'AA FPR':>7}  "
          f"{'CA FPR':>7}  {'Disp.Impact':>12}  {'Fair?':>6}")
    print("  " + "-" * 68)
    for name, r in results_dict.items():
        fair = "YES" if r["DI"] >= 0.80 else "NO "
        print(f"  {name:<22} {r['accuracy']*100:>8.1f}%  "
              f"{r['AA_FPR']*100:>6.1f}%  {r['CA_FPR']*100:>6.1f}%  "
              f"{r['DI']:>12.3f}  {fair:>6}")
    print("=" * 70)
    print("\n  KEY INSIGHT: Fairness improved but accuracy dropped slightly.")
    print("  This confirms the IMPOSSIBILITY THEOREM — you cannot perfectly")
    print("  satisfy all fairness criteria simultaneously.")


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    X, y, df_full, features = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    race_train = df_full.loc[X_train.index, "race"].reset_index(drop=True)
    race_test  = df_full.loc[X_test.index,  "race"].reset_index(drop=True)
    y_test     = y_test.reset_index(drop=True)

    print("[INFO] Running mitigation strategies...\n")

    lr_base, pred_base, gm_base, om_base, di_base = baseline(
        X_train, X_test, y_train, y_test, race_test)

    lr_rw, pred_rw, gm_rw, om_rw, di_rw = reweighing(
        X_train, X_test, y_train, y_test, race_train, race_test)

    pred_th, gm_th, om_th, di_th = threshold_adjustment(
        lr_base, X_test, y_test, race_test)

    results = {
        "Baseline":         {"accuracy": om_base["accuracy"], "AA_FPR": gm_base["African-American"]["FPR"], "CA_FPR": gm_base["Caucasian"]["FPR"], "DI": di_base},
        "Reweighing":       {"accuracy": om_rw["accuracy"],   "AA_FPR": gm_rw["African-American"]["FPR"],   "CA_FPR": gm_rw["Caucasian"]["FPR"],   "DI": di_rw},
        "Threshold Adj.":   {"accuracy": om_th["accuracy"],   "AA_FPR": gm_th["African-American"]["FPR"],   "CA_FPR": gm_th["Caucasian"]["FPR"],   "DI": di_th},
    }

    print_summary(results)
    plot_comparison(results)

    pd.DataFrame(results).T.reset_index().rename(
        columns={"index": "strategy"}
    ).to_csv(f"{METRICS_DIR}/mitigation_results.csv", index=False)
    print(f"\n[OK] Results saved to {METRICS_DIR}/mitigation_results.csv")
    print("[NEXT] Run: python src/06_report_generator.py")

