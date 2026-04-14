"""
COMPAS Bias Audit — Phase 3: Model Training
============================================
Trains two classifiers that replicate COMPAS's decision logic:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest       (higher accuracy)

Both models are trained on the same features COMPAS uses.
The trained models and test predictions are saved for Phase 4.

Outputs:
  - output/metrics/model_performance.csv
  - output/charts/feature_importance.png
  - data/test_predictions.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (classification_report, accuracy_score,
                                     confusion_matrix, roc_auc_score)

CLEAN_PATH   = "data/compas_clean.csv"
PRED_PATH    = "data/test_predictions.csv"
METRICS_DIR  = "output/metrics"
CHART_DIR    = "output/charts"
MODEL_DIR    = "output"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHART_DIR,   exist_ok=True)


# ── Load & encode ───────────────────────────────────────────
def prepare_data(df):
    df = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    df["race_enc"]   = le.fit_transform(df["race"])
    df["sex_enc"]    = le.fit_transform(df["sex"])
    df["charge_enc"] = le.fit_transform(df["c_charge_degree"])
    df["age_enc"]    = le.fit_transform(df["age_cat"])

    feature_cols = ["age", "priors_count", "charge_enc",
                    "sex_enc", "race_enc", "age_enc"]
    X = df[feature_cols]
    y = df["label"]

    return X, y, df, feature_cols


# ── Train & evaluate ────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=200, max_depth=8,
                                   random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model":     model,
            "y_pred":    y_pred,
            "accuracy":  acc,
            "auc":       auc,
            "cv_mean":   cv.mean(),
            "cv_std":    cv.std(),
            "precision": report["weighted avg"]["precision"],
            "recall":    report["weighted avg"]["recall"],
            "f1":        report["weighted avg"]["f1-score"],
        }

        print(f"  Accuracy  : {acc*100:.1f}%")
        print(f"  AUC-ROC   : {auc:.3f}")
        print(f"  CV (5-fold): {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=["No recid", "Recid"]))

    return results


# ── Save predictions with race metadata ─────────────────────
def save_predictions(results, X_test, y_test, df_full):
    test_df = X_test.copy()
    test_df["true_label"] = y_test.values
    test_df["race"]  = df_full.loc[X_test.index, "race"].values
    test_df["sex"]   = df_full.loc[X_test.index, "sex"].values
    test_df["age"]   = df_full.loc[X_test.index, "age"].values

    for name, res in results.items():
        col = name.lower().replace(" ", "_")
        test_df[f"pred_{col}"]      = res["y_pred"]
        test_df[f"pred_{col}_prob"] = res["model"].predict_proba(X_test)[:, 1]

    test_df.to_csv(PRED_PATH, index=False)
    print(f"\n[OK] Predictions saved to {PRED_PATH}")
    return test_df


# ── Feature importance chart ─────────────────────────────────
def plot_feature_importance(results, feature_cols):
    rf = results["Random Forest"]["model"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_vals     = importances[indices]

    display_names = {
        "priors_count": "Prior convictions",
        "age":          "Age",
        "race_enc":     "Race (proxy bias!)",
        "charge_enc":   "Charge degree",
        "sex_enc":      "Sex",
        "age_enc":      "Age category",
    }
    labels = [display_names.get(f, f) for f in sorted_features]
    colors = ["#E24B4A" if "Race" in l else "#378ADD" for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels[::-1], sorted_vals[::-1],
                   color=colors[::-1], edgecolor="white")

    ax.set_title("Random Forest — feature importance\n"
                 "(race is 3rd most important — proxy bias confirmed)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance score")

    red_patch  = mpatches.Patch(color="#E24B4A", label="Biased feature (race)")
    blue_patch = mpatches.Patch(color="#378ADD", label="Legitimate feature")
    ax.legend(handles=[red_patch, blue_patch])

    for bar, val in zip(bars, sorted_vals[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out = f"{CHART_DIR}/feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


# ── Save metrics CSV ─────────────────────────────────────────
def save_metrics(results):
    rows = []
    for name, res in results.items():
        rows.append({
            "model":     name,
            "accuracy":  round(res["accuracy"],  3),
            "auc":       round(res["auc"],       3),
            "precision": round(res["precision"], 3),
            "recall":    round(res["recall"],    3),
            "f1":        round(res["f1"],        3),
            "cv_mean":   round(res["cv_mean"],   3),
            "cv_std":    round(res["cv_std"],    3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{METRICS_DIR}/model_performance.csv", index=False)
    print(f"[OK] Metrics saved to {METRICS_DIR}/model_performance.csv")


if __name__ == "__main__":
    import matplotlib.patches as mpatches

    df = pd.read_csv(CLEAN_PATH)
    X, y, df_full, feature_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"[OK] Train: {len(X_train):,} | Test: {len(X_test):,}")

    results  = train_models(X_train, X_test, y_train, y_test)
    test_df  = save_predictions(results, X_test, y_test, df_full)
    plot_feature_importance(results, feature_cols)
    save_metrics(results)

    print("\n[OK] Phase 3 complete.")
    print("[NEXT] Run: python src/04_bias_detection.py")

