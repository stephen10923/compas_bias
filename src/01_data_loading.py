"""
COMPAS Bias Audit — Phase 1: Data Loading & Cleaning
=====================================================
Challenge: [Unbiased AI Decision] Ensuring Fairness and
           Detecting Bias in Automated Decisions

This script:
  - Downloads the ProPublica COMPAS dataset
  - Applies the same filters ProPublica used in their 2016 investigation
  - Saves the cleaned dataset for subsequent phases
"""

import pandas as pd
import numpy as np
import os

# ─── Config ───────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/propublica/"
    "compas-analysis/master/compas-scores-two-years.csv"
)
RAW_PATH    = "data/compas-scores-two-years.csv"
CLEAN_PATH  = "data/compas_clean.csv"
OUTPUT_DIR  = "output"

os.makedirs("data",   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Step 1: Download ─────────────────────────────────────────
def download_dataset():
    if os.path.exists(RAW_PATH):
        print(f"[INFO] Dataset already at {RAW_PATH}")
        return

    print("[INFO] Downloading ProPublica COMPAS dataset...")
    try:
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        print(f"[OK]   Saved to {RAW_PATH}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[HELP] Download manually from:")
        print("       https://github.com/propublica/compas-analysis")
        raise


# ─── Step 2: Load & Filter ────────────────────────────────────
def load_and_clean():
    print("\n[INFO] Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)
    print(f"[OK]   Raw shape: {df.shape}")   # ~7,214 rows

    # Select relevant columns only
    cols = [
        "id", "age", "c_charge_degree", "race", "age_cat",
        "score_text", "sex", "priors_count",
        "days_b_screening_arrest", "decile_score",
        "is_recid", "two_year_recid"
    ]
    df = df[cols]

    # Apply ProPublica's exact filters
    # (same as their R analysis)
    print("[INFO] Applying ProPublica filters...")
    n_before = len(df)

    df = df[
        (df["days_b_screening_arrest"] <= 30)  &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1)                 &
        (df["c_charge_degree"] != "O")         &
        (df["score_text"] != "N/A")
    ].copy()

    n_after = len(df)
    print(f"[OK]   Rows kept: {n_after} / {n_before} "
          f"({n_before - n_after} removed by filters)")

    # Encode binary label clearly
    df["label"] = df["two_year_recid"].astype(int)

    return df


# ─── Step 3: Summarise ────────────────────────────────────────
def summarise(df):
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)

    print(f"\n  Total defendants:     {len(df):,}")
    print(f"  Recidivism rate:      {df['label'].mean()*100:.1f}%")
    print(f"  Features:             {df.shape[1]}")

    print("\n  Racial breakdown:")
    race_counts = df["race"].value_counts()
    for race, count in race_counts.items():
        pct = count / len(df) * 100
        print(f"    {race:<22} {count:>5,}  ({pct:.1f}%)")

    print("\n  COMPAS score labels:")
    label_counts = df["score_text"].value_counts()
    for label, count in label_counts.items():
        print(f"    {label:<12} {count:>5,}")

    print("\n  Recidivism by race:")
    recid = df.groupby("race")["label"].mean() * 100
    for race, rate in recid.sort_values(ascending=False).items():
        print(f"    {race:<22} {rate:.1f}%")

    print("=" * 50)


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
    df = load_and_clean()
    summarise(df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\n[OK] Cleaned dataset saved to {CLEAN_PATH}")
    print("[NEXT] Run: python src/02_eda.py")

