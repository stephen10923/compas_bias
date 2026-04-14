"""
COMPAS Bias Audit — Run All Phases
===================================
Runs all 6 phases of the project in sequence.
Usage: python src/run_all.py
"""

import subprocess, sys, time

phases = [
    ("Phase 1: Data Loading",    "src/01_data_loading.py"),
    ("Phase 2: EDA",             "src/02_eda.py"),
    ("Phase 3: Model Training",  "src/03_model_training.py"),
    ("Phase 4: Bias Detection",  "src/04_bias_detection.py"),
    ("Phase 5: Mitigation",      "src/05_mitigation.py"),
    ("Phase 6: Report",          "src/06_report_generator.py"),
]

print("\n" + "=" * 55)
print("  COMPAS BIAS AUDIT — FULL PIPELINE")
print("  [Unbiased AI Decision] Challenge")
print("=" * 55 + "\n")

total_start = time.time()

for name, script in phases:
    print(f"\n{'─'*55}")
    print(f"  Running: {name}")
    print(f"{'─'*55}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed. Stopping pipeline.")
        sys.exit(1)
    print(f"  Completed in {elapsed:.1f}s")

total = time.time() - total_start
print(f"\n{'='*55}")
print(f"  ALL PHASES COMPLETE in {total:.0f}s")
print(f"{'='*55}")
print("\n  Outputs:")
print("  - output/charts/       All visualisation charts")
print("  - output/metrics/      CSV metrics files")
print("  - report/audit_report.md  Full audit report")
print()

