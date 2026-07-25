#!/usr/bin/env python3
"""
Verify App E synthetic-pipeline numbers against the actual `compute_robustness_suite.py`
output. Run AFTER `compute_robustness_suite.py` has produced its JSON/CSV outputs.

Paper App E reports the following numbers that this script checks:

  A1_minimal (item-level synthetic, items_per_subscale × alpha grid):
    Mean recovered slope across 9 × 20 = 180 runs:  1.289 ± 0.096
    Per-cell range:                                 1.17 to 1.43
    Specific cells (paper Table):
      items=3, alpha=0.60: 1.430 ± 0.007
      items=3, alpha=0.75: 1.394 ± 0.005
      items=3, alpha=0.85: 1.196 ± 0.006
      items=5, alpha=0.60: 1.343 ± 0.006
      items=5, alpha=0.75: 1.259 ± 0.003
      items=5, alpha=0.85: 1.231 ± 0.002
      items=7, alpha=0.60: 1.380 ± 0.007
      items=7, alpha=0.75: 1.193 ± 0.003
      items=7, alpha=0.85: 1.172 ± 0.002

  A1_full (high-dim SNR sweep, d × SNR grid):
    SNR=0.1, d=896:  1.39      SNR=0.1, d=2048: 1.48
    SNR=0.1, d=4096: 1.54      SNR=0.1, d=5120: 1.56
    SNR=0.3, d=896:  1.32      ...
    SNR=1.0, d=896:  1.22      ...
    SNR=3.0, d=896:  1.16      ...
    SNR=10,  d=896:  1.11      ...
    Mantel r recovered-vs-ground-truth: >= 0.97 throughout

This script:
  1. Auto-locates the A1 outputs under common paths
  2. Loads JSON / CSV
  3. Compares cell by cell against the paper claims
  4. Prints any cell that differs by >0.02 (slope) or >0.01 (Mantel)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from psychometric_inference.paths import BEHAVIOR_OUTPUT_DIR, PROJECT_ROOT, ROBUSTNESS_OUTPUT_DIR

# Paper-claimed values (from App E tables)
PAPER_A1_MINIMAL = {
    # (items_per_subscale, alpha) -> (mean_slope, std)
    (3, 0.60): (1.430, 0.007),
    (3, 0.75): (1.394, 0.005),
    (3, 0.85): (1.196, 0.006),
    (5, 0.60): (1.343, 0.006),
    (5, 0.75): (1.259, 0.003),
    (5, 0.85): (1.231, 0.002),
    (7, 0.60): (1.380, 0.007),
    (7, 0.75): (1.193, 0.003),
    (7, 0.85): (1.172, 0.002),
}
PAPER_A1_MINIMAL_OVERALL_MEAN = 1.289
PAPER_A1_MINIMAL_OVERALL_STD = 0.096

PAPER_A1_FULL = {
    # (snr, d) -> mean_slope
    (0.1,  896):  1.39, (0.1,  2048): 1.48, (0.1,  4096): 1.54, (0.1,  5120): 1.56,
    (0.3,  896):  1.32, (0.3,  2048): 1.30, (0.3,  4096): 1.31, (0.3,  5120): 1.31,
    (1.0,  896):  1.22, (1.0,  2048): 1.17, (1.0,  4096): 1.14, (1.0,  5120): 1.14,
    (3.0,  896):  1.16, (3.0,  2048): 1.10, (3.0,  4096): 1.08, (3.0,  5120): 1.07,
    (10.0, 896):  1.11, (10.0, 2048): 1.07, (10.0, 4096): 1.04, (10.0, 5120): 1.03,
}

SLOPE_TOL = 0.02
MANTEL_TOL = 0.01


def find_outputs() -> dict:
    """Find A1 output files under common locations."""
    candidates = [
        ROBUSTNESS_OUTPUT_DIR,
        BEHAVIOR_OUTPUT_DIR,
        PROJECT_ROOT / "results" / "extra",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "extra_results",
        PROJECT_ROOT,
    ]
    found = {}
    targets = [
        "A1_minimal_summary.json",
        "A1_minimal_by_grid.csv",
        "A1_full_summary.json",
        "A1_full_by_grid.csv",
    ]
    for d in candidates:
        if not d.exists():
            continue
        for t in targets:
            for p in d.rglob(t):
                if t not in found:
                    found[t] = p
    return found


def check_a1_minimal(by_grid_csv: Path) -> None:
    print("\n" + "=" * 70)
    print("  A1_minimal (item-level synthetic): per-cell verification")
    print("=" * 70)
    df = pd.read_csv(by_grid_csv)

    # Detect columns flexibly (script may name them differently)
    items_col = next((c for c in df.columns
                      if c.lower() in ("items_per_subscale", "items", "n_items", "k")), None)
    alpha_col = next((c for c in df.columns
                      if "alpha" in c.lower() and "target" in c.lower()), None)
    if alpha_col is None:
        alpha_col = next((c for c in df.columns
                          if c.lower() in ("alpha", "target_alpha")), None)
    slope_col = next((c for c in df.columns
                      if "slope" in c.lower() and "mean" in c.lower()), None)
    if slope_col is None:
        slope_col = next((c for c in df.columns if c.lower() in ("slope_mean", "slope")), None)
    std_col = next((c for c in df.columns
                    if "slope" in c.lower() and "std" in c.lower()), None)

    if not (items_col and alpha_col and slope_col):
        print(f"  Cannot detect needed columns from {list(df.columns)}")
        return

    print(f"  {'cell (items, alpha)':<22} {'paper':>14} {'actual':>16} {'diff':>8}")
    print("  " + "-" * 64)
    drift = []
    for (k, a), (paper_mean, paper_std) in PAPER_A1_MINIMAL.items():
        sub = df[(df[items_col] == k) & (np.isclose(df[alpha_col], a, atol=1e-4))]
        if not len(sub):
            print(f"  ({k}, {a:.2f})            -- not in CSV --")
            continue
        actual_mean = float(sub[slope_col].iloc[0])
        actual_std = float(sub[std_col].iloc[0]) if std_col else float("nan")
        d = actual_mean - paper_mean
        flag = "  ⚠️" if abs(d) > SLOPE_TOL else ""
        if std_col:
            print(f"  ({k}, {a:.2f}):  paper {paper_mean:.3f}±{paper_std:.3f}  "
                  f"actual {actual_mean:.3f}±{actual_std:.3f}  diff {d:+.3f}{flag}")
        else:
            print(f"  ({k}, {a:.2f}):  paper {paper_mean:.3f}  "
                  f"actual {actual_mean:.3f}  diff {d:+.3f}{flag}")
        if abs(d) > SLOPE_TOL:
            drift.append((k, a, paper_mean, actual_mean, d))

    # Overall summary stats
    overall_mean = float(df[slope_col].mean())
    print(f"\n  Overall mean slope across {len(df)} cells: paper {PAPER_A1_MINIMAL_OVERALL_MEAN:.3f}, "
          f"actual {overall_mean:.3f}, diff {overall_mean - PAPER_A1_MINIMAL_OVERALL_MEAN:+.3f}")

    if drift:
        print(f"\n  ⚠️  {len(drift)} cell(s) drifted >{SLOPE_TOL} from paper claims.")
    else:
        print(f"\n  ✓ All cells within {SLOPE_TOL} of paper claims.")


def check_a1_full(by_grid_csv: Path) -> None:
    print("\n" + "=" * 70)
    print("  A1_full (high-dim SNR sweep): per-cell verification")
    print("=" * 70)
    df = pd.read_csv(by_grid_csv)

    snr_col = next((c for c in df.columns if "snr" in c.lower()), None)
    d_col = next((c for c in df.columns
                  if c.lower() in ("d", "hidden_dim", "dim", "d_hidden")), None)
    slope_col = next((c for c in df.columns
                      if "slope" in c.lower() and "mean" in c.lower()), None)
    if slope_col is None:
        slope_col = next((c for c in df.columns if c.lower() in ("slope_mean", "slope")), None)
    mantel_col = next((c for c in df.columns
                       if "mantel" in c.lower() and "mean" in c.lower()), None)
    if mantel_col is None:
        mantel_col = next((c for c in df.columns if "mantel" in c.lower()), None)

    if not (snr_col and d_col and slope_col):
        print(f"  Cannot detect needed columns from {list(df.columns)}")
        return

    print(f"  {'(SNR, d)':<14} {'paper':>8} {'actual':>10} {'diff':>8}  {'mantel':>8}")
    print("  " + "-" * 50)
    drift = []
    bad_mantel = []
    for (snr, d), paper_slope in PAPER_A1_FULL.items():
        sub = df[(np.isclose(df[snr_col], snr, atol=1e-4)) &
                 (df[d_col] == d)]
        if not len(sub):
            print(f"  ({snr}, {d}):       -- not in CSV --")
            continue
        actual = float(sub[slope_col].iloc[0])
        diff = actual - paper_slope
        flag = "  ⚠️" if abs(diff) > SLOPE_TOL else ""
        mantel = float(sub[mantel_col].iloc[0]) if mantel_col else float("nan")
        m_str = f"{mantel:.3f}" if mantel_col else "  --  "
        m_flag = "  ⚠️ <0.97" if (mantel_col and mantel < 0.97 - MANTEL_TOL) else ""
        print(f"  ({snr:>4}, {d:>4}):  {paper_slope:>6.3f}  {actual:>8.3f}  "
              f"{diff:>+6.3f}{flag}  {m_str}{m_flag}")
        if abs(diff) > SLOPE_TOL:
            drift.append((snr, d, paper_slope, actual, diff))
        if mantel_col and mantel < 0.97 - MANTEL_TOL:
            bad_mantel.append((snr, d, mantel))

    if drift:
        print(f"\n  ⚠️  {len(drift)} slope cell(s) drifted >{SLOPE_TOL}.")
    else:
        print(f"\n  ✓ All slope cells within {SLOPE_TOL} of paper claims.")
    if bad_mantel:
        print(f"  ⚠️  {len(bad_mantel)} cell(s) have Mantel r < {0.97 - MANTEL_TOL:.2f} "
              f"(paper claims ≥0.97 throughout).")


def main():
    print("=" * 70)
    print("  SYNTHETIC PIPELINE VERIFICATION (App E)")
    print("=" * 70)

    found = find_outputs()
    if not found:
        print("\nERROR: Could not find A1 output files. Searched standard locations.")
        print("       Run compute_robustness_suite.py first, then point this script to the")
        print("       correct directory.")
        sys.exit(1)

    print("\nFound:")
    for k, v in found.items():
        print(f"  {k}: {v}")

    if "A1_minimal_by_grid.csv" in found:
        check_a1_minimal(found["A1_minimal_by_grid.csv"])
    else:
        print("\n  (A1_minimal_by_grid.csv not found, skipping minimal check)")

    if "A1_full_by_grid.csv" in found:
        check_a1_full(found["A1_full_by_grid.csv"])
    else:
        print("\n  (A1_full_by_grid.csv not found, skipping full check)")

    print("\nDone.")


if __name__ == "__main__":
    main()
