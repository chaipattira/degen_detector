#!/usr/bin/env python
"""Scan MI matrices across Planck+external dataset combinations.

Tests whether the degeneracy ranking is stable as observations are stacked.
Loads whatever params each chain has — no fixed param list.

Usage:
    python reproduce/science/planck/mi_scan.py
    python reproduce/science/planck/mi_scan.py --top-k 8
"""
import argparse
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/base/plikHM_TTTEEE_lowl_lowE_lensing"
STEM = DATA_DIR / "base_plikHM_TTTEEE_lowl_lowE_lensing"

OBSERVATIONS = [
    ("Planck",        STEM),
    ("+BAO",          STEM.parent / (STEM.name + "_post_BAO")),
    ("+Riess18",      STEM.parent / (STEM.name + "_post_Riess18")),
    ("+Pantheon18",   STEM.parent / (STEM.name + "_post_Pantheon18")),
    ("+BAO+Riess18",  STEM.parent / (STEM.name + "_post_BAO_Riess18")),
    ("+BAO+Pan18",    STEM.parent / (STEM.name + "_post_BAO_Pantheon18")),
    ("+BAO+JLA+R18",  STEM.parent / (STEM.name + "_post_BAO_JLA_Riess18")),
    ("+BAO+Pan+R18",  STEM.parent / (STEM.name + "_post_BAO_Pantheon18_Riess18")),
]

COSMO_PARAMS = [
    "omegabh2", "omegach2", "theta", "tau", "logA", "ns",
    "H0", "omegam", "sigma8",
]


def _load_chain(stem, ignore_rows=0.3):
    """Read CosmoMC chain txt files directly, returning (samples, param_names).

    Column layout: col0=weight, col1=chi2, col2..N=params in .paramnames order.
    Post-processed chains append extra chi2 columns at the end; we ignore those
    by indexing only up to len(param_names).
    """
    import numpy as np

    pfile = Path(str(stem) + ".paramnames")
    all_names = [line.strip().split()[0].rstrip("*")
                 for line in pfile.read_text().splitlines() if line.strip()]

    wanted = [p for p in COSMO_PARAMS if p in all_names]
    # +2 because col0=weight, col1=chi2
    col_indices = [all_names.index(p) + 2 for p in wanted]

    chunks = []
    for txt in sorted(Path(str(stem) + "_1.txt").parent.glob(Path(str(stem)).name + "_[0-9]*.txt")):
        data = np.loadtxt(txt)
        if data.ndim == 1:
            data = data[None, :]
        chunks.append(data)

    all_data = np.vstack(chunks)
    weights = all_data[:, 0]
    burn = int(len(all_data) * ignore_rows)
    all_data, weights = all_data[burn:], weights[burn:]

    # expand weighted rows
    counts = np.round(weights / weights.min()).astype(int)
    rows = np.repeat(all_data, counts, axis=0)
    samples = rows[:, col_indices]
    return samples, wanted


def main():
    parser = argparse.ArgumentParser(description="MI scan across Planck observations")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k pairs to show per obs")
    args = parser.parse_args()

    from degen_detector.analysis import mutual_info_matrix

    print(f"Top-{args.top_k} MI pairs per observation\n" + "=" * 60)
    for label, stem in OBSERVATIONS:
        print(f"\nLoading {label} ...", end=" ", flush=True)
        try:
            samples, param_names = _load_chain(stem)
            mi = mutual_info_matrix(samples, param_names)
            print(f"({samples.shape[0]:,} samples, {len(param_names)} params)")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        n = len(param_names)
        pairs = sorted(
            [(mi.mi_matrix[i, j], param_names[i], param_names[j])
             for i, j in itertools.combinations(range(n), 2)],
            reverse=True,
        )
        for mi_val, p1, p2 in pairs[: args.top_k]:
            print(f"  {p1:14s} <-> {p2:14s}  MI = {mi_val:.4f}")



if __name__ == "__main__":
    main()
