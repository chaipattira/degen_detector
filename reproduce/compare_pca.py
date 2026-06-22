#!/usr/bin/env python
"""PCA baseline comparison across all toy cases.

For each case, regenerates samples from testing.py generators (same fixed seed
that was used to produce the saved outputs), runs PCA, then loads degen_detector
results from the saved pkl for comparison.

Comparison metric: residual_std in Z-scored space
  PCA:  std along the last (most constrained) eigenvector after StandardScaler
  DD:   orthogonal residual std from the best ImplicitFit (also Z-scored)

Run from the degen_detector repo root:
    python reproduce/compare_pca.py
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from degen_detector.testing import (
    generate_banana_degeneracy,
    generate_cubic_degeneracy,
    generate_pixel_mass,
    generate_scurve_separable,
    generate_trig_separable,
)

CASES = [
    ("banana",     generate_banana_degeneracy, {}),
    ("cubic",      generate_cubic_degeneracy,  {}),
    ("trig",       generate_trig_separable,    {}),
    ("scurve",     generate_scurve_separable,  {}),
    ("pixel_mass", generate_pixel_mass,        {}),
]
OUTPUT_ROOT = Path("outputs/reproduce")
N_TOP = 3  # top-loading params to show for PCA


def pca_metrics(samples, param_names):
    """PCA on Z-scored samples; report metrics for most-constrained direction."""
    scaled = StandardScaler().fit_transform(samples)
    pca = PCA().fit(scaled)
    last_vec = pca.components_[-1]
    resid_std = float(np.sqrt(pca.explained_variance_[-1]))
    order = np.argsort(np.abs(last_vec))[::-1]
    top_params = [f"{param_names[i]}({last_vec[i]:+.2f})" for i in order[:N_TOP]]
    return resid_std, top_params


def dd_metrics(result):
    """Extract residual_std, R²_ortho, and detected params from CouplingSearchResult."""
    best = result.fits[0]
    fit = best.fits[0]
    return float(fit.residual_std), float(fit.orthogonal_r2), best.param_names


def run():
    rows = []
    for case_name, generator, kwargs in CASES:
        samples, param_names, gt = generator(**kwargs)

        pkl_path = OUTPUT_ROOT / case_name / "result.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        dd_std, dd_r2, dd_params = dd_metrics(data["result"])

        pca_std, pca_top = pca_metrics(samples, param_names)
        rows.append((case_name, gt, pca_std, pca_top, dd_std, dd_r2, dd_params))

    # ── Comparison table ──────────────────────────────────────────────────
    W = 110
    print("\n" + "=" * W)
    print(f"{'Case':<12} {'PCA resid_std':>14} {'DD resid_std':>13} {'DD R²':>7}  "
          f"{'PCA top params (loading)':<36}  DD params")
    print("=" * W)
    for case_name, gt, pca_std, pca_top, dd_std, dd_r2, dd_params in rows:
        print(f"{case_name:<12} {pca_std:>14.4f} {dd_std:>13.4f} {dd_r2:>7.4f}  "
              f"{', '.join(pca_top):<36}  {dd_params}")
    print("=" * W)

    print("\nGround truth:")
    for case_name, gt, *_ in rows:
        print(f"  {case_name:<12}: {gt['equation']}")
        print(f"               degen params: {gt['degenerate_params']}")
    print()


if __name__ == "__main__":
    run()
