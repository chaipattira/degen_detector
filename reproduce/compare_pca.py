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


def pca_metrics(samples, param_names, n_top):
    """PCA on Z-scored samples; report metrics for most-constrained direction.

    R²_ortho uses the same null model as DD (L=1 in Z-scored space).
    For a linear constraint w·x=c with unit-norm w: L = var(w·x) = last eigenvalue,
    so R²_ortho = 1 - last_eigenvalue.
    """
    scaled = StandardScaler().fit_transform(samples)
    pca = PCA().fit(scaled)
    last_vec = pca.components_[-1]
    last_var = float(pca.explained_variance_[-1])
    resid_std = float(np.sqrt(last_var))
    r2_ortho = float(1.0 - last_var)
    order = np.argsort(np.abs(last_vec))[::-1]
    top_params = [f"{param_names[i]}({last_vec[i]:+.2f})" for i in order[:n_top]]
    return resid_std, r2_ortho, top_params


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

        n_top = len(gt["degenerate_params"])
        pca_std, pca_r2, pca_top = pca_metrics(samples, param_names, n_top)
        rows.append((case_name, gt, pca_std, pca_r2, pca_top, dd_std, dd_r2, dd_params))

    # ── Table 1: numeric metrics ──────────────────────────────────────────
    W1 = 62
    print("\n" + "=" * W1)
    print(f"{'Case':<12} {'PCA resid_std':>14} {'PCA R²':>8} {'DD resid_std':>13} {'DD R²':>8}")
    print("-" * W1)
    for case_name, gt, pca_std, pca_r2, pca_top, dd_std, dd_r2, dd_params in rows:
        print(f"{case_name:<12} {pca_std:>14.4f} {pca_r2:>8.4f} {dd_std:>13.4f} {dd_r2:>8.4f}")
    print("=" * W1)

    # ── Table 2: parameter identification ────────────────────────────────
    W2 = 116
    print("\n" + "=" * W2)
    print(f"{'Case':<12}        {'Ground truth':<36}        {'PCA detected':<36}        DD detected")
    print("-" * W2)
    for case_name, gt, pca_std, pca_r2, pca_top, dd_std, dd_r2, dd_params in rows:
        gt_params = ", ".join(gt["degenerate_params"])
        pca_names = ", ".join(p.split("(")[0] for p in pca_top)
        dd_names  = ", ".join(dd_params)
        print(f"{case_name:<12}        {gt_params:<36}        {pca_names:<36}        {dd_names}")
    print("=" * W2)

    # ── Table 3: ground truth equations ──────────────────────────────────
    print("\nGround truth equations:")
    for case_name, gt, *_ in rows:
        print(f"  {case_name:<12}  {gt['equation']}")
    print()


if __name__ == "__main__":
    run()
