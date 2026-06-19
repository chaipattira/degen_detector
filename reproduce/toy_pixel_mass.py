#!/usr/bin/env python
"""Reproduce: astro pixel-mass experiment (mass conservation degeneracy).

Physical setup: 3×3 grid (9 pixels). Total mass C = Σmᵢ is observed with
noise σ=0.1; individual pixel masses are unresolvable. Posterior samples
satisfy mass conservation: m1 + m2 + ... + m9 ≈ M_obs.

Prior:       P(mᵢ) ~ N(0, 1)     (independent per pixel)
Likelihood:  P(M | {mᵢ}) ~ N(Σmᵢ, σ²)
Posterior:   analytical Gaussian (exact equivalent of converged MCMC)

Ground truth (separable implicit surface with linear components):
    g₁(m₁) + g₂(m₂) + ... + g₉(m₉) = 2.0,   gᵢ(x) = x

Using coupling_depth=9 so degen_detector searches the one 9-tuple (all
pixels simultaneously) and recovers the full mass-conservation constraint.

Outputs to outputs/reproduce/pixel_mass/ (result.pkl, summary.txt, diagnostics/).
"""
from degen_detector import run_pipeline
from degen_detector.testing import generate_pixel_mass

samples, param_names, gt = generate_pixel_mass(grid_size=3, sigma=0.1, M_obs=2.0)
print(f"Ground truth: {gt['equation']}")
print(f"n_pixels={len(param_names)}, sigma={gt['sigma']}, M_obs={gt['M_obs']}")
run_pipeline(
    samples,
    param_names,
    output_dir="outputs/reproduce/pixel_mass",
    coupling_depth=9,
    niterations=100,
    max_fits=1,
)
