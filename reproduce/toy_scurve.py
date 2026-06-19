#!/usr/bin/env python
"""Reproduce: S-curve degeneracy.

Ground truth: (x^3 - 3x) + y + z = 0

Outputs to outputs/reproduce/scurve/ (result.pkl, summary.txt, diagnostics/).
"""
from degen_detector import run_pipeline
from degen_detector.testing import generate_scurve_separable

samples, param_names, gt = generate_scurve_separable()
print(f"Ground truth: {gt['equation']}")
run_pipeline(
    samples,
    param_names,
    output_dir="outputs/reproduce/scurve",
    coupling_depth=3,
    niterations=200,
    max_fits=1,
)
