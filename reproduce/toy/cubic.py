#!/usr/bin/env python
"""Reproduce: cubic degeneracy.

Ground truth: theta1 - (10*theta2)^3 ≈ 0

Outputs to outputs/reproduce/cubic/ (result.pkl, summary.txt, diagnostics/).
"""
from degen_detector import run_detector
from degen_detector.testing import generate_cubic_degeneracy

samples, param_names, gt = generate_cubic_degeneracy()
print(f"Ground truth: {gt['equation']}")
run_detector(
    samples,
    param_names,
    output_dir="outputs/reproduce/cubic",
    coupling_depth=2,
    niterations=200,
    max_fits=1,
)
