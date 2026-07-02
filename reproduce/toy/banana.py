#!/usr/bin/env python
"""Reproduce: banana degeneracy (elliptic paraboloid).

Ground truth: 2*theta1^2 + theta2^2 - theta3 = 0.5

Outputs to outputs/reproduce/banana/ (result.pkl, summary.txt, diagnostics/).
"""
from degen_detector import run_detector
from degen_detector.testing import generate_banana_degeneracy

samples, param_names, gt = generate_banana_degeneracy()
print(f"Ground truth: {gt['equation']}")
run_detector(
    samples,
    param_names,
    output_dir="outputs/reproduce/banana",
    coupling_depth=3,
    niterations=200,
    max_fits=1,
)
