#!/usr/bin/env python
"""Reproduce: trigonometric degeneracy.

Ground truth: 2*sin(x) + cos(y) - z = 1

Outputs to outputs/reproduce/trig/ (result.pkl, summary.txt, diagnostics/).
"""
from degen_detector import run_pipeline
from degen_detector.testing import generate_trig_separable

samples, param_names, gt = generate_trig_separable()
print(f"Ground truth: {gt['equation']}")
run_pipeline(
    samples,
    param_names,
    output_dir="outputs/reproduce/trig",
    coupling_depth=3,
    niterations=200,
    max_fits=1,
)
