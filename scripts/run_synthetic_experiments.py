#!/usr/bin/env python
# ABOUTME: Runs the synthetic benchmark cases and saves results as pickle files.
# ABOUTME: Uses SYNTHETIC_CASES registry from degen_detector.testing.
"""Run synthetic degeneracy benchmark experiments and save results for plotting.

Cases (defined in degen_detector.testing.SYNTHETIC_CASES):
- banana : 2*theta1^2 + theta2^2 - theta3 = 0.5
- cubic  : theta1 ≈ (10*theta2)^3
- trig   : 2*sin(x) + cos(y) - z = 1
- scurve : (x^3 - 3x) + y + z = 0

Usage:
    python scripts/run_synthetic_experiments.py
    python scripts/run_synthetic_experiments.py --experiments banana cubic
    python scripts/run_synthetic_experiments.py --max-fits 3
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from degen_detector import DegenDetector
from degen_detector.diagnostics import DiagnosticsRunner
from degen_detector.io import save_pickle, create_output_dir
from degen_detector.testing import SYNTHETIC_CASES


def run_experiment(case, max_fits=1):
    """Run one benchmark case and return a result dict compatible with DiagnosticsRunner."""
    samples, param_names, ground_truth = case["generator"]()
    print(f"\n{'='*60}")
    print(f"{case['label']}  |  {ground_truth['equation']}")
    print(f"{'='*60}")

    result = DegenDetector(samples, param_names).search_couplings(
        coupling_depth=case["coupling_depth"],
        niterations=case["niterations"],
        max_fits=max_fits,
    )

    top_fit = next((cf for cf in result.fits if cf.fit is not None), None)
    if top_fit:
        print(f"Found: {top_fit.fit.equation_str}  (R²={top_fit.fit.orthogonal_r2:.4f})")
    else:
        print("No fit found.")

    return {
        "name": case["name"],
        "ground_truth": ground_truth,
        "samples": samples,
        "param_names": param_names,
        "result": result,
        "top_fit": top_fit,
    }


def main():
    parser = argparse.ArgumentParser(description="Run synthetic degeneracy benchmark")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/synthetic"),
        help="Base output directory (default: outputs/synthetic)",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=[c["name"] for c in SYNTHETIC_CASES],
        help="Which cases to run (default: all)",
    )
    parser.add_argument(
        "--max-fits", type=int, default=1,
        help="Max MI-ranked tuples to fit per experiment (default: 1)",
    )
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    cases = SYNTHETIC_CASES
    if args.experiments:
        cases = [c for c in SYNTHETIC_CASES if c["name"] in args.experiments]

    all_results = {}
    for case in cases:
        result = run_experiment(case, max_fits=args.max_fits)
        all_results[case["name"]] = result
        save_pickle(result, output_dir / f"{case['name']}_result.pkl")
        print(f"Saved: {case['name']}_result.pkl")

    combined_file = output_dir / "all_results.pkl"
    save_pickle(all_results, combined_file)

    print(f"\n{'='*60}")
    print(f"{'Case':<12} {'Ground Truth':<40} {'R²':>8}")
    print(f"{'='*60}")
    for name, r in all_results.items():
        gt_eq = r['ground_truth']['equation']
        r2 = r['top_fit'].fit.orthogonal_r2 if r['top_fit'] else float('nan')
        print(f"{name:<12} {gt_eq:<40} {r2:>8.4f}")

    try:
        runner = DiagnosticsRunner(combined_file)
        runner.run(output_dir=output_dir / "diagnostics")
    except Exception as e:
        print(f"\nDiagnostics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
