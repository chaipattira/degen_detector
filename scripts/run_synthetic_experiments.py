#!/usr/bin/env python
"""Run synthetic degeneracy experiments and save results for plotting.

Four consolidated experiments test different separable function types:
- exp1: Polynomial (cubic, quadratic terms: x^3 + y^2 - z)
- exp2: Nonlinear Mixed (log, exp, sqrt, polynomial: log(x) + exp(y) - sqrt(z) + a^2)
- exp3: Trigonometric (sin, cos functions: sin(x) + cos(y) - z)
- exp4: S-curve (quadratic degeneracy on non-uniform manifold)

Usage:
    python /home/x-ctirapongpra/scratch/degen_detector/scripts/run_synthetic_experiments.py --experiments exp1

Results are saved as pickle files that can be loaded by notebooks for plotting.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Set matplotlib backend before any imports that might use it
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

try:
    import dill as pickle  # dill can pickle lambdas from sympy.lambdify
except ImportError:
    import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from degen_detector import DegenDetector
from degen_detector.synthetic import (
    generate_polynomial_separable,
    generate_nonlinear_mixed,
    generate_trig_separable,
    generate_scurve_separable,
)


EXPERIMENTS = [
    {
        "name": "exp1_polynomial",
        "generator": generate_polynomial_separable,
        "coupling_depth": 3,
        "niterations": 120,
        "max_fits": 1,
    },
    {
        "name": "exp2_nonlinear_mixed",
        "generator": generate_nonlinear_mixed,
        "coupling_depth": 4,
        "niterations": 120,
        "max_fits": 1,
    },
    {
        "name": "exp3_trig",
        "generator": generate_trig_separable,
        "coupling_depth": 3,
        "niterations": 150,
        "max_fits": 1,
    },
    {
        "name": "exp4_scurve",
        "generator": generate_scurve_separable,
        "coupling_depth": 3,
        "niterations": 200,
        "max_fits": 1,
    },
]


def run_experiment(exp_config, output_dir, max_fits=None):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_config['name']}")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate data (uses default n=2000, noise=0.1 from generators)
    print("Generating synthetic data...")
    samples, param_names, ground_truth = exp_config["generator"]()

    print(f"Ground truth: {ground_truth['equation']}")
    print(f"Sample shape: {samples.shape}")

    # Run detector
    print(f"\nInitializing DegenDetector...")
    det = DegenDetector(samples, param_names)

    print(f"Running search_couplings with:")
    print(f"  coupling_depth={exp_config['coupling_depth']}")
    print(f"  niterations={exp_config['niterations']}")
    print(f"  max_fits={max_fits}")
    print("This may take several minutes (PySR symbolic regression)...")

    result = det.search_couplings(
        coupling_depth=exp_config["coupling_depth"],
        niterations=exp_config["niterations"],
        max_complexity=20,
        max_fits=max_fits,
    )

    print(f"\nSearch completed!")
    print(f"Fits attempted: {result.n_fits_attempted}/{result.n_tuples_total}")

    # Results are ranked by MI - first valid fit is the top candidate
    top_fit = next((cf for cf in result.fits if cf.fit is not None), None)
    if top_fit:
        print(f"Top fit (by MI={top_fit.mi_score:.4f}): {top_fit.fit.equation_str}")
        print(f"R²_ortho = {top_fit.fit.orthogonal_r2:.4f}")

    return {
        "name": exp_config["name"],
        "ground_truth": ground_truth,
        "samples": samples,
        "param_names": param_names,
        "result": result,
        "top_fit": top_fit,  # First valid fit by MI ranking
    }


def main():
    parser = argparse.ArgumentParser(description="Run synthetic degeneracy experiments")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/synthetic"),
        help="Base output directory (default: outputs/synthetic)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["exp1", "exp2", "exp3", "exp4"],
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--max-fits",
        type=int,
        default=None,
        help="Maximum number of fits to try (default: no limit)",
    )
    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Filter experiments if specified
    experiments = EXPERIMENTS
    if args.experiments:
        experiments = [e for e in EXPERIMENTS if e["name"].startswith(tuple(args.experiments))]

    # Run experiments
    all_results = {}
    for exp_config in experiments:
        # Use command-line max_fits if provided, otherwise use experiment's default
        max_fits = args.max_fits if args.max_fits is not None else exp_config.get('max_fits')
        result = run_experiment(exp_config, output_dir, max_fits=max_fits)
        # Use exp1, exp2, exp3, exp4 as keys (extract from name like "exp1_linear")
        key = exp_config["name"].split("_")[0]
        all_results[key] = result

        # Save individual result
        result_file = output_dir / f"{exp_config['name']}_result.pkl"
        with open(result_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved: {result_file}")

    # Save combined results
    combined_file = output_dir / "all_results.pkl"
    with open(combined_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved combined results: {combined_file}")

    # Print summary
    print("\n" + "="*80)
    print(f"{'Experiment':<25} {'Ground Truth':<40} {'MI':>8} {'R²_ortho':>10}")
    print("="*80)
    for key, r in all_results.items():
        gt_eq = r['ground_truth']['equation']
        if r['top_fit']:
            mi = r['top_fit'].mi_score
            r2 = r['top_fit'].fit.orthogonal_r2
            print(f"{r['name']:<25} {gt_eq:<40} {mi:>8.4f} {r2:>10.4f}")
        else:
            print(f"{r['name']:<25} {gt_eq:<40} {'N/A':>8} {'N/A':>10}")
    print("="*80)


if __name__ == "__main__":
    main()
