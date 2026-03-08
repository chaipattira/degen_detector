#!/usr/bin/env python
"""Run synthetic degeneracy experiments and save results for plotting.

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
    generate_linear_separable,
    generate_log_separable,
    generate_power_law,
    generate_exp_linear,
    generate_quadratic_separable,
    generate_trig_separable,
)


EXPERIMENTS = [
    {
        "name": "exp1_linear",
        "generator": generate_linear_separable,
        "coupling_depth": 3,
        "niterations": 40,
        "max_fits": 5,
    },
    {
        "name": "exp2_log",
        "generator": generate_log_separable,
        "coupling_depth": 3,
        "niterations": 40,
        "max_fits": 5,
    },
    {
        "name": "exp3_power_law",
        "generator": generate_power_law,
        "coupling_depth": 2,
        "niterations": 40,
        "max_fits": 5,
    },
    {
        "name": "exp4_exp_linear",
        "generator": generate_exp_linear,
        "coupling_depth": 3,
        "niterations": 40,
        "max_fits": 5,
    },
    {
        "name": "exp5_quadratic",
        "generator": generate_quadratic_separable,
        "coupling_depth": 3,
        "niterations": 80,
        "max_fits": 5,
    },
    {
        "name": "exp6_trig",
        "generator": generate_trig_separable,
        "coupling_depth": 3,
        "niterations": 80,
        "max_fits": 5,
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
    print(f"Best fit: {result.best_fit.fit.equation_str}")
    print(f"R²_ortho = {result.best_fit.fit.orthogonal_r2:.4f}")
    print(f"Fits attempted: {result.n_fits_attempted}/{result.n_fits_total}")

    return {
        "name": exp_config["name"],
        "ground_truth": ground_truth,
        "samples": samples,
        "param_names": param_names,
        "result": result,
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
        choices=["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"],
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
        result = run_experiment(exp_config, output_dir, max_fits=args.max_fits)
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
    print(f"{'Experiment':<25} {'Ground Truth':<40} {'R²_ortho':>10}")
    print("="*80)
    for key, r in all_results.items():
        gt_eq = r['ground_truth']['equation']
        print(f"{r['name']:<25} {gt_eq:<40} {r['result'].best_fit.fit.orthogonal_r2:>10.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
