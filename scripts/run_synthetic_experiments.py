#!/usr/bin/env python
"""Run synthetic degeneracy experiments and save results for plotting.

Usage:
    python scripts/run_synthetic_experiments.py [--output-dir DIR]

Results are saved as pickle files that can be loaded by notebooks for plotting.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import dill as pickle  # dill can pickle lambdas from sympy.lambdify
except ImportError:
    import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from degen_detector import DegenDetector
from degen_detector.synthetic import (
    generate_3param_polynomial,
    generate_3param_exp_log,
    load_sbibm_slcp,
)


EXPERIMENTS = [
    {
        "name": "exp1_polynomial",
        "generator": generate_3param_polynomial,
        "params": ["x", "y", "z"],
        "ground_truth": "z = x² + y",
        "coupling_depth": 3,
        "niterations": 40,
        "is_sbibm": False,
    },
    {
        "name": "exp2_exp_log",
        "generator": generate_3param_exp_log,
        "params": ["x", "y", "z"],
        "ground_truth": "z = exp(x) + log(y)",
        "coupling_depth": 3,
        "niterations": 40,
        "is_sbibm": False,
    },
    {
        "name": "exp3_sbibm_slcp",
        "generator": load_sbibm_slcp,
        "params": None,  # Use all 5 parameters
        "ground_truth": "Unknown (SBIBM SLCP benchmark)",
        "coupling_depth": 2,  # Start with pairs
        "niterations": 100,
        "is_sbibm": True,
        "batching": True,  # Use batching for large dataset (10k samples)
    },
]


def run_experiment(exp_config, output_dir, n_samples=2000, noise=0.3):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_config['name']}")
    print(f"Ground truth: {exp_config['ground_truth']}")
    print(f"{'='*60}")

    # Generate or load data
    if exp_config.get("is_sbibm", False):
        samples, names = exp_config["generator"]()
    else:
        samples, names = exp_config["generator"](n=n_samples, noise=noise)

    # Create experiment-specific output dir
    exp_output_dir = output_dir / exp_config["name"]
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Run detector
    det = DegenDetector(samples, names)
    result = det.search_couplings(
        params=exp_config["params"],
        coupling_depth=exp_config["coupling_depth"],
        niterations=exp_config["niterations"],
        max_complexity=20,
        batching=exp_config.get("batching", False),
        output_dir=exp_output_dir,
    )

    print(f"Best fit: {result.best_fit.fit.equation_str}")
    print(f"R² = {result.best_fit.fit.r_squared:.4f}")

    return {
        "name": exp_config["name"],
        "ground_truth": exp_config["ground_truth"],
        "samples": samples,
        "param_names": names,
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
        "--n-samples",
        type=int,
        default=2000,
        help="Number of samples per experiment (default: 2000)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.3,
        help="Noise level (default: 0.3)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["exp1", "exp2", "exp3"],
        default=None,
        help="Specific experiments to run (default: all)",
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
    all_results = []
    for exp_config in experiments:
        result = run_experiment(exp_config, output_dir, args.n_samples, args.noise)
        all_results.append(result)

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
    print(f"{'Experiment':<20} {'Ground Truth':<35} {'R²':>8}")
    print("="*80)
    for r in all_results:
        print(f"{r['name']:<20} {r['ground_truth']:<35} {r['result'].best_fit.fit.r_squared:>8.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
