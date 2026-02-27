#!/usr/bin/env python
"""Run degeneracy detection on Planck 2018 ΛCDM chains and save results for plotting.

Usage:
    python scripts/run_planck_analysis.py [--output-dir DIR]

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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from degen_detector import DegenDetector
from getdist import loadMCSamples


def run_planck_analysis(chain_root, output_dir, target_params, coupling_depth=3,
                        r2_threshold=0.95, niterations=100, max_complexity=15):
    """Run degeneracy detection on Planck chains."""
    print(f"\n{'='*60}")
    print("Loading Planck 2018 ΛCDM chains")
    print(f"{'='*60}")

    # Load chains
    mc_samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})

    # Extract parameters
    param_obj = mc_samples.getParams()
    samples = np.column_stack([getattr(param_obj, p) for p in target_params])

    print(f'Samples shape: {samples.shape}')
    print(f'Parameters: {target_params}')
    print(f'\nMeans:')
    for i, p in enumerate(target_params):
        print(f'  {p:>10s} = {np.mean(samples[:, i]):.5f} ± {np.std(samples[:, i]):.5f}')

    # Run detector
    print(f"\n{'='*60}")
    print("Running degeneracy detector")
    print(f"{'='*60}")

    detector = DegenDetector(samples, target_params)
    results = detector.search_couplings(
        params=target_params,
        coupling_depth=coupling_depth,
        r2_threshold=r2_threshold,
        niterations=niterations,
        max_complexity=max_complexity,
        batching=True,
        output_dir=output_dir,
    )

    print(f'\nFits attempted: {results.n_fits_attempted}/{results.n_fits_total}')
    print(f'Early stop: {results.stopped_early}')

    # Show all successful fits ranked by R²
    successful = [f for f in results.fits if f.fit is not None]
    successful.sort(key=lambda f: f.fit.r_squared, reverse=True)

    print(f'\nSuccessful fits: {len(successful)}')
    for i, cf in enumerate(successful[:10]):  # Show top 10
        print(f'  {i+1}. {cf.fit.equation_str}')
        print(f'     R² = {cf.fit.r_squared:.4f}, MI = {cf.mi_score:.3f}')

    return {
        "name": "planck_2018_lcdm",
        "chain_root": chain_root,
        "samples": samples,
        "param_names": target_params,
        "results": results,
        "successful_fits": successful,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Planck degeneracy analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/x-ctirapongpra/scratch/outputs/planck"),
        help="Base output directory",
    )
    parser.add_argument(
        "--chain-root",
        type=str,
        default="/home/x-ctirapongpra/scratch/degen_detector/data/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing",
        help="Path to Planck chain root",
    )
    parser.add_argument(
        "--coupling-depth",
        type=int,
        default=3,
        help="Maximum coupling depth (default: 3)",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.95,
        help="Early stop R² threshold (default: 0.95)",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=100,
        help="Number of SR iterations (default: 100)",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=15,
        help="Maximum equation complexity (default: 15)",
    )
    args = parser.parse_args()

    # Target cosmological parameters
    target_params = ['omegam', 'H0', 'sigma8', 'ns', 'tau', 'logA']

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run analysis
    result = run_planck_analysis(
        args.chain_root,
        output_dir,
        target_params,
        coupling_depth=args.coupling_depth,
        r2_threshold=args.r2_threshold,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
    )

    # Save result
    result_file = output_dir / "planck_result.pkl"
    with open(result_file, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved: {result_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best fit: {result['results'].best_fit.fit.equation_str}")
    print(f"R² = {result['results'].best_fit.fit.r_squared:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
