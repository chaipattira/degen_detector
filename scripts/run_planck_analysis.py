#!/usr/bin/env python
"""Run degeneracy detection on Planck 2018 ΛCDM chains and save results for plotting.

This script loads Planck MCMC chains, runs the DegenDetector algorithm to search for
parameter degeneracies, and saves results for visualization.

Usage:
    python scripts/run_planck_analysis.py [--output-dir DIR] [--coupling-depth N]

Results are saved as pickle files that can be loaded by plot_planck_results.py for plotting.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from degen_detector import DegenDetector
from getdist import loadMCSamples


def run_planck_analysis(chain_root, param_names, output_dir,
                        coupling_depth=3, niterations=100,
                        max_complexity=15, max_fits=None):
    """Run degeneracy detection on Planck chains.

    Parameters
    ----------
    chain_root : str
        Path to Planck chain root file.
    param_names : list[str]
        List of parameter names to analyze.
    output_dir : Path
        Directory to save results.
    coupling_depth : int
        Maximum coupling depth (number of parameters in a degeneracy).
    niterations : int
        Number of symbolic regression iterations.
    max_complexity : int
        Maximum equation complexity for symbolic regression.
    max_fits : int or None
        Maximum number of fits to attempt (None = no limit).

    Returns
    -------
    dict
        Results dictionary containing samples, parameters, and fits.
    """
    print(f"\n{'='*60}")
    print("Loading Planck 2018 ΛCDM chains")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load chains (ignore first 30% as burn-in)
    print(f"Loading from: {chain_root}")
    mc_samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})

    # Extract parameters
    param_obj = mc_samples.getParams()
    samples = np.column_stack([getattr(param_obj, p) for p in param_names])

    print(f"\nSample shape: {samples.shape}")
    print(f"Parameters: {param_names}")
    print(f"\nParameter statistics:")
    for i, p in enumerate(param_names):
        print(f"  {p:>10s} = {np.mean(samples[:, i]):.5f} ± {np.std(samples[:, i]):.5f}")

    # Run detector
    print(f"\n{'='*60}")
    print("Running DegenDetector")
    print(f"{'='*60}")
    print(f"Initializing DegenDetector...")

    detector = DegenDetector(samples, param_names)

    print(f"Running search_couplings with:")
    print(f"  coupling_depth={coupling_depth}")
    print(f"  niterations={niterations}")
    print(f"  max_complexity={max_complexity}")
    print(f"  max_fits={max_fits}")

    result = detector.search_couplings(
        coupling_depth=coupling_depth,
        niterations=niterations,
        max_complexity=max_complexity,
        max_fits=max_fits,
    )

    print(f"\nSearch completed!")
    print(f"Fits attempted: {result.n_fits_attempted}/{result.n_tuples_total}")

    # Get all successful fits sorted by MI score (default ordering)
    successful_fits = [cf for cf in result.fits if cf.fit is not None]

    print(f"\nSuccessful fits: {len(successful_fits)}")

    # Show top 10 fits ranked by MI score
    if successful_fits:
        print("\nTop fits (ranked by MI score):")
        for i, cf in enumerate(successful_fits[:10]):
            print(f"  {i+1}. {cf.fit.equation_str}")
            print(f"     MI = {cf.mi_score:.4f}, R²_ortho = {cf.fit.orthogonal_r2:.4f}")

    # Top fit is first successful fit (highest MI)
    top_fit = successful_fits[0] if successful_fits else None

    return {
        "name": "planck_2018_lcdm",
        "chain_root": chain_root,
        "samples": samples,
        "param_names": param_names,
        "result": result,
        "successful_fits": successful_fits,
        "top_fit": top_fit,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Planck degeneracy analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/planck"),
        help="Base output directory (default: outputs/planck)",
    )
    parser.add_argument(
        "--chain-root",
        type=str,
        default="/home/x-ctirapongpra/scratch/degen_detector/data/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing",
        help="Path to Planck chain root",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Parameters to analyze (default: omegam H0 sigma8 ns tau logA)",
    )
    parser.add_argument(
        "--coupling-depth",
        type=int,
        default=4,
        help="Maximum coupling depth (default: 4)",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=150,
        help="Number of SR iterations (default: 150)",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=20,
        help="Maximum equation complexity (default: 20)",
    )
    parser.add_argument(
        "--max-fits",
        type=int,
        default=4,
        help="Maximum number of fits to try (default: 4)",
    )
    args = parser.parse_args()

    # Target cosmological parameters (default)
    param_names = args.params if args.params else ['omegam', 'H0', 'sigma8', 'ns', 'tau', 'logA']

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run analysis
    result = run_planck_analysis(
        args.chain_root,
        param_names,
        output_dir,
        coupling_depth=args.coupling_depth,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
        max_fits=args.max_fits,
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
    if result['top_fit']:
        print(f"Top fit (by MI score):")
        print(f"  Equation: {result['top_fit'].fit.equation_str}")
        print(f"  MI score: {result['top_fit'].mi_score:.4f}")
        print(f"  R²_ortho: {result['top_fit'].fit.orthogonal_r2:.4f}")
        print(f"\nTotal successful fits: {len(result['successful_fits'])}")
    else:
        print("No successful fits found")
    print("="*80)


if __name__ == "__main__":
    main()
