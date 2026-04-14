#!/usr/bin/env python
# ABOUTME: Runs DegenLogMode on Planck TTTEEE+lowl+lowE+lensing chains to find log-space degeneracies.
# ABOUTME: Targets cosmological params where multiplicative constraints (e.g. sigma8*Omega_m^0.5) appear.
"""Search for log-space degeneracies in Planck 2018 chains.

Loads base_plikHM_TTTEEE_lowl_lowE_lensing chains via getdist, extracts
cosmological parameters, and runs DegenLogMode (log-transform + pipeline).

Known degeneracies to recover:
  - sigma8 * Omega_m^0.5 = S8 ~ const  =>  log(sigma8) + 0.5*log(omegam) = const
  - Omega_m h^3 ~ const (geometric)     =>  log(omegam) + 3*log(H0) = const

Usage:
    python scripts/run_planck_logmode.py
    python scripts/run_planck_logmode.py --max-fits 3 --coupling-depth 3
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from getdist import loadMCSamples

from degen_detector import DegenLogMode, LOG_TRANSFORM
from degen_detector.diagnostics import DiagnosticsRunner
from degen_detector.io import save_pickle, create_output_dir


CHAIN_ROOT = (
    Path(__file__).parent.parent
    / "data/base/plikHM_TTTEEE_lowl_lowE_lensing"
    / "base_plikHM_TTTEEE_lowl_lowE_lensing"
)

# Cosmological parameters to analyse — all strictly positive, sensible for log transform.
# Derived params (marked * in .paramnames) included because they carry the geometric info.
COSMO_PARAMS = [
    "omegabh2",   # Omega_b h^2
    "omegach2",   # Omega_c h^2
    "H0",         # Hubble constant (km/s/Mpc)
    "omegam",     # Omega_matter (derived)
    "sigma8",     # sigma_8 (derived)
    "ns",         # spectral index
    "tau",        # optical depth to reionisation
]


def load_planck_samples(chain_root: Path, params: list, ignore_rows: float = 0.3):
    """Load Planck chain and return (samples array, param_names).

    Parameters
    ----------
    chain_root : Path
        Path stem for the chain files (without _1.txt etc.).
    params : list[str]
        Parameter names to extract (must exist in .paramnames).
    ignore_rows : float
        Fraction of chain to discard as burn-in (passed to getdist).

    Returns
    -------
    samples : ndarray, shape (n_samples, len(params))
    param_names : list[str]
    """
    mc = loadMCSamples(
        str(chain_root),
        settings={"ignore_rows": ignore_rows},
    )
    p = mc.getParams()

    cols = []
    available = []
    for name in params:
        arr = getattr(p, name, None)
        if arr is None:
            print(f"  Warning: '{name}' not found in chain, skipping.")
            continue
        cols.append(arr)
        available.append(name)

    return np.column_stack(cols), available


def main():
    parser = argparse.ArgumentParser(
        description="DegenLogMode on Planck 2018 chains"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/planck_logmode"),
        help="Base output directory (default: outputs/planck_logmode)",
    )
    parser.add_argument(
        "--coupling-depth",
        type=int,
        default=2,
        help="Tuple size: 2=pairs, 3=triplets (default: 2)",
    )
    parser.add_argument(
        "--max-fits",
        type=int,
        default=None,
        help="Maximum number of tuples to fit by MI rank (default: all)",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=200,
        help="PySR evolution iterations per component (default: 200)",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=15,
        help="Maximum equation complexity for PySR (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="PySR batch size (default: 1000, tuned for ~25k Planck samples)",
    )
    parser.add_argument(
        "--ignore-rows",
        type=float,
        default=0.3,
        help="Burn-in fraction to discard (default: 0.3)",
    )
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # ── Load chain ──────────────────────────────────────────────────────────
    print(f"\nLoading chain: {CHAIN_ROOT}")
    samples, param_names = load_planck_samples(
        CHAIN_ROOT, COSMO_PARAMS, ignore_rows=args.ignore_rows
    )
    print(f"Loaded {samples.shape[0]} samples, {len(param_names)} params: {param_names}")

    # Sanity check: all values must be positive for log transform
    for j, name in enumerate(param_names):
        col = samples[:, j]
        n_bad = np.sum(col <= 0)
        if n_bad > 0:
            print(f"  Warning: '{name}' has {n_bad} non-positive values — will be skipped in log transform.")

    # ── Run DegenLogMode ─────────────────────────────────────────────────────
    # Only log-transform params that are strictly positive across the chain.
    transforms = {
        name: LOG_TRANSFORM
        for j, name in enumerate(param_names)
        if np.all(samples[:, j] > 0)
    }
    skipped = [n for n in param_names if n not in transforms]
    if skipped:
        print(f"  Skipping log transform for: {skipped} (non-positive values present)")

    print(f"\nRunning DegenLogMode with coupling_depth={args.coupling_depth}, "
          f"niterations={args.niterations}, max_fits={args.max_fits}, "
          f"batch_size={args.batch_size}")

    detector = DegenLogMode(samples, param_names, transforms=transforms)
    result = detector.search_couplings(
        coupling_depth=args.coupling_depth,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
        max_fits=args.max_fits,
        batch_size=args.batch_size,
    )

    print(f"\nSearch completed. Fits attempted: {result.n_fits_attempted}/{result.n_tuples_total}")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Params':<30} {'MI':>8} {'R²_ortho':>10}  Equation")
    print("=" * 80)
    for cf in result.fits:
        if cf.fit:
            print(
                f"{str(cf.param_names):<30} {cf.mi_score:>8.4f} "
                f"{cf.fit.orthogonal_r2:>10.4f}  {cf.fit.equation_str}"
            )
        else:
            print(f"{str(cf.param_names):<30} {cf.mi_score:>8.4f} {'N/A':>10}  (fit failed)")
    print("=" * 80)

    # ── Save results ─────────────────────────────────────────────────────────
    payload = {
        "samples": samples,
        "param_names": param_names,
        "result": result,
    }
    result_file = output_dir / "planck_logmode_result.pkl"
    save_pickle(payload, result_file)
    print(f"\nSaved: {result_file}")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("Running diagnostics...")
    try:
        runner = DiagnosticsRunner(result_file)
        runner.run(output_dir=output_dir / "diagnostics")
        print(f"Diagnostics saved to: {output_dir / 'diagnostics'}")
    except Exception as e:
        print(f"Warning: Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
