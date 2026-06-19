#!/usr/bin/env python
# ABOUTME: Runs DegenLogMode on DES Y3 3x2pt chains to find the S8 degeneracy.
# ABOUTME: Uses Omega_m + sigma_8 (and optionally h) to recover sigma8 ~ Omega_m^(-0.5).
"""Search for log-space degeneracies in DES Y3 3x2pt LCDM chains.

Note: DES Y6 chains are not publicly available as of April 2026. DES Y3
3x2pt LCDM MagLim is used as the weak-lensing observational example.

Loads the DES Y3 3x2pt LCDM chain (polychord weighted samples), resamples
to equal weight, and runs DegenLogMode on cosmological parameters.

Data source:
    https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/chains/
    File: chain_3x2pt_lcdm_SR_maglim.txt
    Reference: DES Y3 (Abbott et al. 2022, arXiv:2207.10900)

Chain format (DES Y3 CosmoSIS polychord output, plain ASCII):
    - No header; each row is one sample.
    - Column 0:    Omega_m (matter density fraction)
    - Column ~1-4: h, Omega_b, n_s, A_s (cosmo params, order varies by run)
    - Column ~28+: sigma_8 (derived by CosmoSIS, position auto-detected)
    - Last column: sample weight (polychord importance weight)

Known degeneracy to recover:
  - sigma_8 * Omega_m^0.5 = S8 ~ const  =>  log(sigma_8) + 0.5*log(Omega_m) = const

Usage:
    python scripts/run_des_y3_logmode.py
    python scripts/run_des_y3_logmode.py --max-fits 3

Download chain first:
    bash scripts/download_obs_chains.sh
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from degen_detector import DegenLogMode
from degen_detector.diagnostics import DiagnosticsRunner
from degen_detector.io import save_pickle, create_output_dir


# ── Chain location ────────────────────────────────────────────────────────────
CHAIN_FILE = (
    Path(__file__).parent.parent
    / "data" / "des_y3_3x2pt"
    / "chain_3x2pt_lcdm_SR_maglim.txt"
)

# ── Known column positions ────────────────────────────────────────────────────
# Omega_m is always column 0 in DES Y3 CosmoSIS chains (confirmed from
# read_chains_y3a2.py utility provided by the DES collaboration).
# Weight is the last column (polychord importance weight).
# sigma_8 position is auto-detected (derived parameter, position depends on
# the number of sampled nuisance parameters which varies by analysis type).
COL_OMEGAM = 0
COL_H      = 1
COL_OMEGAB = 2
COL_NS     = 3

# Expected physical ranges for auto-detection
_COSMO_RANGES = {
    "Omega_m": (0.05,  0.70,  0.29),   # (min, max, fiducial)
    "h":       (0.45,  0.95,  0.68),
    "Omega_b": (0.020, 0.080, 0.048),
    "n_s":     (0.80,  1.10,  0.97),
    "sigma_8": (0.50,  1.10,  0.80),
}


def detect_sigma8_column(data: np.ndarray, weight_col: int) -> int:
    """Find the column index of sigma_8 by scanning for expected physical range.

    Searches columns 5..weight_col-1 for a column with median ∈ (0.65, 1.00)
    and std ∈ (0.01, 0.15), consistent with sigma_8 from a WL posterior.
    Falls back to the first match beyond column 20 to skip h and n_s.

    Parameters
    ----------
    data : ndarray
        Full chain array (rows = samples, cols = parameters + weight).
    weight_col : int
        Index of the weight column (last column).

    Returns
    -------
    int
        Column index of sigma_8.

    Raises
    ------
    RuntimeError
        If no plausible sigma_8 column is found.
    """
    candidates = []
    for col in range(5, weight_col):
        vals = data[:, col]
        med = float(np.median(vals))
        std = float(np.std(vals))
        # sigma_8 is in [0.60, 1.05] with std [0.025, 0.15].
        # Lower bound on std (> 0.025) excludes tightly-constrained derived
        # combinations like S8 = sigma8*sqrt(Omega_m/0.3) (std ~ 0.02 in DES),
        # which are stored after sigma_8 in CosmoSIS output.
        if 0.60 < med < 1.05 and 0.025 < std < 0.15:
            candidates.append((col, med, std))

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect sigma_8 column. "
            "Please specify --sigma8-col manually."
        )

    # Return the FIRST candidate: sigma_8 precedes derived combinations
    # (e.g. S8, s8h5) in CosmoSIS chain output ordering.
    col, med, std = candidates[0]
    print(f"  Auto-detected sigma_8 at column {col} (median={med:.3f}, std={std:.3f})")
    return col


def load_des_samples(
    chain_file: Path,
    sigma8_col: int = None,
    n_resample: int = None,
    seed: int = 42,
) -> tuple:
    """Load DES Y3 chain and return (samples, param_names).

    Parameters
    ----------
    chain_file : Path
        Path to the plain-text chain file.
    sigma8_col : int or None
        Column index for sigma_8. If None, auto-detected.
    n_resample : int or None
        Number of equal-weight samples to draw (with replacement).
        If None, uses the total effective sample size.
    seed : int
        Random seed for resampling.

    Returns
    -------
    samples : ndarray, shape (n_resample, n_cosmo)
    param_names : list[str]
    """
    if not chain_file.exists():
        raise FileNotFoundError(
            f"Chain file not found: {chain_file}\n"
            f"Run: bash scripts/download_obs_chains.sh"
        )

    print(f"  Reading {chain_file} ...")
    # DES Y3 chain files have no header; some may have comment lines starting '#'
    data = np.loadtxt(chain_file, comments="#")
    print(f"  Raw shape: {data.shape[0]} samples × {data.shape[1]} columns")

    weight_col = data.shape[1] - 1  # weights are always the last column
    weights = data[:, weight_col]

    # Omega_m: column 0 (known)
    om = data[:, COL_OMEGAM]
    print(f"  Omega_m (col 0):  median={np.median(om):.3f}, std={np.std(om):.3f}")

    # sigma_8: auto-detect or use provided index
    if sigma8_col is None:
        sigma8_col = detect_sigma8_column(data, weight_col)
    s8 = data[:, sigma8_col]
    print(f"  sigma_8 (col {sigma8_col}): median={np.median(s8):.3f}, std={np.std(s8):.3f}")

    # Build parameter matrix: [Omega_m, h, Omega_b, n_s, sigma_8]
    param_cols = [COL_OMEGAM, COL_H, COL_OMEGAB, COL_NS, sigma8_col]
    param_names = ["Omega_m", "h", "Omega_b", "n_s", "sigma_8"]
    cosmo = data[:, param_cols]

    # ── Resample by importance weights ────────────────────────────────────────
    # DES polychord chains use importance weights; must resample to get
    # equal-weight samples for DegenLogMode (which assumes unweighted input).
    w = weights.copy()
    w_neg = np.sum(w < 0)
    if w_neg > 0:
        print(f"  Warning: {w_neg} negative weights found; clipping to 0.")
        w = np.maximum(w, 0.0)
    w /= w.sum()

    eff_n = int(1.0 / np.sum(w ** 2))  # effective sample size
    n_draw = n_resample or min(eff_n, 20000)
    print(f"  Effective sample size: {eff_n}. Drawing {n_draw} equal-weight samples.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cosmo), size=n_draw, replace=True, p=w)
    samples = cosmo[idx]

    return samples, param_names


def main():
    parser = argparse.ArgumentParser(
        description="DegenLogMode on DES Y3 3x2pt LCDM chains"
    )
    parser.add_argument(
        "--chain-file",
        type=Path,
        default=CHAIN_FILE,
        help=f"Path to chain_3x2pt_lcdm_SR_maglim.txt (default: {CHAIN_FILE})",
    )
    parser.add_argument(
        "--sigma8-col",
        type=int,
        default=None,
        help="Column index of sigma_8 (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/des_y3_logmode"),
        help="Base output directory (default: outputs/des_y3_logmode)",
    )
    parser.add_argument(
        "--coupling-depth",
        type=int,
        default=2,
        help="Tuple size: 2=pairs, 3=triplets (default: 3)",
    )
    parser.add_argument(
        "--max-fits",
        type=int,
        default=None,
        help="Maximum number of tuples to fit",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=200,
        help="PySR iterations per component (default: 200)",
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
        help="PySR batch size (default: 1000)",
    )
    parser.add_argument(
        "--n-resample",
        type=int,
        default=20000,
        help="Number of equal-weight samples drawn from polychord output (default: 20000)",
    )
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # ── Load chain ────────────────────────────────────────────────────────────
    print(f"\nLoading DES Y3 chain: {args.chain_file}")
    samples, param_names = load_des_samples(
        args.chain_file,
        sigma8_col=args.sigma8_col,
        n_resample=args.n_resample,
    )
    print(f"  Final: {samples.shape[0]} samples, {len(param_names)} params: {param_names}")

    # ── Run DegenLogMode ──────────────────────────────────────────────────────
    # DegenLogMode applies log to all parameters by default and automatically
    # drops any samples where a parameter is non-positive before transforming.
    print(f"\nRunning DegenLogMode with coupling_depth={args.coupling_depth}, "
          f"niterations={args.niterations}, max_fits={args.max_fits}")
    print("  Expected to recover: log(sigma_8) + 0.5*log(Omega_m) = const  (S8 degeneracy)")

    detector = DegenLogMode(samples, param_names)
    result = detector.search_couplings(
        coupling_depth=args.coupling_depth,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
        max_fits=args.max_fits,
        batch_size=args.batch_size,
    )

    print(f"\nSearch completed. Fits attempted: {result.n_fits_attempted}/{result.n_tuples_total}")

    # ── Summary ───────────────────────────────────────────────────────────────
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

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "samples": samples,
        "param_names": param_names,
        "result": result,
        "dataset": "DES Y3 3x2pt LCDM MagLim (Abbott et al. 2022, arXiv:2207.10900)",
    }
    result_file = output_dir / "des_y3_logmode_result.pkl"
    save_pickle(payload, result_file)
    print(f"\nSaved: {result_file}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
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
