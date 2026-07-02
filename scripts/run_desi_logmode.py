#!/usr/bin/env python
# ABOUTME: Runs DegenLogMode on DESI DR1 full-shape chains to find log-space degeneracies.
# ABOUTME: Targets cosmological params where power-law constraints (e.g. sigma8*Omega_m^0.5) appear.
"""Search for log-space degeneracies in DESI DR1 full-shape chains.

Loads DESI DR1 cobaya MCMC chains (all tracers, LCDM, velocileptors),
extracts cosmological parameters, and runs DegenLogMode.

Data source:
    https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-cosmo-params/
    Dataset: desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10
    Reference: DESI 2024 VII (arXiv:2411.12022)

Known degeneracies to recover:
  - sigma8 * Omega_m^0.5 = S8  =>  log(sigma8) + 0.5*log(omegam) = const
  - Omega_m * h^2 ~ const      =>  log(omegam) + 2*log(H0) = const  (shape param)

The run uses 16 cosmological params — the three S8-family columns (s8h5, s8omegamp5, s8omegamp25),
omegamh2, A, and clamp are excluded (see column notes below).

Parameter note:
    DESI chains fix tau=0.0544 (BBN prior) and use Schoneberg2024 BBN prior on ombh2.
    The free cosmological parameters are: H0, ombh2, omch2, logA, ns.
    Derived parameters (sigma8, omegam, etc.) are stored in the chain by cobaya.

Column order in chain files (cobaya output, 0-indexed):
    0: weight (multiplicity)
    1: -log(posterior)
    2: logA       (sampled)
    3: ns         (sampled)
    4: H0         (sampled)
    5: ombh2      (sampled)
    6: omch2      (sampled)
  7-24: EFT nuisance params per tracer (b1p, b2p, bsp × 6 tracers) — not loaded
         pre_QSO_z0.{b1p,b2p,bsp}   cols 7-9
         pre_ELG_z1.{b1p,b2p,bsp}   cols 10-12
         pre_LRG_z2.{b1p,b2p,bsp}   cols 13-15
         pre_LRG_z1.{b1p,b2p,bsp}   cols 16-18
         pre_LRG_z0.{b1p,b2p,bsp}   cols 19-21
         pre_BGS_z0.{b1p,b2p,bsp}   cols 22-24
   25: As         (derived)
   26: omegam     (derived)
   27: omegamh2   (derived)
   28: omegal     (derived)
   29: zrei       (derived, fixed prior ~10, skipped)
   30: YHe        (derived)
   31: Y_p        (derived)
   32: DHBBN      (derived, stored as 10^5 × D/H ratio, so ~2.65)
   33: sigma8     (derived)
   34: s8h5       (derived)
   35: s8omegamp5 (derived)
   36: s8omegamp25 (derived)
   37: A          (derived, skipped)
   38: clamp      (derived, skipped)
   39: age        (derived)
   40: rdrag      (derived)
   41: zdrag      (derived)
   42: H0rdrag    (derived)

    Column order derived from chain.margestats (getdist output reflects chain column order).

Usage:
    python scripts/run_desi_logmode.py
    python scripts/run_desi_logmode.py --max-fits 3 --coupling-depth 3

Download chains first:
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
CHAIN_DIR = (
    Path(__file__).parent.parent / "data" / "desi_dr1_fs"
)

# ── Column indices in chain.N.txt (0-indexed, including weight and -logpost) ──
# Derived from chain.margestats; verified by sanity-checking physical ranges.
# Cols 18 (A) and 19 (clamp) are skipped: clamp = sigma8*exp(-tau) is redundant
# with sigma8 since tau is fixed at 0.0544, and col-18 'A' is unconfirmed.
COL_WEIGHT = 0
COL_LOGPOST = 1
# Sampled cosmological:
COL_LOGA        = 2   # ln(10^10 * As)  ~ 3.04
COL_NS          = 3   # spectral index   ~ 0.97
COL_H0          = 4   # Hubble constant km/s/Mpc
COL_OMBH2       = 5   # Omega_b * h^2
COL_OMCH2       = 6   # Omega_c * h^2
# EFT nuisance params occupy cols 7-24 (18 cols: 6 tracers × 3 params each).
# Derived cosmological (cols 25-41):
COL_AS          = 25  # As  ~ 2.1e-9
COL_OMEGAM      = 26  # Omega_matter
COL_OMEGAMH2    = 27  # Omega_m * h^2
COL_OMEGAL      = 28  # Omega_Lambda
COL_ZREI        = 29  # Reionization redshift (fixed prior, ~10)  — skipped
COL_YHE         = 30  # Helium fraction (BBN)
COL_YP          = 31  # Primordial helium (BBN)
COL_DHBBN       = 32  # D/H ratio × 10^5 (BBN convention in cobaya)  ~ 2.65
COL_SIGMA8      = 33  # sigma_8
COL_S8H5        = 34  # sigma8 * h^0.5   ~ 1.0
COL_S8OMEGAMP5  = 35  # sigma8 * Omega_m^0.5  ~ 0.45  (S8 up to factor sqrt(1/0.3))
COL_S8OMEGAMP25 = 36  # sigma8 * Omega_m^0.25 ~ 0.61
# col 37: A (skipped), col 38: clamp (skipped)
COL_AGE         = 39  # Age of universe (Gyr)  ~ 13.7
COL_RDRAG       = 40  # Sound horizon at drag epoch (Mpc)  ~ 148
COL_ZDRAG       = 41  # Drag redshift  ~ 1059
COL_H0RDRAG     = 42  # H0 * rdrag  ~ 10390

# Sanity-check ranges for cosmological parameters (physical)
SANITY = {
    "logA":         (2.50,  3.50),
    "ns":           (0.85,  1.05),
    "H0":           (40.0,  100.0),
    #"ombh2":        (0.015, 0.030),
    #"omch2":        (0.08,  0.20),
    "As":           (1e-9,  4e-9),
    "omegam":       (0.15,  0.55),
    #"omegamh2":     (0.08,  0.22),
    "omegal":       (0.45,  0.85),
    "YHe":          (0.22,  0.27),
    "Y_p":          (0.22,  0.27),
    "DHBBN":        (1.0,   5.0),   # stored as 10^5 × D/H in cobaya chains
    "sigma8":       (0.60,  1.10),
    "age":          (12.0,  15.0),
    "rdrag":        (130.0, 160.0),
    "zdrag":        (900.0, 1100.0),
    "H0rdrag":      (8000., 12000.),
}


def load_desi_chains(chain_dir: Path, n_chains: int = 4) -> tuple:
    """Load DESI cobaya chains and return (samples, param_names).

    Reads all chain.N.txt files, stacks them, expands by multiplicity
    weights, and returns 16 cosmological parameters.

    Parameters
    ----------
    chain_dir : Path
        Directory containing chain.1.txt … chain.N.txt files.
    n_chains : int
        Number of chain files to load (default 4).

    Returns
    -------
    samples : ndarray, shape (n_expanded, 16)
        Rows are equal-weight samples.
    param_names : list[str]
    """
    chains = []
    for i in range(1, n_chains + 1):
        fname = chain_dir / f"chain.{i}.txt"
        if not fname.exists():
            raise FileNotFoundError(
                f"Chain file not found: {fname}\n"
                f"Run: bash scripts/download_obs_chains.sh"
            )
        data = np.loadtxt(fname)
        chains.append(data)
    data = np.vstack(chains)

    n_cols = data.shape[1]
    print(f"  Loaded {data.shape[0]} raw rows, {n_cols} columns total.")

    # Sanity check: verify expected columns exist (need at least through col 41 = H0rdrag)
    required_cols = 43
    if n_cols < required_cols:
        raise ValueError(
            f"Chain has only {n_cols} columns; expected at least {required_cols}. "
            f"Check column mapping in this script."
        )

    weights = data[:, COL_WEIGHT]
    raw_params = {
        "logA":         data[:, COL_LOGA],
        "ns":           data[:, COL_NS],
        "H0":           data[:, COL_H0],
       # "ombh2":        data[:, COL_OMBH2],
       # "omch2":        data[:, COL_OMCH2],
        "As":           data[:, COL_AS],
        "omegam":       data[:, COL_OMEGAM],
        "omegal":       data[:, COL_OMEGAL],
        "YHe":          data[:, COL_YHE],
        "Y_p":          data[:, COL_YP],
        "DHBBN":        data[:, COL_DHBBN],
        "sigma8":       data[:, COL_SIGMA8],
        # s8h5, s8omegamp5, s8omegamp25 excluded: they ARE the S8 constraint.
        # Excluding them forces the algorithm to discover it from omegam+sigma8.
        "age":          data[:, COL_AGE],
        "rdrag":        data[:, COL_RDRAG],
        "zdrag":        data[:, COL_ZDRAG],
        "H0rdrag":      data[:, COL_H0RDRAG],
    }

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("  Parameter sanity checks:")
    for name, arr in raw_params.items():
        if name not in SANITY:
            continue  # skip nuisance params
        med = float(np.median(arr))
        lo, hi = SANITY[name]
        status = "OK" if lo < med < hi else "WARNING: out of expected range!"
        print(f"    {name:14s} median={med:.5g}  expected ({lo}, {hi})  {status}")

    # ── Expand by multiplicity ─────────────────────────────────────────────────
    counts = np.round(weights).astype(int)
    counts = np.clip(counts, 0, None)

    param_matrix = np.column_stack(list(raw_params.values()))
    expanded = np.repeat(param_matrix, counts, axis=0)
    print(f"  Expanded by multiplicity: {data.shape[0]} rows → {expanded.shape[0]} samples.")

    param_names = list(raw_params.keys())
    return expanded, param_names


def main():
    parser = argparse.ArgumentParser(
        description="DegenLogMode on DESI DR1 full-shape chains"
    )
    parser.add_argument(
        "--chain-dir",
        type=Path,
        default=CHAIN_DIR,
        help=f"Directory with chain.N.txt files (default: {CHAIN_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/desi_dr1_logmode"),
        help="Base output directory (default: outputs/desi_dr1_logmode)",
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
        default=2,
        help="Maximum number of tuples to fit by MI rank (default: 2)",
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
        help="PySR batch size (default: 1000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Subsample to at most N samples after weight expansion (default: 20000)",
    )
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # ── Load chains ───────────────────────────────────────────────────────────
    print(f"\nLoading DESI DR1 chains from: {args.chain_dir}")
    samples, param_names = load_desi_chains(args.chain_dir)

    # Subsample if needed to keep runtime manageable
    if args.max_samples and len(samples) > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), size=args.max_samples, replace=False)
        samples = samples[idx]
        print(f"  Subsampled to {len(samples)} samples.")

    print(f"  Final: {samples.shape[0]} samples, {len(param_names)} params: {param_names}")

    # ── Run DegenLogMode ──────────────────────────────────────────────────────
    # DegenLogMode applies log to all parameters by default and automatically
    # drops any samples where a parameter is non-positive before transforming.
    print(f"\nRunning DegenLogMode with coupling_depth={args.coupling_depth}, "
          f"niterations={args.niterations}, max_fits={args.max_fits}, "
          f"batch_size={args.batch_size}")

    detector = DegenLogMode(samples, param_names)
    ranking = detector.rank_couplings(coupling_depth=args.coupling_depth)
    result = detector.fit_couplings(
        ranking,
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
        "dataset": "DESI DR1 full-shape, all tracers, LCDM (arXiv:2411.12022)",
    }
    result_file = output_dir / "desi_dr1_logmode_result.pkl"
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
