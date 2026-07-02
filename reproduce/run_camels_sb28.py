#!/usr/bin/env python
# ABOUTME: Inference script: load saved NPE posterior → draw samples → run degen_detector.
# ABOUTME: Runs both linear and log modes; writes outputs to {out_dir}/{field}/{linear,log}/.
"""Load a saved NPE posterior, draw 20k samples, and run degen_detector.

Runs run_detector() twice per field:
  - linear mode: outputs to {out_dir}/{field}/linear/
  - log mode:    outputs to {out_dir}/{field}/log/

Prints per-parameter [min, max] of drawn samples before the degen_detector run
so the user can verify the posterior is within prior bounds.

Run from the degen_detector repo root:
    python reproduce/run_camels_sb28.py --field Mcdm \\
        --sbi-dir outputs/reproduce/camels_sb28/Mcdm/sbi \\
        --params-file /anvil/scratch/x-ctirapongpra/degen_detector/data/camels_sb28/params_SB28_IllustrisTNG.txt
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from degen_detector import run_detector


def _load_param_names(params_file: Path, sbi_dir: Path) -> list:
    """Read parameter names: try param_names.npy saved by train_camels_sbi.py, then params file header, then generic."""
    npy_path = sbi_dir / "param_names.npy"
    if npy_path.exists():
        return list(np.load(npy_path, allow_pickle=True))

    if params_file is not None and params_file.exists():
        with open(params_file) as f:
            first_line = f.readline().strip()
        try:
            float(first_line.split()[0])
            # No header
        except ValueError:
            return first_line.split()

    # Fallback: count columns from params file
    if params_file is not None and params_file.exists():
        data = np.loadtxt(params_file)
        names = [f"theta_{i}" for i in range(data.shape[1])]
        import warnings
        warnings.warn("Using generic parameter names theta_0..theta_N.", UserWarning)
        return names

    raise FileNotFoundError(
        "Cannot determine parameter names: neither param_names.npy in --sbi-dir "
        "nor a --params-file with header was found."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Draw posterior samples from saved NPE and run degen_detector"
    )
    parser.add_argument("--field", choices=["Mcdm", "Mgas", "Mstar"], required=True)
    parser.add_argument(
        "--sbi-dir",
        type=Path,
        help="Directory with posterior.pkl (output of train_camels_sbi.py). "
             "Defaults to outputs/reproduce/camels_sb28/{field}/sbi",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="Path to params_SB28_IllustrisTNG.txt (for parameter names).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reproduce/camels_sb28"),
    )
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of posterior samples to draw")
    parser.add_argument("--coupling-depth", type=int, default=2,
                        help="Coupling depth for degen_detector (default 2 → all pairs)")
    parser.add_argument("--max-fits", type=int, default=5)
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--log-mode", action="store_true",
                        help="Run only log mode (skip linear); default: run both")
    parser.add_argument("--linear-only", action="store_true",
                        help="Run only linear mode (skip log); default: run both")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("ERROR: torch not installed. pip install torch", file=sys.stderr)
        sys.exit(1)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    sbi_dir = args.sbi_dir or (args.out_dir / args.field / "sbi")
    posterior_path = sbi_dir / "posterior.pkl"
    x_obs_path = sbi_dir / "x_obs.npy"

    if not posterior_path.exists():
        print(f"ERROR: posterior.pkl not found at {posterior_path}", file=sys.stderr)
        print("Run train_camels_sbi.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Field:          {args.field}")
    print(f"SBI dir:        {sbi_dir}")
    print(f"Posterior:      {posterior_path}")
    print(f"n_samples:      {args.n_samples}")
    print(f"coupling_depth: {args.coupling_depth}")

    # ── Load parameter names ───────────────────────────────────────────────────
    param_names = _load_param_names(args.params_file, sbi_dir)
    print(f"Params ({len(param_names)}): {param_names[:6]}...")

    # ── Load posterior and x_obs ───────────────────────────────────────────────
    print("\nLoading posterior...")
    with open(posterior_path, "rb") as f:
        posterior_ensemble = pickle.load(f)

    x_obs = np.load(x_obs_path)    # (256, 256)
    x_obs_tensor = torch.tensor(x_obs[None], dtype=torch.float32)  # (1, 256, 256)

    # ── Draw samples ──────────────────────────────────────────────────────────
    print(f"Drawing {args.n_samples} posterior samples...")
    with torch.no_grad():
        samples_tensor = posterior_ensemble.sample((args.n_samples,), x=x_obs_tensor)
    samples = samples_tensor.cpu().numpy()   # (n_samples, 28)
    print(f"Samples shape: {samples.shape}")

    # ── Sanity check: print per-parameter bounds ────────────────────────────────
    print("\nPosterior sample ranges (sanity check):")
    max_name_len = max(len(n) for n in param_names)
    for i, name in enumerate(param_names):
        lo, hi = samples[:, i].min(), samples[:, i].max()
        print(f"  {name:<{max_name_len}} : [{lo:.4g}, {hi:.4g}]")

    # ── Run degen_detector ─────────────────────────────────────────────────────
    run_kwargs = dict(
        coupling_depth=args.coupling_depth,
        max_fits=args.max_fits,
        niterations=args.niterations,
    )

    modes = []
    if not args.log_mode:
        modes.append(("linear", False))
    if not args.linear_only:
        modes.append(("log", True))

    for mode_name, log_mode in modes:
        out_dir = args.out_dir / args.field / mode_name / f"depth_{args.coupling_depth}"
        print(f"\n{'='*60}")
        print(f"Running degen_detector: {mode_name} mode → {out_dir}")
        print(f"{'='*60}")
        run_detector(
            samples,
            param_names,
            output_dir=out_dir,
            log_mode=log_mode,
            **run_kwargs,
        )
        print(f"Done. Results: {out_dir}/summary.txt")


if __name__ == "__main__":
    main()
