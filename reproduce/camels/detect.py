#!/usr/bin/env python
# ABOUTME: Run degen_detector on pre-drawn posterior samples for a CAMELS SB28 field.
# ABOUTME: Expects samples.npy + param_names.npy from sample_camels_sb28.py.
"""Run degen_detector on posterior samples drawn by sample_camels_sb28.py.

Runs run_detector() in linear and/or log mode; writes outputs to
{out_dir}/{field}/{linear,log}/depth_{coupling_depth}/.

Run from the degen_detector repo root:
    python reproduce/run_camels_sb28.py --field Mcdm \\
        --out-dir outputs/reproduce/camels_sb28
"""

import argparse
from pathlib import Path

import numpy as np

from degen_detector import run_detector


def main():
    parser = argparse.ArgumentParser(
        description="Run degen_detector on pre-drawn CAMELS SB28 posterior samples"
    )
    parser.add_argument("--field", choices=["Mcdm", "Mgas", "Mstar"], required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reproduce/camels_sb28"),
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=None,
        help="Path to samples.npy. Defaults to {out_dir}/{field}/samples.npy",
    )
    parser.add_argument("--coupling-depth", type=int, default=2)
    parser.add_argument("--max-fits", type=int, default=5)
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--log-mode", action="store_true",
                        help="Run only log mode (skip linear); default: run both")
    parser.add_argument("--linear-only", action="store_true",
                        help="Run only linear mode (skip log); default: run both")
    args = parser.parse_args()

    field_dir = args.out_dir / args.field
    samples_path = args.samples_file or (field_dir / "samples.npy")
    param_names_path = field_dir / "param_names.npy"

    if not samples_path.exists():
        raise FileNotFoundError(
            f"samples.npy not found at {samples_path}. "
            "Run sample_camels_sb28.py first."
        )
    if not param_names_path.exists():
        raise FileNotFoundError(
            f"param_names.npy not found at {param_names_path}. "
            "Run sample_camels_sb28.py first."
        )

    samples = np.load(samples_path)
    param_names = list(np.load(param_names_path, allow_pickle=True))

    print(f"Field:          {args.field}")
    print(f"Samples:        {samples_path}  {samples.shape}")
    print(f"Params ({len(param_names)}):    {param_names[:6]}...")
    print(f"coupling_depth: {args.coupling_depth}")

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

        run_samples = samples
        run_param_names = param_names
        if log_mode:
            keep = [i for i in range(len(param_names)) if samples[:, i].min() > 0]
            excluded = [param_names[i] for i in range(len(param_names)) if i not in keep]
            if excluded:
                print(f"  Log mode: excluding params with non-positive values: {excluded}")
                run_samples = samples[:, keep]
                run_param_names = [param_names[i] for i in keep]

        run_detector(
            run_samples,
            run_param_names,
            output_dir=out_dir,
            log_mode=log_mode,
            **run_kwargs,
        )
        print(f"Done. Results: {out_dir}/summary.txt")


if __name__ == "__main__":
    main()
