"""Command-line interface for degen_detector.

Usage:
    degen-detect <source> --format <fmt> [options]

Formats:
    getdist   getdist/CosmoMC chain stem (requires --params)
    emcee     emcee HDFBackend HDF5 file
    numpy     .npy or .npz file (requires --param-names)

Examples:
    degen-detect data/planck/base_plik --format getdist \\
        --params sigma8 omegam H0 --output-dir out/planck --log-mode

    degen-detect chains.h5 --format emcee \\
        --params sigma8 omegam --burn-in 200 --thin 5 --output-dir out/emcee

    degen-detect samples.npy --format numpy \\
        --param-names theta1 theta2 theta3 --output-dir out/synth
"""

# Must come before any matplotlib import — ensures Agg backend on headless HPC nodes
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

from degen_detector.loaders import load_getdist, load_emcee, load_numpy
from degen_detector.pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        prog="degen-detect",
        description="Detect parameter degeneracies in posterior samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", type=Path, help="Chain root, HDF5 file, or .npy/.npz file")
    parser.add_argument(
        "--format", required=True, choices=["getdist", "emcee", "numpy"],
        help="Input format",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Output directory (default: ./outputs)")
    # getdist / emcee params
    parser.add_argument("--params", nargs="+", metavar="PARAM",
                        help="Parameter names to extract (required for --format getdist)")
    parser.add_argument("--ignore-rows", type=float, default=0.3,
                        help="Burn-in fraction for getdist (default: 0.3)")
    # emcee options
    parser.add_argument("--burn-in", type=int, default=0,
                        help="Steps to discard as burn-in for emcee (default: 0)")
    parser.add_argument("--thin", type=int, default=1,
                        help="Thinning factor for emcee (default: 1)")
    # numpy options
    parser.add_argument("--param-names", nargs="+", metavar="NAME",
                        help="Parameter names (required for --format numpy)")
    # pipeline options
    parser.add_argument("--log-mode", action="store_true",
                        help="Use DegenLogMode (log-transforms positive params)")
    parser.add_argument("--coupling-depth", type=int, default=2)
    parser.add_argument("--max-fits", type=int, default=2)
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Conditional required argument enforcement
    if args.format == "getdist" and not args.params:
        parser.error("--params is required when --format getdist")
    if args.format == "numpy" and not args.param_names:
        parser.error("--param-names is required when --format numpy")

    # Load samples
    if args.format == "getdist":
        samples, param_names = load_getdist(
            args.source, args.params, ignore_rows=args.ignore_rows
        )
    elif args.format == "emcee":
        samples, param_names = load_emcee(
            args.source, params=args.params, burn_in=args.burn_in, thin=args.thin
        )
    else:  # numpy
        samples, param_names = load_numpy(args.source, args.param_names)

    run_pipeline(
        samples,
        param_names,
        args.output_dir,
        log_mode=args.log_mode,
        coupling_depth=args.coupling_depth,
        max_fits=args.max_fits,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
