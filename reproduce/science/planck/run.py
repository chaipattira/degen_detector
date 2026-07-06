#!/usr/bin/env python
"""Reproduce: Planck 2018 TTTEEE+lowl+lowE+lensing log-space degeneracies.

Known degeneracies to recover:
  - sigma8 * Omega_m^0.5 = S8  =>  log(sigma8) + 0.5*log(omegam) = const
  - Omega_m h^3 ~ const        =>  log(omegam) + 3*log(H0) = const
  - A_s * exp(-2*tau) ~ const  =>  log(A) - 2*tau = const  (mixed, found via PySR)

"""
import argparse
from pathlib import Path

from degen_detector import load_posterior, run_detector

CHAIN_ROOT = (
    Path(__file__).parent.parent.parent.parent
    / "data/base/plikHM_TTTEEE_lowl_lowE_lensing"
    / "base_plikHM_TTTEEE_lowl_lowE_lensing"
)

# Derived combos (omegamh3, S8, clamp, etc.) excluded intentionally because we want to find them.
COSMO_PARAMS = [
    "omegabh2",  # Omega_b h^2
    "omegach2",  # Omega_c h^2
    "theta",     # 100*theta_MC (acoustic scale ratio); geometric degeneracy with H0/Omega_m
    "tau",       # optical depth to reionisation
    "logA",      # ln(10^10 A_s); tests library skip-transform logic for log_*/ln_* names
    "ns",        # spectral index
    "H0",        # Hubble constant (derived)
    "omegam",    # Omega_matter (derived)
    "omegamh2",  # Omega_m h^2 (derived)
    "sigma8",    # sigma_8 (derived)
]


def main():
    parser = argparse.ArgumentParser(description="Degen detector on Planck 2018 chains")
    parser.add_argument("--log-mode", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--coupling-depth", type=int, default=3)
    parser.add_argument("--max-fits", type=int, default=3)
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--ignore-rows", type=float, default=0.3)
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "logmode" if args.log_mode else "linmode"
        args.output_dir = Path(f"outputs/reproduce/planck_{suffix}")

    samples, param_names = load_posterior(
        CHAIN_ROOT, params=COSMO_PARAMS, ignore_rows=args.ignore_rows
    )

    run_detector(
        samples,
        param_names,
        output_dir=args.output_dir,
        log_mode=args.log_mode,
        coupling_depth=args.coupling_depth,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
        max_fits=args.max_fits,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
