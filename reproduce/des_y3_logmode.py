#!/usr/bin/env python
"""Reproduce: DES Y3 3x2pt LCDM log-space degeneracy.

Data: chain_3x2pt_lcdm_SR_maglim.txt (Abbott et al. 2022, arXiv:2105.13549)
Download: bash scripts/download_obs_chains.sh

Chain format: headerless ASCII, polychord importance weights in last column.
Column layout (CosmoSIS output): Omega_m | h | Omega_b | n_s | ... | sigma_8 | ... | weight

Known degeneracy to recover:
  - sigma_8 * Omega_m^0.5 = S8  =>  log(sigma_8) + 0.5*log(Omega_m) = const

Usage:
    python reproduce/des_y3_logmode.py
    python reproduce/des_y3_logmode.py --max-fits 3 --coupling-depth 3
"""
import argparse
from pathlib import Path

import numpy as np

from degen_detector import run_detector, load_weighted

CHAIN_FILE = (
    Path(__file__).parent.parent
    / "data/des_y3_3x2pt/chain_3x2pt_lcdm_SR_maglim.txt"
)

# Column positions (from chain header; weight is col 37)
# Cosmological (sampled): 0=Omega_m, 1=h, 2=Omega_b, 3=n_s, 4=A_s, 5=OmNuh2
# Nuisance (shear cal 6-9, source photo-z 10-13, lens photo-z 14-21): prior-dominated, skip
# Galaxy bias (well-constrained): 22=b1, 23=b2, 24=b3, 25=b4
# Intrinsic alignment: 26=a1, 27=a2, 28=alpha1, 29=alpha2, 30=bias_ta
# Derived: 31=sigma_8, 32=sigma_12

_COLS = [
    0,  1,  2,  3,  4,  5,   # cosmological
    22, 23, 24, 25,           # galaxy bias b1-b4
    26, 27,                   # IA amplitudes a1, a2
    31,                       # sigma_8 (derived)
]
_NAMES = [
    "Omega_m", "h", "Omega_b", "n_s", "A_s", "OmNuh2",
    "b1", "b2", "b3", "b4",
    "a1_IA", "a2_IA",
    "sigma_8",
]


def main():
    parser = argparse.ArgumentParser(description="DegenLogMode on DES Y3 3x2pt chains")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproduce/des_y3_logmode"))
    parser.add_argument("--coupling-depth", type=int, default=2)
    parser.add_argument("--max-fits", type=int, default=4)
    parser.add_argument("--n-resample", type=int, default=20000)
    args = parser.parse_args()

    param_names = _NAMES

    data = np.loadtxt(CHAIN_FILE, comments="#")
    print(f"  Raw shape: {data.shape[0]} samples × {data.shape[1]} columns")

    samples = load_weighted(data[:, _COLS], data[:, -1], n_samples=args.n_resample, seed=42)
    print(f"  Resampled to {samples.shape[0]} equal-weight samples.")

    run_detector(
        samples,
        param_names,
        output_dir=args.output_dir,
        log_mode=True,
        coupling_depth=args.coupling_depth,
        max_fits=args.max_fits,
    )


if __name__ == "__main__":
    main()
