#!/usr/bin/env python
# ABOUTME: Draw posterior samples from a saved NPE and write samples.npy to disk.
# ABOUTME: Run this once per field before run_camels_sb28.py.
"""Load a saved NPE posterior, draw N samples, and save to {out_dir}/{field}/samples.npy.

Run from the degen_detector repo root:
    python reproduce/sample_camels_sb28.py --field Mcdm \\
        --sbi-dir outputs/reproduce/camels_sb28/Mcdm/sbi \\
        --out-dir outputs/reproduce/camels_sb28 \\
        --n-samples 10000
"""

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn


class _ConvNet(nn.Module):
    """64-dim CNN embedding for 256×256 maps — must match train.py exactly for pickle."""

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _load_param_names(params_file: Path, sbi_dir: Path) -> list:
    """Read parameter names: try param_names.npy saved by train_camels_sbi.py, then params file header."""
    npy_path = sbi_dir / "param_names.npy"
    if npy_path.exists():
        return list(np.load(npy_path, allow_pickle=True))

    if params_file is not None and params_file.exists():
        with open(params_file) as f:
            first_line = f.readline().strip()
        try:
            float(first_line.split()[0])
        except ValueError:
            return first_line.split()

    if params_file is not None and params_file.exists():
        data = np.loadtxt(params_file)
        import warnings
        warnings.warn("Using generic parameter names theta_0..theta_N.", UserWarning)
        return [f"theta_{i}" for i in range(data.shape[1])]

    raise FileNotFoundError(
        "Cannot determine parameter names: neither param_names.npy in --sbi-dir "
        "nor a --params-file with header was found."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Draw posterior samples from saved NPE and save to disk"
    )
    parser.add_argument("--field", choices=["Mcdm", "Mgas", "Mstar"], required=True)
    parser.add_argument(
        "--sbi-dir",
        type=Path,
        help="Directory with posterior.pkl (output of train_camels_sbi.py). "
             "Defaults to {out_dir}/{field}/sbi",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="Path to params_SB28_IllustrisTNG.txt (for parameter names fallback).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reproduce/camels_sb28"),
    )
    parser.add_argument("--n-samples", type=int, default=10000)
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("ERROR: torch not installed. pip install torch", file=sys.stderr)
        sys.exit(1)

    sbi_dir = args.sbi_dir or (args.out_dir / args.field / "sbi")
    posterior_path = sbi_dir / "posterior.pkl"
    x_obs_path = sbi_dir / "x_obs.npy"
    out_dir = args.out_dir / args.field

    if not posterior_path.exists():
        print(f"ERROR: posterior.pkl not found at {posterior_path}", file=sys.stderr)
        print("Run train_camels_sbi.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Field:     {args.field}")
    print(f"SBI dir:   {sbi_dir}")
    print(f"n_samples: {args.n_samples}")

    param_names = _load_param_names(args.params_file, sbi_dir)
    print(f"Params ({len(param_names)}): {param_names[:6]}...")

    print("\nLoading posterior...")
    # posterior.pkl is a plain pickle (not torch.save) with CUDA-saved storages.
    # Patch _load_from_bytes so each storage is remapped to CPU during pickle.load.
    import torch.storage as _ts
    _orig_lfb = _ts._load_from_bytes
    _ts._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    try:
        with open(posterior_path, "rb") as f:
            posterior_ensemble = pickle.load(f)
    finally:
        _ts._load_from_bytes = _orig_lfb

    x_obs = np.load(x_obs_path)
    x_obs_tensor = torch.tensor(x_obs[None], dtype=torch.float32)

    print(f"Drawing {args.n_samples} posterior samples...")
    with torch.no_grad():
        samples_tensor = posterior_ensemble.sample((args.n_samples,), x=x_obs_tensor)
    samples = samples_tensor.cpu().numpy()
    print(f"Samples shape: {samples.shape}")

    print("\nPosterior sample ranges (sanity check):")
    max_name_len = max(len(n) for n in param_names)
    for i, name in enumerate(param_names):
        lo, hi = samples[:, i].min(), samples[:, i].max()
        print(f"  {name:<{max_name_len}} : [{lo:.4g}, {hi:.4g}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.npy"
    param_names_path = out_dir / "param_names.npy"
    np.save(samples_path, samples)
    np.save(param_names_path, np.array(param_names))
    print(f"\nSaved {samples_path}")
    print(f"Saved {param_names_path}")


if __name__ == "__main__":
    main()
