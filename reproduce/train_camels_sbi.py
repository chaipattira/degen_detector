#!/usr/bin/env python
# ABOUTME: Train CNN+NPE ensemble on one CAMELS SB28 field for SBI.
# ABOUTME: Saves posterior.pkl, x_obs.npy, theta_obs_true.npy to --out-dir.
"""Train CNN + NPE ensemble on one CAMELS SB28 field using ltu-ili.

Architecture (agreed in issue #4):
  Conv stack: 5 conv layers with MaxPool, BatchNorm2d after every Conv2d
    Input (1, 256, 256) → flat 128-dim
  FC stack: Dropout(0.2) → Linear(128→64) → Linear(64→64) → Linear(64→32) → Linear(32→64)
  Output: 64-dim embedding

Two density estimators share one embedding net:
  - MAF: 50 hidden features, 5 transforms
  - MDN: 50 hidden features, 6 components

Run from the degen_detector repo root:
    python reproduce/train_camels_sbi.py --field Mcdm \\
        --data-dir /anvil/scratch/x-ctirapongpra/degen_detector/data/camels_sb28 \\
        --out-dir outputs/reproduce/camels_sb28/Mcdm/sbi
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _require(pkg, install_hint):
    try:
        return __import__(pkg)
    except ModuleNotFoundError:
        print(f"ERROR: '{pkg}' not installed. {install_hint}", file=sys.stderr)
        sys.exit(1)


def _load_param_names(params_path: Path) -> list:
    """Read parameter names from header of params file, or generate generic names."""
    with open(params_path) as f:
        first_line = f.readline().strip()
    try:
        float(first_line.split()[0])
        # No header — use generated names
        import warnings
        warnings.warn(
            f"params file has no header; using generic names theta_0..theta_N. "
            "Edit the output summary.txt to rename parameters.",
            UserWarning,
        )
        data = np.loadtxt(params_path)
        return [f"theta_{i}" for i in range(data.shape[1])]
    except ValueError:
        # First line is a header
        return first_line.split()


class ConvNet:
    """CNN embedding network — instantiated as a torch nn.Module."""

    @staticmethod
    def build():
        """Return the agreed 64-dim embedding network for 256×256 maps."""
        import torch.nn as nn

        class _ConvNet(nn.Module):
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
                # 256→126→63→30→15→7→2 spatial; 32 channels → 128 flat
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
                    x = x.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
                x = self.convs(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return _ConvNet()


def main():
    parser = argparse.ArgumentParser(description="Train CNN+NPE on CAMELS SB28 field")
    parser.add_argument("--field", choices=["Mcdm", "Mgas", "Mstar"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/anvil/scratch/x-ctirapongpra/degen_detector/data/camels_sb28"),
    )
    parser.add_argument("--obs-idx", type=int, default=0,
                        help="Simulation index to hold out as the observation")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/reproduce/camels_sb28"))
    parser.add_argument("--device", default="cuda",
                        help="torch device: 'cuda' for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    torch = _require("torch", "pip install torch")
    _require("ili", "pip install ltu-ili")
    _require("sbi", "pip install sbi")
    from ili.inference import SBIRunner
    from ili.dataloaders import NumpyLoader
    from ili.utils import load_nde_sbi
    from sbi.utils import BoxUniform

    # ── Resolve paths ─────────────────────────────────────────────────────────
    maps_path = args.data_dir / f"Maps_{args.field}_IllustrisTNG_SB28_z=0.00.npy"
    params_path = args.data_dir / "params_SB28_IllustrisTNG.txt"
    out_dir = args.out_dir / args.field / "sbi"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Field:     {args.field}")
    print(f"Maps:      {maps_path}")
    print(f"Params:    {params_path}")
    print(f"Obs idx:   {args.obs_idx}")
    print(f"Out dir:   {out_dir}")
    print(f"Device:    {args.device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading maps...")
    maps_raw = np.load(maps_path)           # (30720, 256, 256)
    print(f"  Raw maps shape: {maps_raw.shape}, dtype: {maps_raw.dtype}")

    # log10-transform as per CAMELS convention
    maps = np.log10(maps_raw + 1e-10)      # guard against zeros

    print("Loading params...")
    theta = np.loadtxt(params_path)         # (2048, 28)
    param_names = _load_param_names(params_path)
    print(f"  Params shape: {theta.shape}, {len(param_names)} params: {param_names[:6]}...")

    n_sims = theta.shape[0]
    n_proj = maps.shape[0] // n_sims        # should be 15
    n_params = theta.shape[1]

    # Repeat theta to align with flat map array
    theta_rep = np.repeat(theta, n_proj, axis=0)   # (30720, 28)

    # ── Hold out one simulation as the observation ─────────────────────────────
    obs_start = args.obs_idx * n_proj
    obs_end = obs_start + n_proj

    x_obs = maps[obs_start].astype(np.float32)     # (256, 256) — first projection
    theta_obs = theta[args.obs_idx]                  # (28,)

    mask = np.ones(len(maps), dtype=bool)
    mask[obs_start:obs_end] = False
    x_train = maps[mask].astype(np.float32)         # (30720-15, 256, 256)
    theta_train = theta_rep[mask].astype(np.float32) # (30720-15, 28)

    print(f"\nTraining set: {x_train.shape[0]} maps, {theta_train.shape[1]} params")
    print(f"Observation:  {x_obs.shape} (sim index {args.obs_idx})")

    # Save obs data alongside posterior for the inference step
    np.save(out_dir / "x_obs.npy", x_obs)
    np.save(out_dir / "theta_obs_true.npy", theta_obs)
    np.save(out_dir / "param_names.npy", np.array(param_names))
    print(f"Saved x_obs.npy, theta_obs_true.npy, param_names.npy → {out_dir}")

    # ── Build prior from training-set parameter ranges ─────────────────────────
    theta_min = torch.tensor(theta_train.min(axis=0), dtype=torch.float32)
    theta_max = torch.tensor(theta_train.max(axis=0), dtype=torch.float32)
    prior = BoxUniform(low=theta_min, high=theta_max, device=args.device)
    print(f"\nPrior bounds (first 4 params): {theta_min[:4].tolist()} → {theta_max[:4].tolist()}")

    # ── Build embedding network ───────────────────────────────────────────────
    embed_net = ConvNet.build()

    # Dry-run check before committing to training
    import torch as _torch
    dummy = _torch.zeros(1, 1, 256, 256)
    out_shape = embed_net(dummy).shape
    assert out_shape == (1, 64), f"ConvNet output shape mismatch: {out_shape}"
    print(f"ConvNet forward-pass check: input (1,1,256,256) → output {tuple(out_shape)} ✓")

    # ── Build density estimators (MAF + MDN sharing one embedding net) ─────────
    maf = load_nde_sbi('NPE', 'maf', embedding_net=embed_net,
                        hidden_features=50, num_transforms=5)
    mdn = load_nde_sbi('NPE', 'mdn', embedding_net=embed_net,
                        hidden_features=50, num_components=6)

    # ── Train ─────────────────────────────────────────────────────────────────
    runner = SBIRunner(
        prior=prior,
        engine='NPE',
        nets=[maf, mdn],
        train_args=dict(
            training_batch_size=64,
            learning_rate=5e-4,
            stop_after_epochs=30,
        ),
        out_dir=out_dir,
        device=args.device,
    )

    loader = NumpyLoader(x=x_train, theta=theta_train)
    print(f"\nStarting training on {args.device}...")
    posterior_ensemble, _ = runner(loader, seed=42)
    print("Training complete.")
    print(f"Saved posterior.pkl → {out_dir}")

    # ── Sampling sanity check ─────────────────────────────────────────────────
    x_obs_tensor = _torch.tensor(x_obs[None], dtype=_torch.float32).to(args.device)
    test_samples = posterior_ensemble.sample((100,), x=x_obs_tensor)
    assert test_samples.shape == (100, n_params), \
        f"Posterior sample shape mismatch: {test_samples.shape}"
    samples_np = test_samples.cpu().numpy()
    all_in_bounds = (
        (samples_np >= theta_min.numpy() - 0.1).all() and
        (samples_np <= theta_max.numpy() + 0.1).all()
    )
    print(f"Sanity check: 100 posterior samples shape {test_samples.shape} — "
          f"in-bounds: {all_in_bounds}")
    if not all_in_bounds:
        print("  WARNING: some samples outside prior bounds (may be OK near edges)")


if __name__ == "__main__":
    main()
