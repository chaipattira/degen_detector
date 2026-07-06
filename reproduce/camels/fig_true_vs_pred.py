#!/usr/bin/env python3
"""True-vs-predicted scatter: individual params vs. degeneracy directions.

Runs the trained NPE amortised posterior on N_TEST held-out simulations and
computes posterior mean / 68 % uncertainty for every parameter and every
discovered degeneracy direction.  Shows two rows of panels:

  Top row:    individual parameter recovery  (scattered → poorly constrained)
  Bottom row: degeneracy direction recovery   (tight     → well constrained)

Usage
-----
    # first run – samples and caches results, then draws the figure
    python fig_true_vs_pred.py --field Mgas

    # subsequent runs – skip expensive sampling
    python fig_true_vs_pred.py --field Mgas --use-cache

    # more test observations (slower but smoother scatter)
    python fig_true_vs_pred.py --field Mgas --n-test 100 --n-samples 2000

    # explicit GPU
    python fig_true_vs_pred.py --field Mgas --device cuda
"""

import argparse
import io
import os
import pickle
import re
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sympy
from scipy.stats import pearsonr

sys.path.insert(0, "/anvil/scratch/x-ctirapongpra/degen_detector")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = "/anvil/scratch/x-ctirapongpra/degen_detector/data/camels_sb28"
OUTPUTS_ROOT   = "/home/x-ctirapongpra/scratch/degen_detector/outputs/reproduce/camels_sb28"
FIGURES_DIR    = "/anvil/scratch/x-ctirapongpra/degen_detector/figures"

# ---------------------------------------------------------------------------
# CAMELS SB28 metadata
# ---------------------------------------------------------------------------
CAMELS_LATEX = {
    0:  r"\Omega_m",      1:  r"\sigma_8",
    2:  r"A_{\mathrm{SN1}}", 3:  r"A_{\mathrm{AGN1}}",
    4:  r"A_{\mathrm{SN2}}", 5:  r"A_{\mathrm{AGN2}}",
    6:  r"\Omega_b",      7:  r"h",
    8:  r"n_s",           9:  r"t_{\mathrm{sf}}",
    10: r"q_{\mathrm{eos}}",
    11: r"\alpha_{\mathrm{imf}}",
    12: r"M_{\mathrm{snii}}",
    13: r"f_{\mathrm{thwind}}",
    14: r"p_{\mathrm{wind}}",
    15: r"\rho_{\mathrm{wind}}",
    16: r"v_{\mathrm{wind}}",
    17: r"f_{\mathrm{windE}}",
    18: r"Z_{\mathrm{windE}}",
    19: r"n_{\mathrm{windE}}",
    20: r"f_{\mathrm{metals}}",
    21: r"M_{\mathrm{bh,seed}}",
    22: r"f_{\mathrm{bh,acc}}",
    23: r"f_{\mathrm{edd}}",
    24: r"\epsilon_{\mathrm{bh}}",
    25: r"\epsilon_r",
    26: r"\chi_{\mathrm{QSO}}",
    27: r"n_{\mathrm{QSO}}",
}

N_SIM   = 2048
N_PROJ  = 15        # projections per simulation
N_FITS  = 3         # degeneracy directions to display
N_INDIV = 4         # individual params to display

# Okabe-Ito colorblind-safe
BLUE       = "#0072B2"
GREEN      = "#009E73"
VERMILLION = "#D55E00"
GRAY       = "#AAAAAA"

_PUB_RC = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    7,
    "axes.titleweight":  "normal",
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.width": 0.5,
    "ytick.major.size":  2.5,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "lines.linewidth":   1.0,
}


# ---------------------------------------------------------------------------
# CNN definition (must match train.py exactly for unpickling)
# ---------------------------------------------------------------------------
import torch.nn as nn


class _ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8,  kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(8),  nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Helpers: degen-fit
# ---------------------------------------------------------------------------

def _camels_latex(idx: int) -> str:
    return CAMELS_LATEX.get(idx, rf"\theta_{{{idx}}}")


def _fit_param_indices(fit) -> list:
    indices = []
    for pname in fit.param_names:
        try:
            indices.append(int(str(pname).split("_")[1]))
        except (ValueError, IndexError):
            pass
    return indices


def _equation_latex(fit) -> str:
    lhs = sum(fit.component_exprs, sympy.Integer(0))
    for atom in lhs.atoms(sympy.Float):
        lhs = lhs.subs(atom, float(f"{float(atom):.3g}"))
    latex_str = sympy.latex(lhs, full_prec=False)
    latex_str = re.sub(
        r'\\theta_\{(\d+)\}',
        lambda m: _camels_latex(int(m.group(1))),
        latex_str,
    )
    return f"${latex_str} = {fit.constant:.3g}$"


def auto_indiv_indices(result, n_fits: int = N_FITS) -> list:
    seen = []
    for cf in result.fits[:n_fits]:
        if cf.fit is None:
            continue
        for idx in _fit_param_indices(cf.fit):
            if idx not in seen:
                seen.append(idx)
    return seen


# ---------------------------------------------------------------------------
# Load posterior (CPU-safe pickle)
# ---------------------------------------------------------------------------

def load_posterior(field: str):
    import torch
    import torch.storage as _ts

    path = os.path.join(OUTPUTS_ROOT, field, "sbi", "posterior.pkl")
    _orig = _ts._load_from_bytes
    _ts._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    try:
        with open(path, "rb") as fh:
            posterior = pickle.load(fh)
    finally:
        _ts._load_from_bytes = _orig
    return posterior


# ---------------------------------------------------------------------------
# Load maps (memory-mapped, log10-transformed on the fly)
# ---------------------------------------------------------------------------

def load_maps_mmap(field: str):
    path = os.path.join(DATA_DIR, f"Maps_{field}_IllustrisTNG_SB28_z=0.00.npy")
    return np.load(path, mmap_mode="r")   # (30720, 256, 256)


def load_params():
    path = os.path.join(DATA_DIR, "params_SB28_IllustrisTNG.txt")
    return np.loadtxt(path)               # (2048, 28)


# ---------------------------------------------------------------------------
# Load degen result
# ---------------------------------------------------------------------------

def load_result(field: str, mode: str = "linear", depth: int = 2):
    path = os.path.join(OUTPUTS_ROOT, field, mode, f"depth_{depth}", "result.pkl")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data["result"]


# ---------------------------------------------------------------------------
# Sample posteriors for N_test observations
# ---------------------------------------------------------------------------

def sample_test_posteriors(
    posterior,
    maps_mmap,
    theta_all,
    test_sim_indices,
    n_samples: int,
    device: str,
    proj_idx: int = 1,   # use 2nd projection (less likely to be in training)
):
    """Draw posterior samples for each test simulation.

    Returns
    -------
    theta_true : (N_test, 28)
    post_mean  : (N_test, 28)   posterior mean per param
    post_lo    : (N_test, 28)   16th percentile
    post_hi    : (N_test, 28)   84th percentile
    all_samples: list of (n_samples, 28) arrays, one per test obs
    """
    import torch

    n_test = len(test_sim_indices)
    n_params = theta_all.shape[1]

    theta_true = np.zeros((n_test, n_params), dtype=np.float32)
    post_mean  = np.zeros((n_test, n_params), dtype=np.float32)
    post_lo    = np.zeros((n_test, n_params), dtype=np.float32)
    post_hi    = np.zeros((n_test, n_params), dtype=np.float32)
    all_samples = []

    for i, sim_idx in enumerate(test_sim_indices):
        map_flat_idx = sim_idx * N_PROJ + proj_idx
        x_raw = maps_mmap[map_flat_idx].astype(np.float32)      # (256, 256)
        x_log = np.log10(x_raw + 1e-10)
        x_t   = torch.tensor(x_log[None], dtype=torch.float32)  # (1, 256, 256)

        with torch.no_grad():
            samp = posterior.sample((n_samples,), x=x_t)        # (N, 28)
        samp_np = samp.cpu().numpy()

        theta_true[i] = theta_all[sim_idx]
        post_mean[i]  = samp_np.mean(axis=0)
        post_lo[i]    = np.percentile(samp_np, 16, axis=0)
        post_hi[i]    = np.percentile(samp_np, 84, axis=0)
        all_samples.append(samp_np)

        if (i + 1) % 10 == 0 or i == n_test - 1:
            print(f"  sampled {i+1}/{n_test}", flush=True)

    return theta_true, post_mean, post_lo, post_hi, all_samples


# ---------------------------------------------------------------------------
# Evaluate degeneracy directions across test observations
# ---------------------------------------------------------------------------

def eval_degen_directions(result, theta_true, all_samples, n_fits: int = N_FITS):
    """Compute true and posterior-mean degeneracy values.

    Returns
    -------
    fits_used  : list of (cf, orig_indices) for each valid fit
    d_true     : (N_test, n_fits)
    d_mean     : (N_test, n_fits)
    d_lo       : (N_test, n_fits)
    d_hi       : (N_test, n_fits)
    """
    n_test = theta_true.shape[0]
    fits_used = []

    for cf in result.fits[:n_fits]:
        if cf.fit is None:
            continue
        orig_indices = _fit_param_indices(cf.fit)
        fits_used.append((cf, orig_indices))

    n_valid = len(fits_used)
    d_true = np.zeros((n_test, n_valid), dtype=np.float32)
    d_mean = np.zeros((n_test, n_valid), dtype=np.float32)
    d_lo   = np.zeros((n_test, n_valid), dtype=np.float32)
    d_hi   = np.zeros((n_test, n_valid), dtype=np.float32)

    for k, (cf, orig_indices) in enumerate(fits_used):
        fit = cf.fit
        for i in range(n_test):
            theta_sub = theta_true[i, orig_indices][None, :].astype(float)
            d_true[i, k] = float(fit.evaluate(theta_sub)[0])

            samp_sub = all_samples[i][:, orig_indices].astype(float)
            d_samp = fit.evaluate(samp_sub)
            d_mean[i, k] = d_samp.mean()
            d_lo[i, k]   = np.percentile(d_samp, 16)
            d_hi[i, k]   = np.percentile(d_samp, 84)

    return fits_used, d_true, d_mean, d_lo, d_hi


# ---------------------------------------------------------------------------
# Scatter panel helper
# ---------------------------------------------------------------------------

def _scatter_panel(ax, x_true, y_pred, y_lo, y_hi, color,
                   xlabel=None, ylabel=None, title=None):
    yerr_lo = y_pred - y_lo
    yerr_hi = y_hi  - y_pred

    ax.errorbar(
        x_true, y_pred,
        yerr=[yerr_lo, yerr_hi],
        fmt="o", ms=2.5, color=color, alpha=0.55,
        elinewidth=0.6, capsize=0, ecolor=color,
        zorder=2,
    )

    # diagonal
    lo = min(x_true.min(), y_pred.min())
    hi = max(x_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="#BBBBBB", lw=0.7, ls="--", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    # Pearson r
    r, _ = pearsonr(x_true, y_pred)
    ax.text(0.05, 0.93, f"$r={r:.2f}$", transform=ax.transAxes,
            fontsize=6, va="top", ha="left", color=color)

    # axes
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1.5, width=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=6.5, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6.5, labelpad=2)
    if title:
        ax.set_title(title, fontsize=7, pad=3)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(
    field, mode, depth, out_dir,
    theta_true, post_mean, post_lo, post_hi,
    fits_used, d_true, d_mean, d_lo, d_hi,
    result,
):
    plt.rcParams.update(_PUB_RC)

    indiv_indices = auto_indiv_indices(result)
    n_indiv = len(indiv_indices)
    n_degen = len(fits_used)
    n_cols  = max(n_indiv, n_degen)

    if n_cols == 0:
        print(f"  No valid fits for depth={depth}, skipping figure.")
        return

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(6.75, 5.0),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.09,
        hspace=0.65, wspace=0.40,
    )

    # ── Row 0: individual parameters ────────────────────────────────────────
    for col, pidx in enumerate(indiv_indices):
        ax = axes[0, col]
        _scatter_panel(
            ax,
            x_true  = theta_true[:, pidx],
            y_pred  = post_mean[:, pidx],
            y_lo    = post_lo[:, pidx],
            y_hi    = post_hi[:, pidx],
            color   = BLUE,
            xlabel  = f"true ${_camels_latex(pidx)}$",
            ylabel  = "posterior mean" if col == 0 else None,
            title   = f"${_camels_latex(pidx)}$",
        )

    # hide unused top-row axes
    for col in range(n_indiv, n_cols):
        axes[0, col].set_visible(False)

    # ── Row 1: degeneracy directions ─────────────────────────────────────────
    for col, (cf, orig_indices) in enumerate(fits_used):
        ax = axes[1, col]
        fit = cf.fit
        params_str = ", ".join(f"${_camels_latex(i)}$" for i in orig_indices)
        title = f"$d_{{{col+1}}}$: ({params_str})"
        _scatter_panel(
            ax,
            x_true  = d_true[:, col],
            y_pred  = d_mean[:, col],
            y_lo    = d_lo[:, col],
            y_hi    = d_hi[:, col],
            color   = GREEN,
            xlabel  = f"true $d_{{{col+1}}}$",
            ylabel  = "posterior mean" if col == 0 else None,
            title   = title,
        )

    # hide unused bottom-row axes
    for col in range(n_degen, n_cols):
        axes[1, col].set_visible(False)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    n_test = theta_true.shape[0]
    mode_label = "linear" if mode == "linear" else "log-space"
    field_label = {"Mgas": r"$M_\mathrm{gas}$",
                   "Mcdm": r"$M_\mathrm{cdm}$",
                   "Mstar": r"$M_\star$"}[field]
    fig.suptitle(
        f"{field_label}: true vs. posterior mean "
        f"({n_test} test sims, depth-{depth} {mode_label} degens)",
        fontsize=9, fontweight="bold", y=0.97,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fig_true_vs_pred_{field}_{mode}_d{depth}")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png")
    plt.close(fig)
    print(f"Saved: {stem}.pdf")
    print(f"Saved: {stem}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="True-vs-predicted scatter for params and degen directions."
    )
    p.add_argument("--field",     default="Mgas", choices=["Mgas", "Mcdm", "Mstar"])
    p.add_argument("--mode",      default="linear", choices=["linear", "log"])
    p.add_argument("--depths",    default=[2], nargs="+", type=int,
                   choices=[2, 3, 4, 5, 6],
                   help="Coupling depths to plot (one figure per depth)")
    p.add_argument("--n-test",    default=50, type=int,
                   help="Number of held-out simulations to evaluate")
    p.add_argument("--n-samples", default=2000, type=int,
                   help="Posterior samples to draw per observation")
    p.add_argument("--obs-idx",   default=0, type=int,
                   help="Sim index used as the training observation (excluded from test)")
    p.add_argument("--device",    default="cpu",
                   help="torch device: 'cuda' or 'cpu'")
    p.add_argument("--use-cache", action="store_true",
                   help="Load cached sampling results instead of re-running NPE")
    p.add_argument("--out-dir",   default=FIGURES_DIR)
    return p.parse_args()


def _cache_path(field, n_test, n_samples):
    d = os.path.join(OUTPUTS_ROOT, field, "true_vs_pred_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"ntest{n_test}_nsamp{n_samples}.npz")


def main():
    args = parse_args()
    print(f"field={args.field}  mode={args.mode}  depths={args.depths}  "
          f"n_test={args.n_test}  n_samples={args.n_samples}  device={args.device}")

    cache = _cache_path(args.field, args.n_test, args.n_samples)

    if args.use_cache and os.path.exists(cache):
        print(f"Loading cached results from {cache}")
        c = np.load(cache, allow_pickle=True)
        theta_true   = c["theta_true"]
        post_mean    = c["post_mean"]
        post_lo      = c["post_lo"]
        post_hi      = c["post_hi"]
        all_samples  = list(c["all_samples"])
    else:
        print("Loading maps (memory-mapped)...")
        maps_mmap = load_maps_mmap(args.field)
        theta_all = load_params()

        # test sims: exclude obs_idx, pick first n_test from the rest
        all_sims    = [i for i in range(N_SIM) if i != args.obs_idx]
        test_sims   = all_sims[:args.n_test]

        print(f"Loading posterior for field={args.field}...")
        posterior = load_posterior(args.field)

        print(f"Sampling posterior for {args.n_test} test simulations...")
        theta_true, post_mean, post_lo, post_hi, all_samples = sample_test_posteriors(
            posterior, maps_mmap, theta_all, test_sims,
            n_samples=args.n_samples, device=args.device,
        )

        np.savez(
            cache,
            theta_true  = theta_true,
            post_mean   = post_mean,
            post_lo     = post_lo,
            post_hi     = post_hi,
            all_samples = np.array(all_samples),
        )
        print(f"Cached to {cache}")

    for depth in args.depths:
        print(f"\n--- depth={depth} ---")
        result = load_result(args.field, args.mode, depth)

        print("Evaluating degeneracy directions...")
        fits_used, d_true, d_mean, d_lo, d_hi = eval_degen_directions(
            result, theta_true, all_samples, n_fits=N_FITS
        )

        print("Building figure...")
        build_figure(
            field=args.field, mode=args.mode, depth=depth,
            out_dir=args.out_dir,
            theta_true=theta_true,
            post_mean=post_mean, post_lo=post_lo, post_hi=post_hi,
            fits_used=fits_used,
            d_true=d_true, d_mean=d_mean, d_lo=d_lo, d_hi=d_hi,
            result=result,
        )


if __name__ == "__main__":
    main()
