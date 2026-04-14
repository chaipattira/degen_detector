#!/usr/bin/env python
# ABOUTME: Generates a 3-panel publication figure for the Planck CMB degeneracy analysis.
# ABOUTME: Shows log-space manifold, and true-vs-predicted diagnostics for each component.
"""Generate 3-panel figure for the Planck 2018 CMB log-mode degeneracy section.

Layout (1 row, 3 panels):
  (a)  2D manifold — log(H0) vs log(Omega_m) with fitted linear degeneracy
  (b)  True vs Predicted for g1(H0)
  (c)  True vs Predicted for g2(Omega_m)

Usage:
    python reproduce/make_figures_planck.py
    # from the project root: /home/x-ctirapongpra/scratch/degen_detector
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from degen_detector.io import load_pickle
from degen_detector.diagnostics.analyzer import FitAnalyzer

# ---------------------------------------------------------------------------
# Publication rcParams (matching make_figures_toy.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'text.usetex': False,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKL_PATH = PROJECT_ROOT / 'outputs' / 'planck_logmode' / '20260410_145705' / 'planck_logmode_result.pkl'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'planck_cmb_degeneracy.pdf'


# ---------------------------------------------------------------------------
# R² helper
# ---------------------------------------------------------------------------
def _compute_r2(true_g, pred_g):
    """Compute R² for true vs predicted arrays."""
    valid = np.isfinite(true_g) & np.isfinite(pred_g)
    if np.sum(valid) == 0:
        return np.nan
    residuals = true_g[valid] - pred_g[valid]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((true_g[valid] - np.mean(true_g[valid])) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

def _draw_manifold(ax, analyzer, samples, param_names):
    """Panel (a): log(H0) vs log(omegam) scatter + fitted line in log space."""
    fit = analyzer.fit
    p1, p2 = fit.param_names   # H0, omegam

    idx1 = param_names.index(p1)
    idx2 = param_names.index(p2)

    x_raw = samples[:, idx1]   # H0 values
    y_raw = samples[:, idx2]   # omegam values
    x_log = np.log(x_raw)      # log(H0)
    y_log = np.log(y_raw)      # log(omegam)

    # Scatter of data in log space
    ax.scatter(x_log, y_log, alpha=0.2, s=4, c='steelblue', label='Planck 2018', zorder=1,
               rasterized=True)

    # Fitted degeneracy is LINEAR in log space.
    # g1(H0) + g2(omegam) = const
    # g1 = a1*log(H0) + b1, g2 = a2*log(omegam) + b2
    # => a2*log(omegam) + b2 = const - (a1*log(H0) + b1)
    # => log(omegam) = (const - b2 - a1*log(H0) - b1) / a2
    # Extract coefficients from the component expressions directly
    import sympy as sp
    sym_H0 = sp.Symbol('H0')
    sym_om = sp.Symbol('omegam')
    expr_H0 = fit.component_exprs[0]   # a1*log(H0) + b1
    expr_om = fit.component_exprs[1]   # a2*log(omegam) + b2

    # Coefficients: expr_H0 = a1*log(H0) + b1
    a1 = float(expr_H0.coeff(sp.log(sym_H0)))
    b1 = float(expr_H0.subs(sym_H0, sp.E))   # at H0=e, log(H0)=1 => a1*1 + b1
    b1 = b1 - a1

    a2 = float(expr_om.coeff(sp.log(sym_om)))
    b2 = float(expr_om.subs(sym_om, sp.E))
    b2 = b2 - a2

    c = fit.constant

    x_range = np.linspace(x_log.min(), x_log.max(), 200)
    # log(omegam) = (c - b2 - a1*x_range - b1) / a2
    y_fit_log = (c - b2 - a1 * x_range - b1) / a2

    valid = np.isfinite(y_fit_log)
    r2 = fit.orthogonal_r2
    r2_str = f'{r2:.4f}' if np.isfinite(r2) else 'N/A'
    ax.plot(x_range[valid], y_fit_log[valid], 'r-', lw=2,
            label=f'Fit ($R^2_\\perp$={r2_str})', zorder=3)

    ax.set_xlabel(r'$\log H_0$', fontsize=12)
    ax.set_ylabel(r'$\log \Omega_m$', fontsize=12)
    ax.set_title(r'$\text{Planck 2018: } \Omega_m h^3 \approx \mathrm{const}\text{ degeneracy}$', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_true_vs_predicted(ax, true_g, pred_g, pname, comp_num):
    """Panels (b) and (c): true vs predicted scatter + y=x reference line."""
    ax.scatter(true_g, pred_g, alpha=0.2, s=4, c='steelblue', zorder=1, rasterized=True)

    all_vals = np.concatenate([true_g[np.isfinite(true_g)], pred_g[np.isfinite(pred_g)]])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label='Perfect fit', zorder=3)

    r2 = _compute_r2(true_g, pred_g)
    r2_str = f'{r2:.4f}' if np.isfinite(r2) else 'N/A'

    # Label mapping for nicer LaTeX display
    label_map = {'H0': r'H_0', 'omegam': r'\Omega_m', 'omegabh2': r'\Omega_b h^2',
                 'omegach2': r'\Omega_c h^2', 'sigma8': r'\sigma_8', 'ns': r'n_s', 'tau': r'\tau'}
    tex = label_map.get(pname, pname)

    ax.set_xlabel(f'True $g_{{{comp_num}}}({tex})$', fontsize=12)
    ax.set_ylabel(f'Predicted $g_{{{comp_num}}}({tex})$', fontsize=12)
    ax.set_title(f'$R^2 = {r2_str}$', fontsize=12)
    ax.legend(loc='best', fontsize=8, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_planck_figure(output_path):
    # Load data
    data = load_pickle(PKL_PATH)
    samples = data['samples']
    param_names = data['param_names']
    result = data['result']

    # Best fit for (H0, omegam)
    best_cf = result.fits[0]
    best_fit = best_cf.fits[0]
    analyzer = FitAnalyzer(best_fit)

    # Compute true vs predicted for both components
    predictions = analyzer.predict_all_components(samples, param_names)
    # predictions = [(true_g, pred_g, pname, comp_num), ...]

    # Build figure: 1 row × 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.14, wspace=0.38)

    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # (a) Log-space manifold
    # ------------------------------------------------------------------
    _draw_manifold(ax_a, analyzer, samples, param_names)
    ax_a.text(-0.10, 1.06, '(a)', transform=ax_a.transAxes, fontsize=12, va='top')

    # ------------------------------------------------------------------
    # (b) True vs Predicted for g1(H0)
    # ------------------------------------------------------------------
    true_g1, pred_g1, pname1, comp1 = predictions[0]
    _draw_true_vs_predicted(ax_b, true_g1, pred_g1, pname1, comp1)
    ax_b.text(-0.10, 1.06, '(b)', transform=ax_b.transAxes, fontsize=12, va='top')

    # ------------------------------------------------------------------
    # (c) True vs Predicted for g2(omegam)
    # ------------------------------------------------------------------
    true_g2, pred_g2, pname2, comp2 = predictions[1]
    _draw_true_vs_predicted(ax_c, true_g2, pred_g2, pname2, comp2)
    ax_c.text(-0.10, 1.06, '(c)', transform=ax_c.transAxes, fontsize=12, va='top')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    make_planck_figure(OUTPUT_PATH)
    print("Done. Figure written to:", OUTPUT_PATH)
