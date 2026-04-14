#!/usr/bin/env python
# ABOUTME: Generates a single 4-panel publication figure for the toy problems section.
# ABOUTME: Row 0 (S-curve): full pipeline — MI matrix + fitted surface.
# ABOUTME: Row 1 (others):  fitted surface only — Banana (3D) and Cubic (2D).
"""Generate combined 4-panel figure for the toy problems section.

Layout (2×2 GridSpec, width_ratios=[1, 2]):
  A  MI Matrix (S-curve)     |  B  Fitted Surface (S-curve, 3D)
  C  Fitted Surface (Banana) |  D  Fitted Curve   (Cubic, 2D)

Usage:
    python reproduce/make_figures_toy.py
    # from the project root: /home/x-ctirapongpra/scratch/degen_detector
"""

import re
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
# Publication rcParams
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
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {'pkl': 'exp4_scurve_result.pkl',  'label': 'S-curve', 'hero': True},
    {'pkl': 'exp1_banana_result.pkl',  'label': 'Banana',  'hero': False},
    {'pkl': 'exp3_trig_result.pkl',    'label': 'Trig',    'hero': False},
    {'pkl': 'exp2_cubic_result.pkl',   'label': 'Cubic',   'hero': False},
]


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------
def _to_latex(name):
    m = re.match(r'^theta(\d+)$', name)
    if m:
        return r'$\theta_{' + m.group(1) + r'}$'
    if re.match(r'^[a-zA-Z]$', name):
        return f'${name}$'
    return name


# ---------------------------------------------------------------------------
# Axes-level draw helpers (private to this script)
# ---------------------------------------------------------------------------

def _draw_mi(ax, mi_result):
    matrix = mi_result.mi_matrix
    names = mi_result.param_names
    n = len(names)

    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='MI (nats)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_to_latex(p) for p in names], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([_to_latex(p) for p in names], fontsize=9)

    max_val = matrix.max() if matrix.max() != 0 else 1.0
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'black' if val > max_val * 0.5 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)


def _draw_surface_3d(ax, analyzer, samples, param_names):
    from matplotlib.patches import Patch
    fit = analyzer.fit
    p1, p2, p3 = fit.param_names
    x = samples[:, param_names.index(p1)]
    y = samples[:, param_names.index(p2)]
    z = samples[:, param_names.index(p3)]

    ax.scatter(x, y, z, c='steelblue', alpha=0.3, s=5)

    r2 = getattr(fit, 'orthogonal_r2', float('nan'))
    r2_str = f'{r2:.3f}' if np.isfinite(r2) else 'N/A'

    try:
        X_grid, Y_grid = np.meshgrid(
            np.linspace(x.min(), x.max(), 50),
            np.linspace(y.min(), y.max(), 50),
        )
        if analyzer.can_solve_analytically(p3):
            Z_grid = analyzer.solve_for_param(p3, {p1: X_grid, p2: Y_grid})
        else:
            Z_grid = analyzer.solve_for_param(p3, {p1: X_grid, p2: Y_grid},
                                               bounds=(z.min(), z.max()))
        valid_frac = np.sum(np.isfinite(Z_grid)) / Z_grid.size
        if valid_frac > 0.5:
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.4, color='green')
            ax.legend(handles=[
                Patch(facecolor='steelblue', alpha=0.4, label='Data'),
                Patch(facecolor='green',     alpha=0.4, label=f'Fit ($R^2_\\perp$={r2_str})'),
            ], loc='upper left', fontsize=8)
        else:
            ax.text2D(0.5, 0.9, f'Surface: {valid_frac*100:.0f}% valid',
                      transform=ax.transAxes, ha='center', fontsize=8)
    except Exception as e:
        ax.text2D(0.5, 0.9, f'Error: {e}', transform=ax.transAxes, ha='center', fontsize=8)

    ax.set_xlabel(_to_latex(p1), fontsize=12, labelpad=8)
    ax.set_ylabel(_to_latex(p2), fontsize=12, labelpad=8)
    ax.set_zlabel(_to_latex(p3), fontsize=12, labelpad=8)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=20, azim=45)


def _draw_proj_xz(ax, analyzer, samples, param_names):
    """2D projection: p1 vs p3, holding p2 at Q25 / median / Q75."""
    fit = analyzer.fit
    p1, p2, p3 = fit.param_names
    x = samples[:, param_names.index(p1)]
    y = samples[:, param_names.index(p2)]
    z = samples[:, param_names.index(p3)]

    ax.scatter(x, z, alpha=0.3, s=5, c='steelblue', label='Data')

    x_range = np.linspace(x.min(), x.max(), 200)
    y_q25, y_med, y_q75 = np.percentile(y, [25, 50, 75])
    curve_styles = [
        (y_q25, '--', 0.55, f'{_to_latex(p2)}=Q25'),
        (y_med,  '-', 1.0,  f'{_to_latex(p2)}=med'),
        (y_q75, '--', 0.55, f'{_to_latex(p2)}=Q75'),
    ]

    try:
        for p2_val, ls, alpha, label in curve_styles:
            if analyzer.can_solve_analytically(p3):
                z_fit = analyzer.solve_for_param(p3, {p1: x_range, p2: p2_val})
            else:
                z_fit = analyzer.solve_for_param(p3, {p1: x_range, p2: p2_val},
                                                  bounds=(z.min(), z.max()))
            valid = np.isfinite(z_fit)
            if valid.sum() > 10:
                ax.plot(x_range[valid], z_fit[valid], 'r', ls=ls, lw=2,
                        alpha=alpha, label=label)
    except Exception:
        pass

    ax.set_xlabel(_to_latex(p1), fontsize=11)
    ax.set_ylabel(_to_latex(p3), fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_curve_2d(ax, analyzer, samples, param_names):
    fit = analyzer.fit
    p1, p2 = fit.param_names
    x = samples[:, param_names.index(p1)]
    y = samples[:, param_names.index(p2)]

    ax.scatter(x, y, alpha=0.3, s=8, c='steelblue', label='Data', zorder=1)

    x_range = np.linspace(x.min(), x.max(), 200)
    r2 = getattr(fit, 'orthogonal_r2', float('nan'))
    r2_str = f'{r2:.3f}' if np.isfinite(r2) else 'N/A'

    try:
        if analyzer.can_solve_analytically(p2):
            y_fit = analyzer.solve_for_param(p2, {p1: x_range})
            ax.plot(x_range, y_fit, 'r-', lw=2,
                    label=f'Fit ($R^2_\\perp$={r2_str})', zorder=3)
        else:
            y_fit = analyzer.solve_for_param(p2, {p1: x_range}, bounds=(y.min(), y.max()))
            valid = np.isfinite(y_fit)
            if valid.sum() > 10:
                ax.plot(x_range[valid], y_fit[valid], 'r-', lw=2,
                        label=f'Fit ($R^2_\\perp$={r2_str})', zorder=3)
    except Exception as e:
        ax.text(0.5, 0.95, f'Error: {e}', transform=ax.transAxes, ha='center', fontsize=8)

    ax.set_xlabel(_to_latex(p1), fontsize=11)
    ax.set_ylabel(_to_latex(p2), fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def _load(pkl_path):
    data = load_pickle(pkl_path)
    best_cf = data['result'].fits[0]
    return {
        'samples':     data['samples'],
        'param_names': data['param_names'],
        'mi_result':   data['result'].mi_result,
        'analyzer':    FitAnalyzer(best_cf.fit),
        'n_fit':       len(best_cf.param_names),
    }


def make_combined_figure(output_path):
    synthetic_dir = PROJECT_ROOT / 'outputs' / 'synthetic'
    datasets = {exp['label']: _load(synthetic_dir / exp['pkl']) for exp in EXPERIMENTS}

    scurve = datasets['S-curve']
    banana = datasets['Banana']
    trig   = datasets['Trig']
    cubic  = datasets['Cubic']

    # 6-column grid: row 0 = [MI(2col) | 3D(4col)], row 1 = [3D(2col) | 3D(2col) | 2D(2col)]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.2, wspace=0.35)

    # ------------------------------------------------------------------
    # A — S-curve MI matrix  (cols 0-1)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0:2])
    _draw_mi(ax_a, scurve['mi_result'])
    ax_a.set_title('S-curve: MI Matrix', fontsize=11)
    ax_a.text(-0.14, 1.10, '(a)', transform=ax_a.transAxes,
              fontsize=12, va='top')

    # ------------------------------------------------------------------
    # B — S-curve fitted surface (3D)  (cols 2-3)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2:4], projection='3d')
    _draw_surface_3d(ax_b, scurve['analyzer'], scurve['samples'], scurve['param_names'])
    ax_b.set_title(r'S-curve: $(x^3 - 3x) + y + z = 0$', fontsize=11, pad=12)
    ax_b.text2D(-0.05, 1.06, '(b)', transform=ax_b.transAxes,
                fontsize=12, va='top')

    # ------------------------------------------------------------------
    # C — S-curve x vs z projection (2D)  (cols 4-5)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 4:6])
    _draw_proj_xz(ax_c, scurve['analyzer'], scurve['samples'], scurve['param_names'])
    ax_c.set_title(r'S-curve: $x$ vs $z$ projection', fontsize=11)
    ax_c.text(-0.08, 1.08, '(c)', transform=ax_c.transAxes,
              fontsize=12, va='top')

    # ------------------------------------------------------------------
    # D — Banana fitted surface (3D)  (cols 0-1)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0:2], projection='3d')
    _draw_surface_3d(ax_d, banana['analyzer'], banana['samples'], banana['param_names'])
    ax_d.set_title(r'Banana: $2\theta_1^2 + \theta_2^2 - \theta_3 = 0.5$', fontsize=11, pad=12)
    ax_d.text2D(-0.05, 1.06, '(d)', transform=ax_d.transAxes,
                fontsize=12, va='top')

    # ------------------------------------------------------------------
    # E — Trig fitted surface (3D)  (cols 2-3)
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 2:4], projection='3d')
    _draw_surface_3d(ax_e, trig['analyzer'], trig['samples'], trig['param_names'])
    ax_e.set_title(r'Trig: $2\sin(x) + \cos(y) - z = 1$', fontsize=11, pad=12)
    ax_e.text2D(-0.05, 1.06, '(e)', transform=ax_e.transAxes,
                fontsize=12, va='top')

    # ------------------------------------------------------------------
    # F — Cubic fitted curve (2D)  (cols 4-5)
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 4:6])
    _draw_curve_2d(ax_f, cubic['analyzer'], cubic['samples'], cubic['param_names'])
    ax_f.set_title(r'Cubic: $\theta_1 - (10\theta_2)^3 \approx 0$', fontsize=11)
    ax_f.text(-0.08, 1.08, '(f)', transform=ax_f.transAxes,
              fontsize=12, va='top')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    make_combined_figure(OUTPUT_DIR / 'toy_combined.pdf')
    print("Done. Figure written to:", OUTPUT_DIR)
