#!/usr/bin/env python
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from scipy.stats import gaussian_kde

CHAIN = Path(__file__).parent.parent / "data/des_y3_3x2pt/chain_3x2pt_lcdm_SR_maglim.txt"
OUT   = Path(__file__).parent.parent / "outputs/reproduce/des_y3_logmode/corner_annotated.pdf"
N_RESAMPLE = 20_000
RNG = np.random.default_rng(42)

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         14,
    "axes.labelsize":    17,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "axes.linewidth":    1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "pdf.fonttype":      42,
})

C_POST  = "#2166AC"
C_FILL1 = "#6BAED6"
C_FILL2 = "#2166AC"
C_EQ    = "#D73027"
C_S8    = "#4DAC26"

# ── Load & resample ────────────────────────────────────────────────────────────
data    = np.loadtxt(CHAIN, comments="#")
weights = data[:, -1];  weights /= weights.sum()
idx = RNG.choice(len(data), size=N_RESAMPLE, p=weights)
om  = data[idx, 0]
s8  = data[idx, 31]
S8  = s8 * np.sqrt(om / 0.3)

# ── KDE helpers ───────────────────────────────────────────────────────────────
def make_kde2d(x, y, grid_pts=350, bw=0.22):
    fn = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    xg = np.linspace(x.min(), x.max(), grid_pts)
    yg = np.linspace(y.min(), y.max(), grid_pts)
    xx, yy = np.meshgrid(xg, yg)
    zz = fn(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zflat  = np.sort(zz.ravel())[::-1]
    cum    = np.cumsum(zflat) / zflat.sum()
    return xx, yy, zz, zflat[np.searchsorted(cum, 0.95)], zflat[np.searchsorted(cum, 0.68)]

def kde_1d(x, bw=0.22, grid_pts=300):
    fn  = gaussian_kde(x, bw_method=bw)
    pad = 0.12 * (x.max() - x.min())
    xg  = np.linspace(x.min() - pad, x.max() + pad, grid_pts)
    return xg, fn(xg)

def padded(arr, lo=0.5, hi=99.5, frac=0.12):
    a, b = np.percentile(arr, lo), np.percentile(arr, hi)
    m = frac * (b - a)
    return a - m, b + m

def add_contours(ax, x, y, bw=0.22):
    xx, yy, zz, l95, l68 = make_kde2d(x, y, bw=bw)
    ax.contourf(xx, yy, zz, levels=[l95, zz.max()], colors=[C_FILL1], alpha=0.35)
    ax.contourf(xx, yy, zz, levels=[l68, zz.max()], colors=[C_FILL2], alpha=0.55)
    cs = ax.contour(xx, yy, zz, levels=[l95, l68], colors=[C_POST], linewidths=1.5)
    return cs, l95

def clip_and_plot(ax, cs, om_line, y_line, color, lw, ls, label):
    paths = cs.get_paths()
    clip  = PathPatch(paths[0], transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    ln, = ax.plot(om_line, y_line, color=color, lw=lw, ls=ls, label=label)
    ln.set_clip_path(clip)
    return ln

# ── Degeneracy curves ─────────────────────────────────────────────────────────
def eq_detected_s8(om_arr):        # σ8(Ωm) from detected equation
    lom = np.log(om_arr)
    return np.exp((0.7243*lom**2 - 8.7895*lom - 15.3140) / 15.8186)

S8_fid = np.median(S8)
def eq_s8_sigma8(om_arr):          # σ8(Ωm) for S8 = const
    return S8_fid / np.sqrt(om_arr / 0.3)

om_line = np.linspace(np.percentile(om, 0.2), np.percentile(om, 99.8), 500)

# Curves in (Ωm, S8) space
def eq_detected_S8(om_arr):        # S8 = σ8_detected * sqrt(Ωm/0.3)
    return eq_detected_s8(om_arr) * np.sqrt(om_arr / 0.3)

def eq_s8_S8(om_arr):              # S8 = S8_fid (horizontal)
    return np.full_like(om_arr, S8_fid)

# ── Axis limits ───────────────────────────────────────────────────────────────
xlim   = padded(om)
ylim1  = padded(s8)
ylim2  = padded(S8)

# ── Figure: 3 rows × 2 cols ───────────────────────────────────────────────────
# Row 0: Ωm marginal (shared)
# Row 1: 2D(Ωm, σ8) | σ8 marginal
# Row 2: 2D(Ωm, S8) | S8 marginal
fig = plt.figure(figsize=(7, 11))
gs  = fig.add_gridspec(3, 2,
                       height_ratios=[0.7, 2.5, 2.5],
                       width_ratios=[2.8, 1],
                       hspace=0.08, wspace=0.06)

ax_om   = fig.add_subplot(gs[0, 0])
ax_s8p  = fig.add_subplot(gs[1, 0])           # σ8 vs Ωm
ax_s8m  = fig.add_subplot(gs[1, 1])           # σ8 marginal
ax_S8p  = fig.add_subplot(gs[2, 0], sharex=ax_s8p)   # S8 vs Ωm
ax_S8m  = fig.add_subplot(gs[2, 1])           # S8 marginal
fig.add_subplot(gs[0, 1]).set_visible(False)

# ── Ωm marginal (top) ─────────────────────────────────────────────────────────
xg_om, pg_om = kde_1d(om)
ax_om.fill_between(xg_om, pg_om, alpha=0.22, color=C_POST)
ax_om.plot(xg_om, pg_om, color=C_POST, lw=2)
ax_om.set_xlim(xlim)
ax_om.set_xticks([]);  ax_om.set_yticks([])
for sp in ax_om.spines.values(): sp.set_visible(False)

# ── σ8 marginal (right of top 2D panel) ───────────────────────────────────────
xg_s8, pg_s8 = kde_1d(s8)
ax_s8m.fill_betweenx(xg_s8, pg_s8, alpha=0.22, color=C_POST)
ax_s8m.plot(pg_s8, xg_s8, color=C_POST, lw=2)
ax_s8m.set_ylim(ylim1)
ax_s8m.set_xticks([]);  ax_s8m.set_yticks([])
for sp in ax_s8m.spines.values(): sp.set_visible(False)

# ── S8 marginal (right of bottom 2D panel) ────────────────────────────────────
xg_S8, pg_S8 = kde_1d(S8)
ax_S8m.fill_betweenx(xg_S8, pg_S8, alpha=0.22, color=C_POST)
ax_S8m.plot(pg_S8, xg_S8, color=C_POST, lw=2)
ax_S8m.set_ylim(ylim2)
ax_S8m.set_xticks([]);  ax_S8m.set_yticks([])
for sp in ax_S8m.spines.values(): sp.set_visible(False)

# ── Top 2D panel: σ8 vs Ωm ───────────────────────────────────────────────────
cs1, _ = add_contours(ax_s8p, om, s8)
l1 = clip_and_plot(ax_s8p, cs1, om_line, eq_detected_s8(om_line),
                   C_EQ, 2.5, "--", "Recovered eqn.")
l2 = clip_and_plot(ax_s8p, cs1, om_line, eq_s8_sigma8(om_line),
                   C_S8, 2.0, ":", r"$\sigma_8\,(\Omega_m/0.3)^{0.5} \equiv S_8$")
ax_s8p.set_ylabel(r"$\sigma_8$")
ax_s8p.set_xlim(xlim);  ax_s8p.set_ylim(ylim1)
ax_s8p.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax_s8p.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax_s8p.tick_params(which="both", direction="in")
plt.setp(ax_s8p.get_xticklabels(), visible=False)   # hide — shared with bottom

# ── Bottom 2D panel: S8 vs Ωm ─────────────────────────────────────────────────
cs2, _ = add_contours(ax_S8p, om, S8, bw=0.35)
l3 = clip_and_plot(ax_S8p, cs2, om_line, eq_detected_S8(om_line),
                   C_EQ, 2.5, "--", "Recovered eqn.")
l4 = clip_and_plot(ax_S8p, cs2, om_line, eq_s8_S8(om_line),
                   C_S8, 2.0, ":", r"$\sigma_8\,(\Omega_m/0.3)^{0.5} \equiv S_8$")
ax_S8p.set_xlabel(r"$\Omega_m$")
ax_S8p.set_ylabel(r"$S_8$")
ax_S8p.set_xlim(xlim);  ax_S8p.set_ylim(ylim2)
ax_S8p.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax_S8p.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
ax_S8p.tick_params(which="both", direction="in")

# ── Shared legend below both panels ───────────────────────────────────────────
fig.legend([l1, l2], [l1.get_label(), l2.get_label()],
           loc="lower center", bbox_to_anchor=(0.38, -0.02),
           fontsize=13, ncol=1, framealpha=0.94,
           edgecolor="0.75", handlelength=2.6)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
