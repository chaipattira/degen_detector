#!/usr/bin/env python
"""Corner plot for DES Y3 (Omega_m, sigma_8) annotated with all SR degeneracy curves.

Three distinct functional forms from symbolic regression, plus the S_8 = const locus:
  Form 1 (dominant, 3/5 runs, complexity 4): linear in log(Omega_m) and log(sigma_8)
  Form 2 (1/5 runs, complexity 7): quadratic in log(sigma_8)
  Form 3 (1/5 runs, complexity 7): quadratic in log(Omega_m)

Output: docs/figures/des_y3_corner_annotated.pdf
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent.parent.parent   # repo root
CHAIN = ROOT / "data/des_y3_3x2pt/chain_3x2pt_lcdm_SR_maglim.txt"
OUT   = ROOT / "docs/figures/des_y3_corner_annotated.pdf"
N_RESAMPLE = 20_000
RNG = np.random.default_rng(42)

# ── Style ─────────────────────────────────────────────────────────────────────
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

# Curve colours: Form 1 red (dominant), Form 2 orange, Form 3 purple, S8 green
C_F1 = "#D73027"        # solid red
C_F2 = "#F46D43"        # dashed orange
C_F3 = "#762A83"        # dotted purple
C_S8 = "#4DAC26"        # solid green

# ── Load & resample ───────────────────────────────────────────────────────────
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
    """Plot a curve clipped to the 95% posterior contour."""
    paths = cs.get_paths()
    clip  = PathPatch(paths[0], transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    ln, = ax.plot(om_line, y_line, color=color, lw=lw, ls=ls, label=label)
    ln.set_clip_path(clip)
    return ln

# ── Degeneracy curve definitions ──────────────────────────────────────────────
# Form 1 (dominant, 3/5 runs): linear in log(Omega_m) and log(sigma_8)
#   10.4076*log(Omega_m) + 15.8178*log(sigma_8) + 16.2175 = 0
#   => sigma_8 = exp(-(10.4076*log(Omega_m) + 16.2175) / 15.8178)
def form1_sigma8(om_arr):
    lom = np.log(om_arr)
    return np.exp(-(10.4076 * lom + 16.2175) / 15.8178)

# Form 2 (1/5 runs): quadratic in log(sigma_8)
#   10.4076*log(Omega_m) - 0.6806*log(sigma_8)^2 + 15.4252*log(sigma_8) + 16.1609 = 0
#   => -0.6806*y^2 + 15.4252*y + c = 0,  y = log(sigma_8), c = 10.4076*log(Omega_m)+16.1609
#   Roots via quadratic formula (a=-0.6806, b=15.4252):
#     y = (-b ± sqrt(b^2-4ac)) / (2a)
#   Physical (smaller-magnitude) root at typical Omega_m ~ 0.3:
#     y = (-b + sqrt(D)) / (2a)  with a<0 gives y ≈ -0.23, i.e. sigma_8 ≈ 0.79
#     (the other root gives y ≈ +22, which is unphysical)
def form2_sigma8(om_arr):
    lom = np.log(om_arr)
    a = -0.6806
    b =  15.4252
    c = 10.4076 * lom + 16.1609
    discriminant = b**2 - 4.0 * a * c   # = b^2 + 4*0.6806*c, always positive here
    with np.errstate(invalid="ignore"):
        valid = discriminant >= 0
    y = np.full_like(om_arr, np.nan)
    # physical root: (-b + sqrt(D)) / (2a), with a<0 this gives the small (negative-log) root
    y[valid] = (-b + np.sqrt(discriminant[valid])) / (2.0 * a)
    return np.exp(y)

# Form 3 (1/5 runs): quadratic in log(Omega_m)
#   -0.8665*log(Omega_m)^2 + 8.4619*log(Omega_m) + 15.8178*log(sigma_8) + 15.1256 = 0
#   => sigma_8 = exp(-(-0.8665*log(Omega_m)^2 + 8.4619*log(Omega_m) + 15.1256) / 15.8178)
def form3_sigma8(om_arr):
    lom = np.log(om_arr)
    num = -0.8665 * lom**2 + 8.4619 * lom + 15.1256
    return np.exp(-num / 15.8178)

# S_8 = const reference
S8_fid = np.median(S8)
def s8_locus_sigma8(om_arr):
    return S8_fid / np.sqrt(om_arr / 0.3)

# Projections into (Omega_m, S_8) space: S_8 = sigma_8 * sqrt(Omega_m/0.3)
def to_S8(sigma8_fn, om_arr):
    return sigma8_fn(om_arr) * np.sqrt(om_arr / 0.3)

# ── Omega_m grid for curves ───────────────────────────────────────────────────
om_line = np.linspace(np.percentile(om, 0.2), np.percentile(om, 99.8), 500)

# ── Axis limits ───────────────────────────────────────────────────────────────
xlim  = padded(om)
ylim1 = padded(s8)
ylim2 = padded(S8)

# ── Figure: 3 rows × 2 cols ───────────────────────────────────────────────────
# Row 0: Omega_m marginal (spans left col only)
# Row 1: 2D(Omega_m, sigma_8) | sigma_8 marginal
# Row 2: 2D(Omega_m, S_8)    | S_8 marginal
fig = plt.figure(figsize=(7, 11))
gs  = fig.add_gridspec(3, 2,
                       height_ratios=[0.7, 2.5, 2.5],
                       width_ratios=[2.8, 1],
                       hspace=0.08, wspace=0.06)

ax_om  = fig.add_subplot(gs[0, 0])
ax_s8p = fig.add_subplot(gs[1, 0])                       # sigma_8 vs Omega_m
ax_s8m = fig.add_subplot(gs[1, 1])                       # sigma_8 marginal
ax_S8p = fig.add_subplot(gs[2, 0], sharex=ax_s8p)        # S_8 vs Omega_m
ax_S8m = fig.add_subplot(gs[2, 1])                       # S_8 marginal
fig.add_subplot(gs[0, 1]).set_visible(False)

# ── Omega_m marginal (top) ────────────────────────────────────────────────────
xg_om, pg_om = kde_1d(om)
ax_om.fill_between(xg_om, pg_om, alpha=0.22, color=C_POST)
ax_om.plot(xg_om, pg_om, color=C_POST, lw=2)
ax_om.set_xlim(xlim)
ax_om.set_xticks([]);  ax_om.set_yticks([])
for sp in ax_om.spines.values():
    sp.set_visible(False)

# ── sigma_8 marginal (right of top 2D panel) ─────────────────────────────────
xg_s8, pg_s8 = kde_1d(s8)
ax_s8m.fill_betweenx(xg_s8, pg_s8, alpha=0.22, color=C_POST)
ax_s8m.plot(pg_s8, xg_s8, color=C_POST, lw=2)
ax_s8m.set_ylim(ylim1)
ax_s8m.set_xticks([]);  ax_s8m.set_yticks([])
for sp in ax_s8m.spines.values():
    sp.set_visible(False)

# ── S_8 marginal (right of bottom 2D panel) ──────────────────────────────────
xg_S8, pg_S8 = kde_1d(S8)
ax_S8m.fill_betweenx(xg_S8, pg_S8, alpha=0.22, color=C_POST)
ax_S8m.plot(pg_S8, xg_S8, color=C_POST, lw=2)
ax_S8m.set_ylim(ylim2)
ax_S8m.set_xticks([]);  ax_S8m.set_yticks([])
for sp in ax_S8m.spines.values():
    sp.set_visible(False)

# ── Top 2D panel: sigma_8 vs Omega_m ─────────────────────────────────────────
cs1, _ = add_contours(ax_s8p, om, s8)

ln_f1 = clip_and_plot(ax_s8p, cs1, om_line, form1_sigma8(om_line),
                      C_F1, 2.5, "-",  r"Best fit (3/5), BIC$=-4939$")
ln_f2 = clip_and_plot(ax_s8p, cs1, om_line, form2_sigma8(om_line),
                      C_F2, 2.0, "--", r"Variant 1 (1/5), BIC$=-4920$")
ln_f3 = clip_and_plot(ax_s8p, cs1, om_line, form3_sigma8(om_line),
                      C_F3, 2.0, ":",  r"Variant 2 (1/5), BIC$=-4920$")
ln_s8 = clip_and_plot(ax_s8p, cs1, om_line, s8_locus_sigma8(om_line),
                      C_S8, 1.8, "-.", r"$S_8 \equiv \sigma_8\,(\Omega_m/0.3)^{0.5}$")

ax_s8p.set_ylabel(r"$\sigma_8$")
ax_s8p.set_xlim(xlim);  ax_s8p.set_ylim(ylim1)
ax_s8p.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax_s8p.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax_s8p.tick_params(which="both", direction="in")
plt.setp(ax_s8p.get_xticklabels(), visible=False)   # shared x with bottom panel

# ── Bottom 2D panel: S_8 vs Omega_m ──────────────────────────────────────────
cs2, _ = add_contours(ax_S8p, om, S8, bw=0.35)

clip_and_plot(ax_S8p, cs2, om_line, to_S8(form1_sigma8, om_line),
              C_F1, 2.5, "-",  r"Best fit (3/5), BIC$=-4939$")
clip_and_plot(ax_S8p, cs2, om_line, to_S8(form2_sigma8, om_line),
              C_F2, 2.0, "--", r"Variant 1 (1/5), BIC$=-4920$")
clip_and_plot(ax_S8p, cs2, om_line, to_S8(form3_sigma8, om_line),
              C_F3, 2.0, ":",  r"Variant 2 (1/5), BIC$=-4920$")
clip_and_plot(ax_S8p, cs2, om_line, np.full_like(om_line, S8_fid),
              C_S8, 1.8, "-.", r"$S_8 \equiv \sigma_8\,(\Omega_m/0.3)^{0.5}$")

ax_S8p.set_xlabel(r"$\Omega_m$")
ax_S8p.set_ylabel(r"$S_8$")
ax_S8p.set_xlim(xlim);  ax_S8p.set_ylim(ylim2)
ax_S8p.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax_S8p.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
ax_S8p.tick_params(which="both", direction="in")

# ── Legend (below the two 2D panels, left-aligned to the 2D column) ──────────
# Place outside the plotting area to avoid overlapping posterior contours.
legend_handles = [ln_f1, ln_f2, ln_f3, ln_s8]
legend_labels  = [h.get_label() for h in legend_handles]
fig.legend(legend_handles, legend_labels,
           loc="lower center",
           bbox_to_anchor=(0.38, -0.05),
           fontsize=12,
           ncol=1,
           framealpha=0.94,
           edgecolor="0.75",
           handlelength=3.0,
           handleheight=1.1)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
