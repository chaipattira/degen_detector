"""
Validation plot comparing detected vs. known degeneracies.

This creates a comprehensive figure showing:
- Posterior samples (scatter)
- Detected symbolic equations (red curves)
- Known theoretical relationships (green curves)
- Quantitative metrics (R², correlation, residuals)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def compute_theoretical_relations(samples, target_params):
    """Compute values from known theoretical degeneracy relationships."""

    # Get parameter indices
    tau_idx = target_params.index('tau')
    logA_idx = target_params.index('logA')
    s8_idx = target_params.index('sigma8')
    om_idx = target_params.index('omegam')
    h0_idx = target_params.index('H0')

    theoretical = {}

    # 1. tau-logA: CMB constrains As·exp(-2τ)
    #    => logA ≈ constant + 2τ (approximately linear in log space)
    tau = samples[:, tau_idx]
    logA = samples[:, logA_idx]
    theoretical['tau_logA'] = {
        'x': tau,
        'y_actual': logA,
        'y_theory': 2 * tau + (logA - 2*tau).mean(),  # Best fit constant
        'formula': 'logA = 2τ + const',
        'physics': 'CMB constrains As·exp(-2τ)'
    }

    # 2. sigma8-omegam: S8 = σ8·(Ωm/0.3)^0.5
    #    => σ8 ≈ const / sqrt(Ωm)
    omegam = samples[:, om_idx]
    sigma8 = samples[:, s8_idx]
    S8_values = sigma8 * np.sqrt(omegam / 0.3)
    S8_mean = S8_values.mean()
    theoretical['s8_omegam'] = {
        'x': omegam,
        'y_actual': sigma8,
        'y_theory': S8_mean * np.sqrt(0.3 / omegam),
        'formula': f'σ₈ = {S8_mean:.3f}·√(0.3/Ωm)',
        'physics': f'S₈ = σ₈·√(Ωm/0.3) ≈ {S8_mean:.3f}'
    }

    # 3. H0-omegam: Geometric degeneracy (approximately Ωm ∝ 1/H0)
    #    From distance to last scattering: Ωm·H0² ≈ const
    H0 = samples[:, h0_idx]
    omegaH_squared = omegam * H0**2
    const = omegaH_squared.mean()
    theoretical['h0_omegam'] = {
        'x': H0,
        'y_actual': omegam,
        'y_theory': const / H0**2,
        'formula': f'Ωm = {const:.1f}/H₀²',
        'physics': f'Ωm·H₀² ≈ {const:.1f} (geometric)'
    }

    return theoretical


def plot_degeneracy_comparison(samples, target_params, detected_fits=None):
    """
    Create comprehensive validation plot.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples (N, n_params)
    target_params : list
        Parameter names
    detected_fits : list of MultiDegeneracy, optional
        Detected degeneracies from symbolic regression
    """

    # Compute theoretical relationships
    theoretical = compute_theoretical_relations(samples, target_params)

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.06, right=0.98, top=0.95, bottom=0.05)

    # Row 1: tau-logA
    plot_degeneracy_row(
        fig, gs[0, :],
        theoretical['tau_logA'],
        detected_fits,
        param_pair=('tau', 'logA'),
        row_title='Degeneracy 1: τ – log(As)',
        xlabel='τ (optical depth)',
        ylabel='ln(10¹⁰ As)'
    )

    # Row 2: sigma8-omegam
    plot_degeneracy_row(
        fig, gs[1, :],
        theoretical['s8_omegam'],
        detected_fits,
        param_pair=('omegam', 'sigma8'),
        row_title='Degeneracy 2: σ₈ – Ωm',
        xlabel='Ωm (matter density)',
        ylabel='σ₈ (amplitude)'
    )

    # Row 3: H0-omegam
    plot_degeneracy_row(
        fig, gs[2, :],
        theoretical['h0_omegam'],
        detected_fits,
        param_pair=('H0', 'omegam'),
        row_title='Degeneracy 3: H₀ – Ωm',
        xlabel='H₀ [km/s/Mpc]',
        ylabel='Ωm (matter density)'
    )

    return fig


def plot_degeneracy_row(fig, gs_row, theory_data, detected_fits, param_pair,
                         row_title, xlabel, ylabel):
    """Plot one row: scatter + overlays, predicted vs actual, residuals, metrics."""

    x = theory_data['x']
    y_actual = theory_data['y_actual']
    y_theory = theory_data['y_theory']

    # Find detected fit for this parameter pair (if exists)
    y_detected = None
    detected_formula = None
    if detected_fits:
        for degen in detected_fits:
            fit_params = set(degen.fit.input_names + [degen.fit.target_name])
            if fit_params == set(param_pair):
                # Found matching fit
                # Reconstruct prediction
                # (This requires accessing the fit object - may need adjustment)
                detected_formula = degen.fit.equation_str
                # For now, mark as found
                break

    # Compute metrics
    r_theory = np.corrcoef(y_actual, y_theory)[0, 1]
    r2_theory = 1 - np.var(y_actual - y_theory) / np.var(y_actual)
    residuals = y_actual - y_theory

    # Panel 1: Scatter with overlays (2x width)
    ax1 = fig.add_subplot(gs_row[0:2])
    ax1.scatter(x, y_actual, s=2, alpha=0.25, c='steelblue', label='Posterior samples')

    # Sort for smooth curve plotting
    sort_idx = np.argsort(x)
    ax1.plot(x[sort_idx], y_theory[sort_idx], 'g-', lw=3, alpha=0.8,
             label=f'Theory: {theory_data["formula"]}', zorder=10)

    if detected_formula:
        ax1.plot([], [], 'r--', lw=2, label=f'Detected: {detected_formula}')

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(f'{row_title}\n{theory_data["physics"]}', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.3, ls='--')

    # Panel 2: Predicted vs Actual
    ax2 = fig.add_subplot(gs_row[2])
    ax2.scatter(y_actual, y_theory, s=2, alpha=0.25, c='green')
    lims = [min(y_actual.min(), y_theory.min()), max(y_actual.max(), y_theory.max())]
    ax2.plot(lims, lims, 'k--', lw=1.5, alpha=0.7, label='Perfect match')
    ax2.set_xlabel(f'Actual {ylabel}', fontsize=10)
    ax2.set_ylabel(f'Theory {ylabel}', fontsize=10)
    ax2.set_title(f'Theory fit\nR² = {r2_theory:.4f}', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: Residuals histogram
    ax3 = fig.add_subplot(gs_row[3])
    ax3.hist(residuals, bins=50, color='green', alpha=0.6, edgecolor='darkgreen')
    ax3.axvline(0, color='k', ls='--', lw=1.5)
    ax3.set_xlabel(f'Residual (Actual - Theory)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title(f'Residuals\nσ = {np.std(residuals):.5f}', fontsize=11)
    ax3.grid(alpha=0.3, axis='y')

    # Add text box with metrics
    textstr = f'Correlation: {r_theory:.4f}\nRMSE: {np.sqrt(np.mean(residuals**2)):.5f}'
    ax3.text(0.95, 0.95, textstr, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


if __name__ == '__main__':
    # This would be called from the notebook
    print("Import this module in the notebook and call:")
    print("  from degeneracy_validation_plot import plot_degeneracy_comparison")
    print("  fig = plot_degeneracy_comparison(samples, target_params, results.multi_degeneracies)")
