#!/usr/bin/env python
"""Plot diagnostics for synthetic degeneracy experiments.

Usage:
    cd /home/x-ctirapongpra/scratch
    module load anaconda
    conda activate degen_detector
    python3 /home/x-ctirapongpra/scratch/degen_detector/scripts/plot_synthetic_results.py /home/x-ctirapongpra/scratch/degen_detector/outputs/synthetic_15710222/20260315_091846

This script loads all .pkl result files from a directory and generates comprehensive diagnostic plots including:
- Simple corner plots (no overlays)
- Corner plots with implicit surface overlays
- Residual corner plots
- Component function plots
- Ground truth vs predicted values for ALL component functions
- Equation comparison text file (ground truth vs fitted with R² and R²_ortho)
- Summary comparison across experiments
- 3D visualizations of the constraint manifold
- 2D projections of the constraint
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq

try:
    import dill as pickle
except ImportError:
    import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_component_functions(fit, samples, figsize=None):
    """Plot each component function g_j(x_j) with data points overlaid."""
    k = len(fit.component_exprs)

    if figsize is None:
        figsize = (4 * k, 4)

    fig, axes = plt.subplots(1, k, figsize=figsize)
    if k == 1:
        axes = [axes]

    for i, (expr, pname) in enumerate(zip(fit.component_exprs, fit.param_names)):
        ax = axes[i]

        if isinstance(samples, dict):
            x_data = samples[pname]
        else:
            x_data = samples[:, i] if samples.shape[1] == k else None

        sym = sp.Symbol(pname)
        g_func = sp.lambdify(sym, expr, modules="numpy")

        if x_data is not None:
            x_min, x_max = x_data.min(), x_data.max()
            margin = 0.1 * (x_max - x_min)
            x_range = np.linspace(x_min - margin, x_max + margin, 200)
            g_data = g_func(x_data)
            ax.scatter(x_data, g_data, alpha=0.3, s=5, c="steelblue", label="Data")
        else:
            x_range = np.linspace(-5, 5, 200)

        g_curve = g_func(x_range)
        ax.plot(x_range, g_curve, "r-", lw=2, label=f"$g_{{{i+1}}}$")

        ax.set_xlabel(f"${pname}$", fontsize=12)
        ax.set_ylabel(f"$g_{{{i+1}}}({pname})$", fontsize=12)
        ax.set_title(f"$g_{{{i+1}}}({pname}) = {sp.latex(expr)}$", fontsize=10)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{fit.equation_str}", fontsize=12, y=1.02)
    plt.tight_layout()

    return fig


def plot_corner_simple(samples, param_names, figsize=None, color="steelblue", **corner_kwargs):
    """Create a simple corner plot without any overlays or annotations.

    Parameters
    ----------
    samples : ndarray
        Posterior samples, shape (N, M).
    param_names : list[str]
        Parameter names for the samples.
    figsize : tuple or None
        Figure size.
    color : str
        Base color for samples.

    Returns
    -------
    fig : matplotlib.Figure
        The corner plot figure.
    """
    import corner

    n_params = len(param_names)
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

    fig = corner.corner(
        samples,
        labels=param_names,
        show_titles=True,
        title_kwargs={"fontsize": 10},
        color=color,
        fig=plt.figure(figsize=figsize),
        **corner_kwargs
    )

    plt.tight_layout()
    return fig


def compute_all_component_predictions(fit, samples, param_names):
    """Compute predicted vs true values for ALL component functions.

    For a k-parameter fit with equation sum(g_j(x_j)) = c,
    we evaluate each component function on the true parameter values
    and compare against what's needed to satisfy the constraint.

    Parameters
    ----------
    fit : ImplicitFit
        The fitted implicit surface.
    samples : ndarray
        Sample data of shape (N, M).
    param_names : list[str]
        All parameter names in samples.

    Returns
    -------
    predictions : list of tuples
        List of (true_g, pred_g, param_name) for each component function.
    """
    k = len(fit.param_names)
    predictions = []

    for i in range(k):
        pname = fit.param_names[i]
        param_idx = param_names.index(pname)
        x_vals = samples[:, param_idx]

        # Evaluate this component function on true values
        sym = sp.Symbol(pname)
        g_func = sp.lambdify(sym, fit.component_exprs[i], modules="numpy")
        true_g = g_func(x_vals)

        # Compute sum of all OTHER component functions
        other_sum = np.zeros(samples.shape[0])
        for j in range(k):
            if j != i:
                other_pname = fit.param_names[j]
                other_idx = param_names.index(other_pname)
                other_vals = samples[:, other_idx]

                other_sym = sp.Symbol(other_pname)
                other_g_func = sp.lambdify(other_sym, fit.component_exprs[j], modules="numpy")
                other_sum += other_g_func(other_vals)

        # For this parameter: g_i(x_i) should equal c - sum(g_j(x_j) for j != i)
        pred_g = fit.constant - other_sum

        predictions.append((true_g, pred_g, pname, i+1))

    return predictions


def plot_all_components_true_vs_predicted(fit, samples, param_names):
    """Plot true vs predicted values for ALL component functions.

    Parameters
    ----------
    fit : ImplicitFit
        The fitted implicit surface.
    samples : ndarray
        Sample data.
    param_names : list[str]
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure
    """
    predictions = compute_all_component_predictions(fit, samples, param_names)
    k = len(predictions)

    # Create subplots - arrange in a row
    ncols = min(k, 4)  # Max 4 columns
    nrows = (k + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if k == 1:
        axes = np.array([axes])
    axes = axes.flatten() if k > 1 else axes

    for idx, (true_g, pred_g, pname, comp_num) in enumerate(predictions):
        ax = axes[idx]

        # Plot
        ax.scatter(true_g, pred_g, alpha=0.3, s=10, c='steelblue')

        # Add y=x reference line
        all_vals = np.concatenate([true_g, pred_g])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label='Perfect fit')

        # Compute R^2
        valid = np.isfinite(true_g) & np.isfinite(pred_g)
        if np.sum(valid) > 0:
            residuals = true_g[valid] - pred_g[valid]
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((true_g[valid] - np.mean(true_g[valid])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r2 = np.nan

        ax.set_xlabel(f'True $g_{{{comp_num}}}({pname})$', fontsize=11)
        ax.set_ylabel(f'Predicted $g_{{{comp_num}}}({pname})$', fontsize=11)
        ax.set_title(f'Component {comp_num}: {pname} (R² = {r2:.3f})', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(k, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def save_equation_comparison(ground_truth, fit, output_path):
    """Save a text comparison of ground truth vs fitted equation.

    Parameters
    ----------
    ground_truth : dict
        Dictionary with 'equation' key and optional 'component_functions' key.
    fit : ImplicitFit
        The fitted implicit surface.
    output_path : Path
        Path to save the text file.
    """
    # Ground truth text
    gt_text = f"Ground Truth:\n{ground_truth['equation']}\n\n"
    if 'component_functions' in ground_truth:
        gt_text += "Component functions:\n"
        for comp in ground_truth['component_functions']:
            gt_text += f"  {comp}\n"

    # Fitted equation text
    fit_text = f"\n\nFitted Equation:\n{fit.equation_str}\n"

    # Include both R² metrics if available
    if hasattr(fit, 'r2'):
        fit_text += f"R² = {fit.r2:.4f}\n"
    if hasattr(fit, 'orthogonal_r2'):
        fit_text += f"R²_ortho = {fit.orthogonal_r2:.4f}\n"

    fit_text += "\nComponent functions:\n"
    for i, (expr, pname) in enumerate(zip(fit.component_exprs, fit.param_names)):
        fit_text += f"  g{i+1}({pname}) = {expr}\n"

    full_text = gt_text + fit_text

    with open(output_path, 'w') as f:
        f.write(full_text)


def plot_scurve_manifold(samples, param_names, ground_truth, fitted_eq=None, output_path=None):
    """Create 3D visualization comparing ground truth and fitted constraint surfaces.

    Works for any 3-parameter degeneracy (extracts first 3 degenerate parameters).

    Parameters
    ----------
    samples : ndarray
        Sample data of shape (N, M).
    param_names : list
        Parameter names.
    ground_truth : dict
        Ground truth equation info with 'degenerate_params' key.
    fitted_eq : ImplicitFit, optional
        Fitted equation to overlay.
    output_path : Path, optional
        Path to save the figure.
    """
    # Extract degenerate parameters (first 3)
    degen_params = ground_truth.get('degenerate_params', param_names[:3])
    if len(degen_params) < 3:
        print(f"Warning: Only {len(degen_params)} degenerate params, need 3 for 3D plot")
        return None

    # Use first 3 degenerate parameters
    param1, param2, param3 = degen_params[:3]
    idx1 = param_names.index(param1)
    idx2 = param_names.index(param2)
    idx3 = param_names.index(param3)

    x = samples[:, idx1]
    y = samples[:, idx2]
    z = samples[:, idx3]

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 6))

    # ============================================================================
    # Plot 1: Data points + Ground truth constraint surface
    # ============================================================================
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot data points
    ax1.scatter(x, y, z, c='steelblue', alpha=0.4, s=10, label='Data')

    # Note: Ground truth surface plotting is experiment-specific
    # For now, just show the data
    gt_eq = ground_truth.get('equation', 'N/A')

    ax1.set_xlabel(param1, fontsize=12, labelpad=10)
    ax1.set_ylabel(param2, fontsize=12, labelpad=10)
    ax1.set_zlabel(param3, fontsize=12, labelpad=10)
    ax1.set_title(f'Data Points\nGround Truth: {gt_eq}', fontsize=11, pad=20)
    ax1.view_init(elev=20, azim=45)

    # ============================================================================
    # Plot 2: Data points + Fitted surface (if available)
    # ============================================================================
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot data points
    ax2.scatter(x, y, z, c='steelblue', alpha=0.4, s=10, label='Data')

    from matplotlib.patches import Patch

    if fitted_eq is not None:
        # Compute fitted surface: g₁(x) + g₂(y) + g₃(z) = c
        # Solve for z: g₃(z) = c - g₁(x) - g₂(y)

        try:
            # Get component functions
            g1_expr = fitted_eq.component_exprs[0]  # function of x
            g2_expr = fitted_eq.component_exprs[1]  # function of y
            g3_expr = fitted_eq.component_exprs[2]  # function of z
            const = fitted_eq.constant

            # Create lambdified functions
            x_sym = sp.Symbol(fitted_eq.param_names[0])
            y_sym = sp.Symbol(fitted_eq.param_names[1])
            z_sym = sp.Symbol(fitted_eq.param_names[2])

            g1_func = sp.lambdify(x_sym, g1_expr, modules='numpy')
            g2_func = sp.lambdify(y_sym, g2_expr, modules='numpy')
            g3_func = sp.lambdify(z_sym, g3_expr, modules='numpy')

            # For surface plot, we need to solve for z given x and y
            # g₃(z) = c - g₁(x) - g₂(y)
            # For linear g₃ (like g₃=a*z+b), we can solve analytically

            # Check if g₃ is linear in z
            g3_poly = sp.Poly(g3_expr, z_sym)
            if g3_poly.degree() == 1:
                # Linear: g₃(z) = a*z + b
                coeffs = g3_poly.all_coeffs()
                a = float(coeffs[0])  # coefficient of z
                b = float(coeffs[1]) if len(coeffs) > 1 else 0.0  # constant term

                # Solve: a*z + b = c - g₁(x) - g₂(y)
                # z = [c - g₁(x) - g₂(y) - b] / a

                x_range = np.linspace(x.min(), x.max(), 50)
                y_range = np.linspace(y.min(), y.max(), 50)
                X_grid, Y_grid = np.meshgrid(x_range, y_range)

                # Compute fitted z values
                Z_grid_fitted = (const - g1_func(X_grid) - g2_func(Y_grid) - b) / a

                # Plot fitted surface
                surf_fitted = ax2.plot_surface(X_grid, Y_grid, Z_grid_fitted,
                                              alpha=0.3, color='green',
                                              label='Fitted Surface')

                legend_elements = [
                    Patch(facecolor='steelblue', alpha=0.4, label='Data points'),
                    Patch(facecolor='green', alpha=0.3, label=f'Fitted: R²={fitted_eq.orthogonal_r2:.3f}')
                ]
                ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
            else:
                # Non-linear g₃, just show data
                ax2.text2D(0.5, 0.95, f'Non-linear {param3} function, cannot plot surface',
                          transform=ax2.transAxes, ha='center', fontsize=9)

        except Exception as e:
            ax2.text2D(0.5, 0.95, f'Could not compute fitted surface',
                      transform=ax2.transAxes, ha='center', fontsize=9)

        ax2.set_xlabel(param1, fontsize=12, labelpad=10)
        ax2.set_ylabel(param2, fontsize=12, labelpad=10)
        ax2.set_zlabel(param3, fontsize=12, labelpad=10)
        title = f'Data + Fitted Surface\nR²_ortho = {fitted_eq.orthogonal_r2:.4f}'
        ax2.set_title(title, fontsize=12, pad=20)
        ax2.view_init(elev=20, azim=45)
    else:
        ax2.text(0.5, 0.5, 0.5, 'No fitted equation available',
                ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D visualization to {output_path}")

    return fig


def plot_2param_degeneracy(samples, param_names, ground_truth, fitted_eq=None, output_path=None):
    """Create theta1 vs theta2 plot with fitted equation overlay (for 2-param degeneracies).

    For banana and cubic degeneracies, plots:
    - Data scatter of theta1 vs theta2
    - Ground truth theoretical curve (dashed line)
    - Fitted equation (solid red line)

    Parameters
    ----------
    samples : ndarray
        Sample data of shape (N, M).
    param_names : list
        Parameter names.
    ground_truth : dict
        Ground truth equation info with 'degenerate_params' and 'constraint' keys.
    fitted_eq : ImplicitFit, optional
        Fitted equation to overlay.
    output_path : Path, optional
        Path to save the figure.
    """
    # Extract degenerate parameters (should be 2)
    degen_params = ground_truth.get('degenerate_params', param_names[:2])
    if len(degen_params) != 2:
        print(f"Warning: Expected 2 degenerate params, got {len(degen_params)}")
        return None

    param1, param2 = degen_params[:2]
    idx1 = param_names.index(param1)
    idx2 = param_names.index(param2)

    theta1 = samples[:, idx1]
    theta2 = samples[:, idx2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot data scatter
    ax.scatter(theta1, theta2, alpha=0.3, s=10, c='steelblue', label='Data', zorder=1)

    # Overlay ground truth theoretical curve (if we know it)
    gt_constraint = ground_truth.get('constraint', '')
    if 'banana' in ground_truth.get('figure', '').lower():
        # Banana: theta2 ≈ 2*theta1^2 - 0.5
        theta1_theory = np.linspace(theta1.min(), theta1.max(), 200)
        theta2_theory = 2 * theta1_theory**2 - 0.5
        ax.plot(theta1_theory, theta2_theory, 'k--', linewidth=2,
                label='Ground truth: $\\theta_2 = 2\\theta_1^2 - 0.5$', zorder=2)
    elif 'cubic' in ground_truth.get('figure', '').lower():
        # Cubic: theta1 ≈ (10*theta2)^3
        theta2_theory = np.linspace(theta2.min(), theta2.max(), 200)
        theta1_theory = (10 * theta2_theory)**3
        # Clip to data range for better visualization
        mask = (theta1_theory >= theta1.min()) & (theta1_theory <= theta1.max())
        ax.plot(theta1_theory[mask], theta2_theory[mask], 'k--', linewidth=2,
                label='Ground truth: $\\theta_1 = (10\\theta_2)^3$', zorder=2)

    # Overlay fitted equation (red line)
    if fitted_eq is not None:
        try:
            # For separable fit: g1(theta1) + g2(theta2) = c
            # Solve for theta2 as a function of theta1
            g1_expr = fitted_eq.component_exprs[0]
            g2_expr = fitted_eq.component_exprs[1]
            const = fitted_eq.constant

            theta1_sym = sp.Symbol(fitted_eq.param_names[0])
            theta2_sym = sp.Symbol(fitted_eq.param_names[1])

            g1_func = sp.lambdify(theta1_sym, g1_expr, modules='numpy')
            g2_func = sp.lambdify(theta2_sym, g2_expr, modules='numpy')

            # Check if g2 is a linear polynomial
            is_linear = False
            try:
                g2_poly = sp.Poly(g2_expr, theta2_sym)
                is_linear = (g2_poly.degree() == 1)
            except:
                # Not a polynomial (e.g., contains exp, log, etc.)
                is_linear = False

            # For linear g2: solve g2(theta2) = c - g1(theta1)
            if is_linear:
                # g2 = a*theta2 + b
                coeffs = g2_poly.all_coeffs()
                a = float(coeffs[0])
                b = float(coeffs[1]) if len(coeffs) > 1 else 0.0

                theta1_range = np.linspace(theta1.min(), theta1.max(), 200)
                # theta2 = (c - g1(theta1) - b) / a
                theta2_fitted = (const - g1_func(theta1_range) - b) / a

                ax.plot(theta1_range, theta2_fitted, 'r-', linewidth=3,
                        label=f'Fitted (R²_ortho={fitted_eq.orthogonal_r2:.3f})', zorder=3)
            else:
                # Try numerical inversion for non-linear g2
                # Sample points and solve numerically
                theta1_range = np.linspace(theta1.min(), theta1.max(), 100)
                theta2_fitted = []

                for t1 in theta1_range:
                    target = const - g1_func(t1)

                    # Define equation to solve: g2(theta2) - target = 0
                    def equation(t2):
                        return g2_func(t2) - target

                    try:
                        # Use brentq to find root in data range
                        t2_sol = brentq(equation, theta2.min(), theta2.max())
                        theta2_fitted.append(t2_sol)
                    except:
                        theta2_fitted.append(np.nan)

                theta2_fitted = np.array(theta2_fitted)
                valid = np.isfinite(theta2_fitted)

                if np.sum(valid) > 10:
                    ax.plot(theta1_range[valid], theta2_fitted[valid], 'r-', linewidth=3,
                            label=f'Fitted (R²_ortho={fitted_eq.orthogonal_r2:.3f})', zorder=3)
                else:
                    ax.text(0.5, 0.95, 'Could not plot fitted curve',
                           transform=ax.transAxes, ha='center', fontsize=10)

        except Exception as e:
            print(f"  Warning: Could not plot fitted curve: {e}")
            ax.text(0.5, 0.95, f'Could not plot fitted curve',
                   transform=ax.transAxes, ha='center', fontsize=10)

    ax.set_xlabel(f'${param1}$', fontsize=14)
    ax.set_ylabel(f'${param2}$', fontsize=14)
    ax.set_title(f'{param1} vs {param2}\nGround Truth: {gt_constraint}', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2-param degeneracy plot to {output_path}")

    return fig


def plot_scurve_2d_projections(samples, param_names, ground_truth, fitted_eq=None, output_path=None):
    """Create 2D projection plots showing the constraint.

    Works for any 3-parameter degeneracy (extracts first 3 degenerate parameters).

    Parameters
    ----------
    samples : ndarray
        Sample data of shape (N, M).
    param_names : list
        Parameter names.
    ground_truth : dict
        Ground truth equation info.
    fitted_eq : ImplicitFit, optional
        Fitted equation to overlay.
    output_path : Path, optional
        Path to save the figure.
    """
    # Extract degenerate parameters (first 3)
    degen_params = ground_truth.get('degenerate_params', param_names[:3])
    if len(degen_params) < 3:
        print(f"Warning: Only {len(degen_params)} degenerate params, need 3 for plots")
        return None

    # Use first 3 degenerate parameters
    param1, param2, param3 = degen_params[:3]
    idx1 = param_names.index(param1)
    idx2 = param_names.index(param2)
    idx3 = param_names.index(param3)

    x = samples[:, idx1]
    y = samples[:, idx2]
    z = samples[:, idx3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gt_eq = ground_truth.get('equation', 'N/A')

    # ============================================================================
    # Panel 1: param1 vs param3
    # ============================================================================
    ax = axes[0]
    ax.scatter(x, z, alpha=0.4, s=5, c='steelblue', label='Data')
    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param3, fontsize=12)
    ax.set_title(f'{param1} vs {param3}', fontsize=11)
    ax.grid(True, alpha=0.3)

    # ============================================================================
    # Panel 2: param2 vs param3
    # ============================================================================
    ax = axes[1]
    ax.scatter(y, z, alpha=0.4, s=5, c='steelblue', label='Data')
    ax.set_xlabel(param2, fontsize=12)
    ax.set_ylabel(param3, fontsize=12)
    ax.set_title(f'{param2} vs {param3}', fontsize=11)
    ax.grid(True, alpha=0.3)

    # ============================================================================
    # Panel 3: param1 vs param2 (colored by param3)
    # ============================================================================
    ax = axes[2]
    scatter = ax.scatter(x, y, c=z, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title(f'{param1} vs {param2}\n(colored by {param3})', fontsize=11)
    plt.colorbar(scatter, ax=ax, label=param3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Ground Truth: {gt_eq}', fontsize=12, y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D projections to {output_path}")

    return fig


def plot_experiment_diagnostics(result_data, output_dir):
    """Create all diagnostic plots for a single experiment.

    Parameters
    ----------
    result_data : dict
        Dictionary containing 'samples', 'param_names', 'result', 'ground_truth', 'name'.
    output_dir : Path
        Directory to save plots.
    """
    exp_name = result_data['name']
    samples = result_data['samples']
    param_names = result_data['param_names']
    result = result_data['result']
    ground_truth = result_data['ground_truth']

    # Get the top fit by MI ranking (first valid fit)
    top_cf = next((cf for cf in result.fits if cf.fit is not None), None)
    if top_cf is None:
        print(f"Warning: No valid fit for {exp_name}, skipping plots")
        return
    best_fit = top_cf.fit

    print(f"\nGenerating plots for {exp_name}...")

    # Create experiment-specific output directory
    exp_plot_dir = output_dir / exp_name
    exp_plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Simple corner plot (no overlays or equations)
    print("  - Corner plot")
    fig1 = plot_corner_simple(samples, param_names)
    fig1.savefig(exp_plot_dir / "corner_with_surface.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. Component functions
    print("  - Component functions")
    # Get subset of samples for fitted parameters
    fit_indices = [param_names.index(p) for p in best_fit.param_names]
    fit_samples = samples[:, fit_indices]
    fig2 = plot_component_functions(best_fit, fit_samples)
    fig2.savefig(exp_plot_dir / "component_functions.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # 3. True vs Predicted for all components
    print("  - True vs predicted plot (all components)")
    fig3 = plot_all_components_true_vs_predicted(best_fit, samples, param_names)
    fig3.savefig(exp_plot_dir / "true_vs_predicted_all_components.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # 4. Equation comparison (save as text file)
    print("  - Equation comparison")
    save_equation_comparison(ground_truth, best_fit, exp_plot_dir / "equation_comparison.txt")

    # 5. Special plot for 2-parameter degeneracies (banana and cubic)
    degen_params = ground_truth.get('degenerate_params', param_names[:2])
    if len(degen_params) == 2:
        print("  - theta1 vs theta2 with fitted curve")
        fig_2param = plot_2param_degeneracy(samples, param_names, ground_truth,
                                            fitted_eq=best_fit,
                                            output_path=exp_plot_dir / "theta1_vs_theta2.png")
        if fig_2param:
            plt.close(fig_2param)

    # 6. 3D visualization (if applicable - 3+ degenerate parameters)
    degen_params = ground_truth.get('degenerate_params', param_names[:3])
    if len(degen_params) >= 3:
        print("  - 3D visualization")
        fig_3d = plot_scurve_manifold(samples, param_names, ground_truth,
                                       fitted_eq=best_fit,
                                       output_path=exp_plot_dir / "3d_visualization.png")
        if fig_3d:
            plt.close(fig_3d)

        # 7. 2D projections
        print("  - 2D projections")
        fig_2d = plot_scurve_2d_projections(samples, param_names, ground_truth,
                                            fitted_eq=best_fit,
                                            output_path=exp_plot_dir / "2d_projections.png")
        if fig_2d:
            plt.close(fig_2d)

    print(f"  Saved all plots to {exp_plot_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for synthetic experiments"
    )
    parser.add_argument(
        "pkl_dir",
        type=Path,
        help="Directory containing .pkl result files",
    )
    args = parser.parse_args()

    pkl_dir = args.pkl_dir
    if not pkl_dir.exists():
        print(f"Error: Directory {pkl_dir} does not exist")
        return

    if not pkl_dir.is_dir():
        print(f"Error: {pkl_dir} is not a directory")
        return

    # Find all .pkl files in the directory
    pkl_files = list(pkl_dir.glob("*.pkl"))

    if not pkl_files:
        print(f"Error: No .pkl files found in {pkl_dir}")
        return

    print(f"Found {len(pkl_files)} .pkl file(s) in {pkl_dir}")

    # Load results from all pkl files
    all_results = {}
    for pkl_file in pkl_files:
        print(f"\nLoading {pkl_file.name}...")
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # Handle both single experiment and multi-experiment formats
            if isinstance(data, dict):
                if 'samples' in data and 'result' in data:
                    # Single experiment format
                    exp_name = pkl_file.stem
                    all_results[exp_name] = data
                    if 'name' not in data:
                        data['name'] = exp_name
                else:
                    # Multi-experiment format (dictionary of experiments)
                    all_results.update(data)
            else:
                print(f"  Warning: Unexpected format in {pkl_file.name}, skipping")
        except Exception as e:
            print(f"  Error loading {pkl_file.name}: {e}")
            continue

    if not all_results:
        print("Error: No valid experiment data loaded")
        return

    print(f"\nLoaded {len(all_results)} experiment(s) total")

    # Create plots directory
    plots_dir = pkl_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate diagnostics for each experiment
    for key, result_data in all_results.items():
        plot_experiment_diagnostics(result_data, plots_dir)

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Output directory: {plots_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
