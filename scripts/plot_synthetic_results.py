#!/usr/bin/env python
"""Plot diagnostics for synthetic degeneracy experiments.

Usage:
    python3 /home/x-ctirapongpra/scratch/degen_detector/scripts/plot_synthetic_results.py /home/x-ctirapongpra/scratch/degen_detector/outputs/synthetic_15627082/20260309_193657

This script loads all .pkl result files from a directory and generates comprehensive diagnostic plots including:
- Simple corner plots (no overlays)
- Corner plots with implicit surface overlays
- Residual corner plots
- Component function plots
- Ground truth vs predicted values for ALL component functions
- Equation comparison text file (ground truth vs fitted with R² and R²_ortho)
- Summary comparison across experiments
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
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
        Dictionary with 'equation' and 'component_functions' keys.
    fit : ImplicitFit
        The fitted implicit surface.
    output_path : Path
        Path to save the text file.
    """
    # Ground truth text
    gt_text = f"Ground Truth:\n{ground_truth['equation']}\n\n"
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

    print(f"  Saved all plots to {exp_plot_dir}")


def plot_summary(all_results, output_dir):
    """Create a summary figure comparing all experiments.

    Parameters
    ----------
    all_results : dict
        Dictionary mapping experiment keys to result data.
    output_dir : Path
        Directory to save the summary plot.
    """
    # Helper to get top fit from result
    def get_top_fit(result):
        return next((cf.fit for cf in result.fits if cf.fit is not None), None)

    # Filter to only results with valid fits
    valid_results = {k: v for k, v in all_results.items() if get_top_fit(v['result']) is not None}
    if not valid_results:
        print("No valid fits to summarize")
        return

    # Create one row per experiment, columns for each component
    max_components = max(len(get_top_fit(rd['result']).param_names) for rd in valid_results.values())

    n_valid = len(valid_results)
    fig, axes = plt.subplots(n_valid, max_components, figsize=(5*max_components, 5*n_valid))
    if n_valid == 1 and max_components == 1:
        axes = np.array([[axes]])
    elif n_valid == 1:
        axes = axes.reshape(1, -1)
    elif max_components == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (key, result_data) in enumerate(valid_results.items()):
        samples = result_data['samples']
        param_names = result_data['param_names']
        best_fit = get_top_fit(result_data['result'])

        predictions = compute_all_component_predictions(best_fit, samples, param_names)

        for col_idx, (true_g, pred_g, pname, comp_num) in enumerate(predictions):
            ax = axes[row_idx, col_idx]

            # Plot
            ax.scatter(true_g, pred_g, alpha=0.3, s=10, c='steelblue')
            all_vals = np.concatenate([true_g, pred_g])
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)

            # Compute R^2
            valid = np.isfinite(true_g) & np.isfinite(pred_g)
            if np.sum(valid) > 0:
                residuals = true_g[valid] - pred_g[valid]
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((true_g[valid] - np.mean(true_g[valid])) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                r2 = np.nan

            title = f"{result_data['name']}: g{comp_num}({pname})\nR² = {r2:.3f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f'True', fontsize=9)
            ax.set_ylabel(f'Predicted', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots in this row
        for col_idx in range(len(predictions), max_components):
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    summary_file = output_dir / "summary_comparison.png"
    fig.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved summary plot to {summary_file}")


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

    # Generate summary plot
    if len(all_results) > 1:
        plot_summary(all_results, plots_dir)

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Output directory: {plots_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
