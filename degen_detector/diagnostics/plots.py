# ABOUTME: Plotting functions for visualizing degeneracy detection results.
# ABOUTME: Provides corner plots, component functions, manifold surfaces, residuals, and MI heatmaps.

import numpy as np


def plot_corner(samples, param_names, figsize=None, color="steelblue", **kwargs):
    """Create a corner plot of samples.

    Parameters
    ----------
    samples : ndarray
        Sample data of shape (n_samples, n_params).
    param_names : list
        Parameter names.
    figsize : tuple, optional
        Figure size. Auto-scales with number of parameters if None.
    color : str
        Color for the samples.
    **kwargs
        Additional arguments passed to corner.corner().

    Returns
    -------
    fig : matplotlib.Figure
    """
    import corner
    import matplotlib.pyplot as plt

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
        **kwargs
    )
    plt.tight_layout()
    return fig


def plot_components(analyzer, samples, param_names=None, figsize=None):
    """Plot each component function g_j(x_j) with data points overlaid.

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data. If 2D, uses param_names to find columns.
        If 1D or dict, uses directly.
    param_names : list, optional
        Parameter names for sample columns. Required if samples is 2D.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    fit = analyzer.fit
    k = len(fit.param_names)

    if figsize is None:
        figsize = (4 * k, 4)

    fig, axes = plt.subplots(1, k, figsize=figsize)
    if k == 1:
        axes = [axes]

    for i, pname in enumerate(fit.param_names):
        ax = axes[i]

        # Get data for this parameter
        if isinstance(samples, dict):
            x_data = samples[pname]
        elif samples.ndim == 2 and param_names is not None:
            idx = param_names.index(pname)
            x_data = samples[:, idx]
        else:
            x_data = samples[:, i] if samples.ndim == 2 else samples

        # Evaluate component function
        g_data = analyzer.g(pname, x_data)

        # Plot data points
        ax.scatter(x_data, g_data, alpha=0.3, s=5, c="steelblue", label="Data")

        # Plot smooth curve
        x_min, x_max = x_data.min(), x_data.max()
        margin = 0.1 * (x_max - x_min)
        x_range = np.linspace(x_min - margin, x_max + margin, 200)
        g_curve = analyzer.g(pname, x_range)
        ax.plot(x_range, g_curve, "r-", lw=2, label=f"$g_{{{i+1}}}$")

        ax.set_xlabel(f"${pname}$", fontsize=12)
        ax.set_ylabel(f"$g_{{{i+1}}}({pname})$", fontsize=12)
        ax.set_title(f"$g_{{{i+1}}}({pname}) = {analyzer.get_component_latex(pname)}$", fontsize=10)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{fit.equation_str}", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_true_vs_predicted(analyzer, samples, param_names):
    """Plot true vs predicted values for all component functions.

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data.
    param_names : list
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    predictions = analyzer.predict_all_components(samples, param_names)
    k = len(predictions)

    ncols = min(k, 4)
    nrows = (k + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if k == 1:
        axes = np.array([axes])
    axes = axes.flatten() if k > 1 else axes

    for idx, (true_g, pred_g, pname, comp_num) in enumerate(predictions):
        ax = axes[idx]

        ax.scatter(true_g, pred_g, alpha=0.3, s=10, c='steelblue')

        # y=x reference line
        all_vals = np.concatenate([true_g, pred_g])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label='Perfect fit')

        # Compute R²
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


def plot_residuals(analyzer, samples, param_names):
    """Plot residual distribution (F(x) = sum(g_j) - c).

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data.
    param_names : list
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    residuals = analyzer.constraint_residual(samples, param_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
    ax.set_xlabel('Residual (F(x) - c)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Residual Distribution\nstd = {np.std(residuals):.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_manifold_2d(analyzer, samples, param_names):
    """Plot 2D constraint curve with data and fitted curve overlay.

    For 2-parameter fits, plots param1 vs param2 with the fitted curve.

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data.
    param_names : list
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    fit = analyzer.fit
    if len(fit.param_names) != 2:
        raise ValueError(f"plot_manifold_2d requires 2 params, got {len(fit.param_names)}")

    p1, p2 = fit.param_names
    idx1 = param_names.index(p1)
    idx2 = param_names.index(p2)

    x = samples[:, idx1]
    y = samples[:, idx2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot data scatter
    ax.scatter(x, y, alpha=0.3, s=10, c='steelblue', label='Data', zorder=1)

    # Compute fitted curve: solve for p2 given p1 values
    x_range = np.linspace(x.min(), x.max(), 200)

    try:
        if analyzer.can_solve_analytically(p2):
            # Analytic solution
            y_fitted = analyzer.solve_for_param(p2, {p1: x_range})
            ax.plot(x_range, y_fitted, 'r-', linewidth=3,
                    label=f'Fitted (R²_ortho={fit.orthogonal_r2:.3f})', zorder=3)
        else:
            # Numerical solution
            y_fitted = analyzer.solve_for_param(p2, {p1: x_range}, bounds=(y.min(), y.max()))
            valid = np.isfinite(y_fitted)
            if np.sum(valid) > 10:
                ax.plot(x_range[valid], y_fitted[valid], 'r-', linewidth=3,
                        label=f'Fitted (R²_ortho={fit.orthogonal_r2:.3f})', zorder=3)
    except Exception as e:
        ax.text(0.5, 0.95, f'Could not plot fitted curve: {e}',
                transform=ax.transAxes, ha='center', fontsize=10)

    ax.set_xlabel(f'${p1}$', fontsize=14)
    ax.set_ylabel(f'${p2}$', fontsize=14)
    ax.set_title(f'{p1} vs {p2}\nFit: {fit.equation_str}', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_manifold_3d(analyzer, samples, param_names):
    """Plot 3D constraint surface with data and fitted surface overlay.

    For 3-parameter fits, plots 3D scatter with fitted surface.

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data.
    param_names : list
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fit = analyzer.fit
    if len(fit.param_names) != 3:
        raise ValueError(f"plot_manifold_3d requires 3 params, got {len(fit.param_names)}")

    p1, p2, p3 = fit.param_names
    idx1 = param_names.index(p1)
    idx2 = param_names.index(p2)
    idx3 = param_names.index(p3)

    x = samples[:, idx1]
    y = samples[:, idx2]
    z = samples[:, idx3]

    fig = plt.figure(figsize=(14, 6))

    # Left panel: data only
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x, y, z, c='steelblue', alpha=0.4, s=10, label='Data')
    ax1.set_xlabel(p1, fontsize=12, labelpad=10)
    ax1.set_ylabel(p2, fontsize=12, labelpad=10)
    ax1.set_zlabel(p3, fontsize=12, labelpad=10)
    ax1.set_title('Data Points', fontsize=11, pad=20)
    ax1.view_init(elev=20, azim=45)

    # Right panel: data + fitted surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x, y, z, c='steelblue', alpha=0.4, s=10, label='Data')

    # Try to plot fitted surface
    try:
        x_range = np.linspace(x.min(), x.max(), 50)
        y_range = np.linspace(y.min(), y.max(), 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)

        if analyzer.can_solve_analytically(p3):
            # Analytic solution (linear g3)
            Z_grid = analyzer.solve_for_param(p3, {p1: X_grid, p2: Y_grid})
        else:
            # Numerical solution (non-linear g3)
            Z_grid = analyzer.solve_for_param(p3, {p1: X_grid, p2: Y_grid},
                                               bounds=(z.min(), z.max()))

        # Check if we got valid solutions
        valid_frac = np.sum(np.isfinite(Z_grid)) / Z_grid.size
        if valid_frac > 0.5:
            ax2.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='green')

            legend_elements = [
                Patch(facecolor='steelblue', alpha=0.4, label='Data points'),
                Patch(facecolor='green', alpha=0.3, label=f'Fitted: R²={fit.orthogonal_r2:.3f}')
            ]
            ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
        else:
            ax2.text2D(0.5, 0.95, f'Could not solve for {p3} (only {valid_frac*100:.0f}% valid)',
                       transform=ax2.transAxes, ha='center', fontsize=9)
    except Exception as e:
        ax2.text2D(0.5, 0.95, f'Could not compute surface: {e}',
                   transform=ax2.transAxes, ha='center', fontsize=9)

    ax2.set_xlabel(p1, fontsize=12, labelpad=10)
    ax2.set_ylabel(p2, fontsize=12, labelpad=10)
    ax2.set_zlabel(p3, fontsize=12, labelpad=10)
    ax2.set_title(f'Data + Fitted Surface\nR²_ortho = {fit.orthogonal_r2:.4f}', fontsize=12, pad=20)
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


def plot_projections_3d(analyzer, samples, param_names):
    """Plot 2D projections of 3-parameter constraint with fitted curves.

    For each 2D projection, overlays the fitted constraint curve by holding
    the third variable at representative values (median and quartiles).

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data.
    param_names : list
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    fit = analyzer.fit
    if len(fit.param_names) != 3:
        raise ValueError(f"plot_projections_3d requires 3 params, got {len(fit.param_names)}")

    p1, p2, p3 = fit.param_names
    idx1 = param_names.index(p1)
    idx2 = param_names.index(p2)
    idx3 = param_names.index(p3)

    x = samples[:, idx1]
    y = samples[:, idx2]
    z = samples[:, idx3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Define representative values for held-constant parameters
    x_median = np.median(x)
    y_median = np.median(y)
    z_median = np.median(z)

    # Quartiles for drawing multiple curves
    x_q25, x_q75 = np.percentile(x, [25, 75])
    y_q25, y_q75 = np.percentile(y, [25, 75])
    z_q25, z_q75 = np.percentile(z, [25, 75])

    # Panel 1: p1 vs p3 (holding p2 at median and quartiles)
    ax = axes[0]
    ax.scatter(x, z, alpha=0.4, s=5, c='steelblue', label='Data')

    # Plot fitted curves for different p2 values
    x_range = np.linspace(x.min(), x.max(), 200)
    curve_styles = [
        (y_q25, '--', 0.6, f'{p2}=Q25'),
        (y_median, '-', 1.0, f'{p2}=median'),
        (y_q75, '--', 0.6, f'{p2}=Q75'),
    ]

    try:
        for p2_val, ls, alpha, label in curve_styles:
            if analyzer.can_solve_analytically(p3):
                z_fitted = analyzer.solve_for_param(p3, {p1: x_range, p2: p2_val})
            else:
                z_fitted = analyzer.solve_for_param(p3, {p1: x_range, p2: p2_val},
                                                     bounds=(z.min(), z.max()))
            valid = np.isfinite(z_fitted)
            if np.sum(valid) > 10:
                ax.plot(x_range[valid], z_fitted[valid], 'r', ls=ls, lw=2, alpha=alpha, label=label)
    except Exception:
        pass  # Skip curve if solving fails

    ax.set_xlabel(p1, fontsize=12)
    ax.set_ylabel(p3, fontsize=12)
    ax.set_title(f'{p1} vs {p3}', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: p2 vs p3 (holding p1 at median and quartiles)
    ax = axes[1]
    ax.scatter(y, z, alpha=0.4, s=5, c='steelblue', label='Data')

    y_range = np.linspace(y.min(), y.max(), 200)
    curve_styles = [
        (x_q25, '--', 0.6, f'{p1}=Q25'),
        (x_median, '-', 1.0, f'{p1}=median'),
        (x_q75, '--', 0.6, f'{p1}=Q75'),
    ]

    try:
        for p1_val, ls, alpha, label in curve_styles:
            if analyzer.can_solve_analytically(p3):
                z_fitted = analyzer.solve_for_param(p3, {p1: p1_val, p2: y_range})
            else:
                z_fitted = analyzer.solve_for_param(p3, {p1: p1_val, p2: y_range},
                                                     bounds=(z.min(), z.max()))
            valid = np.isfinite(z_fitted)
            if np.sum(valid) > 10:
                ax.plot(y_range[valid], z_fitted[valid], 'r', ls=ls, lw=2, alpha=alpha, label=label)
    except Exception:
        pass

    ax.set_xlabel(p2, fontsize=12)
    ax.set_ylabel(p3, fontsize=12)
    ax.set_title(f'{p2} vs {p3}', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: p1 vs p2 with contour lines for different p3 values
    ax = axes[2]
    scatter = ax.scatter(x, y, c=z, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(p1, fontsize=12)
    ax.set_ylabel(p2, fontsize=12)

    # Draw contour lines by solving for p2 at fixed p3 values
    x_grid = np.linspace(x.min(), x.max(), 200)
    contour_z_vals = [z_q25, z_median, z_q75]
    contour_styles = ['--', '-', '--']
    contour_alphas = [0.6, 1.0, 0.6]

    try:
        for z_val, ls, alpha in zip(contour_z_vals, contour_styles, contour_alphas):
            if analyzer.can_solve_analytically(p2):
                y_contour = analyzer.solve_for_param(p2, {p1: x_grid, p3: z_val})
            else:
                y_contour = analyzer.solve_for_param(p2, {p1: x_grid, p3: z_val},
                                                      bounds=(y.min(), y.max()))
            valid = np.isfinite(y_contour)
            if np.sum(valid) > 10:
                ax.plot(x_grid[valid], y_contour[valid], 'r', ls=ls, lw=2, alpha=alpha,
                        label=f'{p3}={z_val:.2f}')
    except Exception:
        pass

    ax.set_title(f'{p1} vs {p2}\n(colored by {p3})', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax, label=p3)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Fit: {fit.equation_str}', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_manifold(analyzer, samples, param_names):
    """Auto-select 2D or 3D manifold plot based on number of parameters.

    Parameters
    ----------
    analyzer : FitAnalyzer
        Analyzer wrapping the fit.
    samples : ndarray
        Sample data.
    param_names : list
        Parameter names.

    Returns
    -------
    fig : matplotlib.Figure or None
        Returns None if fit has < 2 or > 3 parameters.
    """
    n = len(analyzer.fit.param_names)
    if n == 2:
        return plot_manifold_2d(analyzer, samples, param_names)
    elif n == 3:
        return plot_manifold_3d(analyzer, samples, param_names)
    else:
        return None


def plot_mi_matrix(mi_result, output_path=None):
    """Plot the mutual information matrix as a heatmap.

    Parameters
    ----------
    mi_result : MIResult
        Computed mutual information matrix.
    output_path : Path or str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    n = len(mi_result.param_names)
    figsize = (max(6, n), max(5, n - 1))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(mi_result.mi_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mutual Information (nats)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mi_result.param_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(mi_result.param_names, fontsize=10)

    max_val = mi_result.mi_matrix.max()
    for i in range(n):
        for j in range(n):
            val = mi_result.mi_matrix[i, j]
            color = 'black' if val > max_val * 0.5 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_title('Mutual Information Matrix', fontsize=13)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
