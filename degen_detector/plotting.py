# ABOUTME: Visualization utilities for degeneracy detection results.
# ABOUTME: Includes corner plots with equation overlays for posterior verification.

import numpy as np
import matplotlib.pyplot as plt


def plot_corner_with_degeneracy(
    samples,
    param_names,
    result,
    highlight_fit=None,
    figsize=None,
    color="steelblue",
    truth=None,
    **corner_kwargs
):
    """Create corner plot with discovered degeneracy equations overlaid.

    Parameters
    ----------
    samples : ndarray
        Posterior samples, shape (N, M).
    param_names : list[str]
        Parameter names.
    result : CouplingSearchResult
        Result from DegenDetector.search_couplings().
    highlight_fit : CouplingFit or None
        Specific fit to highlight. If None, uses result.best_fit.
    figsize : tuple or None
        Figure size. If None, auto-scaled based on number of parameters.
    color : str
        Base color for samples.
    truth : dict or None
        Ground truth values {param_name: value} to mark.
    **corner_kwargs
        Additional kwargs passed to corner.corner().

    Returns
    -------
    fig : matplotlib.Figure
        The corner plot figure.
    """
    import corner

    n_params = len(param_names)
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

    # Prepare truth values for corner
    truths = None
    if truth is not None:
        truths = [truth.get(name, None) for name in param_names]

    # Create base corner plot
    fig = corner.corner(
        samples,
        labels=param_names,
        show_titles=True,
        title_kwargs={"fontsize": 10},
        color=color,
        truths=truths,
        fig=plt.figure(figsize=figsize),
        **corner_kwargs
    )

    # Get the axes array
    axes = np.array(fig.axes).reshape((n_params, n_params))

    # Determine which fit to highlight
    fit_to_show = highlight_fit if highlight_fit else result.best_fit
    if fit_to_show is None or fit_to_show.fit is None:
        return fig

    fit = fit_to_show.fit

    # Find the relevant 2D panel(s) and add overlays
    target_idx = param_names.index(fit.target_name)

    for input_name in fit.input_names:
        input_idx = param_names.index(input_name)

        # 2D panel is in lower triangle: row > col
        # If target_idx > input_idx: panel at (target_idx, input_idx)
        # If input_idx > target_idx: panel at (input_idx, target_idx)
        if input_idx < target_idx:
            ax = axes[target_idx, input_idx]
            x_idx, y_idx = input_idx, target_idx
        else:
            ax = axes[input_idx, target_idx]
            x_idx, y_idx = target_idx, input_idx

        # Add equation annotation
        eq_text = f"{fit.equation_str}\nR² = {fit.r_squared:.3f}"
        ax.annotate(
            eq_text,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Highlight panel border
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)

    # Overlay predicted curve for single-input fits
    if len(fit.input_names) == 1:
        _overlay_predicted_curve(axes, samples, param_names, fit)

    plt.tight_layout()
    return fig


def _overlay_predicted_curve(axes, samples, param_names, fit):
    """Overlay predicted curve on the relevant 2D panel for single-input fits."""
    input_name = fit.input_names[0]
    target_name = fit.target_name

    input_idx = param_names.index(input_name)
    target_idx = param_names.index(target_name)

    # Determine which axis is x and which is y in the panel
    if input_idx < target_idx:
        ax = axes[target_idx, input_idx]
        x_data = samples[:, input_idx]
        # x-axis is input, y-axis is target -> plot curve directly
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_pred = fit.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, "r-", lw=2, zorder=10)
    else:
        ax = axes[input_idx, target_idx]
        # x-axis is target, y-axis is input -> plot inverted (swap axes)
        x_data = samples[:, input_idx]
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_pred = fit.predict(x_range.reshape(-1, 1))
        # Plot with axes swapped: (predicted_target, input)
        ax.plot(y_pred, x_range, "r-", lw=2, zorder=10)


def plot_residual_corner(
    samples,
    param_names,
    fit,
    figsize=None,
    cmap="coolwarm",
    **corner_kwargs
):
    """Create corner plot with samples colored by equation residual.

    Parameters
    ----------
    samples : ndarray
        Posterior samples.
    param_names : list[str]
        Parameter names.
    fit : MultiSymbolicFit
        The fitted equation.
    figsize : tuple or None
        Figure size.
    cmap : str
        Colormap for residuals.
    **corner_kwargs
        Additional kwargs passed to corner.corner().

    Returns
    -------
    fig : matplotlib.Figure
    """
    import corner

    # Compute residuals
    input_indices = [param_names.index(n) for n in fit.input_names]
    target_idx = param_names.index(fit.target_name)

    X = samples[:, input_indices]
    y_actual = samples[:, target_idx]
    y_pred = fit.predict(X)
    residuals = y_actual - y_pred

    # Normalize residuals for coloring (clip to 3 sigma)
    res_std = residuals.std()
    if res_std > 0:
        res_normalized = residuals / (3 * res_std)
    else:
        res_normalized = np.zeros_like(residuals)
    res_clipped = np.clip(res_normalized, -1, 1)

    n_params = len(param_names)
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

    # Create corner plot with light gray base (will overlay colored scatter)
    fig = corner.corner(
        samples,
        labels=param_names,
        show_titles=True,
        color="lightgray",
        fig=plt.figure(figsize=figsize),
        plot_contours=False,
        plot_density=False,
        **corner_kwargs
    )

    axes = np.array(fig.axes).reshape((n_params, n_params))

    # Get colormap and compute colors
    cm = plt.cm.get_cmap(cmap)
    colors = cm((res_clipped + 1) / 2)  # Map [-1, 1] to [0, 1]

    # Overlay colored scatter on 2D panels (lower triangle)
    for i in range(n_params):
        for j in range(i):
            ax = axes[i, j]
            ax.scatter(
                samples[:, j],
                samples[:, i],
                c=colors,
                s=1,
                alpha=0.5,
                rasterized=True,
            )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(-3 * res_std, 3 * res_std)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="Residual")

    fig.suptitle(f"{fit.equation_str} (R²={fit.r_squared:.3f})", y=1.02)

    return fig
