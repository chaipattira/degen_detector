"""Visualization utilities for implicit separable surface fits.

This module provides plotting functions for visualizing fits of the form:
    sum_j g_j(x_j) = c
where each g_j is a univariate function of parameter x_j.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import sympy as sp


def _solve_implicit_for_plot(g1_expr, g2_expr, c, x_range, param_names):
    """Solve g1(x) + g2(y) = c for y at each x value."""
    x_sym = sp.Symbol(param_names[0])
    y_sym = sp.Symbol(param_names[1])

    g1_func = sp.lambdify(x_sym, g1_expr, modules="numpy")
    g2_func = sp.lambdify(y_sym, g2_expr, modules="numpy")

    x_vals = []
    y_vals = []

    for x_val in x_range:
        g1_val = g1_func(x_val)
        target = c - g1_val

        def residual(y):
            return g2_func(y) - target

        y_search_range = np.linspace(-1e6, 1e6, 1000)
        try:
            g2_vals = g2_func(y_search_range)
            sign_changes = np.where(np.diff(np.sign(g2_vals - target)))[0]

            if len(sign_changes) > 0:
                idx = sign_changes[0]
                y_lo, y_hi = y_search_range[idx], y_search_range[idx + 1]
                y_solution = brentq(residual, y_lo, y_hi)
                x_vals.append(x_val)
                y_vals.append(y_solution)
        except (ValueError, RuntimeError):
            continue

    return np.array(x_vals), np.array(y_vals)


def plot_corner_with_implicit(
    samples,
    param_names,
    fit,
    figsize=None,
    color="steelblue",
    **corner_kwargs
):
    """Create corner plot with implicit surface curve overlaid.

    Parameters
    ----------
    samples : ndarray
        Posterior samples, shape (N, M).
    param_names : list[str]
        Parameter names for the samples.
    fit : ImplicitFit
        Fit object with component_exprs, constant, param_names, orthogonal_r2, equation_str.
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

    axes = np.array(fig.axes).reshape((n_params, n_params))
    fit_param_indices = [param_names.index(p) for p in fit.param_names]

    if len(fit.component_exprs) == 2:
        idx0, idx1 = fit_param_indices[0], fit_param_indices[1]

        if idx1 > idx0:
            ax = axes[idx1, idx0]
            x_idx, y_idx = idx0, idx1
            g1_expr, g2_expr = fit.component_exprs[0], fit.component_exprs[1]
            fit_params_ordered = [fit.param_names[0], fit.param_names[1]]
        else:
            ax = axes[idx0, idx1]
            x_idx, y_idx = idx1, idx0
            g1_expr, g2_expr = fit.component_exprs[1], fit.component_exprs[0]
            fit_params_ordered = [fit.param_names[1], fit.param_names[0]]

        x_data = samples[:, x_idx]
        x_range = np.linspace(x_data.min(), x_data.max(), 200)

        x_curve, y_curve = _solve_implicit_for_plot(
            g1_expr, g2_expr, fit.constant, x_range, fit_params_ordered
        )

        if len(x_curve) > 0:
            ax.plot(x_curve, y_curve, "r-", lw=2, zorder=10)

        eq_text = f"{fit.equation_str}\nR² = {fit.orthogonal_r2:.3f}"
        ax.annotate(
            eq_text,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)

    else:
        for idx_i in fit_param_indices:
            for idx_j in fit_param_indices:
                if idx_i > idx_j:
                    ax = axes[idx_i, idx_j]
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(2)

        if len(fit_param_indices) >= 2:
            idx_i = max(fit_param_indices[0], fit_param_indices[1])
            idx_j = min(fit_param_indices[0], fit_param_indices[1])
            ax = axes[idx_i, idx_j]

            eq_text = f"{fit.equation_str}\nR² = {fit.orthogonal_r2:.3f}"
            ax.annotate(
                eq_text,
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    plt.tight_layout()
    return fig


def _compute_residuals(samples, param_names, fit):
    """Compute residuals F = sum_j g_j(x_j) - c for all samples."""
    n_samples = samples.shape[0]
    total = np.zeros(n_samples)

    for expr, pname in zip(fit.component_exprs, fit.param_names):
        param_idx = param_names.index(pname)
        x_vals = samples[:, param_idx]

        sym = sp.Symbol(pname)
        g_func = sp.lambdify(sym, expr, modules="numpy")
        total += g_func(x_vals)

    return total - fit.constant


def plot_residual_corner(
    samples,
    param_names,
    fit,
    figsize=None,
    cmap="coolwarm",
    **corner_kwargs
):
    """Create corner plot with samples colored by implicit surface residual."""
    import corner

    residuals = _compute_residuals(samples, param_names, fit)

    res_std = residuals.std()
    if res_std > 0:
        res_normalized = residuals / (3 * res_std)
    else:
        res_normalized = np.zeros_like(residuals)
    res_clipped = np.clip(res_normalized, -1, 1)

    n_params = len(param_names)
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

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

    cm = plt.cm.get_cmap(cmap)
    colors = cm((res_clipped + 1) / 2)

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

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(-3 * res_std, 3 * res_std)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="Residual (F)")

    fig.suptitle(f"{fit.equation_str} (R²={fit.orthogonal_r2:.3f})", y=1.02)

    return fig


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
