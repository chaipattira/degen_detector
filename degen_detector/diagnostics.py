# ABOUTME: General diagnostics module for analyzing ImplicitFit results.
# ABOUTME: Provides FitAnalyzer for cached sympy operations and plotting utilities.

"""Diagnostics for ImplicitFit results.

This module provides tools for analyzing and visualizing fitted implicit surfaces.
It can be used on any pkl file containing ImplicitFit objects.
"""

import numpy as np
import sympy as sp


class FitAnalyzer:
    """Analyzer for ImplicitFit with cached sympy evaluation.

    Caches lambdified functions and linearity information on initialization
    to avoid repeated sympy operations during plotting.

    Parameters
    ----------
    fit : ImplicitFit
        The fitted implicit surface to analyze.

    Examples
    --------
    >>> analyzer = FitAnalyzer(fit)
    >>> residuals = analyzer.constraint_residual(samples, param_names)
    >>> true_g, pred_g = analyzer.predict_component('theta1', samples, param_names)
    """

    def __init__(self, fit):
        self.fit = fit
        self.n_params = len(fit.param_names)
        self._cache_functions()

    def _cache_functions(self):
        """Cache lambdified functions and polynomial info on init."""
        self._funcs = {}
        self._symbols = {}
        self._poly_degree = {}  # degree of polynomial (0 if not polynomial)
        self._poly_coeffs = {}  # coefficients for polynomial components

        for i, (expr, pname) in enumerate(zip(self.fit.component_exprs, self.fit.param_names)):
            sym = sp.Symbol(pname)
            self._symbols[pname] = sym
            self._funcs[pname] = sp.lambdify(sym, expr, modules="numpy")

            # Check polynomial degree and cache coefficients
            try:
                poly = sp.Poly(expr, sym)
                degree = poly.degree()
                self._poly_degree[pname] = degree

                # Cache coefficients for linear and quadratic (solvable analytically)
                if degree <= 2:
                    coeffs = poly.all_coeffs()
                    # Pad with zeros: [a, b, c] for ax² + bx + c
                    while len(coeffs) < 3:
                        coeffs.append(0)
                    self._poly_coeffs[pname] = tuple(float(c) for c in coeffs)
            except (sp.PolynomialError, sp.GeneratorsNeeded):
                # Transcendental function (exp, log, etc.)
                self._poly_degree[pname] = 0

    # === Core evaluation ===

    def g(self, param, values):
        """Evaluate component function g_j for given values.

        Parameters
        ----------
        param : str
            Parameter name.
        values : array-like
            Values at which to evaluate.

        Returns
        -------
        result : ndarray
            g_j(values)
        """
        return self._funcs[param](values)

    def constraint_residual(self, samples, param_names):
        """Compute F(x) = sum(g_j) - c for all samples.

        Parameters
        ----------
        samples : ndarray of shape (n_samples, n_params)
            Sample data.
        param_names : list
            Parameter names corresponding to sample columns.

        Returns
        -------
        residuals : ndarray of shape (n_samples,)
            Should be near zero for points on the constraint surface.
        """
        total = np.zeros(len(samples))
        for pname in self.fit.param_names:
            idx = param_names.index(pname)
            total += self.g(pname, samples[:, idx])
        return total - self.fit.constant

    # === Prediction (for true vs predicted plots) ===

    def predict_component(self, target_param, samples, param_names):
        """Compute (true_g, pred_g) for a component.

        For the constraint sum(g_j) = c, computes:
        - true_g = g_i(x_i) using actual sample values
        - pred_g = c - sum(g_j for j != i) what g_i "should" be

        Parameters
        ----------
        target_param : str
            Parameter name for the target component.
        samples : ndarray
            Sample data.
        param_names : list
            Parameter names.

        Returns
        -------
        true_g : ndarray
            Actual component values.
        pred_g : ndarray
            Predicted component values from other components.
        """
        param_idx = param_names.index(target_param)
        true_g = self.g(target_param, samples[:, param_idx])

        other_sum = np.zeros(len(samples))
        for pname in self.fit.param_names:
            if pname != target_param:
                idx = param_names.index(pname)
                other_sum += self.g(pname, samples[:, idx])

        pred_g = self.fit.constant - other_sum
        return true_g, pred_g

    def predict_all_components(self, samples, param_names):
        """Compute (true_g, pred_g) for all components.

        Returns
        -------
        predictions : list of (true_g, pred_g, param_name, component_idx)
        """
        predictions = []
        for i, pname in enumerate(self.fit.param_names):
            true_g, pred_g = self.predict_component(pname, samples, param_names)
            predictions.append((true_g, pred_g, pname, i + 1))
        return predictions

    # === Solving (for curve/surface plotting) ===

    def is_linear(self, param):
        """Check if component is linear in the parameter."""
        return self._poly_degree.get(param, 0) == 1

    def is_quadratic(self, param):
        """Check if component is quadratic in the parameter."""
        return self._poly_degree.get(param, 0) == 2

    def poly_coeffs(self, param):
        """Get polynomial coefficients (a, b, c) for ax² + bx + c.

        Returns None if not a polynomial of degree <= 2.
        For linear, returns (0, a, b) where g(x) = ax + b.
        """
        return self._poly_coeffs.get(param)

    def can_solve_analytically(self, param):
        """Check if we can solve for this parameter analytically."""
        degree = self._poly_degree.get(param, 0)
        return degree in (1, 2)

    def solve_for_param(self, target_param, other_values, bounds=None):
        """Solve for target_param given other parameter values.

        Works analytically for linear/quadratic, numerically otherwise.

        Parameters
        ----------
        target_param : str
            Parameter to solve for.
        other_values : dict
            {param_name: value(s)} for other parameters.
            Can be meshgrid arrays for surface plots.
        bounds : tuple, optional
            (min, max) bounds for numerical solving and selecting quadratic root.

        Returns
        -------
        solution : ndarray
            Solved values (same shape as input arrays).
        """
        # Compute target g value: c - sum(g_j(x_j) for j != target)
        other_sum = sum(
            self.g(p, other_values[p])
            for p in self.fit.param_names if p != target_param
        )
        target_g = self.fit.constant - other_sum

        degree = self._poly_degree.get(target_param, 0)
        coeffs = self.poly_coeffs(target_param)

        if degree == 1 and coeffs is not None:
            # Linear: ax + b = target => x = (target - b) / a
            _, a, b = coeffs  # (0, a, b) for linear
            return (target_g - b) / a

        if degree == 2 and coeffs is not None:
            # Quadratic: ax² + bx + c = target => ax² + bx + (c - target) = 0
            a, b, c = coeffs
            return self._solve_quadratic(a, b, c - target_g, bounds)

        # Non-polynomial: numerical solution
        if bounds is None:
            raise ValueError(f"bounds required for non-polynomial param {target_param}")
        return self._solve_numerical(target_param, target_g, bounds)

    def _solve_quadratic(self, a, b, c, bounds=None):
        """Solve ax² + bx + c = 0 using quadratic formula.

        Returns the root within bounds, or the smaller root if no bounds given.
        """
        discriminant = b**2 - 4*a*c

        # Handle array discriminant
        discriminant = np.asarray(discriminant)
        result = np.full_like(discriminant, np.nan, dtype=float)

        valid = discriminant >= 0
        if not np.any(valid):
            return result

        sqrt_disc = np.sqrt(np.where(valid, discriminant, 0))

        # Two roots: (-b ± sqrt(disc)) / (2a)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)

        if bounds is not None:
            # Select root within bounds
            lo, hi = bounds
            in_bounds1 = (root1 >= lo) & (root1 <= hi)
            in_bounds2 = (root2 >= lo) & (root2 <= hi)

            # Prefer root1 if in bounds, else root2
            result = np.where(valid & in_bounds1, root1, result)
            result = np.where(valid & ~in_bounds1 & in_bounds2, root2, result)
        else:
            # Return smaller root (arbitrary choice)
            result = np.where(valid, np.minimum(root1, root2), result)

        return result

    def _solve_numerical(self, param, target_g, bounds):
        """Vectorized brentq solver for non-polynomial case."""
        from scipy.optimize import brentq

        g_func = self._funcs[param]
        shape = np.shape(target_g)
        flat = np.atleast_1d(target_g).ravel()
        solutions = np.full_like(flat, np.nan, dtype=float)

        for i, target in enumerate(flat):
            if not np.isfinite(target):
                continue
            try:
                solutions[i] = brentq(lambda x: g_func(x) - target, *bounds)
            except (ValueError, RuntimeError):
                pass  # leave as nan

        return solutions.reshape(shape) if shape else float(solutions[0])

    # === Metrics ===

    def get_metrics(self):
        """Get fit quality metrics as a dictionary."""
        return {
            'equation': self.fit.equation_str,
            'r2': getattr(self.fit, 'r2', None),
            'r2_ortho': getattr(self.fit, 'orthogonal_r2', None),
            'residual_std': getattr(self.fit, 'residual_std', None),
            'n_params': len(self.fit.param_names),
            'params': list(self.fit.param_names),
        }

    def get_component_latex(self, param):
        """Get LaTeX representation of a component function."""
        idx = self.fit.param_names.index(param)
        expr = self.fit.component_exprs[idx]
        return sp.latex(expr)


# =============================================================================
# Plotting Functions
# =============================================================================

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
        # Always pass bounds to select correct root for quadratic solutions
        y_bounds = (y.min(), y.max())
        if analyzer.can_solve_analytically(p2):
            # Analytic solution
            y_fitted = analyzer.solve_for_param(p2, {p1: x_range}, bounds=y_bounds)
            ax.plot(x_range, y_fitted, 'r-', linewidth=3,
                    label=f'Fitted (R²_ortho={fit.orthogonal_r2:.3f})', zorder=3)
        else:
            # Numerical solution
            y_fitted = analyzer.solve_for_param(p2, {p1: x_range}, bounds=y_bounds)
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

        # Always pass bounds to select correct root for quadratic solutions
        z_bounds = (z.min(), z.max())
        if analyzer.can_solve_analytically(p3):
            # Analytic solution (linear or quadratic g3)
            Z_grid = analyzer.solve_for_param(p3, {p1: X_grid, p2: Y_grid}, bounds=z_bounds)
        else:
            # Numerical solution (non-linear g3)
            Z_grid = analyzer.solve_for_param(p3, {p1: X_grid, p2: Y_grid}, bounds=z_bounds)

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
    """Plot 2D projections of 3-parameter constraint.

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

    # Panel 1: p1 vs p3
    ax = axes[0]
    ax.scatter(x, z, alpha=0.4, s=5, c='steelblue')
    ax.set_xlabel(p1, fontsize=12)
    ax.set_ylabel(p3, fontsize=12)
    ax.set_title(f'{p1} vs {p3}', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: p2 vs p3
    ax = axes[1]
    ax.scatter(y, z, alpha=0.4, s=5, c='steelblue')
    ax.set_xlabel(p2, fontsize=12)
    ax.set_ylabel(p3, fontsize=12)
    ax.set_title(f'{p2} vs {p3}', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: p1 vs p2 colored by p3
    ax = axes[2]
    scatter = ax.scatter(x, y, c=z, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(p1, fontsize=12)
    ax.set_ylabel(p2, fontsize=12)
    ax.set_title(f'{p1} vs {p2}\n(colored by {p3})', fontsize=11)
    plt.colorbar(scatter, ax=ax, label=p3)
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


# =============================================================================
# Equation Output
# =============================================================================

def format_all_equations(coupling_search_result, ground_truth=None):
    """Format all fitted equations as text.

    Parameters
    ----------
    coupling_search_result : CouplingSearchResult
        Result from DegenDetector.search_couplings().
    ground_truth : dict, optional
        Ground truth info with 'equation' and 'component_functions' keys.

    Returns
    -------
    text : str
        Formatted text with all equations.
    """
    lines = []
    lines.append("DEGENERACY DETECTION RESULTS")

    # Ground truth (if available)
    if ground_truth:
        lines.append("\nGround truth:")
        if 'equation' in ground_truth:
            lines.append(f"  {ground_truth['equation']}")
        if 'component_functions' in ground_truth:
            lines.append("  Component functions:")
            for comp in ground_truth['component_functions']:
                lines.append(f"    {comp}")
        lines.append("")

    # All fitted equations
    lines.append("\nFitted equations (ranked by MI score):")

    n_valid = 0
    for i, cf in enumerate(coupling_search_result.fits):
        rank = i + 1
        params_str = ", ".join(cf.param_names)

        if cf.fit is None:
            lines.append(f"\n[{rank}] Parameters: ({params_str})")
            lines.append(f"    MI score: {cf.mi_score:.4f}")
            lines.append("    Fit: FAILED")
            continue

        n_valid += 1
        fit = cf.fit
        lines.append(f"\n[{rank}] Parameters: ({params_str})")
        lines.append(f"    MI score: {cf.mi_score:.4f}")
        lines.append(f"    Equation: {fit.equation_str}")
        lines.append(f"    R²_ortho: {fit.orthogonal_r2:.4f}")
        lines.append(f"    Residual std: {fit.residual_std:.6f}")
        lines.append("    Components:")
        for j, (expr, pname) in enumerate(zip(fit.component_exprs, fit.param_names)):
            lines.append(f"      g{j+1}({pname}) = {expr}")

    lines.append(f"Total: {len(coupling_search_result.fits)} fits attempted, {n_valid} successful")
    lines.append("-" * 80)

    return "\n".join(lines)


def save_equations(coupling_search_result, output_path, ground_truth=None):
    """Save all equations to a text file.

    Parameters
    ----------
    coupling_search_result : CouplingSearchResult
        Result from DegenDetector.search_couplings().
    output_path : Path or str
        Path to save the text file.
    ground_truth : dict, optional
        Ground truth info.
    """
    text = format_all_equations(coupling_search_result, ground_truth)
    with open(output_path, 'w') as f:
        f.write(text)


# =============================================================================
# DiagnosticsRunner
# =============================================================================

class DiagnosticsRunner:
    """Run diagnostics on pkl files containing fit results.

    Auto-detects pkl structure and generates appropriate plots.

    Parameters
    ----------
    pkl_path : Path or str
        Path to pkl file or directory containing pkl files.

    Examples
    --------
    >>> runner = DiagnosticsRunner("outputs/results.pkl")
    >>> runner.run(output_dir="outputs/plots")
    """

    def __init__(self, pkl_path):
        from pathlib import Path
        self.pkl_path = Path(pkl_path)
        self.data = None
        self._load()

    def _load(self):
        """Load and parse the pkl file."""
        try:
            import dill as pickle
        except ImportError:
            import pickle

        if self.pkl_path.is_dir():
            # Load all pkl files in directory
            pkl_files = list(self.pkl_path.glob("*.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No pkl files in {self.pkl_path}")
            self.data = {}
            for pf in pkl_files:
                with open(pf, 'rb') as f:
                    self.data[pf.stem] = pickle.load(f)
        else:
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)

    def _extract_fit_data(self, data):
        """Extract samples, param_names, result, ground_truth from various formats.

        Returns
        -------
        dict with keys: samples, param_names, result, ground_truth (optional), name
        """
        # Format 1: Direct dict with samples, result, etc.
        if isinstance(data, dict) and 'samples' in data and 'result' in data:
            return {
                'samples': data['samples'],
                'param_names': data['param_names'],
                'result': data['result'],
                'ground_truth': data.get('ground_truth'),
                'name': data.get('name', 'unnamed'),
            }

        # Format 2: Just a CouplingSearchResult
        from degen_detector.core import CouplingSearchResult
        if isinstance(data, CouplingSearchResult):
            return {
                'samples': None,
                'param_names': data.selected_params,
                'result': data,
                'ground_truth': None,
                'name': 'result',
            }

        # Format 3: Dict of experiments
        if isinstance(data, dict):
            # Check if any value looks like experiment data
            for key, val in data.items():
                if isinstance(val, dict) and 'samples' in val:
                    # This is a multi-experiment dict, return as-is
                    return None  # Signal to iterate over dict

        raise ValueError(f"Unknown pkl format: {type(data)}")

    def run(self, output_dir=None, best_only=False):
        """Run diagnostics and generate plots.

        Parameters
        ----------
        output_dir : Path or str, optional
            Directory to save plots. Defaults to pkl_path parent / 'diagnostics'.
        best_only : bool
            If True, only generate plots for the best fit (highest R²_ortho).
            If False, generates plots for top fit by MI ranking.
        """
        from pathlib import Path
        import matplotlib.pyplot as plt

        if output_dir is None:
            if self.pkl_path.is_dir():
                output_dir = self.pkl_path / 'diagnostics'
            else:
                output_dir = self.pkl_path.parent / 'diagnostics'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle different data formats
        if isinstance(self.data, dict):
            # Check if it's a single experiment or multiple
            extracted = self._extract_fit_data(self.data)
            if extracted is not None:
                # Single experiment
                self._run_single(extracted, output_dir)
            else:
                # Multiple experiments
                for name, exp_data in self.data.items():
                    try:
                        extracted = self._extract_fit_data(exp_data)
                        if extracted is not None:
                            extracted['name'] = name
                            exp_dir = output_dir / name
                            exp_dir.mkdir(parents=True, exist_ok=True)
                            self._run_single(extracted, exp_dir)
                    except Exception as e:
                        print(f"Warning: Could not process {name}: {e}")
        else:
            extracted = self._extract_fit_data(self.data)
            self._run_single(extracted, output_dir)

        print(f"\nDiagnostics saved to: {output_dir}")

    def _run_single(self, data, output_dir):
        """Run diagnostics for a single experiment."""
        import matplotlib.pyplot as plt

        name = data['name']
        samples = data['samples']
        param_names = data['param_names']
        result = data['result']
        ground_truth = data.get('ground_truth')

        print(f"\nProcessing: {name}")

        # Get best fit (first valid fit by MI ranking)
        best_cf = next((cf for cf in result.fits if cf.fit is not None), None)
        if best_cf is None:
            print(f"  Warning: No valid fit for {name}, skipping plots")
            return

        best_fit = best_cf.fit
        analyzer = FitAnalyzer(best_fit)

        # 1. Save all equations
        print("  - Equations (all fits)")
        save_equations(result, output_dir / "equations.txt", ground_truth)

        if samples is None:
            print("  - No samples available, skipping plots")
            return

        # 2. Corner plot
        print("  - Corner plot")
        fig = plot_corner(samples, param_names)
        fig.savefig(output_dir / "corner.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 3. Component functions
        print("  - Component functions")
        fig = plot_components(analyzer, samples, param_names)
        fig.savefig(output_dir / "components.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 4. True vs predicted
        print("  - True vs predicted")
        fig = plot_true_vs_predicted(analyzer, samples, param_names)
        fig.savefig(output_dir / "true_vs_predicted.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 5. Residuals
        print("  - Residuals")
        fig = plot_residuals(analyzer, samples, param_names)
        fig.savefig(output_dir / "residuals.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 6. Manifold plot (2D or 3D based on param count)
        n_fit_params = len(best_fit.param_names)
        if n_fit_params == 2:
            print("  - 2D manifold")
            fig = plot_manifold_2d(analyzer, samples, param_names)
            fig.savefig(output_dir / "manifold_2d.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        elif n_fit_params == 3:
            print("  - 3D manifold")
            fig = plot_manifold_3d(analyzer, samples, param_names)
            fig.savefig(output_dir / "manifold_3d.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            print("  - 2D projections")
            fig = plot_projections_3d(analyzer, samples, param_names)
            fig.savefig(output_dir / "projections_2d.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"  Done: {output_dir}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for diagnostics."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for degeneracy detection results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on a single pkl file
    python -m degen_detector.diagnostics results.pkl

    # Run on a directory of pkl files
    python -m degen_detector.diagnostics outputs/synthetic_15710222/20260315_091812

    # Specify output directory
    python -m degen_detector.diagnostics results.pkl -o my_plots/
        """
    )
    parser.add_argument(
        "pkl_path",
        type=Path,
        help="Path to pkl file or directory containing pkl files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for plots (default: <pkl_path>/diagnostics)",
    )

    args = parser.parse_args()

    if not args.pkl_path.exists():
        print(f"Error: {args.pkl_path} does not exist")
        return 1

    runner = DiagnosticsRunner(args.pkl_path)
    runner.run(output_dir=args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
