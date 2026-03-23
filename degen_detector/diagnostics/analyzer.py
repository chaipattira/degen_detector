# ABOUTME: FitAnalyzer class for cached sympy evaluation, solving, and metrics on ImplicitFit results.
# ABOUTME: Caches lambdified functions and polynomial info for fast repeated evaluation.

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
                    # Pad with leading zeros: [a, b, c] for ax² + bx + c
                    coeffs = [0] * (3 - len(coeffs)) + list(coeffs)
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
