"""Separable implicit surface fitting using alternating optimization.

This module finds functions g1, g2, ..., gk and constant c such that:
    g1(x1) + g2(x2) + ... + gk(xk) = c

The algorithm uses alternating optimization:
1. Initialize: gj(xj) = xj (identity), c = mean(sum of xj)
2. For each iteration:
   - For j = 1 to k:
     - Compute partial_sum = sum of gi(xi) for i != j
     - Target for gj: target_j = c - partial_sum
     - If j == 0: normalize target to unit variance (anchoring)
     - Fit gj using PySR (1D symbolic regression)
     - Update component values
   - Update c = mean(sum of gj(xj))
   - Check convergence: if std(sum of gj - c) < threshold, stop
3. Transform expressions back to original coordinates
"""

from dataclasses import dataclass

import numpy as np
import sympy
from pysr import PySRRegressor

from degen_detector.loss import compute_orthogonal_r2


@dataclass
class ImplicitFit:
    """Result of implicit surface fitting.

    Represents a separable implicit surface of the form:
        g1(x1) + g2(x2) + ... + gk(xk) = c
    """

    component_exprs: list
    constant: float
    param_names: list
    residual_std: float
    orthogonal_r2: float
    equation_str: str

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Compute sum of gj(xj) - c for each point.

        Parameters
        ----------
        X : ndarray of shape (n_samples, k)
            Data matrix where each column corresponds to a parameter.

        Returns
        -------
        residuals : ndarray of shape (n_samples,)
            The value of F(x) = g1(x1) + g2(x2) + ... + gk(xk) - c
            for each data point. Perfect fit gives residuals near zero.
        """
        total = np.zeros(X.shape[0])
        for j, expr in enumerate(self.component_exprs):
            sym = sympy.Symbol(self.param_names[j])
            fn = sympy.lambdify(sym, expr, modules="numpy")
            total += fn(X[:, j])
        return total - self.constant

    def predict_component(self, j: int, x_j: np.ndarray) -> np.ndarray:
        """Evaluate gj(xj) for a single component.

        Parameters
        ----------
        j : int
            Index of the component (0 to k-1).
        x_j : ndarray
            Values of the j-th parameter.

        Returns
        -------
        values : ndarray
            The evaluated component function gj(xj).
        """
        expr = self.component_exprs[j]
        sym = sympy.Symbol(self.param_names[j])
        fn = sympy.lambdify(sym, expr, modules="numpy")
        return fn(x_j)


def _make_pysr_model_1d(max_complexity: int, niterations: int) -> PySRRegressor:
    """Create a PySR model configured for 1D symbolic regression."""
    return PySRRegressor(
        binary_operators=["+", "*", "^"],
        unary_operators=["log", "exp"],
        constraints={"^": (-1, 3)},
        maxsize=max_complexity,
        niterations=niterations,
        deterministic=True,
        parallelism='serial',
        random_state=42,
        progress=False,
        verbosity=0,
    )


def fit_separable_implicit(
    samples: np.ndarray,
    param_names: list,
    max_iterations: int = 5,
    max_complexity: int = 15,
    niterations: int = 40,
    convergence_threshold: float = 0.01,
    verbose: bool = True,
) -> ImplicitFit:
    """Fit an implicit surface of separable form.

    Finds functions g1, g2, ..., gk and constant c such that:
        g1(x1) + g2(x2) + ... + gk(xk) = c

    Uses alternating optimization: fix all but one component, fit that
    component, then move to the next.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, k)
        Data matrix where each column corresponds to a parameter.
    param_names : list of str
        Names of the k parameters.
    max_iterations : int, default=5
        Maximum number of alternating optimization iterations.
    max_complexity : int, default=15
        Maximum equation complexity for PySR.
    niterations : int, default=40
        Number of PySR evolution iterations per component fit.
    convergence_threshold : float, default=0.01
        Stop when residual standard deviation falls below this value.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    fit : ImplicitFit
        The fitted implicit surface with component expressions.
    """
    k = len(param_names)
    n_samples = samples.shape[0]

    if samples.shape[1] != k:
        raise ValueError(
            f"samples has {samples.shape[1]} columns but "
            f"{k} param names given"
        )

    # Normalize data to unit variance for balanced fitting
    X_mean = np.mean(samples, axis=0)
    X_std = np.std(samples, axis=0)
    X_std = np.where(X_std < 1e-12, 1.0, X_std)
    X_norm = (samples - X_mean) / X_std

    # Initialize: gj(xj) = xj (identity functions)
    component_values = [X_norm[:, j].copy() for j in range(k)]
    component_exprs = [sympy.Symbol(f"z{j}") for j in range(k)]
    c = float(np.mean(sum(component_values)))

    residual_std = np.inf

    for iteration in range(max_iterations):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iterations}")

        # Fit each component in turn
        for j in range(k):
            # Target for gj: what it should produce
            partial_sum = sum(component_values[i] for i in range(k) if i != j)
            target_j = c - partial_sum

            # Anchoring: if j == 0, normalize target to unit variance
            # to prevent trivial solutions
            if j == 0:
                target_std = np.std(target_j)
                if target_std > 1e-8:
                    target_j = target_j / target_std
                    c = c / target_std
                    # Rescale other components
                    for i in range(1, k):
                        component_values[i] = component_values[i] / target_std

            # Run 1D symbolic regression
            model = _make_pysr_model_1d(max_complexity, niterations)
            x_j = X_norm[:, j].reshape(-1, 1)
            model.fit(x_j, target_j, variable_names=[f"z{j}"])

            # Get best expression
            expr_j = model.sympy()
            component_exprs[j] = expr_j

            # Update component values
            sym_j = sympy.Symbol(f"z{j}")
            fn_j = sympy.lambdify(sym_j, expr_j, modules="numpy")
            component_values[j] = fn_j(X_norm[:, j])

            if verbose:
                print(f"  Component {j} ({param_names[j]}): {expr_j}")

        # Update constant
        total = sum(component_values)
        c = float(np.mean(total))

        # Check convergence
        residual = total - c
        residual_std = float(np.std(residual))

        if verbose:
            print(f"  Residual std: {residual_std:.6f}")

        if residual_std < convergence_threshold:
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break

    # Transform expressions back to original coordinates
    # If X_norm = (X - mu) / sigma, substitute zj = (xj - mu_j) / sigma_j
    component_exprs_orig = []
    for j, expr in enumerate(component_exprs):
        sym_orig = sympy.Symbol(param_names[j])
        sym_norm = sympy.Symbol(f"z{j}")
        # zj = (xj - mu_j) / sigma_j
        substitution = (sym_orig - X_mean[j]) / X_std[j]
        expr_orig = expr.subs(sym_norm, substitution)
        component_exprs_orig.append(sympy.simplify(expr_orig))

    # Build equation string
    terms = []
    for expr in component_exprs_orig:
        term_str = str(expr)
        if term_str.startswith("-"):
            terms.append(f"({term_str})")
        else:
            terms.append(term_str)
    equation_str = " + ".join(terms) + f" = {c:.4f}"

    # Compute orthogonal R^2 using the implicit surface loss
    orthogonal_r2 = compute_orthogonal_r2(
        component_exprs_orig, param_names, samples
    )

    return ImplicitFit(
        component_exprs=component_exprs_orig,
        constant=c,
        param_names=param_names,
        residual_std=residual_std,
        orthogonal_r2=orthogonal_r2,
        equation_str=equation_str,
    )
