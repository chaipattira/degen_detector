# ABOUTME: Separable implicit surface fitting via alternating optimization and PySR.
# ABOUTME: Finds g1(x1) + g2(x2) + ... + gk(xk) = c using 1D symbolic regression per component.

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

from collections import Counter
from dataclasses import dataclass

import numpy as np
import sympy
from pysr import PySRRegressor

from degen_detector.loss import compute_orthogonal_r2


def _polynomial_r2(x_predictor: np.ndarray, x_response: np.ndarray, max_degree: int) -> float:
    """R² of polynomial regression: x_response ~ f(x_predictor), degree <= max_degree.

    Uses ordinary least squares with polynomial features [x, x^2, ..., x^max_degree].
    """
    n = len(x_predictor)
    X = np.column_stack([np.ones(n)] + [x_predictor**d for d in range(1, max_degree + 1)])
    coef, _, _, _ = np.linalg.lstsq(X, x_response, rcond=None)
    ss_res = np.sum((x_response - X @ coef) ** 2)
    ss_tot = np.sum((x_response - np.mean(x_response)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _compute_nonlinearity_scores(X_norm: np.ndarray, max_degree: int = 4) -> np.ndarray:
    """Compute nonlinearity score for each variable using polynomial predictability.

    For each pair (i, j), measures R²(xi ~ poly(xj)) and R²(xj ~ poly(xi)).
    A high R² for xj predicting xi means xi is well-expressed as a polynomial
    of xj, so xj carries the polynomial function and should be fitted first.

    This correctly distinguishes, e.g., theta1 = (10*theta2)^3 by observing that
    R²(theta1 ~ poly3(theta2)) ≈ 1 while R²(theta2 ~ poly3(theta1)) << 1
    (cube root is not polynomial), so theta2 is identified as the cubic variable.

    Parameters
    ----------
    X_norm : ndarray of shape (n_samples, k)
        Normalized data matrix.
    max_degree : int
        Maximum polynomial degree to test.

    Returns
    -------
    scores : ndarray of shape (k,)
        Nonlinearity score per variable. Higher = should be fitted first.
    """
    k = X_norm.shape[1]
    if k == 1:
        return np.zeros(1)

    scores = np.zeros(k)
    for i in range(k):
        for j in range(i + 1, k):
            x_i, x_j = X_norm[:, i], X_norm[:, j]
            # High R² here means xj predicts xi polynomially → xj carries the function
            scores[j] += _polynomial_r2(x_j, x_i, max_degree)
            scores[i] += _polynomial_r2(x_i, x_j, max_degree)

    return scores


def _total_ops(fit):
    """Total sympy operation count across all component expressions."""
    return sum(sympy.count_ops(e) for e in fit.component_exprs)


def _functional_form_key(expr, var_name):
    """Canonical key for functional structure, ignoring numeric constants.

    For polynomial expressions, returns the frozenset of non-zero degrees.
    For non-polynomial expressions (exp, log, etc.), replaces all numeric
    atoms with a placeholder symbol and uses the resulting structure string.
    """
    var = sympy.Symbol(var_name)
    try:
        poly = sympy.Poly(sympy.expand(expr), var)
        return ('poly', frozenset(d[0] for d in poly.monoms()))
    except (sympy.PolynomialError, sympy.GeneratorsNeeded, Exception):
        C = sympy.Symbol('_C')
        form = expr
        for atom in sorted(expr.atoms(sympy.Number), key=lambda x: abs(float(x))):
            form = form.subs(atom, C)
        return ('nonpoly', str(form))


def _rank_by_consensus(candidates):
    """Re-rank candidates so the most common functional form appears first.

    Computes a functional form key for each candidate (ignoring numeric
    constants), counts how many candidates share each form, then sorts so
    the most-common form group comes first. Within each group, sorts by
    orthogonal_r2 (descending), breaking ties by complexity (ascending).

    Returns
    -------
    ranked : list of ImplicitFit
        Re-ranked candidates.
    consensus_count : int
        How many of the input candidates matched the winning form.
    n_forms : int
        Total number of distinct functional forms found.
    """
    form_keys = [
        tuple(
            _functional_form_key(expr, pname)
            for expr, pname in zip(fit.component_exprs, fit.param_names)
        )
        for fit in candidates
    ]

    counts = Counter(form_keys)
    most_common_form, consensus_count = counts.most_common(1)[0]

    def sort_key(idx):
        return (
            0 if form_keys[idx] == most_common_form else 1,
            -candidates[idx].orthogonal_r2,
            _total_ops(candidates[idx]),
        )

    ranked = [candidates[i] for i in sorted(range(len(candidates)), key=sort_key)]
    return ranked, consensus_count, len(counts)


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
    complexity: int  # total sympy operation count across all component expressions

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


def _make_pysr_model_1d(
    max_complexity: int, niterations: int, batch_size: int = 50, log_mode: bool = False
) -> PySRRegressor:
    """Create a PySR model configured for 1D symbolic regression.

    Focuses on polynomial expressions (x, x^2, x^3) which are most common in
    cosmological degeneracies. By default exp is included but heavily penalized
    (complexity cap 6). In log_mode, exp is excluded entirely because the data
    is already in log space — power-law degeneracies are polynomial there, and
    exp would just recover the original variable (exp(log(x)) = x).
    """
    if log_mode:
        unary_operators = ["square", "cube"]
        constraints = {"square": 2, "cube": 2}
    else:
        unary_operators = ["square", "exp", "cube"]
        constraints = {"exp": 6, "square": 2, "cube": 2}

    return PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=unary_operators,
        maxsize=max_complexity,
        niterations=niterations,
        parsimony=0.01,  # Favor simpler expressions
        constraints=constraints,
        deterministic=True,
        parallelism='serial',
        random_state=42,
        progress=False,
        verbosity=0,
        batching=True,
        batch_size=batch_size,
    )


def fit_separable_implicit(
    samples: np.ndarray,
    param_names: list,
    max_iterations: int = 5,
    max_complexity: int = 20,
    niterations: int = 100,
    convergence_threshold: float = 0.01,
    verbose: bool = True,
    n_candidates: int = 5,
    batch_size: int = 50,
    log_mode: bool = False,
) -> list:
    """Fit an implicit surface of separable form.

    Finds functions g1, g2, ..., gk and constant c such that:
        g1(x1) + g2(x2) + ... + gk(xk) = c

    Uses alternating optimization: fix all but one component, fit that
    component, then move to the next. Returns top n_candidates complete
    surfaces by exploring PySR's hall of fame for each component.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, k)
        Data matrix where each column corresponds to a parameter.
    param_names : list of str
        Names of the k parameters.
    max_iterations : int, default=5
        Maximum number of alternating optimization iterations.
    max_complexity : int, default=20
        Maximum equation complexity for PySR.
    niterations : int, default=100
        Number of PySR evolution iterations per component fit.
    convergence_threshold : float, default=0.01
        Stop when residual standard deviation falls below this value.
    verbose : bool, default=True
        Print progress information.
    n_candidates : int, default=5
        Number of candidate equations to return, ranked by orthogonal_r2.
    batch_size : int, default=50
        PySR batch size (number of data points per gradient step). Increase
        for large datasets (e.g. 500–2000 for n > 10k) to reduce noise in
        fitness evaluation while keeping iteration speed acceptable.
    log_mode : bool, default=False
        If True, exclude exp from PySR unary operators. Use when samples are
        already in log space (e.g. from DegenLogMode), where power-law
        degeneracies are polynomial and exp is redundant.

    Returns
    -------
    fits : list of ImplicitFit
        Top n_candidates fitted implicit surfaces. Ranked by functional-form
        consensus first (most common form across candidates promoted to front),
        then by orthogonal_r2 descending within each form group.
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

    # Determine optimal variable ordering based on nonlinearity scores
    # Most nonlinear variables should be fitted first
    nonlin_scores = _compute_nonlinearity_scores(X_norm)
    order = np.argsort(nonlin_scores)[::-1]  # descending: most nonlinear first
    inverse_order = np.argsort(order)  # to map back to original order

    if verbose:
        print(f"Nonlinearity scores: {dict(zip(param_names, nonlin_scores))}")
        print(f"Fitting order: {[param_names[i] for i in order]}")

    # Reorder data and tracking arrays
    X_norm = X_norm[:, order]
    X_mean = X_mean[order]
    X_std = X_std[order]
    param_names_ordered = [param_names[i] for i in order]

    # Initialize: gj(xj) = xj (identity/linear)
    # Simpler, more general starting point for alternating optimization
    component_values = [X_norm[:, j].copy() for j in range(k)]
    component_exprs = [sympy.Symbol(f"z{j}") for j in range(k)]
    c = float(np.mean(sum(component_values)))

    residual_std = np.inf

    # Store PySR models for each component to access hall of fame later
    component_models = [None] * k

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
            model = _make_pysr_model_1d(max_complexity, niterations, batch_size, log_mode)
            x_j = X_norm[:, j].reshape(-1, 1)
            model.fit(x_j, target_j, variable_names=[f"z{j}"])

            # Store model for later access to hall of fame
            component_models[j] = model

            # Get best expression
            expr_j = model.sympy()
            component_exprs[j] = expr_j

            # Update component values
            sym_j = sympy.Symbol(f"z{j}")
            fn_j = sympy.lambdify(sym_j, expr_j, modules="numpy")
            component_values[j] = fn_j(X_norm[:, j])

            if verbose:
                print(f"  Component {j} ({param_names_ordered[j]}): {expr_j}")

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

    # Helper function to create ImplicitFit from component expressions
    # Note: param_names is the ORIGINAL order, param_names_ordered is the fitting order
    original_param_names = param_names  # Capture original order before any reassignment

    def create_implicit_fit(comp_exprs_norm, c_value):
        """Create ImplicitFit from normalized component expressions."""
        # Transform expressions back to original coordinates (in reordered space)
        comp_exprs_reordered = []
        for j, expr in enumerate(comp_exprs_norm):
            sym_orig = sympy.Symbol(param_names_ordered[j])
            sym_norm = sympy.Symbol(f"z{j}")
            # zj = (xj - mu_j) / sigma_j (X_mean/X_std are already reordered)
            substitution = (sym_orig - X_mean[j]) / X_std[j]
            expr_orig = expr.subs(sym_norm, substitution)
            comp_exprs_reordered.append(sympy.simplify(expr_orig))

        # Reorder expressions back to original parameter order
        # inverse_order[i] gives the reordered position of original index i
        comp_exprs_orig = [comp_exprs_reordered[inverse_order[i]] for i in range(k)]

        # Build equation string (in original parameter order)
        terms = []
        for expr in comp_exprs_orig:
            term_str = str(expr)
            if term_str.startswith("-"):
                terms.append(f"({term_str})")
            else:
                terms.append(term_str)
        equation_str = " + ".join(terms) + f" = {c_value:.4f}"

        # Compute residual_std in normalized space (still in reordered space)
        comp_vals_norm = []
        for j, expr in enumerate(comp_exprs_norm):
            sym_j = sympy.Symbol(f"z{j}")
            fn_j = sympy.lambdify(sym_j, expr, modules="numpy")
            comp_vals_norm.append(fn_j(X_norm[:, j]))
        total_norm = sum(comp_vals_norm)
        residual_norm = total_norm - c_value
        res_std = float(np.std(residual_norm))

        # Compute orthogonal R^2 using the implicit surface loss
        # comp_exprs_orig is in original order, samples is in original order
        orthogonal_r2 = compute_orthogonal_r2(
            comp_exprs_orig, original_param_names, samples, c=c_value, normalize=False
        )

        complexity = int(sum(sympy.count_ops(e) for e in comp_exprs_orig))

        return ImplicitFit(
            component_exprs=comp_exprs_orig,
            constant=c_value,
            param_names=list(original_param_names),
            residual_std=res_std,
            orthogonal_r2=orthogonal_r2,
            equation_str=equation_str,
            complexity=complexity,
        )

    # Generate candidate fits by exploring PySR hall of fame
    candidates = []

    # Always include the best fit (default converged solution)
    best_fit = create_implicit_fit(component_exprs, c)
    candidates.append(best_fit)

    if verbose:
        print(f"\nGenerating top {n_candidates} candidate fits from hall of fame...")

    # For each component, extract alternative equations from hall of fame
    max_alternatives = min(3, n_candidates)  # Get top 3 alternatives per component
    for j in range(k):
        model = component_models[j]
        if model is None or not hasattr(model, 'equations_'):
            continue

        # Get equations DataFrame sorted by loss
        eqs = model.equations_
        if len(eqs) <= 1:
            continue  # Only one equation available

        # Take top equations (skip the first one as it's already in best_fit)
        n_alts = min(max_alternatives, len(eqs) - 1)
        for alt_idx in range(1, n_alts + 1):
            if alt_idx >= len(eqs):
                break

            # Create alternative component expressions by swapping j-th component
            alt_comp_exprs = component_exprs.copy()
            try:
                alt_expr = eqs.iloc[alt_idx]['sympy_format']
                alt_comp_exprs[j] = alt_expr

                # Create ImplicitFit with alternative expression
                alt_fit = create_implicit_fit(alt_comp_exprs, c)
                candidates.append(alt_fit)

                if verbose:
                    print(f"  Component {j} alt {alt_idx}: R²_ortho={alt_fit.orthogonal_r2:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Component {j} alt {alt_idx}: Failed - {e}")
                continue

    # First sort by R²_ortho to get top n_candidates
    candidates.sort(key=lambda f: (-f.orthogonal_r2, _total_ops(f)))
    top_candidates = candidates[:n_candidates]

    # Re-rank by functional form consensus: the most common functional form
    # (ignoring numeric constants) is promoted to the front. Within each
    # form group, R²_ortho (descending) and complexity (ascending) apply.
    top_candidates, consensus_count, n_forms = _rank_by_consensus(top_candidates)

    if verbose:
        print(f"\nReturning top {len(top_candidates)} fits "
              f"(consensus form: {consensus_count}/{len(top_candidates)} candidates, "
              f"{n_forms} distinct form(s)):")
        for i, fit in enumerate(top_candidates):
            print(f"  [{i+1}] R²_ortho={fit.orthogonal_r2:.4f}, residual_std={fit.residual_std:.6f}")

    return top_candidates
