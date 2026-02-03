# ABOUTME: Fits symbolic equations to degenerate parameter groups using genetic programming.
# ABOUTME: Wraps PySR to find human-readable relationships like sigma8 = f(Om).

from dataclasses import dataclass

import numpy as np
import sympy
from pysr import PySRRegressor


@dataclass
class MultiSymbolicFit:
    """A symbolic equation relating degenerate parameters."""
    equation_str: str
    r_squared: float
    complexity: int
    predict: object  # callable: (X: ndarray) -> ndarray
    input_names: list
    target_name: str


def _make_pysr_model(max_complexity=25, niterations=60):
    """Create a PySR model with the standard configuration."""
    return PySRRegressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp"],
        constraints={"^": (-1, 1), "/": (-1, 9)},
        maxsize=max_complexity,
        niterations=niterations,
        deterministic=True,
        parallelism="serial",
        populations=15,
        progress=False,
    )


def _select_equation(equations_df):
    """Select the simplest equation within 1.5x of the best loss."""
    min_loss = equations_df["loss"].min()
    candidates = equations_df[equations_df["loss"] < min_loss * 1.5]
    best_idx = candidates["complexity"].idxmin()
    return equations_df.loc[best_idx]


def _build_predict(sympy_expr, input_names):
    """Build a numpy-compatible predict callable from a sympy expression."""
    symbols = sympy.symbols(input_names)
    fn = sympy.lambdify(symbols, sympy_expr, modules=["numpy"])

    def predict(X):
        return np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]),
                          dtype=float)
    return predict


def fit_group_all_targets(group, samples, param_names,
                          max_complexity=25, niterations=60):
    """Try each parameter as target, return the fit with highest R².

    For a group of k parameters, runs k separate PySR fits. Each fit
    uses k-1 parameters as inputs and 1 as the target. Returns the
    combination that yields the highest R².
    """
    best_fit = None
    best_r2 = -np.inf

    for target_idx in group.param_indices:
        input_indices = [j for j in group.param_indices if j != target_idx]
        X = samples[:, input_indices]
        y = samples[:, target_idx]
        input_names = [param_names[j] for j in input_indices]
        target_name = param_names[target_idx]

        model = _make_pysr_model(max_complexity, niterations)
        model.fit(X, y, variable_names=input_names)

        eq = _select_equation(model.equations_)

        # Build predict from sympy (safer than lambda_format)
        predict_fn = _build_predict(eq["sympy_format"], input_names)

        # Compute R²
        y_pred = predict_fn(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        if r2 > best_r2:
            best_r2 = r2
            equation_str = f"{target_name} = {eq['equation']}"
            best_fit = MultiSymbolicFit(
                equation_str=equation_str,
                r_squared=r2,
                complexity=int(eq["complexity"]),
                predict=predict_fn,
                input_names=input_names,
                target_name=target_name,
            )

    return best_fit
