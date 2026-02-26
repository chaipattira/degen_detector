# ABOUTME: Fits symbolic equations to parameter tuples using genetic programming.
# ABOUTME: Wraps PySR to find human-readable relationships like sigma8 = f(Om).

from dataclasses import dataclass
from pathlib import Path

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


def _make_pysr_model(max_complexity=25, niterations=60, batching=False, tempdir=None):
    """Create a PySR model with the standard configuration."""
    kwargs = dict(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp"],
        constraints={"^": (-1, 1), "/": (-1, 9)},
        maxsize=max_complexity,
        niterations=niterations,
        deterministic=True,
        parallelism="serial",
        populations=15,
        progress=False,
        batching=batching,
    )
    if tempdir is not None:
        kwargs["tempdir"] = str(tempdir)
        kwargs["temp_equation_file"] = True
    return PySRRegressor(**kwargs)


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


def fit_tuple(samples, param_names, param_indices,
              max_complexity=25, niterations=60, batching=False, output_dir=None):
    """Fit symbolic equation for a specific parameter tuple.

    Tries each parameter as target, returns the fit with highest R².

    Parameters
    ----------
    samples : ndarray
        Full samples array (N, M).
    param_names : list[str]
        All parameter names.
    param_indices : list[int]
        Indices of parameters in this tuple.
    max_complexity : int
        Maximum equation complexity (PySR maxsize).
    niterations : int
        Number of PySR evolution iterations.
    batching : bool
        Enable batching for large datasets.
    output_dir : str or Path, optional
        Base directory for PySR temp files. Subfolder names are based on
        the parameter tuple being fit.

    Returns
    -------
    fit : MultiSymbolicFit or None
        Best fit (highest R²), or None if all fits fail.
    """
    best_fit = None
    best_r2 = -np.inf

    # Build tuple name for output folder
    tuple_names = [param_names[j] for j in param_indices]
    tuple_label = "_".join(tuple_names)

    for target_idx in param_indices:
        input_indices = [j for j in param_indices if j != target_idx]
        X = samples[:, input_indices]
        y = samples[:, target_idx]
        input_names_local = [param_names[j] for j in input_indices]
        target_name = param_names[target_idx]

        # Create descriptive subfolder for this fit
        tempdir = None
        if output_dir is not None:
            tempdir = Path(output_dir) / f"{tuple_label}_target_{target_name}"
            tempdir.mkdir(parents=True, exist_ok=True)

        try:
            model = _make_pysr_model(max_complexity, niterations, batching, tempdir)
            model.fit(X, y, variable_names=input_names_local)

            eq = _select_equation(model.equations_)
            predict_fn = _build_predict(eq["sympy_format"], input_names_local)

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
                    input_names=input_names_local,
                    target_name=target_name,
                )
        except Exception:
            # Skip failed fits
            continue

    return best_fit
