"""
Orthogonal loss computation for separable implicit surfaces.

This module computes orthogonal loss for surfaces of the form:
    F(x) = g1(x1) + g2(x2) + ... + gk(xk) - c = 0

The orthogonal loss measures perpendicular distance to this surface:
    L_ortho = mean(F^2 / ||grad F||^2)

Where ||grad F||^2 = (dg1/dx1)^2 + (dg2/dx2)^2 + ... (each gj is univariate)
"""

import numpy as np
import sympy as sp
from typing import List, Tuple


GRADIENT_FLOOR = 1e-8


def z_score_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize data for scale invariance.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, k).

    Returns
    -------
    X_normalized : np.ndarray
        Z-score normalized data.
    means : np.ndarray
        Column means used for normalization.
    stds : np.ndarray
        Column standard deviations used for normalization.
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_normalized = (X - means) / stds
    return X_normalized, means, stds


def compute_orthogonal_loss(
    expressions: List[sp.Expr],
    param_names: List[str],
    X: np.ndarray,
    c: float,
    normalize: bool = True
) -> float:
    """
    Compute orthogonal loss for a separable implicit surface.

    The surface is defined as F(x) = sum(gj(xj)) - c = 0.
    Orthogonal loss is L = mean(F^2 / ||grad F||^2).

    Parameters
    ----------
    expressions : List[sp.Expr]
        List of k sympy expressions [g1, g2, ..., gk], each univariate.
    param_names : List[str]
        List of k parameter names corresponding to each expression.
    X : np.ndarray
        Data matrix of shape (N, k) where N is number of samples.
    c : float
        Constant offset in the implicit surface equation.
    normalize : bool, optional
        Whether to z-score normalize data before computing loss. Default True.

    Returns
    -------
    float
        Orthogonal loss value. Returns np.nan if all points are invalid.
    """
    k = len(expressions)
    if X.shape[1] != k:
        raise ValueError(
            f"Number of expressions ({k}) must match number of columns in X ({X.shape[1]})"
        )
    if len(param_names) != k:
        raise ValueError(
            f"Number of parameter names ({len(param_names)}) must match number of expressions ({k})"
        )

    if normalize:
        X_norm, _, _ = z_score_normalize(X)
    else:
        X_norm = X.copy()

    symbols = [sp.Symbol(name) for name in param_names]

    derivatives = []
    for expr, sym in zip(expressions, symbols):
        deriv = sp.diff(expr, sym)
        derivatives.append(deriv)

    g_funcs = [sp.lambdify(sym, expr, modules='numpy')
               for expr, sym in zip(expressions, symbols)]
    dg_funcs = [sp.lambdify(sym, deriv, modules='numpy')
                for deriv, sym in zip(derivatives, symbols)]

    N = X_norm.shape[0]
    g_values = np.zeros((N, k))
    dg_values = np.zeros((N, k))

    for j in range(k):
        with np.errstate(invalid='ignore', divide='ignore'):
            g_values[:, j] = g_funcs[j](X_norm[:, j])
            dg_values[:, j] = dg_funcs[j](X_norm[:, j])

    F = np.sum(g_values, axis=1) - c
    grad_norm_sq = np.sum(dg_values ** 2, axis=1)
    grad_norm_sq = np.maximum(grad_norm_sq, GRADIENT_FLOOR)

    pointwise_loss = F ** 2 / grad_norm_sq
    valid_mask = np.isfinite(pointwise_loss)

    if not np.any(valid_mask):
        return np.nan

    return np.mean(pointwise_loss[valid_mask])


def compute_orthogonal_r2(
    expressions: List[sp.Expr],
    param_names: List[str],
    X: np.ndarray,
    c: float = 0.0,
    normalize: bool = True
) -> float:
    """
    Compute orthogonal R^2 for a separable implicit surface.

    In normalized space, the null model loss is 1 (unit variance).
    R^2 = 1 - L_ortho

    Parameters
    ----------
    expressions : List[sp.Expr]
        List of k sympy expressions [g1, g2, ..., gk], each univariate.
    param_names : List[str]
        List of k parameter names corresponding to each expression.
    X : np.ndarray
        Data matrix of shape (N, k) where N is number of samples.
    c : float, optional
        Constant offset in the implicit surface equation. Default 0.0.
    normalize : bool, optional
        Whether to z-score normalize data before computing loss. Default True.

    Returns
    -------
    float
        Orthogonal R^2 value. Returns np.nan if all points are invalid.
    """
    loss = compute_orthogonal_loss(expressions, param_names, X, c, normalize)

    if np.isnan(loss):
        return np.nan

    return 1.0 - loss
