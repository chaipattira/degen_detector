# ABOUTME: Computes mutual information matrices and parameter selection for degeneracy analysis.
# ABOUTME: Quantifies statistical dependencies between parameters, including nonlinear ones.

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_regression


@dataclass
class MIResult:
    """Mutual information between all parameter pairs."""
    mi_matrix: np.ndarray  # (M, M) symmetric, zero diagonal
    param_names: list


def mutual_info_matrix(samples, param_names):
    """Compute pairwise mutual information for all parameters.

    Uses sklearn's k-NN MI estimator. Returns a symmetric matrix with
    zero diagonal, measured in nats.
    """
    n_params = len(param_names)
    if samples.shape[1] != n_params:
        raise ValueError(
            f"samples has {samples.shape[1]} columns but "
            f"{n_params} param names given"
        )

    mi_raw = np.zeros((n_params, n_params))
    for i in range(n_params):
        mi_row = mutual_info_regression(samples, samples[:, i], random_state=42)
        mi_raw[i, :] = mi_row

    # Symmetrize and clean up
    mi_matrix = (mi_raw + mi_raw.T) / 2
    np.fill_diagonal(mi_matrix, 0)
    mi_matrix = np.maximum(mi_matrix, 0)

    return MIResult(mi_matrix=mi_matrix, param_names=list(param_names))


def select_params_by_mi(mi_result, top_n):
    """Select top-N parameters by total mutual information.

    Parameters with highest sum of MI to all other parameters are selected,
    i.e., those most correlated with the rest of the parameter space.

    Parameters
    ----------
    mi_result : MIResult
        Precomputed mutual information matrix.
    top_n : int
        Number of parameters to select.

    Returns
    -------
    selected_indices : list[int]
        Indices of selected parameters (sorted ascending).
    selected_names : list[str]
        Names of selected parameters.
    """
    total_mi = mi_result.mi_matrix.sum(axis=1)
    top_indices = np.argsort(total_mi)[::-1][:top_n].tolist()
    top_indices.sort()  # Keep original order for consistency
    selected_names = [mi_result.param_names[i] for i in top_indices]
    return top_indices, selected_names
