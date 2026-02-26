# ABOUTME: Computes mutual information matrices and local PCA for parameter analysis.
# ABOUTME: Quantifies statistical dependencies between parameters, including nonlinear ones.

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


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


def local_pca_intrinsic_dim(samples, variance_threshold=0.95):
    """Estimate intrinsic dimensionality via PCA on a local patch.

    Takes the 20% of samples closest to the median (in standardized space),
    runs PCA, and counts dimensions needed for the given variance threshold.
    """
    # Standardize so distance is not dominated by high-variance params
    stds = np.std(samples, axis=0)
    stds[stds == 0] = 1.0
    standardized = (samples - np.median(samples, axis=0)) / stds

    # Select 20% closest to median
    dists = np.linalg.norm(standardized, axis=1)
    cutoff = np.percentile(dists, 20)
    mask = dists <= cutoff
    local_samples = standardized[mask]

    # PCA on local patch
    pca = PCA()
    pca.fit(local_samples)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = int(np.searchsorted(cumvar, variance_threshold) + 1)

    return intrinsic_dim


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
