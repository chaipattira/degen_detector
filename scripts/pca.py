# ABOUTME: PCA-based degeneracy detection for linear correlations
# ABOUTME: Serves as baseline comparison for symbolic regression approach

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PCAResult:
    """Results from PCA degeneracy analysis."""
    components: np.ndarray      # (n_params, n_params) - principal components as rows
    eigenvalues: np.ndarray     # (n_params,) - variance along each PC
    explained_variance: np.ndarray  # (n_params,) - fraction of variance explained
    param_names: list[str]
    mean: np.ndarray           # (n_params,) - mean of samples
    std: np.ndarray            # (n_params,) - std of samples

    def print_summary(self, n_components: Optional[int] = None):
        """Print ranked degeneracy directions."""
        if n_components is None:
            n_components = len(self.eigenvalues)

        print(f"{'PC':<4} {'Var Explained':<15} {'Direction'}")
        print("-" * 50)

        for i in range(n_components):
            # Format direction as linear combination
            direction = self._format_direction(i)
            print(f"PC{i+1:<3} {self.explained_variance[i]:>12.1%}    {direction}")

    def _format_direction(self, pc_idx: int, threshold: float = 0.1) -> str:
        """Format a principal component as a linear combination of parameters."""
        components = self.components[pc_idx]
        terms = []
        for j, (coef, name) in enumerate(zip(components, self.param_names)):
            if abs(coef) > threshold:
                sign = "+" if coef > 0 else "-"
                if len(terms) == 0 and coef > 0:
                    sign = ""
                terms.append(f"{sign}{abs(coef):.2f}*{name}")
        return " ".join(terms) if terms else "0"

    def get_degeneracy_pairs(self) -> list[tuple[str, str, float]]:
        """
        Extract pairwise correlations from PCA.
        Returns list of (param1, param2, correlation_strength).
        """
        # Correlation matrix from PCA reconstruction
        # High correlation = potential degeneracy
        n = len(self.param_names)
        pairs = []

        # Compute correlation from eigenvectors and eigenvalues
        # Cov = V @ diag(eigenvalues) @ V.T
        cov = self.components.T @ np.diag(self.eigenvalues) @ self.components
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)

        for i in range(n):
            for j in range(i+1, n):
                pairs.append((self.param_names[i], self.param_names[j], abs(corr[i, j])))

        # Sort by correlation strength (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


def pca_analysis(
    samples: np.ndarray,
    param_names: Optional[list[str]] = None
) -> PCAResult:
    """
    Perform PCA on posterior samples to find linear degeneracy directions.

    Parameters
    ----------
    samples : np.ndarray
        Shape (N, M) where N is number of samples, M is number of parameters
    param_names : list[str], optional
        Names for each parameter. Defaults to ['p0', 'p1', ...]

    Returns
    -------
    PCAResult
        Contains principal components ranked by variance explained
    """
    n_samples, n_params = samples.shape

    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]

    # Standardize samples
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    samples_standardized = (samples - mean) / std

    # Compute covariance matrix
    cov = np.cov(samples_standardized, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance
    total_var = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_var

    return PCAResult(
        components=eigenvectors.T,  # Rows are components
        eigenvalues=eigenvalues,
        explained_variance=explained_variance,
        param_names=param_names,
        mean=mean,
        std=std
    )
