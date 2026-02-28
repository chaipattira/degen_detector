# ABOUTME: Generates synthetic datasets with known multi-dimensional degeneracies.
# ABOUTME: Used for testing and validating the degeneracy detection pipeline.

import numpy as np


def generate_3param_polynomial(n=2000, noise=0.1, seed=42):
    """Generate 7 params: z = x² + y + noise, with 4 independent params.

    Degenerate: x, y, z (polynomial relationship)
    Independent: a, b, c, d (uncorrelated noise)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = x**2 + y + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c, d])
    return samples, ["x", "y", "z", "a", "b", "c", "d"]


def generate_3param_exp_log(n=2000, noise=0.1, seed=42):
    """Generate 7 params: z = exp(x) + log(y) + noise, with 4 independent params.

    Degenerate: x, y, z (exp/log relationship)
    Independent: a, b, c, d (uncorrelated noise)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1, 1, n)  # Keep x small so exp(x) doesn't explode
    y = rng.uniform(0.5, 3, n)  # Keep y > 0 for log
    z = np.exp(x) + np.log(y) + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c, d])
    return samples, ["x", "y", "z", "a", "b", "c", "d"]


def generate_s_curve(n=2000, noise=0.1, seed=42):
    """Generate S-curve manifold with degeneracy in 3D space.

    The S-curve is a 2D manifold embedded in 3D space where the 3 coordinates
    (x, y, z) are related through the manifold structure, creating a clear
    multi-dimensional degeneracy.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the S-curve.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        S-curve samples (n, 3) representing [x, y, z] coordinates.
    names : list[str]
        Parameter names ["x", "y", "z"].
    """
    from sklearn.datasets import make_s_curve

    # Generate S-curve
    samples, t = make_s_curve(n_samples=n, noise=noise, random_state=seed)
    names = ["x", "y", "z"]
    return samples, names
