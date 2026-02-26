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


def load_sbibm_slcp(num_observation=1):
    """Load SLCP reference posterior from SBIBM.

    SLCP (Simple Likelihood Complex Posterior) has 5 parameters with
    complex nonlinear correlations. No known analytical ground truth -
    useful for testing the detector on realistic SBI posteriors.

    Parameters
    ----------
    num_observation : int
        Which observation to use (1-10 available in SBIBM).

    Returns
    -------
    samples : ndarray
        Reference posterior samples (10000, 5).
    names : list[str]
        Parameter names ["theta_0", "theta_1", ..., "theta_4"].
    """
    import sbibm

    task = sbibm.get_task("slcp")
    samples = task.get_reference_posterior_samples(num_observation=num_observation)
    samples = samples.numpy()
    names = [f"theta_{i}" for i in range(5)]
    return samples, names
