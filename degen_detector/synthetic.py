"""Synthetic dataset generators with known separable implicit degeneracies.

This module generates test datasets where parameters follow the separable form:
    g_1(x_1) + g_2(x_2) + ... + g_k(x_k) = c
"""

import numpy as np

def generate_scurve_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset using S-curve manifold with separable degeneracy: x^2 + y^2 + z = 6.

    Uses sklearn's S-curve manifold to generate a non-uniform sampling distribution,
    then imposes a separable quadratic constraint: x^2 + y^2 + z = c.

    This tests whether the detector can identify separable degeneracies when the
    parameter space is sampled from a complex manifold rather than uniform distributions.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the constraint.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n, 7) containing all parameter samples.
    param_names : list of str
        Names of the parameters: ['x', 'y', 'z', 'a', 'b', 'c', 'd'].
    ground_truth : dict
        Dictionary containing equation, component_functions, constant, degenerate_params.
    """
    from sklearn.datasets import make_s_curve

    # Generate S-curve manifold (returns 3D points and parameter t)
    X_scurve, _ = make_s_curve(n_samples=n, noise=0.0, random_state=seed)

    rng = np.random.default_rng(seed)

    # Use S-curve coordinates for non-uniform sampling
    # S-curve x is in [-1, 1] roughly, y is in [0, 2]
    x = X_scurve[:, 0]
    y = X_scurve[:, 1]

    # Impose separable constraint: x^2 + y^2 + z = c
    c = 6.0  # Chosen to keep z in reasonable range given x, y distributions
    z = c - x**2 - y**2 + rng.normal(0, noise, n)

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'x^2 + y^2 + z = 6',
        'component_functions': ['g1(x) = x^2', 'g2(y) = y^2', 'g3(z) = z'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth

def generate_polynomial_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with polynomial terms (separable): x^3 + y^2 - z = 3.

    Combines cubic and quadratic terms to test whether the detector
    can identify separable polynomial degeneracies.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the constraint.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n, 7) containing all parameter samples.
    param_names : list of str
        Names of the parameters: ['x', 'y', 'z', 'a', 'b', 'c', 'd'].
    ground_truth : dict
        Dictionary containing equation, component_functions, constant, degenerate_params.
    """
    rng = np.random.default_rng(seed)

    c = 3.0
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    # z = x^3 + y^2 - c + noise
    z = x**3 + y**2 - c + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'x^3 + y^2 - z = 3',
        'component_functions': ['g1(x) = x^3', 'g2(y) = y^2', 'g3(z) = -z'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth


def generate_nonlinear_mixed(n=2000, noise=0.1, seed=42):
    """Generate dataset with log/exp/power/polynomial mixed: log(x) + exp(y) - sqrt(z) + a^2 = 2.

    Combines logarithmic, exponential, power-law (sqrt), and polynomial (quadratic) transformations
    to test detector's ability to find diverse nonlinear separable degeneracies.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the constraint.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n, 7) containing all parameter samples.
    param_names : list of str
        Names of the parameters: ['x', 'y', 'z', 'a', 'b', 'c', 'd'].
    ground_truth : dict
        Dictionary containing equation, component_functions, constant, degenerate_params.
    """
    rng = np.random.default_rng(seed)

    c = 2.0
    x = rng.uniform(0.5, 3, n)  # Positive for log
    y = rng.uniform(-1, 1, n)   # Limited range for exp
    a = rng.uniform(-1.5, 1.5, n)  # Moderate range for polynomial term

    # z = (log(x) + exp(y) + a^2 - c)^2, then sqrt(z) = log(x) + exp(y) + a^2 - c
    z = (np.log(x) + np.exp(y) + a**2 - c + rng.normal(0, noise, n)) ** 2
    z = np.abs(z)  # Ensure positive for sqrt

    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'log(x) + exp(y) - sqrt(z) + a^2 = 2',
        'component_functions': ['g1(x) = log(x)', 'g2(y) = exp(y)', 'g3(z) = -sqrt(z)', 'g4(a) = a^2'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z', 'a']
    }

    return samples, param_names, ground_truth


def generate_trig_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with trigonometric functions (separable): sin(x) + cos(y) - z = 1.

    Combines trigonometric functions to test whether the detector can identify
    periodic separable degeneracies.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the constraint.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n, 7) containing all parameter samples.
    param_names : list of str
        Names of the parameters: ['x', 'y', 'z', 'a', 'b', 'c', 'd'].
    ground_truth : dict
        Dictionary containing equation, component_functions, constant, degenerate_params.
    """
    rng = np.random.default_rng(seed)

    c = 1.0
    x = rng.uniform(0, 2 * np.pi, n)
    y = rng.uniform(0, 2 * np.pi, n)
    z = np.sin(x) + np.cos(y) - c + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'sin(x) + cos(y) - z = 1',
        'component_functions': ['g1(x) = sin(x)', 'g2(y) = cos(y)', 'g3(z) = -z'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth
