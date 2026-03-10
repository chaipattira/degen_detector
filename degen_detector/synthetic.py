"""Synthetic dataset generators with known separable implicit degeneracies.

This module generates test datasets where parameters follow the separable form:
    g_1(x_1) + g_2(x_2) + ... + g_k(x_k) = c
"""

import numpy as np

def generate_scurve_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with S-curve separable degeneracy: (x^3 - 3x) + y + z = 0.

    Uses cubic function to create a pronounced S-curve constraint in the x-z plane.
    The cubic term (x^3 - 3x) creates dramatic curves at top and bottom, resembling
    the letter "S". This tests whether the detector can identify separable degeneracies
    with nonlinear S-shaped transformations.

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

    # Sample x and y uniformly
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)

    # Impose S-curve constraint: (x^3 - 3x) + y + z = 0
    # Solve for z: z = -(x^3 - 3x) - y + noise = -x^3 + 3x - y + noise
    c = 0.0
    z = -(x**3 - 3*x) - y + rng.normal(0, noise, n)

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': '(x^3 - 3x) + y + z = 0',
        'component_functions': ['g1(x) = x^3 - 3x', 'g2(y) = y', 'g3(z) = z'],
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
    """Generate dataset with log/exp mixed: exp(x) + log(y) - z = 2.

    Combines exponential and logarithmic transformations to test detector's
    ability to find diverse nonlinear separable degeneracies.

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
    x = rng.uniform(-1, 1, n)   # Limited range for exp
    y = rng.uniform(0.5, 3, n)  # Positive for log

    # Impose constraint: exp(x) + log(y) - z = c
    # Solve for z: z = exp(x) + log(y) - c + noise
    z = np.exp(x) + np.log(y) - c + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'exp(x) + log(y) - z = 2',
        'component_functions': ['g1(x) = exp(x)', 'g2(y) = log(y)', 'g3(z) = -z'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z']
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
