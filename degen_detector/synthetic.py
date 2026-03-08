"""Synthetic dataset generators with known separable implicit degeneracies.

This module generates test datasets where parameters follow the separable form:
    g_1(x_1) + g_2(x_2) + ... + g_k(x_k) = c
"""

import numpy as np


def generate_linear_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with linear separable degeneracy: x + 2*y - z = 0.

    Creates 7 parameters where 3 follow the linear constraint x + 2*y - z = 0
    (plus noise), and 4 are independent random variables.

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

    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = x + 2 * y + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'x + 2*y - z = 0',
        'component_functions': ['g1(x) = x', 'g2(y) = 2*y', 'g3(z) = -z'],
        'constant': 0.0,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth


def generate_log_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with logarithmic separable degeneracy: log(x) + log(y) - z = 0.

    Creates 7 parameters where 3 follow the constraint log(x) + log(y) - z = 0,
    equivalently z = log(x*y). Both x and y are positive.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.5, 3, n)
    y = rng.uniform(0.5, 3, n)
    z = np.log(x) + np.log(y) + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'log(x) + log(y) - z = 0',
        'component_functions': ['g1(x) = log(x)', 'g2(y) = log(y)', 'g3(z) = -z'],
        'constant': 0.0,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth


def generate_power_law(n=2000, noise=0.1, seed=42):
    """Generate dataset with cosmological power-law degeneracy.

    Creates 5 parameters modeling sigma8 ~ Om^0.5 degeneracy:
    log(sigma8) - 0.5*log(Om) = c
    """
    rng = np.random.default_rng(seed)

    alpha = 0.5
    Om = rng.uniform(0.2, 0.4, n)
    c_norm = 0.8
    sigma8 = c_norm * (Om ** alpha) + rng.normal(0, noise * 0.1, n)

    constant = np.log(c_norm)

    H0 = rng.normal(70, 5, n)
    ns = rng.normal(0.96, 0.02, n)
    Ob = rng.uniform(0.04, 0.06, n)

    samples = np.column_stack([Om, sigma8, H0, ns, Ob])
    param_names = ['Om', 'sigma8', 'H0', 'ns', 'Ob']

    ground_truth = {
        'equation': 'log(sigma8) - 0.5*log(Om) = log(0.8)',
        'component_functions': ['g1(sigma8) = log(sigma8)', 'g2(Om) = -0.5*log(Om)'],
        'constant': constant,
        'degenerate_params': ['Om', 'sigma8']
    }

    return samples, param_names, ground_truth


def generate_exp_linear(n=2000, noise=0.1, seed=42):
    """Generate dataset with mixed exp-linear separable degeneracy: exp(x) + y - z = 0."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-2, 2, n)
    z = np.exp(x) + y + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'exp(x) + y - z = 0',
        'component_functions': ['g1(x) = exp(x)', 'g2(y) = y', 'g3(z) = -z'],
        'constant': 0.0,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth


def generate_quadratic_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with quadratic separable degeneracy: x^2 + y^2 - z = 4.

    Creates a degeneracy following the implicit surface x^2 + y^2 - z = r^2
    (paraboloid), which is separable as g1(x) + g2(y) + g3(z) = c.

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

    r_squared = 4.0
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = x**2 + y**2 - r_squared + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'x^2 + y^2 - z = 4',
        'component_functions': ['g1(x) = x^2', 'g2(y) = y^2', 'g3(z) = -z'],
        'constant': r_squared,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth


def generate_trig_separable(n=2000, noise=0.1, seed=42):
    """Generate dataset with trigonometric separable degeneracy: sin(x) + cos(y) + z = 1.

    Creates a degeneracy following the implicit surface sin(x) + cos(y) + z = c,
    which is separable with periodic component functions.

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
    z = c - np.sin(x) - np.cos(y) + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': 'sin(x) + cos(y) + z = 1',
        'component_functions': ['g1(x) = sin(x)', 'g2(y) = cos(y)', 'g3(z) = z'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth
