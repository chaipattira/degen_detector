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

def generate_banana_degeneracy(n=2000, noise=0.5, seed=42):
    """Generate dataset with banana-shaped non-Gaussian degeneracy (Figure 11).

    From the paper (Section IV.A): P(θ1, θ2) ∝ N(√(θ₁² + 20(2θ₁² - θ₂ - 1/2)²), 1/4)

    This creates a strong banana-shaped degeneracy where the constraint is:
    θ₁² + 20(2θ₁² - θ₂ - 1/2)² ≈ 0

    Solving for θ₂: θ₂ ≈ 2θ₁² - 1/2

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the constraint.
        Default 0.25 corresponds to variance 1/16 ≈ 1/4 from the paper.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n, 7) containing all parameter samples.
    param_names : list of str
        Names of the parameters: ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e'].
    ground_truth : dict
        Dictionary containing equation, constraint, and degenerate_params.
    """
    rng = np.random.default_rng(seed)

    # Sample θ1 uniformly from prior range
    theta1 = rng.uniform(-3, 3, n)

    # Impose banana-shaped constraint: θ₂ = 2θ₁² - 1/2 + noise
    # This makes θ₁² + 20(2θ₁² - θ₂ - 1/2)² ≈ 0
    theta2 = 2 * theta1**2 - 0.5 + rng.normal(0, noise, n)

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)

    samples = np.column_stack([theta1, theta2, a, b, c_param, d, e])
    param_names = ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e']

    ground_truth = {
        'equation': 'theta1^2 + 20(2*theta1^2 - theta2 - 0.5)^2 ≈ 0',
        'constraint': 'theta2 ≈ 2*theta1^2 - 0.5',
        'prior_range': 'theta1, theta2 ∈ [-3, 3]',
        'degenerate_params': ['theta1', 'theta2'],
        'figure': 'Figure 11 - Banana-shaped non-Gaussian degeneracy'
    }

    return samples, param_names, ground_truth


def generate_cubic_degeneracy(n=2000, noise=0.5, seed=42):
    """Generate dataset with cubic degeneracy and informative prior (Figure 12).

    From the paper (Section IV.B): P(θ1, θ2) ∝ N(θ1 - (10θ2)³, 1/2)

    This creates a non-linear degeneracy with constraint:
    θ1 - (10θ2)³ ≈ 0

    The prior is informative with θ₁ ∈ [-2, 2] and θ₂ ∈ [-0.2, 0.2].
    One parameter is fully prior constrained.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise. Default sqrt(0.5) ≈ 0.707
        corresponds to variance 1/2 from the paper.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n, 7) containing all parameter samples.
    param_names : list of str
        Names of the parameters: ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e'].
    ground_truth : dict
        Dictionary containing equation, constraint, and degenerate_params.
    """
    rng = np.random.default_rng(seed)

    # Sample θ2 uniformly from tight prior range (informative prior)
    theta2 = rng.uniform(-0.2, 0.2, n)

    # Impose cubic constraint: θ1 = (10θ2)³ + noise
    theta1 = (10 * theta2)**3 + rng.normal(0, noise, n)

    # Clip theta1 to respect its prior bounds
    theta1 = np.clip(theta1, -2, 2)

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)

    samples = np.column_stack([theta1, theta2, a, b, c, d, e])
    param_names = ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e']

    ground_truth = {
        'equation': 'theta1 - (10*theta2)^3 ≈ 0',
        'constraint': 'theta1 ≈ (10*theta2)^3',
        'prior_range': 'theta1 ∈ [-2, 2], theta2 ∈ [-0.2, 0.2]',
        'degenerate_params': ['theta1', 'theta2'],
        'figure': 'Figure 12 - Cubic degeneracy with informative prior'
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
