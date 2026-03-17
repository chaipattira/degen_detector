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

def generate_banana_degeneracy(n=2000, seed=42):
    """Generate dataset with banana-shaped non-Gaussian degeneracy (Figure 11).

    From the paper (Section IV.A): P(θ1, θ2) ∝ N(√(θ₁² + 20(2θ₁² - θ₂ - 1/2)²), 1/4)

    This creates a strong banana-shaped degeneracy. The posterior is Gaussian in the
    distance from the constraint surface θ₂ = 2θ₁² - 1/2.

    Implementation uses change of variables:
        u ~ N(0, 0.25),  v ~ N(0, 0.0125)
        θ₁ = u,  θ₂ = v + 2u² - 0.5

    This makes d² = θ₁² + 20(2θ₁² - θ₂ - 1/2)² = u² + 20v² separable.

    Parameters
    ----------
    n : int
        Number of samples to generate.
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

    samples_theta1 = []
    samples_theta2 = []

    # Oversample to account for rejection outside prior bounds
    n_samples = int(n * 2)

    while len(samples_theta1) < n:
        # Sample u ~ N(0, 0.25), so σ = 0.5
        u = rng.normal(0, 0.5, n_samples)

        # Sample v ~ N(0, 0.0125), so σ ≈ 0.1118
        v = rng.normal(0, np.sqrt(0.0125), n_samples)

        # Transform to (θ₁, θ₂)
        theta1 = u
        theta2 = v + 2 * u**2 - 0.5

        # Accept only if within prior bounds [-3, 3]²
        mask = (theta1 >= -3) & (theta1 <= 3) & (theta2 >= -3) & (theta2 <= 3)

        samples_theta1.extend(theta1[mask].tolist())
        samples_theta2.extend(theta2[mask].tolist())

    # Trim to exact size
    samples_theta1 = np.array(samples_theta1[:n])
    samples_theta2 = np.array(samples_theta2[:n])

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)

    samples = np.column_stack([samples_theta1, samples_theta2, a, b, c_param, d, e])
    param_names = ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e']

    ground_truth = {
        'equation': 'theta1^2 + 20(2*theta1^2 - theta2 - 0.5)^2 ≈ 0',
        'constraint': 'theta2 ≈ 2*theta1^2 - 0.5',
        'prior_range': 'theta1, theta2 ∈ [-3, 3]',
        'degenerate_params': ['theta1', 'theta2'],
        'figure': 'Figure 11 - Banana-shaped non-Gaussian degeneracy'
    }

    return samples, param_names, ground_truth


def generate_cubic_degeneracy(n=2000, seed=42):
    """Generate dataset with cubic degeneracy and informative prior (Figure 12).

    From the paper (Section IV.B): P(θ1, θ2) ∝ N(θ1 - (10θ2)³, 1/2)

    This creates a non-linear degeneracy with constraint θ1 = (10θ2)³.
    The posterior is Gaussian in the residual θ1 - (10θ2)³.

    The prior is informative with θ₁ ∈ [-2, 2] and θ₂ ∈ [-0.1, 0.1].
    Note: Paper says θ₂ ∈ [-0.1, 0.1], not [-0.2, 0.2].

    Implementation:
        Sample θ₂ ~ Uniform([-0.1, 0.1])
        Sample θ₁ | θ₂ ~ N((10θ₂)³, 0.5)
        Reject if θ₁ ∉ [-2, 2]

    Parameters
    ----------
    n : int
        Number of samples to generate.
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

    samples_theta1 = []
    samples_theta2 = []

    # Oversample to account for rejection outside prior bounds
    n_samples = int(n * 1.5)  # Minimal rejection expected

    while len(samples_theta1) < n:
        # Sample θ₂ from uniform prior
        theta2 = rng.uniform(-0.1, 0.1, n_samples)

        # Sample θ₁ from conditional Gaussian: θ₁ | θ₂ ~ N((10θ₂)³, 0.5)
        mean = (10 * theta2)**3
        std = np.sqrt(0.5)
        theta1 = rng.normal(mean, std)

        # Accept only if within prior bounds
        mask = (theta1 >= -2) & (theta1 <= 2)

        samples_theta1.extend(theta1[mask].tolist())
        samples_theta2.extend(theta2[mask].tolist())

    # Trim to exact size
    samples_theta1 = np.array(samples_theta1[:n])
    samples_theta2 = np.array(samples_theta2[:n])

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)

    samples = np.column_stack([samples_theta1, samples_theta2, a, b, c, d, e])
    param_names = ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e']

    ground_truth = {
        'equation': 'theta1 - (10*theta2)^3 ≈ 0',
        'constraint': 'theta1 ≈ (10*theta2)^3',
        'prior_range': 'theta1 ∈ [-2, 2], theta2 ∈ [-0.1, 0.1]',
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
