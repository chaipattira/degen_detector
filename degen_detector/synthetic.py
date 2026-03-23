# ABOUTME: Synthetic dataset generators for testing separable implicit degeneracy detection.
# ABOUTME: Each generator produces samples following g1(x1) + ... + gk(xk) = c with known ground truth.

"""Synthetic dataset generators with known separable implicit degeneracies.

This module generates test datasets where parameters follow the separable form:
    g_1(x_1) + g_2(x_2) + ... + g_k(x_k) = c
"""

import numpy as np

def generate_scurve_separable(n=2000, noise=0.2, seed=42):
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
    """Generate dataset with 3D paraboloid surface degeneracy.

    Extends the 2D banana degeneracy to a 2D surface embedded in 3D space.
    The constraint is: 2θ₁² + θ₂² - θ₃ = 0.5

    This creates an elliptic paraboloid (bowl-shaped) surface that preserves
    the characteristic quadratic curvature of the banana in the θ₁ direction.

    Separable form: g₁(θ₁) + g₂(θ₂) + g₃(θ₃) = c
        where g₁(θ₁) = 2θ₁², g₂(θ₂) = θ₂², g₃(θ₃) = -θ₃, c = 0.5

    Implementation uses change of variables:
        θ₁ ~ N(0, 0.5), θ₂ ~ N(0, 0.5)
        w ~ N(0, 0.0125)  (noise perpendicular to surface)
        θ₃ = 2θ₁² + θ₂² - 0.5 + w

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
        Names of the parameters: ['theta1', 'theta2', 'theta3', 'a', 'b', 'c', 'd'].
    ground_truth : dict
        Dictionary containing equation, constraint, and degenerate_params.
    """
    rng = np.random.default_rng(seed)

    samples_theta1 = []
    samples_theta2 = []
    samples_theta3 = []

    # Oversample to account for rejection outside prior bounds
    n_samples = int(n * 2)

    while len(samples_theta1) < n:
        # Sample θ₁, θ₂ from Gaussian (similar spread to 2D banana)
        theta1 = rng.normal(0, 0.5, n_samples)
        theta2 = rng.normal(0, 0.5, n_samples)

        # Sample noise perpendicular to the surface
        w = rng.normal(0, np.sqrt(0.0125), n_samples)

        # Compute θ₃ from constraint: θ₃ = 2θ₁² + θ₂² - 0.5 + noise
        theta3 = 2 * theta1**2 + theta2**2 - 0.5 + w

        # Accept only if within prior bounds [-3, 3]³
        mask = (
            (theta1 >= -3) & (theta1 <= 3) &
            (theta2 >= -3) & (theta2 <= 3) &
            (theta3 >= -3) & (theta3 <= 3)
        )

        samples_theta1.extend(theta1[mask].tolist())
        samples_theta2.extend(theta2[mask].tolist())
        samples_theta3.extend(theta3[mask].tolist())

    # Trim to exact size
    samples_theta1 = np.array(samples_theta1[:n])
    samples_theta2 = np.array(samples_theta2[:n])
    samples_theta3 = np.array(samples_theta3[:n])

    # Add independent parameters
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([samples_theta1, samples_theta2, samples_theta3, a, b, c_param, d])
    param_names = ['theta1', 'theta2', 'theta3', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': '2*theta1^2 + theta2^2 - theta3 = 0.5',
        'constraint': 'theta3 ≈ 2*theta1^2 + theta2^2 - 0.5',
        'component_functions': ['g1(theta1) = 2*theta1^2', 'g2(theta2) = theta2^2', 'g3(theta3) = -theta3'],
        'constant': 0.5,
        'prior_range': 'theta1, theta2, theta3 ∈ [-3, 3]',
        'degenerate_params': ['theta1', 'theta2', 'theta3'],
        'surface_type': 'Elliptic paraboloid (3D extension of banana)'
    }

    return samples, param_names, ground_truth


def generate_cubic_degeneracy(n=2000, seed=42):
    """Generate dataset with cubic degeneracy (Figure 12).

    Inspired by the paper (Section IV.B): P(θ1, θ2) ∝ N(θ1 - (10θ2)³, σ²)

    Constraint: θ1 = (10θ2)³, creating a strongly nonlinear separable degeneracy.
    θ₂ ∈ [-0.1, 0.1] so the cubic value (10*0.1)^3 = 1 stays within the θ1 prior
    [-2, 2] without truncation.
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

    theta2 = rng.uniform(-0.1, 0.1, n)
    theta1 = (10 * theta2)**3 + rng.normal(0, 0.15, n)

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
        'prior_range': 'theta1 ∈ [-2, 2], theta2 ∈ [-0.1, 0.1]',
        'degenerate_params': ['theta1', 'theta2'],
        'figure': 'Figure 12 - Cubic degeneracy with informative prior'
    }

    return samples, param_names, ground_truth


def generate_trig_separable(n=2000, noise=0.2, seed=42):
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
    x = rng.uniform(0, 2 * np.pi, n) + rng.normal(0, noise, n)
    y = rng.uniform(0, 2 * np.pi, n) + rng.normal(0, noise, n)
    z = 2 * np.sin(x) + np.cos(y) - c + rng.normal(0, noise, n)

    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c_param = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)

    samples = np.column_stack([x, y, z, a, b, c_param, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': '2*sin(x) + cos(y) - z = 1',
        'component_functions': ['g1(x) = 2*sin(x)', 'g2(y) = cos(y)', 'g3(z) = -z'],
        'constant': c,
        'degenerate_params': ['x', 'y', 'z']
    }

    return samples, param_names, ground_truth
