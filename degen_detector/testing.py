# ABOUTME: Synthetic dataset generators with known degeneracies, for testing and benchmarking.
# ABOUTME: Not part of the core detection API — import from degen_detector.testing.

"""Synthetic dataset generators for benchmarking and tests.

Each generator returns (samples, param_names, ground_truth) where the degeneracy
equation and degenerate parameter names are known in advance.

SYNTHETIC_CASES is the canonical benchmark registry used by run_synthetic_experiments.py.
"""

import numpy as np


def generate_banana_degeneracy(n=2000, seed=42):
    """3D paraboloid (banana) degeneracy: 2*theta1^2 + theta2^2 - theta3 = 0.5."""
    rng = np.random.default_rng(seed)

    samples_theta1, samples_theta2, samples_theta3 = [], [], []
    n_batch = int(n * 2)

    while len(samples_theta1) < n:
        theta1 = rng.normal(0, 0.5, n_batch)
        theta2 = rng.normal(0, 0.5, n_batch)
        w = rng.normal(0, np.sqrt(0.0125), n_batch)
        theta3 = 2 * theta1**2 + theta2**2 - 0.5 + w

        mask = (
            (theta1 >= -3) & (theta1 <= 3) &
            (theta2 >= -3) & (theta2 <= 3) &
            (theta3 >= -3) & (theta3 <= 3)
        )
        samples_theta1.extend(theta1[mask].tolist())
        samples_theta2.extend(theta2[mask].tolist())
        samples_theta3.extend(theta3[mask].tolist())

    theta1 = np.array(samples_theta1[:n])
    theta2 = np.array(samples_theta2[:n])
    theta3 = np.array(samples_theta3[:n])

    a, b, c, d = (rng.normal(0, 1, n) for _ in range(4))
    samples = np.column_stack([theta1, theta2, theta3, a, b, c, d])
    param_names = ['theta1', 'theta2', 'theta3', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': '2*theta1^2 + theta2^2 - theta3 = 0.5',
        'component_functions': ['g1(theta1) = 2*theta1^2', 'g2(theta2) = theta2^2', 'g3(theta3) = -theta3'],
        'constant': 0.5,
        'degenerate_params': ['theta1', 'theta2', 'theta3'],
    }
    return samples, param_names, ground_truth


def generate_cubic_degeneracy(n=2000, seed=42):
    """Cubic degeneracy: theta1 ≈ (10*theta2)^3."""
    rng = np.random.default_rng(seed)

    theta2 = rng.uniform(-0.1, 0.1, n)
    theta1 = (10 * theta2)**3 + rng.normal(0, 0.15, n)

    a, b, c, d, e = (rng.normal(0, 1, n) for _ in range(5))
    samples = np.column_stack([theta1, theta2, a, b, c, d, e])
    param_names = ['theta1', 'theta2', 'a', 'b', 'c', 'd', 'e']

    ground_truth = {
        'equation': 'theta1 - (10*theta2)^3 ≈ 0',
        'constraint': 'theta1 ≈ (10*theta2)^3',
        'degenerate_params': ['theta1', 'theta2'],
    }
    return samples, param_names, ground_truth


def generate_trig_separable(n=2000, noise=0.2, seed=42):
    """Trigonometric degeneracy: 2*sin(x) + cos(y) - z = 1."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n) + rng.normal(0, noise, n)
    y = rng.uniform(0, 2 * np.pi, n) + rng.normal(0, noise, n)
    z = 2 * np.sin(x) + np.cos(y) - 1.0 + rng.normal(0, noise, n)

    a, b, c, d = (rng.normal(0, 1, n) for _ in range(4))
    samples = np.column_stack([x, y, z, a, b, c, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': '2*sin(x) + cos(y) - z = 1',
        'component_functions': ['g1(x) = 2*sin(x)', 'g2(y) = cos(y)', 'g3(z) = -z'],
        'constant': 1.0,
        'degenerate_params': ['x', 'y', 'z'],
    }
    return samples, param_names, ground_truth


def generate_scurve_separable(n=2000, noise=0.2, seed=42):
    """S-curve degeneracy: (x^3 - 3x) + y + z = 0."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = -(x**3 - 3 * x) - y + rng.normal(0, noise, n)

    a, b, c, d = (rng.normal(0, 1, n) for _ in range(4))
    samples = np.column_stack([x, y, z, a, b, c, d])
    param_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd']

    ground_truth = {
        'equation': '(x^3 - 3x) + y + z = 0',
        'component_functions': ['g1(x) = x^3 - 3x', 'g2(y) = y', 'g3(z) = z'],
        'constant': 0.0,
        'degenerate_params': ['x', 'y', 'z'],
    }
    return samples, param_names, ground_truth


# Canonical benchmark registry: used by run_synthetic_experiments.py
SYNTHETIC_CASES = [
    {
        "name": "banana",
        "label": "Elliptic paraboloid (banana)",
        "generator": generate_banana_degeneracy,
        "coupling_depth": 3,
        "niterations": 200,
    },
    {
        "name": "cubic",
        "label": "Cubic degeneracy",
        "generator": generate_cubic_degeneracy,
        "coupling_depth": 2,
        "niterations": 200,
    },
    {
        "name": "trig",
        "label": "Trigonometric (sin/cos)",
        "generator": generate_trig_separable,
        "coupling_depth": 3,
        "niterations": 200,
    },
    {
        "name": "scurve",
        "label": "S-curve cubic",
        "generator": generate_scurve_separable,
        "coupling_depth": 3,
        "niterations": 200,
    },
]
