# ABOUTME: Generates synthetic datasets with known parameter degeneracies.
# ABOUTME: Used for testing and validating the degeneracy detection pipeline.

import numpy as np


def generate_linear_degeneracy(n=2000, noise=0.01, seed=42):
    """Generate 3 params: y = 2*x + 1 + noise, z independent."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 2 * x + 1 + rng.normal(0, noise, n)
    z = rng.normal(0, 1, n)
    samples = np.column_stack([x, y, z])
    return samples, ["x", "y", "z"]


def generate_power_law_degeneracy(n=2000, noise=0.01, seed=42):
    """Generate 3 params: sigma8 * Om^0.5 ~ const, ns independent."""
    rng = np.random.default_rng(seed)
    Om = rng.uniform(0.2, 0.4, n)
    S8_const = 0.83
    sigma8 = S8_const / Om**0.5 + rng.normal(0, noise, n)
    ns = rng.normal(0.96, 0.01, n)
    samples = np.column_stack([Om, sigma8, ns])
    return samples, ["Om", "sigma8", "ns"]


def generate_multigroup_degeneracy(n=2000, noise=0.01, seed=42):
    """Generate 5 params: (a,b) with b=3a-2, (c,d) with d=exp(c), e independent."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = 3 * a - 2 + rng.normal(0, noise, n)
    c = rng.uniform(-1, 1, n)
    d = np.exp(c) + rng.normal(0, noise, n)
    e = rng.normal(0, 1, n)
    samples = np.column_stack([a, b, c, d, e])
    return samples, ["a", "b", "c", "d", "e"]


def generate_three_param_degeneracy(n=2000, noise=0.01, seed=42):
    """Generate 3 params: z = x^2 + y."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = x**2 + y + rng.normal(0, noise, n)
    samples = np.column_stack([x, y, z])
    return samples, ["x", "y", "z"]
