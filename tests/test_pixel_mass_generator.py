import numpy as np
import pytest
from degen_detector.testing import generate_pixel_mass


def test_output_shapes_default():
    samples, param_names, gt = generate_pixel_mass()
    assert samples.shape == (2000, 9), "Default 3x3 grid → 9 pixels, 2000 samples"
    assert len(param_names) == 9


def test_param_names():
    _, param_names, _ = generate_pixel_mass(grid_size=2)
    assert param_names == ["m1", "m2", "m3", "m4"]


def test_ground_truth_keys():
    _, _, gt = generate_pixel_mass()
    for key in ("equation", "M_obs", "sigma", "grid_size", "degenerate_params"):
        assert key in gt, f"Missing ground truth key: {key}"


def test_mass_conservation_tight():
    """Posterior sum should be near M_obs with small spread for small sigma."""
    M_obs = 2.0
    samples, _, _ = generate_pixel_mass(grid_size=3, sigma=0.1, M_obs=M_obs, seed=0)
    total_mass = samples.sum(axis=1)
    assert abs(total_mass.mean() - M_obs) < 0.1
    # Var(Σmᵢ|M) = n·σ²/(σ²+n) = 9·0.01/9.01 ≈ 0.010  → std ≈ 0.10
    assert total_mass.std() < 0.15


def test_mass_conservation_loose():
    """With large sigma, posterior total shrinks toward prior mean (0), not M_obs.

    E[Σmᵢ | M] = n*M/(σ²+n).  For σ=5, n=9, M=5: 9*5/34 ≈ 1.32.
    """
    sigma, M_obs, n = 5.0, 5.0, 9
    samples, _, _ = generate_pixel_mass(grid_size=3, sigma=sigma, M_obs=M_obs, seed=0)
    total_mass = samples.sum(axis=1)
    expected_mean = n * M_obs / (sigma**2 + n)  # ≈ 1.32
    assert abs(total_mass.mean() - expected_mean) < 0.3


def test_custom_grid_size():
    samples, param_names, gt = generate_pixel_mass(grid_size=2, n_samples=500)
    assert samples.shape == (500, 4)
    assert gt["grid_size"] == 2


def test_reproducible():
    s1, _, _ = generate_pixel_mass(seed=99)
    s2, _, _ = generate_pixel_mass(seed=99)
    np.testing.assert_array_equal(s1, s2)
