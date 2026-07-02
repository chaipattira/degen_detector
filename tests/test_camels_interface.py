# ABOUTME: Integration test for the CAMELS SBI → degen_detector interface.
# ABOUTME: Verifies that a (n_samples, 28) posterior array feeds correctly into run_detector.
"""Test that degen_detector correctly accepts SBI posterior output shape.

Issue #2 testing decision:
  Given a synthetic (20000, 28) samples array where columns 0 and 1 satisfy
  sigma8 * Omega_m^0.5 ≈ const (the S8 degeneracy), run_detector() with
  coupling_depth=2, max_fits=1 must identify the top-ranked pair as (0, 1)
  with R²_ortho > 0.95.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_camels_like_samples(n: int = 1000, seed: int = 42) -> tuple:
    """Synthetic (n, 28) posterior with S8 degeneracy in columns 0 and 1."""
    rng = np.random.default_rng(seed)

    # Column 0: Omega_m ~ Uniform(0.1, 0.5)
    omega_m = rng.uniform(0.1, 0.5, n)
    # Column 1: sigma_8 such that sigma_8 * omega_m^0.5 = 0.8 + small noise
    s8_const = 0.8
    sigma_8 = s8_const / np.sqrt(omega_m) + rng.normal(0, 0.005, n)

    # 26 independent nuisance parameters (CAMELS astrophysical params)
    nuisance = rng.uniform(0.5, 2.0, (n, 26))

    samples = np.column_stack([omega_m, sigma_8, nuisance])

    # Realistic CAMELS SB28 param names (first 2 cosmological)
    param_names = ["Omega_m", "sigma_8"] + [f"theta_{i}" for i in range(2, 28)]

    return samples, param_names


def test_camels_like_input_shape():
    """run_detector accepts (n, 28) array without error."""
    from degen_detector import run_detector

    samples, param_names = _make_camels_like_samples(n=500)
    assert samples.shape == (500, 28)
    assert len(param_names) == 28

    with tempfile.TemporaryDirectory() as tmp:
        result = run_detector(
            samples,
            param_names,
            output_dir=Path(tmp) / "out",
            coupling_depth=2,
            max_fits=1,
            niterations=1,
            batch_size=50,
        )
    assert result is not None
    assert len(result.fits) >= 1


def test_camels_s8_degeneracy_detected():
    """Top-ranked pair is (Omega_m, sigma_8) and R²_ortho > 0.95 for S8 degeneracy."""
    from degen_detector import run_detector

    samples, param_names = _make_camels_like_samples(n=2000)

    with tempfile.TemporaryDirectory() as tmp:
        result = run_detector(
            samples,
            param_names,
            output_dir=Path(tmp) / "out",
            coupling_depth=2,
            max_fits=1,
            niterations=5,
            batch_size=100,
            log_mode=True,  # S8 is a multiplicative degeneracy
        )

    assert len(result.fits) >= 1
    top = result.fits[0]
    assert set(top.param_names) == {"Omega_m", "sigma_8"}, (
        f"Expected {{Omega_m, sigma_8}} as top pair, got {top.param_names}"
    )
    best_fit = top.fit
    assert best_fit is not None
    assert best_fit.orthogonal_r2 > 0.95, (
        f"Expected R²_ortho > 0.95 for S8 degeneracy, got {best_fit.orthogonal_r2:.4f}"
    )


def test_load_weighted_resamples_to_target_size():
    """load_weighted resamples importance-weighted posterior to equal-weight output."""
    from degen_detector import load_weighted

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 5))
    weights = rng.uniform(0, 1, 500)

    resampled = load_weighted(samples, weights, n_samples=200, seed=0)
    assert resampled.shape == (200, 5)
    assert resampled.dtype == samples.dtype or np.issubdtype(resampled.dtype, np.floating)
