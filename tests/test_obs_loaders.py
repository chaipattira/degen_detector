# ABOUTME: Tests for observational chain loaders (DESI DR1, DES Y3).
# ABOUTME: Covers column detection, weight expansion, and sanity checks using synthetic data.
"""Tests for scripts/run_desi_logmode.py and scripts/run_des_y3_logmode.py loaders."""
import importlib.util
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

# ── Import loader functions from scripts (not installed packages) ─────────────
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _import_script(name):
    """Import a script module by filename."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


desi_mod = _import_script("run_desi_logmode")
des_mod = _import_script("run_des_y3_logmode")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_desi_chain(n_rows=200, n_extra_cols=20, seed=42):
    """Create a synthetic DESI-like chain array with correct column layout.

    Columns: weight, -logpost, logA, ns, H0, ombh2, omch2, As, omegam, omegamh2,
             omegal, YHe, Y_p, DHBBN, sigma8, [rest filled with random noise]
    """
    rng = np.random.default_rng(seed)
    n_cols = 2 + 13 + n_extra_cols  # 2 header + 13 before EFT + extras
    data = rng.standard_normal((n_rows, n_cols))

    # Column 0: weight (positive integers 1-5)
    data[:, 0] = rng.integers(1, 6, size=n_rows).astype(float)
    # Column 1: -logpost (large positive)
    data[:, 1] = rng.uniform(100, 200, size=n_rows)
    # Column 2: logA ~ 3.0
    data[:, 2] = rng.normal(3.0, 0.02, size=n_rows)
    # Column 3: ns ~ 0.97
    data[:, 3] = rng.normal(0.97, 0.005, size=n_rows)
    # Column 4: H0 ~ 68
    data[:, 4] = rng.normal(68.0, 0.8, size=n_rows)
    # Column 5: ombh2 ~ 0.0224
    data[:, 5] = rng.normal(0.0224, 0.0002, size=n_rows)
    # Column 6: omch2 ~ 0.12
    data[:, 6] = rng.normal(0.12, 0.002, size=n_rows)
    # Column 7: As ~ 2.1e-9 (derived)
    data[:, 7] = rng.normal(2.1e-9, 0.05e-9, size=n_rows)
    # Column 8: omegam ~ 0.31 (derived)
    data[:, 8] = rng.normal(0.31, 0.01, size=n_rows)
    # Columns 9-13: other derived cosmological quantities (plausible values)
    data[:, 9] = rng.normal(0.155, 0.005, size=n_rows)   # omegamh2
    data[:, 10] = rng.normal(0.69, 0.01, size=n_rows)    # omegal
    data[:, 11] = rng.normal(0.245, 0.001, size=n_rows)  # YHe
    data[:, 12] = rng.normal(0.244, 0.001, size=n_rows)  # Y_p
    data[:, 13] = rng.normal(25.3, 0.1, size=n_rows)     # DHBBN
    # Column 14: sigma8 ~ 0.81 (derived)
    data[:, 14] = rng.normal(0.81, 0.01, size=n_rows)

    return data


def _make_des_chain(n_rows=1000, n_nuisance=28, seed=42):
    """Create a synthetic DES Y3-like chain array.

    Columns: Omega_m (0), nuisance params (1..n_nuisance-1),
             sigma_8 (n_nuisance), derived quantities, weight (last).
    """
    rng = np.random.default_rng(seed)
    sigma8_col = n_nuisance
    n_derived = 3  # S8, s8omegamp25, etc.
    weight_col = n_nuisance + n_derived + 1
    n_cols = weight_col + 1

    data = rng.standard_normal((n_rows, n_cols))

    # Column 0: Omega_m ~ 0.29
    data[:, 0] = rng.normal(0.29, 0.03, size=n_rows)
    # Nuisance params (columns 1..n_nuisance-1): varied physical ranges
    for col in range(1, n_nuisance):
        data[:, col] = rng.uniform(-1.0, 2.0, size=n_rows)
    # sigma_8: ~ 0.78
    data[:, sigma8_col] = rng.normal(0.78, 0.04, size=n_rows)
    # Derived quantities after sigma_8
    data[:, sigma8_col + 1] = rng.normal(0.74, 0.02, size=n_rows)   # S8
    data[:, sigma8_col + 2] = rng.normal(0.80, 0.02, size=n_rows)   # other
    data[:, sigma8_col + 3] = rng.normal(0.82, 0.02, size=n_rows)   # other
    # Last column: weights (positive)
    data[:, weight_col] = rng.exponential(scale=1.0 / n_rows, size=n_rows)
    data[:, weight_col] += 1e-10  # avoid zero weights

    return data, sigma8_col, weight_col


# ─────────────────────────────────────────────────────────────────────────────
# DESI loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDesiSanityCheck:
    """Tests for physical sanity-check ranges in DESI loader."""

    def test_h0_in_expected_range(self):
        lo, hi = desi_mod.SANITY["H0"]
        assert lo < 68.0 < hi

    def test_ombh2_in_expected_range(self):
        lo, hi = desi_mod.SANITY["ombh2"]
        assert lo < 0.0224 < hi

    def test_sigma8_in_expected_range(self):
        lo, hi = desi_mod.SANITY["sigma8"]
        assert lo < 0.81 < hi

    def test_column_indices_within_42_cols(self):
        """All expected column indices fit within a 42-column chain."""
        max_col = max(
            desi_mod.COL_H0,
            desi_mod.COL_OMBH2,
            desi_mod.COL_OMCH2,
            desi_mod.COL_OMEGAM,
            desi_mod.COL_SIGMA8,
        )
        assert max_col < 42


class TestDesiChainLoading:
    """Tests for load_desi_chains() using tmp files."""

    def _write_chain_files(self, tmp_path, data, n_files=4):
        """Write synthetic data as multiple chain.N.txt files."""
        n_per_file = len(data) // n_files
        for i in range(1, n_files + 1):
            chunk = data[(i - 1) * n_per_file : i * n_per_file]
            np.savetxt(tmp_path / f"chain.{i}.txt", chunk)

    def test_loads_correct_shape(self, tmp_path):
        data = _make_desi_chain(n_rows=200)
        self._write_chain_files(tmp_path, data, n_files=4)

        samples, param_names = desi_mod.load_desi_chains(tmp_path, n_chains=4)

        assert samples.ndim == 2
        assert samples.shape[1] == 5  # H0, ombh2, omch2, omegam, sigma8
        assert param_names == ["H0", "ombh2", "omch2", "omegam", "sigma8"]

    def test_weight_expansion(self, tmp_path):
        """Samples with weight=3 should appear 3× after expansion."""
        data = _make_desi_chain(n_rows=40)
        # Set all weights to exactly 2
        data[:, desi_mod.COL_WEIGHT] = 2.0
        self._write_chain_files(tmp_path, data, n_files=1)

        samples, _ = desi_mod.load_desi_chains(tmp_path, n_chains=1)

        # Should have 2× the raw rows
        assert len(samples) == 40 * 2

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="chain.1.txt"):
            desi_mod.load_desi_chains(tmp_path, n_chains=1)

    def test_raises_if_too_few_columns(self, tmp_path):
        data = np.random.randn(20, 5)  # only 5 cols, needs at least 15
        np.savetxt(tmp_path / "chain.1.txt", data)
        with pytest.raises(ValueError, match="columns"):
            desi_mod.load_desi_chains(tmp_path, n_chains=1)

    def test_h0_values_are_physical(self, tmp_path):
        data = _make_desi_chain(n_rows=100)
        self._write_chain_files(tmp_path, data, n_files=2)
        samples, param_names = desi_mod.load_desi_chains(tmp_path, n_chains=2)

        h0_idx = param_names.index("H0")
        median_h0 = float(np.median(samples[:, h0_idx]))
        assert 40 < median_h0 < 100, f"H0 median {median_h0} out of physical range"

    def test_sigma8_values_are_physical(self, tmp_path):
        data = _make_desi_chain(n_rows=100)
        self._write_chain_files(tmp_path, data, n_files=2)
        samples, param_names = desi_mod.load_desi_chains(tmp_path, n_chains=2)

        s8_idx = param_names.index("sigma8")
        median_s8 = float(np.median(samples[:, s8_idx]))
        assert 0.6 < median_s8 < 1.1, f"sigma8 median {median_s8} out of physical range"


# ─────────────────────────────────────────────────────────────────────────────
# DES Y3 loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectSigma8Column:
    """Tests for detect_sigma8_column() auto-detection logic."""

    def test_detects_known_sigma8_column(self):
        data, sigma8_col, weight_col = _make_des_chain(n_rows=500, n_nuisance=28)
        detected = des_mod.detect_sigma8_column(data, weight_col)
        assert detected == sigma8_col

    def test_ignores_weight_column(self):
        """Weight column (last) should not be returned as sigma_8."""
        data, _, weight_col = _make_des_chain(n_rows=500, n_nuisance=10)
        detected = des_mod.detect_sigma8_column(data, weight_col)
        assert detected != weight_col

    def test_raises_if_no_candidate(self):
        """All columns are clearly outside sigma_8 range → RuntimeError."""
        rng = np.random.default_rng(0)
        # All columns have median far from [0.65, 1.00]
        data = np.column_stack([
            rng.normal(0.3, 0.02, 200),   # col 0: Omega_m-like
            rng.normal(5.0, 0.1, 200),    # col 1: too large
            rng.normal(0.02, 0.001, 200), # col 2: too small
            rng.uniform(0, 1, 200),       # col 3: weights
        ])
        with pytest.raises(RuntimeError, match="sigma_8"):
            des_mod.detect_sigma8_column(data, weight_col=3)

    def test_skips_early_cosmo_columns(self):
        """h (~0.68) is in the sigma_8 range but appears in the first 5 columns; must be skipped."""
        rng = np.random.default_rng(42)
        n = 500
        data = np.column_stack([
            rng.normal(0.3, 0.03, n),   # col 0: Omega_m
            rng.normal(0.68, 0.04, n),  # col 1: h (same range as sigma_8; col < 5 → skipped)
            rng.normal(0.05, 0.005, n), # col 2: Omega_b (too small)
            rng.normal(0.97, 0.01, n),  # col 3: ns (std too small after filter)
            rng.normal(2.1e-9, 0.05e-9, n),  # col 4: A_s (way too small)
            rng.uniform(-1, 1, n),      # cols 5-9: nuisance (outside median range)
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            rng.normal(0.78, 0.04, n),  # col 10: sigma_8 (first valid candidate)
            rng.normal(0.77, 0.019, n), # col 11: S8 (std too small → excluded)
            rng.uniform(0, 0.1, n),     # col 12: weights
        ])
        detected = des_mod.detect_sigma8_column(data, weight_col=12)
        assert detected == 10


class TestLoadDesSamples:
    """Tests for load_des_samples() using tmp files."""

    def _write_des_chain(self, tmp_path, data, filename="chain.txt"):
        fpath = tmp_path / filename
        np.savetxt(fpath, data)
        return fpath

    def test_loads_correct_params(self, tmp_path):
        data, sigma8_col, _ = _make_des_chain(n_rows=500, n_nuisance=28)
        fpath = self._write_des_chain(tmp_path, data)

        samples, param_names = des_mod.load_des_samples(
            fpath, sigma8_col=sigma8_col, n_resample=200
        )

        assert samples.shape == (200, 2)
        assert param_names == ["Omega_m", "sigma_8"]

    def test_omega_m_is_col_0(self, tmp_path):
        data, sigma8_col, _ = _make_des_chain(n_rows=500, n_nuisance=28)
        fpath = self._write_des_chain(tmp_path, data)

        samples, param_names = des_mod.load_des_samples(
            fpath, sigma8_col=sigma8_col, n_resample=300
        )

        om_idx = param_names.index("Omega_m")
        median_om = float(np.median(samples[:, om_idx]))
        assert 0.15 < median_om < 0.55

    def test_sigma8_values_physical(self, tmp_path):
        data, sigma8_col, _ = _make_des_chain(n_rows=500, n_nuisance=28)
        fpath = self._write_des_chain(tmp_path, data)

        samples, param_names = des_mod.load_des_samples(
            fpath, sigma8_col=sigma8_col, n_resample=300
        )

        s8_idx = param_names.index("sigma_8")
        median_s8 = float(np.median(samples[:, s8_idx]))
        assert 0.5 < median_s8 < 1.1

    def test_resampling_respects_seed(self, tmp_path):
        data, sigma8_col, _ = _make_des_chain(n_rows=500, n_nuisance=28)
        fpath = self._write_des_chain(tmp_path, data)

        s1, _ = des_mod.load_des_samples(fpath, sigma8_col=sigma8_col, n_resample=100, seed=7)
        s2, _ = des_mod.load_des_samples(fpath, sigma8_col=sigma8_col, n_resample=100, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_samples(self, tmp_path):
        data, sigma8_col, _ = _make_des_chain(n_rows=500, n_nuisance=28)
        fpath = self._write_des_chain(tmp_path, data)

        s1, _ = des_mod.load_des_samples(fpath, sigma8_col=sigma8_col, n_resample=100, seed=1)
        s2, _ = des_mod.load_des_samples(fpath, sigma8_col=sigma8_col, n_resample=100, seed=2)
        assert not np.array_equal(s1, s2)

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            des_mod.load_des_samples(tmp_path / "nonexistent.txt")

    def test_zero_weights_clipped(self, tmp_path):
        """Rows with weight=0 should be clipped; function should not crash."""
        data, sigma8_col, weight_col = _make_des_chain(n_rows=200, n_nuisance=10)
        data[:10, weight_col] = 0.0   # force 10 zero weights
        fpath = self._write_des_chain(tmp_path, data)

        # Should not raise
        samples, _ = des_mod.load_des_samples(
            fpath, sigma8_col=sigma8_col, n_resample=50
        )
        assert len(samples) == 50

    def test_auto_detects_sigma8_column(self, tmp_path):
        """load_des_samples can auto-detect sigma_8 if sigma8_col=None."""
        data, sigma8_col, _ = _make_des_chain(n_rows=500, n_nuisance=28)
        fpath = self._write_des_chain(tmp_path, data)

        # Pass sigma8_col=None to trigger auto-detection
        samples, param_names = des_mod.load_des_samples(
            fpath, sigma8_col=None, n_resample=100
        )
        assert "sigma_8" in param_names
