import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from degen_detector.loaders import load_numpy


class TestLoadNumpy:
    def test_accepts_ndarray_directly(self):
        arr = np.random.randn(100, 3)
        samples, names = load_numpy(arr, ["a", "b", "c"])
        np.testing.assert_array_equal(samples, arr)
        assert names == ["a", "b", "c"]

    def test_loads_npy_file(self, tmp_path):
        arr = np.random.randn(50, 2)
        path = tmp_path / "samples.npy"
        np.save(path, arr)
        samples, names = load_numpy(path, ["x", "y"])
        np.testing.assert_array_equal(samples, arr)
        assert names == ["x", "y"]

    def test_loads_npz_samples_key(self, tmp_path):
        arr = np.random.randn(40, 2)
        path = tmp_path / "data.npz"
        np.savez(path, samples=arr, other=np.zeros(5))
        samples, names = load_numpy(path, ["p", "q"])
        np.testing.assert_array_equal(samples, arr)

    def test_loads_npz_first_array_with_warning(self, tmp_path):
        arr = np.random.randn(30, 2)
        path = tmp_path / "data.npz"
        np.savez(path, mydata=arr)  # no "samples" key
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            samples, names = load_numpy(path, ["a", "b"])
        assert any("first array" in str(warning.message).lower() for warning in w)
        np.testing.assert_array_equal(samples, arr)

    def test_raises_on_shape_mismatch(self):
        arr = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="param_names"):
            load_numpy(arr, ["a", "b"])  # 3 cols but 2 names

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_numpy(tmp_path / "nonexistent.npy", ["a"])

    def test_raises_on_1d_array(self):
        arr = np.random.randn(50)
        with pytest.raises(ValueError, match="2-D"):
            load_numpy(arr, ["a"])


getdist = pytest.importorskip("getdist", reason="getdist not installed")

from degen_detector.loaders import load_getdist


class TestLoadGetdist:
    def test_loads_params_from_chain(self, tmp_path):
        """Write a minimal getdist chain and verify params load correctly."""
        import numpy as np
        from getdist import MCSamples

        rng = np.random.default_rng(42)
        n = 200
        s8 = rng.normal(0.81, 0.02, n)
        om = rng.normal(0.31, 0.01, n)
        mc = MCSamples(
            samples=np.column_stack([s8, om]),
            names=["sigma8", "omegam"],
            labels=["\\sigma_8", "\\Omega_m"],
        )
        root = str(tmp_path / "test_chain")
        mc.saveAsText(root)

        samples, names = load_getdist(root, ["sigma8", "omegam"])
        assert samples.shape[1] == 2
        assert names == ["sigma8", "omegam"]
        assert abs(float(np.median(samples[:, 0])) - 0.81) < 0.05

    def test_warns_and_skips_missing_param(self, tmp_path):
        import numpy as np
        from getdist import MCSamples

        rng = np.random.default_rng(1)
        mc = MCSamples(
            samples=rng.normal(size=(100, 1)),
            names=["sigma8"],
        )
        root = str(tmp_path / "chain")
        mc.saveAsText(root)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            samples, names = load_getdist(root, ["sigma8", "nonexistent"])
        assert any("nonexistent" in str(warning.message) for warning in w)
        assert "nonexistent" not in names
        assert "sigma8" in names

    def test_raises_if_no_params_loaded(self, tmp_path):
        import numpy as np
        from getdist import MCSamples

        rng = np.random.default_rng(2)
        mc = MCSamples(samples=rng.normal(size=(50, 1)), names=["sigma8"])
        root = str(tmp_path / "chain")
        mc.saveAsText(root)

        with pytest.raises(ValueError, match="No params"):
            load_getdist(root, ["nonexistent_param"])


emcee = pytest.importorskip("emcee", reason="emcee not installed")

from degen_detector.loaders import load_emcee


class TestLoadEmcee:
    def _make_backend(self, tmp_path, n_walkers=10, n_steps=50, n_dim=3, labels=None):
        """Create a minimal emcee HDFBackend file."""
        import emcee
        import numpy.random as npr

        path = tmp_path / "chain.h5"
        backend = emcee.backends.HDFBackend(str(path))
        backend.reset(n_walkers, n_dim)

        rng = np.random.default_rng(42)
        rs = npr.RandomState()
        coords = rng.standard_normal((n_walkers, n_dim))
        log_prob = -0.5 * np.sum(coords**2, axis=1)

        for _ in range(n_steps):
            state = emcee.State(coords, log_prob=log_prob, random_state=rs.get_state())
            accepted = np.ones(n_walkers, dtype=bool)
            backend.grow(1, None)
            backend.save_step(state, accepted)
            coords = coords + rng.standard_normal(coords.shape) * 0.1
            log_prob = -0.5 * np.sum(coords**2, axis=1)

        if labels is not None:
            import h5py
            with h5py.File(str(path), "a") as f:
                f.attrs["param_names"] = labels

        return path

    def test_loads_with_explicit_params(self, tmp_path):
        path = self._make_backend(tmp_path, n_dim=3)
        samples, names = load_emcee(path, params=["a", "b", "c"])
        assert samples.shape[1] == 3
        assert names == ["a", "b", "c"]

    def test_loads_labels_from_file(self, tmp_path):
        path = self._make_backend(tmp_path, n_dim=2, labels=["sigma8", "omegam"])
        samples, names = load_emcee(path)
        assert names == ["sigma8", "omegam"]

    def test_auto_generates_theta_names_with_warning(self, tmp_path):
        path = self._make_backend(tmp_path, n_dim=2)  # no labels stored
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            samples, names = load_emcee(path)
        assert any("theta_" in str(warning.message) for warning in w)
        assert names == ["theta_0", "theta_1"]

    def test_burn_in_and_thin(self, tmp_path):
        path = self._make_backend(tmp_path, n_walkers=8, n_steps=40, n_dim=2)
        full, _ = load_emcee(path, params=["a", "b"])
        thinned, _ = load_emcee(path, params=["a", "b"], burn_in=10, thin=2)
        assert len(thinned) < len(full)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_emcee(tmp_path / "nonexistent.h5")
