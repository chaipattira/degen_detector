import tempfile
from pathlib import Path

import numpy as np
import pytest

from degen_detector import run_detector


def _banana_samples(n=300, seed=42):
    """Minimal samples with a quadratic degeneracy: 2*t1^2 + t2^2 - t3 = 0.5."""
    rng = np.random.default_rng(seed)
    t1 = rng.uniform(-1, 1, n)
    t2 = rng.uniform(-1, 1, n)
    t3 = 2 * t1**2 + t2**2 - 0.5 + rng.normal(0, 0.01, n)
    return np.column_stack([t1, t2, t3]), ["t1", "t2", "t3"]


class TestRunPipeline:
    def test_returns_coupling_search_result(self, tmp_path):
        from degen_detector import CouplingSearchResult
        samples, names = _banana_samples()
        result = run_detector(
            samples, names, tmp_path / "out",
            coupling_depth=2, max_fits=1, niterations=1, batch_size=50,
        )
        assert isinstance(result, CouplingSearchResult)

    def test_creates_output_dir(self, tmp_path):
        samples, names = _banana_samples()
        out = tmp_path / "nested" / "output"
        run_detector(
            samples, names, out,
            max_fits=1, niterations=1, batch_size=50,
        )
        assert out.exists()

    def test_saves_result_pkl(self, tmp_path):
        from degen_detector.io import load_pickle
        samples, names = _banana_samples()
        out = tmp_path / "out"
        run_detector(samples, names, out, max_fits=1, niterations=1, batch_size=50)
        pkl = out / "result.pkl"
        assert pkl.exists()
        data = load_pickle(pkl)
        assert "samples" in data
        assert "param_names" in data
        assert "result" in data

    def test_saves_summary_txt(self, tmp_path):
        samples, names = _banana_samples()
        out = tmp_path / "out"
        run_detector(samples, names, out, max_fits=1, niterations=1, batch_size=50)
        summary = out / "summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "MI" in text
        assert "R²" in text

    def test_no_timestamp_subdir_added(self, tmp_path):
        """output_dir is used directly — no create_output_dir timestamping."""
        samples, names = _banana_samples()
        out = tmp_path / "exact_dir"
        run_detector(samples, names, out, max_fits=1, niterations=1, batch_size=50)
        assert (out / "result.pkl").exists()
        # Should NOT have created a timestamped subdir
        subdirs = [p for p in out.iterdir() if p.is_dir() and p.name != "diagnostics"]
        assert len(subdirs) == 0

    def test_log_mode_runs_without_error(self, tmp_path):
        """log_mode=True should use DegenLogMode on all-positive samples."""
        rng = np.random.default_rng(0)
        samples = rng.uniform(0.1, 2.0, (200, 3))
        names = ["a", "b", "c"]
        run_detector(
            samples, names, tmp_path / "log_out",
            log_mode=True, max_fits=1, niterations=1, batch_size=50,
        )
        assert (tmp_path / "log_out" / "result.pkl").exists()
