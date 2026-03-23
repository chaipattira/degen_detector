# ABOUTME: Tests for shared I/O utilities (pickle save/load, output directory creation).
# ABOUTME: Verifies round-trip pickle serialization and timestamped directory naming.

"""Tests for degen_detector.io module."""

import tempfile
from pathlib import Path

from degen_detector.io import save_pickle, load_pickle, create_output_dir


def test_pickle_round_trip():
    """Save and load a dict through pickle."""
    data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pkl"
        save_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded == data


def test_pickle_with_numpy():
    """Save and load numpy arrays through pickle."""
    import numpy as np
    data = {"samples": np.array([[1.0, 2.0], [3.0, 4.0]])}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pkl"
        save_pickle(data, path)
        loaded = load_pickle(path)
        np.testing.assert_array_equal(loaded["samples"], data["samples"])


def test_create_output_dir():
    """Create timestamped output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "outputs"
        output_dir = create_output_dir(base)
        assert output_dir.exists()
        assert output_dir.parent == base
        # Directory name should be a timestamp (YYYYMMDD_HHMMSS format)
        assert len(output_dir.name) == 15  # 8 date + 1 underscore + 6 time


def test_create_output_dir_creates_parents():
    """create_output_dir creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "deep" / "nested" / "outputs"
        output_dir = create_output_dir(base)
        assert output_dir.exists()
