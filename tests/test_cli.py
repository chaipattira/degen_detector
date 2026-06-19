import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestCliNumpy:
    def test_numpy_format_runs(self, tmp_path):
        """CLI with --format numpy completes without error on tiny data."""
        import matplotlib
        matplotlib.use("Agg")
        from degen_detector.cli import main

        arr = np.random.randn(200, 2)
        samples_path = tmp_path / "samples.npy"
        np.save(samples_path, arr)
        out = tmp_path / "out"

        argv = [
            "degen-detect", str(samples_path),
            "--format", "numpy",
            "--param-names", "a", "b",
            "--output-dir", str(out),
            "--max-fits", "1",
            "--niterations", "1",
            "--batch-size", "50",
        ]
        with patch.object(sys, "argv", argv):
            main()

        assert (out / "result.pkl").exists()
        assert (out / "summary.txt").exists()

    def test_numpy_missing_param_names_errors(self, tmp_path):
        """--param-names is required for --format numpy."""
        import matplotlib
        matplotlib.use("Agg")
        from degen_detector.cli import main

        arr = np.random.randn(50, 2)
        path = tmp_path / "s.npy"
        np.save(path, arr)

        argv = ["degen-detect", str(path), "--format", "numpy", "--output-dir", str(tmp_path / "o")]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                main()

    def test_getdist_missing_params_errors(self, tmp_path):
        """--params is required for --format getdist."""
        import matplotlib
        matplotlib.use("Agg")
        from degen_detector.cli import main

        argv = ["degen-detect", str(tmp_path / "chain"), "--format", "getdist",
                "--output-dir", str(tmp_path / "o")]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                main()
