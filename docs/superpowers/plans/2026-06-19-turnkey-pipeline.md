# Turnkey Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `degen_detector/loaders.py`, `degen_detector/pipeline.py`, and `degen_detector/cli.py` so users can point at a posterior chain and get plots + pkl + summary with a single function call or CLI command.

**Architecture:** Three new files with clear single responsibilities: loaders translate format-specific chain files into `(ndarray, list[str])`, `run_pipeline` orchestrates the full detect→save→summarise→diagnose sequence, and `cli.py` is a thin argparse wrapper over the loaders + pipeline. All three are exposed as public exports from the top-level package.

**Tech Stack:** Python 3.11, numpy, pytest, getdist (optional), emcee (optional), matplotlib (Agg backend for headless HPC)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `degen_detector/__init__.py` | Remove broken `detect` import; add new public exports |
| **Create** | `degen_detector/loaders.py` | `load_numpy`, `load_getdist`, `load_emcee` |
| **Create** | `degen_detector/pipeline.py` | `run_pipeline` orchestrator |
| **Create** | `degen_detector/cli.py` | `degen-detect` CLI entry point |
| **Create** | `pyproject.toml` | Package metadata + `degen-detect` console script entry point |
| **Modify** | `tests/test_imports.py` | Remove `detect`; add new exports to import test |
| **Create** | `tests/test_loaders.py` | Tests for all three loaders |
| **Create** | `tests/test_pipeline.py` | Tests for `run_pipeline` |
| **Create** | `tests/test_cli.py` | Tests for `degen-detect` CLI |

---

## Task 1: Fix broken `detect` import

The existing `degen_detector/__init__.py` imports `detect` from `core.py` but that symbol does not exist — this causes an `ImportError` and breaks `test_imports.py`. Fix it before anything else.

**Files:**
- Modify: `degen_detector/__init__.py:3`
- Modify: `tests/test_imports.py:13`

- [ ] **Step 1: Remove `detect` from `__init__.py`**

Change line 3 of `degen_detector/__init__.py` from:
```python
from degen_detector.core import DegenDetector, CouplingFit, CouplingSearchResult, detect
```
to:
```python
from degen_detector.core import DegenDetector, CouplingFit, CouplingSearchResult
```

- [ ] **Step 2: Remove `detect` from the import test**

In `tests/test_imports.py`, remove `detect,` from `test_top_level_imports`.

- [ ] **Step 3: Run import tests**

```bash
cd /home/x-ctirapongpra/scratch/degen_detector
source .venv/bin/activate
python -m pytest tests/test_imports.py -v
```
Expected: all 5 tests PASS (previously hanging/erroring on `detect`).

- [ ] **Step 4: Commit**

```bash
git add degen_detector/__init__.py tests/test_imports.py
git commit -m "fix: remove broken detect import from __init__ and test_imports"
```

---

## Task 2: Create `degen_detector/loaders.py` — `load_numpy`

Start with the simplest loader: no external deps, pure numpy.

**Files:**
- Create: `degen_detector/loaders.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests for `load_numpy`**

Create `tests/test_loaders.py`:

```python
# ABOUTME: Tests for degen_detector.loaders — load_numpy, load_getdist, load_emcee.

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_loaders.py::TestLoadNumpy -v
```
Expected: `ImportError` — `loaders` module doesn't exist yet.

- [ ] **Step 3: Create `degen_detector/loaders.py` with `load_numpy`**

```python
# ABOUTME: Format-specific loaders that return (ndarray, list[str]) for use with run_pipeline.

"""Sample loaders for degen_detector pipeline.

Each loader returns (samples, param_names) where samples is an ndarray of
shape (n_samples, n_params) and param_names is a list of strings.
"""

import warnings
from pathlib import Path

import numpy as np


def load_numpy(array_or_path, param_names):
    """Load samples from a numpy array or .npy/.npz file.

    Parameters
    ----------
    array_or_path : ndarray or path-like
        Array directly, or path to a .npy or .npz file.
        For .npz files, loads the array named "samples" if present,
        otherwise the first array (with a UserWarning).
    param_names : list[str]
        Names for each column. Required — no metadata to infer from.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    if isinstance(array_or_path, np.ndarray):
        samples = array_or_path
    else:
        path = Path(array_or_path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if path.suffix == ".npz":
            data = np.load(path)
            if "samples" in data:
                samples = data["samples"]
            else:
                keys = list(data.keys())
                warnings.warn(
                    f"No 'samples' key in {path.name}; loading first array '{keys[0]}'. "
                    f"Available keys: {keys}",
                    UserWarning,
                    stacklevel=2,
                )
                samples = data[keys[0]]
        else:
            samples = np.load(path)

    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array, got shape {samples.shape}. "
            f"Samples must have shape (n_samples, n_params)."
        )
    if samples.shape[1] != len(param_names):
        raise ValueError(
            f"Shape mismatch: array has {samples.shape[1]} columns but "
            f"param_names has {len(param_names)} entries: {param_names}"
        )
    return samples, list(param_names)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_loaders.py::TestLoadNumpy -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add degen_detector/loaders.py tests/test_loaders.py
git commit -m "feat: add load_numpy loader with tests"
```

---

## Task 3: Add `load_getdist` to loaders

`getdist` is installed (1.7.7). The loader still guards the import with a helpful `ImportError` message so users who install the package without getdist get a clear error.

**Files:**
- Modify: `degen_detector/loaders.py`
- Modify: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests for `load_getdist`**

Append to `tests/test_loaders.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_loaders.py::TestLoadGetdist -v
```
Expected: `ImportError` — `load_getdist` not yet implemented.

- [ ] **Step 3: Add `load_getdist` to `degen_detector/loaders.py`**

Append after `load_numpy`:

```python
def load_getdist(chain_root, params, ignore_rows=0.3):
    """Load samples from a getdist/CosmoMC chain.

    Parameters
    ----------
    chain_root : str or path-like
        Path stem for the chain files (without _1.txt, _2.txt, etc.).
        Same convention used by getdist.loadMCSamples.
    params : list[str]
        Parameter names to extract. Required — no default.
    ignore_rows : float
        Fraction of chain to discard as burn-in (default 0.3).

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    try:
        from getdist import loadMCSamples
    except ImportError as exc:
        raise ImportError(
            "getdist is required for load_getdist. "
            "Install it with: pip install getdist"
        ) from exc

    mc = loadMCSamples(str(chain_root), settings={"ignore_rows": ignore_rows})
    p = mc.getParams()

    cols, available = [], []
    for name in params:
        arr = getattr(p, name, None)
        if arr is None:
            warnings.warn(
                f"Parameter '{name}' not found in chain at {chain_root}; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        cols.append(arr)
        available.append(name)

    if not available:
        raise ValueError(
            f"No params could be loaded from {chain_root}. "
            f"Requested: {params}. Check the chain's .paramnames file."
        )

    return np.column_stack(cols), available
```

- [ ] **Step 4: Run all loader tests**

```bash
python -m pytest tests/test_loaders.py -v
```
Expected: `TestLoadNumpy` and `TestLoadGetdist` all PASS.

- [ ] **Step 5: Commit**

```bash
git add degen_detector/loaders.py tests/test_loaders.py
git commit -m "feat: add load_getdist loader with tests"
```

---

## Task 4: Add `load_emcee` to loaders

`emcee` is optional. The loader handles missing backend labels by generating `theta_i` names.

**Files:**
- Modify: `degen_detector/loaders.py`
- Modify: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests for `load_emcee`**

Append to `tests/test_loaders.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_loaders.py::TestLoadEmcee -v
```
Expected: `ImportError` — `load_emcee` not yet implemented.

- [ ] **Step 3: Add `load_emcee` to `degen_detector/loaders.py`**

Append after `load_getdist`:

```python
def load_emcee(h5_path, params=None, burn_in=0, thin=1):
    """Load samples from an emcee HDFBackend HDF5 file.

    Parameters
    ----------
    h5_path : path-like
        Path to the emcee HDFBackend HDF5 file.
    params : list[str] or None
        Parameter names. If None, tries to read from the file's
        "param_names" attribute; falls back to theta_0, theta_1, …
        with a UserWarning if no labels are stored.
    burn_in : int
        Number of steps to discard as burn-in before flattening.
    thin : int
        Keep every `thin`-th step before flattening.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee is required for load_emcee. "
            "Install it with: pip install emcee"
        ) from exc

    path = Path(h5_path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    chain = backend.get_chain(discard=burn_in, thin=thin, flat=True)
    # chain shape: (n_samples, n_dim)

    n_dim = chain.shape[1]

    if params is not None:
        if len(params) != n_dim:
            raise ValueError(
                f"params has {len(params)} entries but chain has {n_dim} dimensions."
            )
        param_names = list(params)
    else:
        # Try to read labels stored as a file attribute
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                stored = f.attrs.get("param_names", None)
            if stored is not None:
                param_names = list(stored)
            else:
                raise KeyError("no labels")
        except Exception:
            param_names = [f"theta_{i}" for i in range(n_dim)]
            warnings.warn(
                f"No parameter labels found in {path.name}; "
                f"using auto-generated names: {param_names}. "
                f"Pass params=[...] explicitly to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )

    return chain, param_names
```

- [ ] **Step 4: Run all loader tests**

```bash
python -m pytest tests/test_loaders.py -v
```
Expected: all loader tests PASS.

- [ ] **Step 5: Commit**

```bash
git add degen_detector/loaders.py tests/test_loaders.py
git commit -m "feat: add load_emcee loader with tests"
```

---

## Task 5: Create `degen_detector/pipeline.py` — `run_pipeline`

`run_pipeline` is the core orchestrator. Tests use tiny synthetic data and `niterations=1` to keep runtime under a few seconds.

**Files:**
- Create: `degen_detector/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
# ABOUTME: Tests for run_pipeline — the full detect→save→summarise→diagnose orchestrator.

import tempfile
from pathlib import Path

import numpy as np
import pytest

from degen_detector import run_pipeline


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
        result = run_pipeline(
            samples, names, tmp_path / "out",
            coupling_depth=2, max_fits=1, niterations=1, batch_size=50,
        )
        assert isinstance(result, CouplingSearchResult)

    def test_creates_output_dir(self, tmp_path):
        samples, names = _banana_samples()
        out = tmp_path / "nested" / "output"
        run_pipeline(
            samples, names, out,
            max_fits=1, niterations=1, batch_size=50,
        )
        assert out.exists()

    def test_saves_result_pkl(self, tmp_path):
        from degen_detector.io import load_pickle
        samples, names = _banana_samples()
        out = tmp_path / "out"
        run_pipeline(samples, names, out, max_fits=1, niterations=1, batch_size=50)
        pkl = out / "result.pkl"
        assert pkl.exists()
        data = load_pickle(pkl)
        assert "samples" in data
        assert "param_names" in data
        assert "result" in data

    def test_saves_summary_txt(self, tmp_path):
        samples, names = _banana_samples()
        out = tmp_path / "out"
        run_pipeline(samples, names, out, max_fits=1, niterations=1, batch_size=50)
        summary = out / "summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "MI" in text
        assert "R²" in text

    def test_no_timestamp_subdir_added(self, tmp_path):
        """output_dir is used directly — no create_output_dir timestamping."""
        samples, names = _banana_samples()
        out = tmp_path / "exact_dir"
        run_pipeline(samples, names, out, max_fits=1, niterations=1, batch_size=50)
        assert (out / "result.pkl").exists()
        # Should NOT have created a timestamped subdir
        subdirs = [p for p in out.iterdir() if p.is_dir() and p.name != "diagnostics"]
        assert len(subdirs) == 0

    def test_log_mode_runs_without_error(self, tmp_path):
        """log_mode=True should use DegenLogMode on all-positive samples."""
        rng = np.random.default_rng(0)
        samples = rng.uniform(0.1, 2.0, (200, 3))
        names = ["a", "b", "c"]
        run_pipeline(
            samples, names, tmp_path / "log_out",
            log_mode=True, max_fits=1, niterations=1, batch_size=50,
        )
        assert (tmp_path / "log_out" / "result.pkl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pipeline.py -v
```
Expected: `ImportError` — `run_pipeline` not yet exported.

- [ ] **Step 3: Create `degen_detector/pipeline.py`**

```python
# ABOUTME: Full pipeline orchestrator — runs detection, saves pkl, writes summary, runs diagnostics.

"""run_pipeline: point at samples, get diagnostics."""

import sys
from pathlib import Path

import numpy as np

from degen_detector.core import DegenDetector
from degen_detector.diagnostics.runner import DiagnosticsRunner
from degen_detector.io import save_pickle
from degen_detector.transforms import DegenLogMode


def run_pipeline(
    samples,
    param_names,
    output_dir,
    *,
    log_mode=False,
    coupling_depth=2,
    max_fits=2,
    niterations=200,
    max_complexity=15,
    batch_size=1000,
):
    """Run the full degeneracy detection pipeline and save all outputs.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples.
    param_names : list[str]
        Names matching columns of samples.
    output_dir : path-like
        Directory to write outputs. Created if it does not exist.
        No timestamp subdirectory is added.
    log_mode : bool
        If True, use DegenLogMode (log-transforms all params not already
        named log_* / ln_*, drops rows with non-finite values after transform).
        If False, use DegenDetector.
    coupling_depth : int
        2 = search pairs, 3 = search triplets.
    max_fits : int
        Number of MI-ranked tuples to attempt fitting.
    niterations : int
        PySR iterations per component. Default 200 (higher than class default
        of 40 — tuned for real posteriors with complex degeneracies).
    max_complexity : int
        PySR complexity cap.
    batch_size : int
        PySR batch size. Default 1000 (higher than class default of 50 —
        tuned for O(10k–100k) sample chains).

    Returns
    -------
    CouplingSearchResult
    """
    samples = np.asarray(samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run detection
    if log_mode:
        detector = DegenLogMode(samples, param_names)
    else:
        detector = DegenDetector(samples, param_names)

    result = detector.search_couplings(
        coupling_depth=coupling_depth,
        niterations=niterations,
        max_complexity=max_complexity,
        max_fits=max_fits,
        batch_size=batch_size,
    )

    # 2. Save pkl
    save_pickle(
        {"samples": samples, "param_names": param_names, "result": result},
        output_dir / "result.pkl",
    )

    # 3. Write + print summary
    _write_summary(result, output_dir / "summary.txt")

    # 4. Run diagnostics (DiagnosticsRunner takes the pkl path, not the result object)
    try:
        runner = DiagnosticsRunner(output_dir / "result.pkl")
        runner.run(output_dir=output_dir / "diagnostics")
    except Exception as e:
        print(f"Warning: Diagnostics failed: {e}", file=sys.stderr)

    return result


def _write_summary(result, path):
    """Print equation table to stdout and write to path simultaneously."""
    header = f"\n{'='*80}"
    col_header = f"{'Params':<30} {'MI':>8} {'R²_ortho':>10}  Equation"
    sep = "=" * 80

    lines = [header, col_header, sep]
    for cf in result.fits:
        if cf.fit:
            row = (
                f"{str(cf.param_names):<30} {cf.mi_score:>8.4f} "
                f"{cf.fit.orthogonal_r2:>10.4f}  {cf.fit.equation_str}"
            )
        else:
            row = f"{str(cf.param_names):<30} {cf.mi_score:>8.4f} {'N/A':>10}  (fit failed)"
        lines.append(row)
    lines.append(sep)

    output = "\n".join(lines)
    print(output)
    Path(path).write_text(output + "\n")
```

- [ ] **Step 4: Temporarily add `run_pipeline` to `__init__.py` for the test import**

Add to `degen_detector/__init__.py`:
```python
from degen_detector.pipeline import run_pipeline
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_pipeline.py -v
```
Expected: all 6 tests PASS (they use `niterations=1` so should complete in under 60s).

- [ ] **Step 6: Commit**

```bash
git add degen_detector/pipeline.py degen_detector/__init__.py tests/test_pipeline.py
git commit -m "feat: add run_pipeline orchestrator with tests"
```

---

## Task 6: Create `degen_detector/cli.py`

The CLI is a thin wrapper over loaders + `run_pipeline`. Tests invoke `cli.main()` directly (not subprocess) for speed.

**Files:**
- Create: `degen_detector/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
# ABOUTME: Tests for the degen-detect CLI entry point.

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cli.py -v
```
Expected: `ImportError` — `cli` module doesn't exist yet.

- [ ] **Step 3: Create `degen_detector/cli.py`**

```python
# ABOUTME: CLI entry point for degen-detect command.
# ABOUTME: Thin argparse wrapper over loaders + run_pipeline.

"""Command-line interface for degen_detector.

Usage:
    degen-detect <source> --format <fmt> [options]

Formats:
    getdist   getdist/CosmoMC chain stem (requires --params)
    emcee     emcee HDFBackend HDF5 file
    numpy     .npy or .npz file (requires --param-names)

Examples:
    degen-detect data/planck/base_plik --format getdist \\
        --params sigma8 omegam H0 --output-dir out/planck --log-mode

    degen-detect chains.h5 --format emcee \\
        --params sigma8 omegam --burn-in 200 --thin 5 --output-dir out/emcee

    degen-detect samples.npy --format numpy \\
        --param-names theta1 theta2 theta3 --output-dir out/synth
"""

# Must come before any matplotlib import — ensures Agg backend on headless HPC nodes
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

from degen_detector.loaders import load_getdist, load_emcee, load_numpy
from degen_detector.pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        prog="degen-detect",
        description="Detect parameter degeneracies in posterior samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", type=Path, help="Chain root, HDF5 file, or .npy/.npz file")
    parser.add_argument(
        "--format", required=True, choices=["getdist", "emcee", "numpy"],
        help="Input format",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Output directory (default: ./outputs)")
    # getdist / emcee params
    parser.add_argument("--params", nargs="+", metavar="PARAM",
                        help="Parameter names to extract (required for --format getdist)")
    parser.add_argument("--ignore-rows", type=float, default=0.3,
                        help="Burn-in fraction for getdist (default: 0.3)")
    # emcee options
    parser.add_argument("--burn-in", type=int, default=0,
                        help="Steps to discard as burn-in for emcee (default: 0)")
    parser.add_argument("--thin", type=int, default=1,
                        help="Thinning factor for emcee (default: 1)")
    # numpy options
    parser.add_argument("--param-names", nargs="+", metavar="NAME",
                        help="Parameter names (required for --format numpy)")
    # pipeline options
    parser.add_argument("--log-mode", action="store_true",
                        help="Use DegenLogMode (log-transforms positive params)")
    parser.add_argument("--coupling-depth", type=int, default=2)
    parser.add_argument("--max-fits", type=int, default=2)
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Conditional required argument enforcement
    if args.format == "getdist" and not args.params:
        parser.error("--params is required when --format getdist")
    if args.format == "numpy" and not args.param_names:
        parser.error("--param-names is required when --format numpy")

    # Load samples
    if args.format == "getdist":
        samples, param_names = load_getdist(
            args.source, args.params, ignore_rows=args.ignore_rows
        )
    elif args.format == "emcee":
        samples, param_names = load_emcee(
            args.source, params=args.params, burn_in=args.burn_in, thin=args.thin
        )
    else:  # numpy
        samples, param_names = load_numpy(args.source, args.param_names)

    run_pipeline(
        samples,
        param_names,
        args.output_dir,
        log_mode=args.log_mode,
        coupling_depth=args.coupling_depth,
        max_fits=args.max_fits,
        niterations=args.niterations,
        max_complexity=args.max_complexity,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

```bash
python -m pytest tests/test_cli.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add degen_detector/cli.py tests/test_cli.py
git commit -m "feat: add degen-detect CLI entry point with tests"
```

---

## Task 7: Create `pyproject.toml` + finalise `__init__.py` exports + update `test_imports.py`

Wire the `degen-detect` console script and expose all new public names.

**Files:**
- Create: `pyproject.toml`
- Modify: `degen_detector/__init__.py`
- Modify: `tests/test_imports.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "degen_detector"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "sympy>=1.9",
    "pysr>=0.16.0",
    "matplotlib>=3.3.0",
    "corner>=2.2.0",
]

[project.optional-dependencies]
getdist = ["getdist"]
emcee = ["emcee", "h5py"]

[project.scripts]
degen-detect = "degen_detector.cli:main"
```

- [ ] **Step 2: Install the package in editable mode**

```bash
cd /home/x-ctirapongpra/scratch/degen_detector
source .venv/bin/activate
uv pip install -e .
```
Expected: installs successfully; `degen-detect --help` works.

- [ ] **Step 3: Verify CLI is on PATH**

```bash
degen-detect --help
```
Expected: prints usage with format choices.

- [ ] **Step 4: Finalise `__init__.py` exports**

Ensure `degen_detector/__init__.py` contains all new public exports:
```python
from degen_detector.pipeline import run_pipeline
from degen_detector.loaders import load_getdist, load_emcee, load_numpy
```
(The `run_pipeline` line was added in Task 5 Step 4; just add the loaders line.)

- [ ] **Step 5: Update `tests/test_imports.py`**

Add a new test function after the existing ones:
```python
def test_pipeline_and_loader_imports():
    """New public exports importable from top-level package."""
    from degen_detector import run_pipeline, load_getdist, load_emcee, load_numpy
```

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_obs_loaders.py
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml degen_detector/__init__.py tests/test_imports.py
git commit -m "feat: add pyproject.toml, wire degen-detect entry point, finalise __init__ exports"
```

---

## Smoke Test

After all tasks are complete, do a quick end-to-end smoke test with numpy format on synthetic data:

```bash
source /home/x-ctirapongpra/scratch/degen_detector/.venv/bin/activate
python -c "
import numpy as np
rng = np.random.default_rng(0)
t1 = rng.uniform(-1,1,300); t2 = rng.uniform(-1,1,300)
t3 = 2*t1**2 + t2**2 - 0.5 + rng.normal(0,0.02,300)
np.save('/tmp/smoke_samples.npy', np.column_stack([t1,t2,t3]))
"
degen-detect /tmp/smoke_samples.npy \
    --format numpy \
    --param-names t1 t2 t3 \
    --output-dir /tmp/smoke_out \
    --max-fits 1 \
    --niterations 5 \
    --batch-size 50
ls /tmp/smoke_out/
```
Expected: `result.pkl`, `summary.txt`, `diagnostics/` directory with plots.
