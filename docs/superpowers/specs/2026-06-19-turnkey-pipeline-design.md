# Turnkey Pipeline Design

**Date:** 2026-06-19
**Status:** Approved (rev 3 — post spec review round 2)

## Problem

Users who want to run degen_detector on a real posterior must write ~200-line bespoke scripts
(see `scripts/run_planck_logmode.py`) that are mostly boilerplate around ~10 lines of actual
logic. The goal is a turnkey interface: point at a posterior, get diagnostics.

## Design

### New files

| File | Purpose |
|---|---|
| `degen_detector/loaders.py` | Format-specific sample loaders |
| `degen_detector/pipeline.py` | Full pipeline orchestration |
| `degen_detector/cli.py` | CLI entry point (`degen-detect`) |

`degen_detector/__init__.py` gets new public exports: `run_pipeline`, `load_getdist`,
`load_emcee`, `load_numpy`. **Note:** the existing `from degen_detector.core import detect`
line in `__init__.py` is a broken import (`detect` does not exist in `core.py`) and will
cause an `ImportError` on package load. Remove this import as part of this work.

`pyproject.toml` gets a new entry point: `degen-detect = "degen_detector.cli:main"`.

Existing scripts in `scripts/` are untouched.

---

## Section 1: Loaders (`degen_detector/loaders.py`)

Three functions, each returning `(ndarray, list[str])`:

### `load_getdist(chain_root, params, ignore_rows=0.3)`

- `chain_root`: path stem to getdist/CosmoMC chain (same convention as existing scripts)
- `params`: list of parameter name strings to extract — **required, no default**
- Warns and skips any param not found in the chain
- Raises `ValueError` on missing file or if no params could be loaded

### `load_emcee(h5_path, params=None, burn_in=0, thin=1)`

- `h5_path`: path to emcee `backends.HDFBackend` HDF5 file
- `params`: optional list of names; if `None`, tries to read labels stored in the backend
  as a custom HDF5 attribute. If no labels are stored, auto-generates names
  `[f"theta_{i}" for i in range(ndim)]` and emits a `UserWarning`.
- `burn_in`, `thin`: control chain slicing before flattening
- Raises `ValueError` if the file is missing or the shape is wrong

### `load_numpy(array_or_path, param_names)`

- `array_or_path`: an `ndarray` directly, or a path to:
  - `.npy` — loaded as-is
  - `.npz` — loads the array named `"samples"` if present, otherwise the first array;
    emits a `UserWarning` if falling back to the first array
- `param_names`: required (no metadata to infer from)
- Raises `ValueError` on shape mismatch (columns ≠ len(param_names)) or missing file

All loaders raise `ValueError` with a clear message rather than silently dropping data.

---

## Section 2: Pipeline (`degen_detector/pipeline.py`)

### `run_pipeline(samples, param_names, output_dir, *, log_mode=False, coupling_depth=2, max_fits=2, niterations=200, max_complexity=15, batch_size=1000)`

**Parameters:**
- `samples`: `ndarray` of shape `(n_samples, n_params)`
- `param_names`: list of strings matching columns of `samples`
- `output_dir`: directory to write outputs — created with `Path(output_dir).mkdir(parents=True, exist_ok=True)`. **Does NOT call `create_output_dir()`.** No timestamp subdirectory is added; the caller controls directory naming.
- `log_mode`: if `True`, instantiates `DegenLogMode(samples, param_names)` with `transforms=None`.
  `DegenLogMode` automatically skips params whose name already starts with `log`/`ln`
  (e.g. `logA`), log-transforms everything else, and **drops rows** (not params) where any
  transform produces a non-finite value (e.g. log of zero or negative). Raises `ValueError`
  if all rows are dropped. If `False`, uses `DegenDetector(samples, param_names)`.
- `coupling_depth`: 2 = pairs, 3 = triplets
- `max_fits`: number of MI-ranked tuples to attempt fitting
- `niterations`: PySR iterations per component. **Default 200** (deliberately higher than the class default of 40) — real posteriors need more evolution steps than synthetic experiments
- `max_complexity`: PySR complexity cap
- `batch_size`: PySR batch size. **Default 1000** (deliberately higher than class default of 50) — tuned for real MCMC chains with O(10k–100k) samples

**Returns:** `CouplingSearchResult` (so callers can inspect programmatically)

**Internal steps — in this order:**

1. Create `output_dir` with `mkdir(parents=True, exist_ok=True)` (not `create_output_dir`)
2. Run detection: `DegenLogMode(...)` or `DegenDetector(...)` → `.search_couplings(...)`
3. Save pkl: `save_pickle({"samples": samples, "param_names": param_names, "result": result}, output_dir / "result.pkl")`
4. Write and print summary: write the equation table to `output_dir / "summary.txt"` and simultaneously print to stdout. For failed fits (where `cf.fit is None`), write `N/A` in the R² column and `(fit failed)` in the Equation column, matching the existing script convention (`run_planck_logmode.py` line 191).
5. Run diagnostics: `DiagnosticsRunner(output_dir / "result.pkl").run(output_dir=output_dir / "diagnostics")` — note `DiagnosticsRunner` takes a **pkl path**, not a result object; the pkl must be saved first.

**Output structure written to `output_dir`:**

```
output_dir/
  result.pkl        ← dict with samples, param_names, CouplingSearchResult
  summary.txt       ← equation table (params | MI | R² | equation)
  diagnostics/      ← all plots from DiagnosticsRunner
```

**Summary table format:**

```
================================================================
Params                         MI       R²_ortho  Equation
================================================================
['sigma8', 'omegam']        0.6421     0.9957  ...
['H0', 'omegam']            0.1203       N/A   (fit failed)
================================================================
```

---

## Section 3: CLI (`degen_detector/cli.py`)

Entry point: `degen-detect <source> --format <fmt> [options]`

`cli.py` must call `matplotlib.use("Agg")` before any import that triggers matplotlib,
to ensure headless operation on HPC nodes.

### Formats

| `--format` | Loader called | Required flags | Optional flags |
|---|---|---|---|
| `getdist` | `load_getdist` | `--params` | `--ignore-rows` |
| `emcee` | `load_emcee` | _(none)_ | `--params`, `--burn-in`, `--thin` |
| `numpy` | `load_numpy` | `--param-names` | _(none)_ |

**`--param-names` is required when `--format numpy`** — enforced via a manual check:
```python
if args.format == "numpy" and not args.param_names:
    parser.error("--param-names is required when --format numpy")
```

**`--params` is required when `--format getdist`** — enforced via a manual check:
```python
if args.format == "getdist" and not args.params:
    parser.error("--params is required when --format getdist")
```

### Common options

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | `./outputs` | Where to write results (no timestamp added) |
| `--log-mode` | off | Use `DegenLogMode` (auto log-transforms positive params) |
| `--coupling-depth` | 2 | 2 = pairs, 3 = triplets |
| `--max-fits` | 2 | MI-ranked tuples to fit |
| `--niterations` | 200 | PySR iterations per component |
| `--max-complexity` | 15 | PySR complexity cap |
| `--batch-size` | 1000 | PySR batch size |

### Usage examples

```bash
# getdist chains
degen-detect data/planck/base_plik --format getdist \
    --params sigma8 omegam H0 --output-dir out/planck --log-mode

# emcee HDF5
degen-detect chains.h5 --format emcee \
    --params sigma8 omegam --burn-in 200 --thin 5 --output-dir out/emcee

# numpy array (param names required)
degen-detect samples.npy --format numpy \
    --param-names theta1 theta2 theta3 --output-dir out/synth
```

### Python API equivalents

```python
from degen_detector import run_pipeline
from degen_detector.loaders import load_getdist, load_emcee, load_numpy

# getdist
samples, params = load_getdist("data/planck/base_plik", ["sigma8", "omegam", "H0"])
result = run_pipeline(samples, params, "out/planck", log_mode=True)

# emcee
samples, params = load_emcee("chains.h5", burn_in=200, thin=5)
result = run_pipeline(samples, params, "out/emcee")

# numpy
samples, params = load_numpy("samples.npy", ["theta1", "theta2", "theta3"])
result = run_pipeline(samples, params, "out/synth")
```

---

## What is NOT in scope

- YAML/config file interface (not needed given CLI + argparse)
- Changes to existing `scripts/` — they remain as examples
- New plot types or changes to `DiagnosticsRunner`
- Format auto-detection (user must always pass `--format`)
- Exposing a caller-supplied `transforms` dict in `run_pipeline` — `log_mode=True` always uses an internally built positivity-check dict (not `DegenLogMode` defaults)
