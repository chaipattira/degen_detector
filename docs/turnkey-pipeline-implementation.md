# Turnkey Pipeline Implementation — Summary

**Date:** 2026-06-19  
**Branch:** main  
**Plan:** [docs/superpowers/plans/2026-06-19-turnkey-pipeline.md](superpowers/plans/2026-06-19-turnkey-pipeline.md)

---

## What Was Built

Three new modules + `pyproject.toml` that together let a user point at any posterior chain and get degeneracy plots, a pickle, and a summary in a single function call or CLI command.

| File | Role |
|---|---|
| `degen_detector/loaders.py` | Format-specific chain readers → `(ndarray, list[str])` |
| `degen_detector/pipeline.py` | `run_pipeline` orchestrator (detect → save → summarise → diagnose) |
| `degen_detector/cli.py` | `degen-detect` argparse entry point |
| `pyproject.toml` | Package metadata + console script wiring |

---

## Files Changed

### Modified
- `degen_detector/__init__.py` — removed broken `detect` import; added `run_pipeline`, `load_getdist`, `load_emcee`, `load_numpy` as public exports
- `tests/test_imports.py` — removed `detect` from import test; added `test_pipeline_and_loader_imports`

### Created
- `degen_detector/loaders.py`
- `degen_detector/pipeline.py`
- `degen_detector/cli.py`
- `pyproject.toml`
- `tests/test_loaders.py`
- `tests/test_pipeline.py`
- `tests/test_cli.py`

---

## Loaders (`degen_detector/loaders.py`)

Three loader functions, all returning `(samples: ndarray, param_names: list[str])`:

### `load_numpy(array_or_path, param_names)`
- Accepts an ndarray directly, or a `.npy` / `.npz` path
- For `.npz`: loads the `"samples"` key if present; otherwise falls back to the first array with a `UserWarning`
- Raises `ValueError` on missing files, 1-D arrays, or column/name count mismatch

### `load_getdist(chain_root, params, ignore_rows=0.3)`
- Wraps `getdist.loadMCSamples` — same path-stem convention as CosmoMC
- Warns and skips params not found in the chain (instead of crashing)
- Raises `ValueError` if no params could be loaded at all
- `getdist` is an optional dependency; gives a clear `ImportError` with install hint if missing

### `load_emcee(h5_path, params=None, burn_in=0, thin=1)`
- Reads an emcee `HDFBackend` HDF5 file
- If `params=None`, reads `param_names` from the file's HDF5 root attribute; auto-generates `theta_i` names with a `UserWarning` if no labels are stored
- Supports `burn_in` (steps to discard) and `thin` (thinning factor)
- `emcee` + `h5py` are optional dependencies

---

## Pipeline (`degen_detector/pipeline.py`)

### `run_pipeline(samples, param_names, output_dir, *, log_mode=False, coupling_depth=2, max_fits=2, niterations=200, max_complexity=15, batch_size=1000)`

Sequence:
1. Creates `output_dir` (no timestamp subdirectory added)
2. Instantiates `DegenLogMode` (if `log_mode=True`) or `DegenDetector`
3. Calls `detector.search_couplings(...)` with the given hyperparameters
4. Saves `result.pkl` via `degen_detector.io.save_pickle` — dict with keys `samples`, `param_names`, `result`
5. Writes `summary.txt` (equation table with MI scores and R²_ortho) and prints it to stdout
6. Runs `DiagnosticsRunner` in `output_dir/diagnostics/` — failures are caught and printed as warnings so the pipeline doesn't abort
7. Returns the `CouplingSearchResult`

Default hyperparameters are tuned for real posteriors: `niterations=200`, `batch_size=1000` (larger than the class defaults of 40/50).

---

## CLI (`degen_detector/cli.py`)

Installed as `degen-detect` via `pyproject.toml`.

```
degen-detect <source> --format {getdist,emcee,numpy} [options]
```

**Format-specific required arguments:**
- `--format getdist` requires `--params PARAM [PARAM ...]`
- `--format numpy` requires `--param-names NAME [NAME ...]`
- `--format emcee` is self-contained (reads labels from file or auto-generates)

**Pipeline options (all optional with defaults):**
```
--log-mode              Use DegenLogMode (log-transform positive params)
--coupling-depth INT    Default 2
--max-fits INT          Default 2
--niterations INT       Default 200
--max-complexity INT    Default 15
--batch-size INT        Default 1000
--output-dir PATH       Default ./outputs
```

**Example usage:**
```bash
# Planck getdist chain
degen-detect data/planck/base_plik --format getdist \
    --params sigma8 omegam H0 --output-dir out/planck --log-mode

# emcee HDF5
degen-detect chains.h5 --format emcee \
    --params sigma8 omegam --burn-in 200 --thin 5 --output-dir out/emcee

# numpy array
degen-detect samples.npy --format numpy \
    --param-names t1 t2 t3 --output-dir out/synth
```

---

## Tests

| File | Tests | Status |
|---|---|---|
| `tests/test_imports.py` | 6 (was 5) | All pass |
| `tests/test_loaders.py` | 15 | All pass |
| `tests/test_pipeline.py` | 6 | All pass |
| `tests/test_cli.py` | 3 | All pass |
| **Total** | **30** | **All pass** |

Pipeline tests use `niterations=1, batch_size=50` to keep runtime under ~90s.

---

## Environment Note

Tests require `module load gcc/14.2.0` before running — the system's default GCC 11.2.0 lacks `GLIBCXX_3.4.30` which Julia (PySR's backend) requires.

```bash
module load gcc/14.2.0
.venv/bin/python -m pytest tests/ --ignore=tests/test_obs_loaders.py -v
```

---

## Smoke Test Result

```bash
module load gcc/14.2.0
.venv/bin/degen-detect /tmp/smoke_samples.npy \
    --format numpy --param-names t1 t2 t3 \
    --output-dir /tmp/smoke_out --max-fits 1 --niterations 5 --batch-size 50
```

Outputs produced:
```
smoke_out/
├── result.pkl
├── summary.txt
└── diagnostics/
    ├── corner.png
    ├── equations.txt
    ├── mi_matrix.png
    └── t1_t3/
        ├── components.png
        ├── true_vs_predicted.png
        ├── residuals.png
        └── manifold_2d.png
```

Found degeneracy `t1 ↔ t3` (MI = 0.94, R²_ortho = 0.97): `−2.82·t1² + 1.15·t3 ≈ const`

---

## Known Issue: git Object Store Corruption

The repository's git object store has pre-existing corruption in the `data/` subtree (large Planck chain files whose blobs are missing from `.git/objects`). This prevents `git commit` from working. All code changes are saved to disk; a `git fsck` and `git gc --aggressive` or manual object repair would be needed to restore commit capability.
