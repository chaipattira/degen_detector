
## Progress Log

### 2026-04-10 (observational probes)
- Added scripts for two cosmological observational datasets (paralleling `run_planck_logmode.py`):
  - `scripts/run_desi_logmode.py` + `.job` — DESI DR1 full-shape chains (all tracers, LCDM, velocileptors); loads cobaya MCMC chains with known column order from `chain.margestats`; expands by multiplicity weights; expects to recover Ω_m · H₀³ and σ₈ · Ω_m^0.5 degeneracies
  - `scripts/run_des_y3_logmode.py` + `.job` — DES Y3 3x2pt LCDM MagLim (DES Y6 chains not yet public as of April 2026); auto-detects σ₈ column by range/std filter; resamples polychord importance weights; recovers S8 = σ₈ · Ω_m^0.5
  - `scripts/download_obs_chains.sh` — downloads both datasets from public servers
- Added `tests/test_obs_loaders.py` with 22 unit tests covering DESI column extraction, weight expansion, DES σ₈ auto-detection (std-filter distinguishes σ₈ from tighter S8), and polychord resampling

### 2026-04-10
- Added `DegenLogMode` in `degen_detector/transforms.py` — coordinate-transform wrapper for `DegenDetector`
- Applies per-parameter transforms (default: `LOG_TRANSFORM` = `np.log`) before running the full MI + PySR pipeline
- Back-substitutes transformed symbols into original parameterization: `Symbol("log_Omega_m")` → `log(Symbol("Omega_m"))` in all `ImplicitFit.component_exprs`
- Returned fits are in original coordinates; `evaluate(original_samples)` works correctly
- `search_couplings(params=["Omega_m", "h"])` accepts original names — translated to transformed names automatically
- Raises `ValueError` with informative message if any transformed column contains non-finite values (log of zero/negative)
- Designed to recover multiplicative degeneracies like `Omega_m * h^2 ≈ const` (linear in log space)
- Exported `DegenLogMode`, `LOG_TRANSFORM`, `ParameterTransform` from top-level package
- Added `tests/test_transforms.py` with 18 unit tests covering transform application, param renaming, back-substitution, evaluate round-trip, edge cases, and params translation

### 2026-03-23 (restructure)
- Split 1263-line `diagnostics.py` monolith into `diagnostics/` package: `analyzer.py`, `plots.py`, `equations.py`, `runner.py`
- Added `degen_detector/io.py` with `save_pickle`, `load_pickle`, `create_output_dir` — shared by both scripts
- Scripts `run_synthetic_experiments.py` and `run_planck_analysis.py` now use `io.py` instead of inline pickle boilerplate
- Added `tests/` with 25 tests covering imports, I/O utilities, FitAnalyzer, equation formatting, and plot smoke tests
- All existing import paths and CLI (`python -m degen_detector.diagnostics`) preserved

### 2026-03-23
- `equations.txt` now shows a `Top form:` line after the candidates header, with all floating-point constants replaced by `c_1, c_2, ...` placeholders (same value → same label; integer exponents preserved)
- Added `_make_form_string()` helper to `diagnostics.py` with tests in `tests/test_make_form_string.py`
- Added `complexity` field to `ImplicitFit` (sympy operation count across all component expressions)
- `equations.txt` now shows **all candidate fits per tuple** (not just the best), each with R²_ortho, residual_std, complexity, and components
- Added `plot_mi_matrix()` to `diagnostics.py`; `DiagnosticsRunner` now saves `mi_matrix.png` alongside other plots
- Added ABOUTME headers to `core.py`, `implicit_fit.py`, `synthetic.py`, `loss.py`, `run_synthetic_experiments.py`
- Cleaned up `CouplingFit` docstring (removed temporal/backward-compat language)

### 2026-03-19
- Added **variable orientation detection** via nonlinearity scoring to `fit_separable_implicit()`
- New `_compute_nonlinearity_scores()` function uses ensemble of correlation + MI with power transformations
- Tests powers k ∈ {0.5, 1/3, 2, 3} to detect both polynomial and inverse relationships
- Variables reordered by descending nonlinearity score (most nonlinear fitted first)
- Fixes issue where algorithm would incorrectly assign cubic to wrong variable (e.g., fitting `cubic(θ₁) + linear(θ₂)` when ground truth is `linear(θ₁) + cubic(θ₂)`)

### 2026-03-16
- Modified `fit_separable_implicit()` to return top 5 candidate equations from PySR hall of fame (ranked by R²_ortho)
- Updated `CouplingFit` to store multiple fits per tuple in `fits` list; added `fit` property for backward compatibility
- Added `n_candidates` parameter to `search_couplings()` to control number of candidate equations saved per tuple
- Integrated diagnostics plotting into `run_synthetic_experiments.py` - automatically generates plots after experiments
- Created `degen_detector.diagnostics` module with `FitAnalyzer` class for cached sympy operations
- Added plotting functions: corner, components, true_vs_predicted, residuals, manifold_2d/3d, projections
- `equations.txt` now outputs ALL fitted equations (ranked by MI), not just best fit
- Auto-detects pkl format and chooses 2D/3D plots based on parameter count
- CLI entry point: `python -m degen_detector.diagnostics <pkl_path>`
- `FitAnalyzer.solve_for_param()` now solves quadratic components analytically using quadratic formula (previously only linear)

### 2026-03-15
- Fixed `save_equation_comparison()` to handle ground truth dictionaries without 'component_functions' key
- Fixed `plot_2param_degeneracy()` to handle non-polynomial g2 expressions (e.g., nested exponentials) by catching exception and using numerical inversion
- Removed summary plot generation from `plot_synthetic_results.py`
