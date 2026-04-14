# degen_detector

Automatic detection of parameter degeneracies in Bayesian posterior samples. Discovers symbolic relationships (e.g., `g1(Om) + g2(sigma8) = c`) using mutual information ranking and symbolic regression.

## Installation

```bash
pip install numpy scikit-learn sympy pysr
```

## Quick Start

```python
from degen_detector import DegenDetector

# Load your MCMC samples (N samples x M parameters)
samples = ...  # shape: (N, M)
param_names = ["Om", "sigma8", "H0", "ns", "tau"]

detector = DegenDetector(samples, param_names)

# Search for pairwise degeneracies
result = detector.search_couplings(
    params=param_names,   # which params to search
    coupling_depth=2,     # 2=pairs, 3=triplets
)

# Results are ranked by MI (strongest dependencies first)
for cf in result.fits:
    if cf.fit:
        print(f"[MI={cf.mi_score:.3f}] {cf.fit.equation_str}")
        print(f"  R²_ortho={cf.fit.orthogonal_r2:.4f}, residual_std={cf.fit.residual_std:.4f}")
```

## How It Works

1. **MI Ranking**: Compute mutual information between all parameter pairs. Rank tuples by MI score (higher = stronger statistical dependency).

2. **Symbolic Fitting**: For each tuple, find functions g1, g2, ..., gk such that:
   ```
   g1(x1) + g2(x2) + ... + gk(xk) = c
   ```
   Uses alternating optimization with PySR for 1D symbolic regression.

3. **Output**: R² and residual_std are quality diagnostics.

## API

### `DegenDetector.search_couplings()`

```python
result = detector.search_couplings(
    params=None,           # list[str] | int | None
    coupling_depth=2,      # tuple size (2=pairs, 3=triplets)
    max_fits=None,         # limit number of fits (top N by MI)
)
```

**Parameter selection (`params`):**
- `["Om", "sigma8"]` — explicit parameter names
- `5` — auto-select top 5 by total MI (most correlated)
- `None` — use all parameters

**Returns:** `CouplingSearchResult` with:
- `fits` — all fits ranked by MI (descending)
- `mi_result` — mutual information matrix
- `n_tuples_total` — total number of tuples considered

### Result Objects

```python
# Iterate fits (ranked by MI)
for cf in result.fits:
    print(f"Params: {cf.param_names}")
    print(f"MI score: {cf.mi_score:.4f}")  # ranking metric

    if cf.fit:
        print(f"Equation: {cf.fit.equation_str}")
        print(f"R²_ortho: {cf.fit.orthogonal_r2:.4f}")  # quality check
        print(f"Residual std: {cf.fit.residual_std:.4f}")  # tightness
```


## Example: Planck Cosmology

```python
from getdist import loadMCSamples
from degen_detector import DegenDetector

# Load Planck chains
mc = loadMCSamples("planck_chains/base_plikHM")
params = ["omegam", "sigma8", "H0", "ns"]
samples = np.column_stack([getattr(mc.getParams(), p) for p in params])

detector = DegenDetector(samples, params)
result = detector.search_couplings(params=params, coupling_depth=2)

# Show fits (ranked by MI)
for cf in result.fits:
    if cf.fit:
        print(f"[MI={cf.mi_score:.3f}] {cf.fit.equation_str}")
```


## Why MI-First?

MI (mutual information) measures intrinsic statistical dependency between variables, regardless of functional form. R² only measures how well a *specific* equation captures the relationship.

- **High MI**: Variables are constrained together (true degeneracy likely)
- **Low MI**: Variables are more independent (weak/no degeneracy)
- **R²**: Tells you if your equation is adequate, not if a degeneracy exists

## Diagnostics

The `degen_detector.diagnostics` module provides tools for analyzing and visualizing fit results.

### Command Line

```bash
# Run on a single pkl file
python -m degen_detector.diagnostics results.pkl

# Run on a directory of pkl files
python -m degen_detector.diagnostics outputs/synthetic_15710222/20260315_091846

# Specify output directory
python -m degen_detector.diagnostics results.pkl -o my_plots/
```

### Generated Outputs

| File | Description |
|------|-------------|
| `equations.txt` | All candidate fits per tuple (ranked by MI then R²), with complexity |
| `mi_matrix.png` | Mutual information heatmap across all parameters |
| `corner.png` | Corner plot of all parameters |
| `components.png` | Component functions g_j(x_j) |
| `true_vs_predicted.png` | True vs predicted for each component |
| `residuals.png` | Residual distribution and Q-Q plot |
| `manifold_2d.png` | 2D constraint curve (for 2-param fits) |
| `manifold_3d.png` | 3D constraint surface (for 3-param fits) |
| `projections_2d.png` | 2D projections (for 3-param fits) |

### Python API

```python
from degen_detector.diagnostics import DiagnosticsRunner, FitAnalyzer

# Run all diagnostics
runner = DiagnosticsRunner("results.pkl")
runner.run(output_dir="plots/")

# Or use FitAnalyzer for custom analysis
from degen_detector.diagnostics import FitAnalyzer, plot_components

analyzer = FitAnalyzer(fit)
residuals = analyzer.constraint_residual(samples, param_names)
true_g, pred_g = analyzer.predict_component('theta1', samples, param_names)
```