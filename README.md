# degen_detector

Automatic detection of parameter degeneracies in Bayesian posterior samples. Discovers symbolic relationships (e.g., `sigma8 = f(Omega_m)`) using mutual information ranking and genetic programming.

## Installation

```bash
pip install numpy scikit-learn sympy pysr sbibm dill corner getdist
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

# Best fit
print(result.best_fit.fit.equation_str)  # e.g., "sigma8 = 0.82 * Om^(-0.5)"
print(f"R² = {result.best_fit.fit.r_squared:.4f}")
```

## API

### `DegenDetector.search_couplings()`

```python
result = detector.search_couplings(
    params=None,           # list[str] | int | None
    coupling_depth=2,      # tuple size (2=pairs, 3=triplets)
    r2_threshold=0.95,     # early stop when R² >= threshold
    max_fits=None,         # max fits to attempt (None=all)
    mi_rank_method="min",  # "min", "avg", "sum", "geometric"
)
```

**Parameter selection (`params`):**
- `["Om", "sigma8"]` — explicit parameter names
- `5` — auto-select top 5 by total MI (most correlated)
- `None` — use all parameters

**Returns:** `CouplingSearchResult` with:
- `best_fit` — highest R² fit
- `fits` — all attempted fits in MI-ranked order
- `mi_result` — mutual information matrix
- `stopped_early` — whether R² threshold was reached

### Result Objects

```python
# Best fit details
fit = result.best_fit.fit
print(fit.equation_str)   # "sigma8 = 0.82 * Om^(-0.5)"
print(fit.r_squared)      # 0.98
print(fit.target_name)    # "sigma8"
print(fit.input_names)    # ["Om"]

# Predict with discovered equation
y_pred = fit.predict(X)   # X shape: (N, n_inputs)
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

# Search pairs with early stopping
result = detector.search_couplings(
    params=params,
    coupling_depth=2,
    r2_threshold=0.90,
)

# Show top fits
for cf in sorted(result.fits, key=lambda x: x.fit.r_squared if x.fit else 0, reverse=True)[:3]:
    if cf.fit:
        print(f"{cf.fit.equation_str}  (R²={cf.fit.r_squared:.3f})")
```

## MI Ranking Methods

| Method | Description |
|--------|-------------|
| `"min"` | Minimum pairwise MI (conservative) |
| `"avg"` | Average pairwise MI |
| `"sum"` | Sum of all pairwise MI |
| `"geometric"` | Geometric mean |
