# DegenDetector

[![Paper](https://img.shields.io/badge/paper-OpenReview-b31b1b.svg)](https://openreview.net/pdf?id=yLfPgQOrLe)


## Motivation

Parameter degeneracy occurs when observed data constrains a combination of parameters more tightly than the individual parameters themselves, posterior samples concentrate along a lower-dimensional manifold. Standard tools like corner plots can show that a correlation exists, but they don't reveal the functional form and will miss degeneracies involving three or more parameters.

`degen_detector` fills this gap. Given posterior samples, it automatically identifies which parameters are coupled and expresses the relationship as a closed-form symbolic equation.

---

## How it works

`degen_detector` runs two stages:

1. **MI screening.** Mutual information between all parameter pairs is estimated with a k-nearest-neighbor estimator. Parameter tuples are ranked by their aggregated MI score — higher means a stronger, more significant degeneracy — cutting the combinatorial search down to a tractable set of candidates.

2. **Alternating symbolic regression.** For each candidate tuple, it fits a separable implicit surface g₁(θ₁) + g₂(θ₂) + ⋯ = c by cycling through one component at a time, reducing the k-dimensional problem to k independent 1D symbolic regressions (via PySR). The result is simplified by SymPy into a human-readable equation.

Fit quality is measured by **orthogonal R²** (i.e., the mean squared perpendicular distance from samples to the fitted surface).

For **multiplicative degeneracies** (e.g., power laws), one can activate `--log-mode` / `log_mode=True`: the pipeline automatically runs in log-transformed coordinates and reports the equation back in the original parameterization.

---

## Installation

```bash
git clone https://github.com/chaipattira/degen_detector.git
cd degen_detector

pip install -e ".[all]"
```

## Usage

### One-shot API

Runs MI screening, symbolic fitting, and diagnostics in one call:

```python
from degen_detector import load_posterior, run_detector

samples, param_names = load_posterior("base_plik", params=["omegam", "H0"])  # GetDist

result = run_detector(samples, param_names, output_dir="out/")
```

### Two-stage pipeline

```python
from degen_detector import DegenDetector, DegenLogMode

detector = DegenDetector(samples, param_names)

# Stage 1 — fast: compute pairwise MI and rank all parameter tuples
ranking = detector.rank_couplings(coupling_depth=2)
print(ranking.tuples[:5])   # inspect top candidates by MI score

# Stage 2 — slow: fit symbolic equations to the top tuples
result = detector.fit_couplings(ranking, max_fits=3)
for cf in result.fits:
    if cf.fit:
        print(cf.fit.equation_str, "  R²=", cf.fit.orthogonal_r2)
```

For power-law / multiplicative degeneracies, use `DegenLogMode` — same two-stage API, fits in log-space and reports back in original coordinates:

```python
detector = DegenLogMode(samples, param_names)
ranking  = detector.rank_couplings(coupling_depth=2)
result   = detector.fit_couplings(ranking, max_fits=3)
```

#### Supported file types

| Format | Extensions | `params` required? |
|--------|--------------|-------------------|
| ArviZ NetCDF | `.nc`, `.netcdf` | no |
| CSV | `.csv` | no |
| NumPy | `.npy`, `.npz` | **yes** |
| emcee HDF5 | `.h5` / `.hdf5` (content-sniffed) | no |
| GetDist / CosmoMC | `<stem>.paramnames` or `<stem>_1.txt` | **yes** |
| PolyChord / MultiNest | any — pass arrays directly | **yes** |

For importance-weighted chains (PolyChord, MultiNest, dynesty), use `load_weighted` to resample to equal weights before passing to `run_detector`.


### Running via CLI

```bash
degen-detect data/planck/base_plik --params omegam H0 sigma8 --log-mode --output-dir out/

```

#### Common flags

```
--output-dir PATH     Where to write outputs (default: ./outputs)
--log-mode            Run in log-space (for power-law / multiplicative degeneracies)
--coupling-depth INT  2 = pairs, 3 = triplets (default: 2)
--max-fits INT        Stop after N fits (default: 2)
--niterations INT     Symbolic regression iterations (default: 200)
--burn-in INT         Steps to discard, emcee only (default: 0)
--thin INT            Thinning factor, emcee only (default: 1)
--ignore-rows FLOAT   Burn-in fraction for getdist (default: 0.3)
```

---

### Outputs

Each run writes to `output_dir/`:

| File | Contents |
|------|----------|
| `result.pkl` | Samples, parameter names, and full `CouplingSearchResult` |
| `summary.txt` | Ranked table of equations with MI scores and R²⊥ |
| `diagnostics/corner.png` | Corner plot of degenerate parameters with fitted surface overlaid |
| `diagnostics/mi_matrix.png` | Pairwise MI heatmap across all parameters |
| `diagnostics/<pair>/` | Per-degeneracy plots: component functions, true vs. predicted, manifold |
