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
pip install -e .
pip install getdist        # for GetDist / CosmoMC chains
pip install emcee h5py     # for emcee HDF5 chains
```


## One-shot Python API

```python
from degen_detector import run_pipeline, load_emcee

# Load samples via load_numpy, load_getdist, or load_emcee
samples, param_names = load_emcee("chains.h5", burn_in=200, thin=5)

# Run — saves result.pkl, summary.txt, and diagnostic plots to output_dir/
result = run_pipeline(samples, param_names, output_dir="out/", log_mode=True)

### For power-law / multiplicative degeneracies (e.g. Ωm h³ ≈ const), run in log-mode
```

## CLI

```
degen-detect <source> --format {getdist,emcee,numpy} [options]
```

| Format | Required flags |
|--------|---------------|
| `getdist` | `--params PARAM [PARAM ...]` |
| `numpy` | `--param-names NAME [NAME ...]` |
| `emcee` | *(none — reads labels from file)* |

**Common options:**

```
--output-dir PATH     Where to write outputs (default: ./outputs)
--log-mode            Run in log-space (for power-law / multiplicative degeneracies)
--coupling-depth INT  2 = pairs, 3 = triplets (default: 2)
--max-fits INT        Stop after N fits (default: 2)
--niterations INT     Symbolic regression iterations (default: 200)
--burn-in INT         Steps to discard, emcee only (default: 0)
--thin INT            Thinning factor, emcee only (default: 1)
```

**Examples:**

```bash
# Planck CMB chain — use log-mode for the Ωm h³ power-law degeneracy
degen-detect data/planck/base_plik --format getdist \
    --params omegam H0 sigma8 --log-mode --output-dir out/planck
```

---

## Outputs

Each run writes to `output_dir/`:

| File | Contents |
|------|----------|
| `result.pkl` | Samples, parameter names, and full `CouplingSearchResult` |
| `summary.txt` | Ranked table of equations with MI scores and R²⊥ |
| `diagnostics/corner.png` | Corner plot of degenerate parameters with fitted surface overlaid |
| `diagnostics/mi_matrix.png` | Pairwise MI heatmap across all parameters |
| `diagnostics/<pair>/` | Per-degeneracy plots: component functions, true vs. predicted, manifold |
