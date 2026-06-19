# degen_detector Algorithm Summary

## Overview

**degen_detector** is a Python package for automatic detection of parameter degeneracies in Bayesian posterior samples. It discovers symbolic relationships between parameters (e.g., `σ₈ = f(Ωₘ)` in cosmology) using mutual information ranking and genetic programming via PySR (Python Symbolic Regression).

The key innovation is finding **separable implicit surfaces** of the form:
```
g₁(x₁) + g₂(x₂) + ... + gₖ(xₖ) = c
```
where each `gⱼ` is a univariate function discovered through symbolic regression.

---

## Codebase Structure

```
/anvil/scratch/x-ctirapongpra/degen_detector/
├── degen_detector/          # Main package
│   ├── core.py             # Main API and DegenDetector class
│   ├── implicit_fit.py     # Separable implicit surface fitting (CORE ALGORITHM)
│   ├── analysis.py         # Mutual information computation
│   ├── groups.py           # Parameter tuple ranking
│   ├── loss.py            # Orthogonal loss computation
│   ├── synthetic.py       # Synthetic test dataset generators
│   └── plotting.py        # Visualization utilities
├── scripts/               # Example usage scripts
├── data/                  # Planck cosmology chain data
└── notebooks/            # Jupyter notebooks for analysis
```

---

## Step-by-Step Algorithm

### Step 1: Mutual Information Analysis
**File:** [analysis.py](degen_detector/analysis.py)

1. Compute pairwise mutual information (MI) between all parameters
2. Build a symmetric MI matrix showing how correlated each parameter pair is
3. Optionally select the top-N most correlated parameters based on total MI

**Key Functions:**
- `mutual_info_matrix()` - Uses scikit-learn's k-NN estimator
- `select_params_by_mi()` - Filters to most relevant parameters

---

### Step 2: Parameter Tuple Ranking
**File:** [groups.py](degen_detector/groups.py)

1. Generate all k-tuples of parameters (e.g., all pairs if `coupling_depth=2`)
2. For each tuple, aggregate the pairwise MI scores using methods like:
   - **"sum"**: Total MI between all pairs in the tuple
   - **"min"**: Minimum MI (all pairs must be correlated)
   - **"avg"** or **"geometric"**: Average measures
3. Rank tuples from highest to lowest aggregated MI

**Key Function:**
- `generate_ranked_tuples()` - Creates sorted list of parameter combinations

**Important:** The number of components (k) is **user-specified** via `coupling_depth`, not discovered by the algorithm.

---

### Step 3: Implicit Surface Fitting (CORE ALGORITHM)
**File:** [implicit_fit.py](degen_detector/implicit_fit.py)

This is the heart of the algorithm. For each parameter tuple, it uses **alternating optimization** to discover the implicit relationship.

#### Algorithm: Alternating Optimization

```
Goal: Find g₁(x₁) + g₂(x₂) + ... + gₖ(xₖ) = c

1. INITIALIZE:
   - Set gⱼ(xⱼ) = xⱼ (identity function) for all j
   - Set c = mean(Σⱼ xⱼ)

2. ITERATE until convergence:
   For j = 0, 1, 2, ..., k-1:
     a. Compute partial_sum = Σᵢ≠ⱼ gᵢ(xᵢ)  # Sum all OTHER components

     b. Define target: target_j = c - partial_sum
        # This is what gⱼ(xⱼ) should equal to satisfy the equation

     c. If j == 0: normalize target to unit variance (anchoring trick)
        # Prevents all functions from scaling together

     d. Fit gⱼ using PySR (symbolic regression):
        - Input: xⱼ (1D array)
        - Target: target_j (1D array)
        - Output: Best symbolic expression for gⱼ

     e. Update component: gⱼ(xⱼ) ← PySR_result(xⱼ)

   # Update the constant
   c = mean(Σⱼ gⱼ(xⱼ))

   # Check convergence
   residuals = Σⱼ gⱼ(xⱼ) - c
   if std(residuals) < convergence_threshold:
       BREAK

3. Return discovered functions g₁, g₂, ..., gₖ and constant c
```

**Key Insight:** By assuming separability, the k-dimensional problem becomes k independent 1D symbolic regression problems at each iteration!

**Key Function:**
- `fit_separable_implicit()` - Main alternating optimization loop

---

### Step 4: Fit Quality Evaluation
**File:** [loss.py](degen_detector/loss.py)

After fitting, evaluate the quality using **orthogonal loss** (perpendicular distance to the surface):

```python
# Define the implicit surface function
F(x) = g₁(x₁) + g₂(x₂) + ... + gₖ(xₖ) - c

# Compute gradient
∇F = [∂g₁/∂x₁, ∂g₂/∂x₂, ..., ∂gₖ/∂xₖ]

# Orthogonal loss (perpendicular distance squared)
L_ortho = mean(F(x)² / ||∇F||²)

# Orthogonal R² (normalized quality metric)
R²_ortho = 1 - L_ortho / var(F)
```

**Why orthogonal loss?**
- Standard R² depends on which variable is the "target"
- Orthogonal R² is **symmetric** - same result regardless of variable choice
- Better for implicit surfaces where no single variable is special

**Key Functions:**
- `compute_orthogonal_loss()` - Perpendicular distance metric
- `compute_orthogonal_r2()` - Normalized score ∈ [0,1]
- `z_score_normalize()` - Scales variables for fair comparison

---

### Step 5: Search Loop Orchestration
**File:** [core.py](degen_detector/core.py)

The main orchestration:

```python
1. Select parameters (via MI analysis)
2. Generate and rank tuples
3. For each tuple (in order of MI ranking):
   a. Fit implicit surface using alternating optimization
   b. Compute orthogonal R²
   c. If R²_ortho ≥ threshold: EARLY STOP (found good degeneracy!)
   d. If max_fits reached: STOP
4. Return all fits, ranked by R²_ortho
```

**Key Classes:**
- `DegenDetector` - Main API wrapper
- `CouplingSearchResult` - Contains all fits and metadata
- `CouplingFit` - Wraps `ImplicitFit` with parameter names and MI scores
- `ImplicitFit` - Result of fitting one implicit surface

**Main Method:**
- `search_couplings()` - Orchestrates entire workflow

---

## Example Usage

```python
from degen_detector.core import DegenDetector

# Load your posterior samples (N samples × M parameters)
detector = DegenDetector(samples, param_names)

# Search for pairwise degeneracies
result = detector.search_couplings(
    coupling_depth=2,          # Search pairs (2 components per fit)
    r2_threshold=0.95,         # Stop when R² ≥ 0.95
    max_fits=10,               # Try top 10 tuples
    mi_rank_method="sum"       # Rank by total MI
)

# Best fit
best = result.best_fit
print(best.equation_str)       # e.g., "log(sigma8) + 0.5*log(Om) = 0.342"
print(best.orthogonal_r2)      # e.g., 0.987
```

---

## Key Algorithm Parameters

### `search_couplings()` Parameters:
- **`params`**: Which parameters to search (list, int for top-N, or None for all)
- **`coupling_depth`**: Size of tuples - **DETERMINES NUMBER OF COMPONENTS**
  - `coupling_depth=2` → pairs → `g₁(x₁) + g₂(x₂) = c`
  - `coupling_depth=3` → triplets → `g₁(x₁) + g₂(x₂) + g₃(x₃) = c`
- **`r2_threshold`**: Early stop when R²_ortho ≥ threshold (default 0.95)
- **`max_fits`**: Limit number of fits attempted
- **`mi_rank_method`**: How to aggregate MI ("min", "avg", "sum", "geometric")
- **`max_complexity`**: PySR equation complexity limit
- **`niterations`**: PySR evolution iterations
- **`max_iterations`**: Alternating optimization iterations
- **`convergence_threshold`**: Residual std threshold for convergence

---

## Key Features

1. **Separability Assumption**: Decomposes k-D problem into k 1-D problems
2. **Symbolic Regression**: Uses PySR (genetic programming) for interpretable equations
3. **Orthogonal Loss**: Symmetric fit metric for implicit surfaces
4. **MI-based Ranking**: Prioritizes likely degeneracies first
5. **Early Stopping**: Stops when good fit found, saving computation

---

## Visualization

**File:** [plotting.py](degen_detector/plotting.py)

Three main plotting functions:
1. `plot_corner_with_implicit()` - Corner plot with discovered degeneracy curve overlaid
2. `plot_residual_corner()` - Corner plot colored by surface residuals
3. `plot_component_functions()` - Individual plots of each gⱼ(xⱼ)

**Plotting Script:** [scripts/plot_synthetic_results.py](scripts/plot_synthetic_results.py)

This script generates comprehensive diagnostics including all the above plots plus:
- True vs predicted values for each component function
- Equation comparison (ground truth vs fitted)

---

## Understanding R² Metrics

### Two Different R² Metrics Are Used:

#### 1. Orthogonal R² (from implicit_fit.py/loss.py)
- **Purpose:** Evaluates the overall fit quality of the implicit surface
- **Measures:** Perpendicular distance to surface `g₁ + g₂ + ... = c`
- **Formula:** `R²_ortho = 1 - L_ortho / var(F)` where `L_ortho = mean(F² / ||∇F||²)`
- **Properties:** Symmetric - same result regardless of which variable is treated as dependent
- **Used in:** Main algorithm results (`fit.orthogonal_r2`)

#### 2. Standard R² (in plot_synthetic_results.py)
- **Purpose:** Evaluates how well each individual component function `gⱼ(xⱼ)` matches expected values
- **Measures:** How well predicted component values match true component values
- **Formula:** Standard least-squares `R² = 1 - SS_res / SS_tot`
- **Properties:** Standard regression metric for 1D predictions
- **Used in:** Component-wise diagnostic plots ("true vs predicted" scatter plots)

### Why Both?

**Example:** If you have `g₁(σ₈) + g₂(Ωₘ) = c`:

- **Orthogonal R²**: How well does the surface `g₁(σ₈) + g₂(Ωₘ) - c = 0` capture the degeneracy? (k-dimensional surface quality)
- **Standard R² for g₁**: If I know all Ωₘ values and constant, can I predict what g₁(σ₈) should be? (1D component quality)
- **Standard R² for g₂**: If I know all σ₈ values and constant, can I predict what g₂(Ωₘ) should be? (1D component quality)

The standard R² values help diagnose **which parts** of the separable fit are working well vs. struggling.

---

## FAQ

### Q: How does the algorithm know to break functions down into many gⱼ's?

**A:** It doesn't discover the number - it's **specified by the user** via `coupling_depth`!

- `coupling_depth=2` → analyzes pairs → 2 components
- `coupling_depth=3` → analyzes triplets → 3 components
- Each `gⱼ` corresponds to one parameter in the tuple being analyzed

The algorithm assumes separability and tries to fit that form. It reports how well it works via R²_ortho.

### Q: What if the degeneracy isn't actually separable?

**A:** The algorithm will still try to fit a separable form, but the R²_ortho will be low, indicating poor fit. The algorithm doesn't validate the separability assumption - it's up to the user to interpret the fit quality.

### Q: How do I choose coupling_depth?

**A:** Start with `coupling_depth=2` (pairs) as most degeneracies involve 2-3 parameters. If you suspect higher-order degeneracies, try `coupling_depth=3` or higher, but note that:
- Computational cost grows combinatorially
- Higher-dimensional fits are harder to interpret
- Most physical degeneracies are low-dimensional

---

## Applications

### Synthetic Testing
**File:** [synthetic.py](degen_detector/synthetic.py)

Generators for known degeneracies:
- `generate_linear_separable()`: x + 2y - z = 0
- `generate_log_separable()`: log(x) + log(y) - z = 0
- `generate_power_law()`: log(σ₈) - 0.5·log(Ωₘ) = c
- `generate_exp_linear()`: exp(x) + y - z = 0
- `generate_quadratic_separable()`: x² + y² - z = 4
- `generate_trig_separable()`: sin(x) + cos(y) + z = 1

### Cosmology Application
Applied to Planck 2018 ΛCDM chains to discover parameter degeneracies like:
- σ₈ ~ Ωₘ^α (matter-structure degeneracy)
- Interdependencies between H₀, ns, τ, etc.

**Script:** [scripts/run_planck_analysis.py](scripts/run_planck_analysis.py)

---

## Technical Highlights

1. **Separable Form**: Decomposes k-D problem into k independent 1-D symbolic regressions at each iteration
2. **Orthogonal Loss**: Perpendicular distance to surface, invariant to variable choice
3. **Normalization**: All variables z-score normalized before computing orthogonal loss
4. **PySR Integration**: Genetic programming discovers symbolic expressions (interpretable)
5. **MI-based Ranking**: Prioritizes tuples most likely to have degeneracies
6. **Early Stopping**: Saves computation when good fit found

---

## Summary

The **degen_detector** package implements a sophisticated algorithm for discovering implicit parameter degeneracies in high-dimensional posterior samples. The main innovation is combining:
- Mutual information analysis for intelligent parameter selection
- Alternating optimization for separable implicit surface fitting
- Symbolic regression (PySR) for interpretable equations
- Orthogonal loss for symmetric fit evaluation

The algorithm is particularly well-suited for cosmological parameter analysis but applicable to any Bayesian inference problem with potential parameter degeneracies.



| Method | Source | Pros | Cons |
|--------|--------|------|------|
| **Implicit Derivative** | [Schmidt & Lipson 2010](https://link.springer.com/chapter/10.1007/978-1-4419-1626-6_5) | Principled, proven on physics problems | Requires dense low-noise data for numerical gradients |
| **Probabilistic Fitness (KL)** | [Roberts et al. GECCO 2024](https://dl.acm.org/doi/10.1145/3638530.3654357) | Noise-robust, no numerical gradients | Complex implementation, less interpretable |
| **Orthogonal Distance (ODR)** | [scipy.odr](https://docs.scipy.org/doc/scipy/reference/odr.html) | Mature library, handles implicit models | Parameter optimization only, no structure search |
| **PySR Custom Objective** | [PySR Discussions](https://github.com/MilesCranmer/PySR/discussions/299) | Full control via `full_objective`, derivative support | Requires Julia code, still searches `y = f(x)` form |

- [Schmidt & Lipson (2010): Symbolic Regression of Implicit Equations](https://link.springer.com/chapter/10.1007/978-1-4419-1626-6_5)
- [Roberts et al. (2024): Implicit Symbolic Regression via Probabilistic Fitness](https://dl.acm.org/doi/10.1145/3638530.3654357)
- [scipy.odr documentation](https://docs.scipy.org/doc/scipy/reference/odr.html)
- [PySR Custom Objectives Discussion](https://github.com/MilesCranmer/PySR/discussions/299)
- [Total Least Squares Overview](https://people.duke.edu/~hpgavin/SystemID/References/Markovsky+VanHuffel-SP-2007.pdf)
