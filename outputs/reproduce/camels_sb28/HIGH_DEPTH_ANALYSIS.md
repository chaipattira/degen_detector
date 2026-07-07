# CAMELS SB28 — High-depth degeneracies (coupling depth 3–6)

Companion to `DEGENERACY_HIGHLIGHTS.md`, focused on the depth-3…6 fits (3+ parameter
couplings). Same citation convention: `[probe/mode/depthN Leq]` links to the numeric
equation line in that run's `diagnostics/equations.txt` (MI/header a few lines above,
R²⊥/components below). Full dumps regenerable via `scratchpad/highdepth.py`.

The depth-2 pairs (previous report) are the *spines*; here the question is what the
extra parameters reveal. Three findings stand out.

---

## A. Mgas measures ONE flat 6-D hyperplane (the strongest high-depth result)

The gas baryon-fraction spine does not saturate — every parameter the MI screen adds
falls onto the **same tight linear surface**, and R²⊥ stays pinned at **1.0000** all
the way to depth 6 while MI climbs 0.74 → **1.06**:

| depth | added params | combination (linear) | R²⊥ | source |
|------:|--------------|----------------------|:---:|--------|
| 2 | Ωₘ, Ωᵦ | `11.58 Ωₘ − 102.66 Ωᵦ` | 1.0000 | [d2 L12](Mgas/linear/depth_2/diagnostics/equations.txt#L12) |
| 3 | +ε_bh | `−13.30 Ωₘ + 87.45 Ωᵦ − 0.85 ε_bh` | 1.0000 | [d3 L12](Mgas/linear/depth_3/diagnostics/equations.txt#L12) |
| 5 | +σ₈,nₛ | `−13.79 Ωₘ + 94.29 Ωᵦ − 3.12 σ₈ − 3.92 nₛ − 0.62 ε_bh` | 1.0000 | [d5 L12](Mgas/linear/depth_5/diagnostics/equations.txt#L12) |
| 6 | +f_edd | above `+ (0.47 f_edd + 0.03) e^{−0.62 f_edd}` | 1.0000 | [d6 L12](Mgas/linear/depth_6/diagnostics/equations.txt#L12) |

**Interpretation.** The gas map constrains essentially *one* combination of
{Ωₘ, Ωᵦ, σ₈, nₛ, ε_bh, f_edd} to machine precision; the other five directions are
prior-dominated. Ωᵦ carries by far the largest coefficient (≈ 90–103, vs ≤ 4 for the
rest), so the **baryon fraction remains the dominant axis** and σ₈/nₛ/ε_bh only *tilt*
the plane. Note the Ωₘ:Ωᵦ slope ratio rotates from 8.87 (depth 2) → 6.9 (depth 6) as
σ₈, nₛ enter — i.e. the "baryon-fraction" direction is not exactly Ωᵦ/Ωₘ once you
account for the amplitude parameters. **This is your cleanest example of a genuinely
high-dimensional but perfectly flat degeneracy** — a 5-D nuisance hyperplane inside a
6-D subspace — and it stays *linear* (no curvature needed) throughout.

### A′. The same thing as a power law (log mode)
Log-mode finds the multiplicative twin, also stable and tight (R²⊥≈0.998):

`−4.35 log Ωₘ − 2.25 log σ₈ + 4.64 log Ωᵦ − 3.22 log nₛ + 7.86 = 0`
→ **Ωᵦ⁴·⁶ / (Ωₘ⁴·⁴ σ₈²·² nₛ³·²) ≈ const**
([Mgas/log/depth4 L12](Mgas/log/depth_4/diagnostics/equations.txt#L12), extends to
[d6 L12](Mgas/log/depth_6/diagnostics/equations.txt#L12)).

The near-equal-and-opposite Ωᵦ/Ωₘ exponents (+4.6 / −4.4) confirm the baryon-fraction
reading: `(Ωᵦ/Ωₘ)^≈4.5 · (amplitude terms)`. Either form is quotable; the log form is
arguably the more physical "the gas map measures a baryon-fraction power law."

---

## B. The probes differ in the *shape* of their degeneracy, not just the parameters

A clean qualitative contrast worth a figure/table:

- **Mgas — flat / affine.** Best fits are pure linear hyperplanes; nonlinear terms give
  no R²⊥ gain (§A). Physically: gas mass responds ~linearly to the baryon budget.
- **Mcdm — curved.** The AGN sector enters through **recurring nonlinear atoms**:
  - χ_QSO as a hard **threshold**: `exp(−112.885·χ_QSO)` appears identically in
    Mcdm depths 3–6 ([d3 L12](Mcdm/linear/depth_3/diagnostics/equations.txt#L12),
    [d4 L158](Mcdm/linear/depth_4/diagnostics/equations.txt#L158),
    [d6 L12](Mcdm/linear/depth_6/diagnostics/equations.txt#L12)). The huge rate (≈113)
    means χ_QSO acts as a **switch** near χ_QSO ≈ 0.009 — below it χ_QSO matters, above
    it saturates. A physically suggestive "QSO threshold behaves like a step."
  - ε_bh as a **quadratic** centered at ε_bh ≈ 0.12: `(15.55·ε_bh − 1.86)²` recurs in
    [d4 L158](Mcdm/linear/depth_4/diagnostics/equations.txt#L158) and
    [d5 L84](Mcdm/linear/depth_5/diagnostics/equations.txt#L84) — an optimum/turnover,
    not a monotonic trend.
  - σ₈ via `exp(−19.11·σ₈)` at [d5 L156](Mcdm/linear/depth_5/diagnostics/equations.txt#L156).
  Consequence: Mcdm surfaces plateau at R²⊥≈0.999 and *need* the curvature; a purely
  linear fit underperforms. Total-matter maps couple to feedback nonlinearly.
- **Mstar — linear but noisier.** Fits are mostly affine in {Ωₘ, A_SN1, α_imf, nₛ} but
  cap at R²⊥≈0.998 and never reach 1.0 — stellar maps are the least constraining probe,
  consistent with the depth-2 story.

---

## C. Mstar splits into two competing degeneracy families

Unlike the other probes, Mstar's top fits fall into **two distinct branches** with
different partners and fit quality — the tool is finding two real, competing directions:

- **α_imf branch** (R²⊥ ≈ 0.979): `−2.70 Ωₘ + 0.55 α_imf + 0.165 A_SN1 = const`,
  extremely **stable coefficients** across depths 3–6
  ([d3 L12](Mstar/linear/depth_3/diagnostics/equations.txt#L12),
  [d6 L12](Mstar/linear/depth_6/diagnostics/equations.txt#L12)). Ties Ωₘ to the **IMF
  slope** and SN feedback — a galaxy-formation degeneracy.
- **nₛ branch** (R²⊥ ≈ 0.994, *better*): `−4.88 Ωₘ + 0.063 A_SN1 − 1.03 nₛ = const`
  ([d3 L77](Mstar/linear/depth_3/diagnostics/equations.txt#L77), with f_bh_acc at
  [d4 L229](Mstar/linear/depth_4/diagnostics/equations.txt#L229)).

That two coherent branches persist (rather than merging) suggests the stellar-mass map
constrains a genuinely **2-D** degenerate manifold, not a single line — a candidate for
"the reparameterization d = f(Ωₘ, α_imf, A_SN1) *and* d′ = g(Ωₘ, nₛ)" (criterion 4).

---

## D. ε_bh is the shared BH-feedback nuisance of Mgas *and* Mcdm

Black-hole feedback efficiency ε_bh enters the high-depth degeneracies of **both** the
gas and CDM maps (never the stellar map):
- Mgas: linearly, small coefficient (−0.6…−0.85), riding the baryon-fraction plane
  ([d3 L12](Mgas/linear/depth_3/diagnostics/equations.txt#L12)).
- Mcdm: nonlinearly, as the quadratic turnover above
  ([d5 L84](Mcdm/linear/depth_5/diagnostics/equations.txt#L84)).

So ε_bh is measured by two probes but through **different functional couplings** — a
nice cross-probe consistency check, and it reinforces criterion 3: a joint Mgas+Mcdm
fit sees ε_bh from two directions and should pin it.

---

## E. What each probe absorbs as depth grows (sensitivity ordering)

The order in which parameters join the top-MI chain ranks each probe's sensitivity:

- **Mgas**: Ωᵦ → ε_bh → σ₈, nₛ → f_edd → (h weakly at
  [d6 L178](Mgas/linear/depth_6/diagnostics/equations.txt#L178)). Cosmology-heavy.
- **Mcdm**: χ_QSO → nₛ → ρ_wind → ε_r, ε_bh → σ₈. AGN + wind feedback + tilt.
- **Mstar**: A_SN1 → α_imf / nₛ → ρ_wind, f_windE, Z_windE → f_bh_acc, v_wind. Almost
  entirely galaxy-formation astrophysics; cosmology (beyond Ωₘ) enters last.

This ordering is itself a result: **gas is the most cosmological probe, stars the least**,
with CDM sitting in between and uniquely sensitive to the AGN sector.

---

## Caveats
- **R²⊥ = 1.0 at high depth is partly extra freedom.** More components = more ways to
  fit a surface; the *striking* part for Mgas is that it stays 1.0 while remaining
  **linear with stable coefficients**, which the added-freedom argument does not explain.
- **Recurring identical constants** (112.885 for χ_QSO, 15.5468 for ε_bh, 34.7359 /
  19.1099 for nₛ / σ₈) partly reflect the pipeline **reusing hall-of-fame 1-D component
  fits** across tuples. But their reuse *across different parameter partners* is genuine
  evidence that each parameter's 1-D contribution shape is stable — i.e. the separable
  g₁+g₂+… ansatz holds here.
- **Some high-depth fits produced no consensus equation** (marked NA in the dumps: e.g.
  several depth-4 Mgas tuples, depth-6 Mstar log). Excluded, not reported.
- Interpretations (baryon fraction, QSO threshold, IMF degeneracy) are physical
  readings of the symbolic forms; confirming them needs the marginal/conditional widths
  and the injected truth θ\* (see `DEGENERACY_HIGHLIGHTS.md` caveats).
