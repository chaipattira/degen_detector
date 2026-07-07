# CAMELS SB28 — Degeneracy highlights for the paper

Analysis of all 150 symbolic fits under `outputs/reproduce/camels_sb28/`
(3 tracer maps × {linear, log} × coupling depths 2–6, top-5 MI tuples each).
Tightness is the orthogonal R²⊥ reported by the detector; MI is the screening
score. Marginal-constraint claims use CAMELS domain knowledge (no raw samples
were available locally) — see *Caveats*.

**Sources.** Every fit below is cited as a clickable link to its
`diagnostics/equations.txt` (paths relative to this file). The link points at the
**equation line** (the `[1] … = 0` line with the numeric coefficients); the fit's
header/MI is a few lines above it and R²⊥/BIC/residual/components are the ~15 lines
below it. Format: `[probe/mode/depthN Leq]`.

**Parameter map** (`reproduce/camels/fig_true_vs_pred.py` lines 52–76, `CAMELS_LATEX`):
cosmology = Ωₘ(0), σ₈(1), Ωᵦ(6), h(7), nₛ(8);
standard feedback = A_SN1(2), A_AGN1(3), A_SN2(4), A_AGN2(5); extended SB28 astro =
α_imf(11), ρ_wind(15), v_wind(16), f_windE(17), Z_windE(18), f_metals(20),
f_edd(23), ε_bh(24), ε_r(25), χ_QSO(26), …

---

## 1. The headline result: each probe pins Ωₘ through a *different* sector

The single most important pattern. Which parameters the MI screen selects (i.e.
which the map is sensitive to) is almost disjoint across the three tracers:

| Parameter        | Mcdm | Mgas | Mstar | Physical sector |
|------------------|:----:|:----:|:-----:|-----------------|
| Ωₘ, σ₈, nₛ       |  ✓   |  ✓   |  ✓/–  | cosmology |
| **Ωᵦ**           |  ✓   | **✓✓** |  –   | baryon fraction |
| **χ_QSO, ε_bh, ε_r** | **✓✓** | ✓ |  –   | AGN / BH feedback |
| **A_SN1, α_imf, v_wind, f_bh_acc, A_SN2** | – | ± | **✓✓** | stellar / SN / wind |
| f_metals         |  –   |  ✓   |   –   | gas enrichment |
| f_edd, Z_windE   |  –   |  ✓   |   ✓   | accretion / winds |

- **Mgas** → the **baryon fraction** (Ωₘ–Ωᵦ), plus metals and Eddington ratio.
- **Mcdm** → **AGN/BH feedback** (χ_QSO, ε_bh, ε_r) that redistributes total matter.
- **Mstar** → **galaxy-formation astrophysics** (A_SN1, α_imf, winds, BH accretion).

*Table source:* cross-tabulated from the `Parameters:` headers of all 15 linear
`equations.txt` files (depths 2–6 × 3 probes) — i.e. which parameters the MI screen
ever selects per probe. Regenerate with `scratchpad/cross.py`. Per-cell fits are the
specific equation lines cited in §2–§3.

This is the backbone of the "combining probes breaks the degeneracies" story
(criterion 3, §4).

---

## 2. Criterion 1 — tight degeneracies rescuing poorly-constrained parameters

Ranked by interest. These involve a parameter that is **prior-dominated in the
marginal** for single-field CAMELS inference, yet lies on a razor-thin surface.

### ⭐ 1a. Ωₘ–Ωᵦ baryon fraction (Mgas) — *the strongest single result*
- MI = **0.739** (10× the next pair), R²⊥ = **1.0000**, linear.
- `11.578 Ωₘ − 102.659 Ωᵦ + 1.255 = 0`  →  **Ωᵦ ≈ 0.113 Ωₘ + 0.012**.
- Ωᵦ is essentially unconstrained by a single field's map individually, but the
  gas map pins the combination `Ωₘ − 8.87 Ωᵦ` (≈ cosmic baryon fraction) to a line.
- Log-mode confirms it (R²⊥=0.996) but linear is cleaner → the relation is affine,
  not a pure power law.
- Source: [Mgas/linear/depth2 L12](Mgas/linear/depth_2/diagnostics/equations.txt#L12)
  (block header + MI at L5); log-mode [Mgas/log/depth2 L12](Mgas/log/depth_2/diagnostics/equations.txt#L12).

### ⭐ 1b. Ωₘ–χ_QSO (Mcdm)
- MI = 0.076, R²⊥ = **0.9999**, linear.
- `1.534 Ωₘ − 28.495 χ_QSO − 0.322 = 0`  →  **χ_QSO ≈ 0.054 Ωₘ − 0.011**.
- χ_QSO (the QSO threshold, an obscure extended-SB28 AGN parameter) is normally a
  flat nuisance, yet here it is a linear function of Ωₘ to 4 nines. Great "who knew
  *that* was constrained" example — highlight the corner/manifold plot.
- Source: [Mcdm/linear/depth2 L75](Mcdm/linear/depth_2/diagnostics/equations.txt#L75) (block L68).

### 1c. Ωₘ–ε_bh (Mcdm)
- MI = 0.049, R²⊥ = 0.998, mildly quadratic:
  `−1.543 Ωₘ − 37.72 ε_bh² + 9.05 ε_bh + 0.145 = 0`.
- A second, distinct BH-feedback parameter tied to Ωₘ (curved surface — nicer figure
  than a straight line).
- Source: [Mcdm/linear/depth2 L267](Mcdm/linear/depth_2/diagnostics/equations.txt#L267) (block L260).

### 1d. ρ_wind–Z_windE (Mstar) — a purely *astrophysical* degeneracy
- MI = 0.041, R²⊥ = **1.0000**: `29.69 Z_windE − 0.722 exp(−8.97 ρ_wind) + 0.309 = 0`.
- Two normally-unconstrained wind parameters lock together (no cosmology involved).
  Shows the tool finding structure entirely inside the nuisance sector.
- Source: [Mstar/linear/depth2 L75](Mstar/linear/depth_2/diagnostics/equations.txt#L75) (block L68).

### 1e. Z_windE–f_edd (Mgas), rho_wind–eps_r (Mcdm)
- Z_windE–f_edd: R²⊥=0.9999, near-linear. rho_wind–eps_r: R²⊥=0.99, exponential
  form `0.602 ε_r + 1.147 exp(−9.52 ρ_wind) = 0.694`. Secondary but clean.
- Sources: Z_windE–f_edd [Mgas/linear/depth2 L264](Mgas/linear/depth_2/diagnostics/equations.txt#L264) (block L257);
  ρ_wind–ε_r [Mcdm/linear/depth2 L138](Mcdm/linear/depth_2/diagnostics/equations.txt#L138) (block L131).

---

## 3. Criterion 2 — tight in one probe, absent/loose in another

### ⭐ 2a. Baryon fraction Ωₘ–Ωᵦ: **Mgas only**
- Mgas R²⊥ = 1.0000 (MI 0.74). Ωᵦ is **never selected** in Mstar and does not form a
  comparable pair in Mcdm. Textbook criterion-2: gas traces baryons, so only the gas
  map measures Ωᵦ/Ωₘ; stellar/CDM maps leave Ωᵦ prior-dominated.
- Source: [Mgas/linear/depth2 L12](Mgas/linear/depth_2/diagnostics/equations.txt#L12); to
  confirm absence, scan the summary tables of
  [Mcdm/linear/depth2](Mcdm/linear/depth_2/summary.txt) and
  [Mstar/linear/depth2](Mstar/linear/depth_2/summary.txt) (Ωᵦ = theta_6 appears in neither).

### ⭐ 2b. Ωₘ–A_SN1 (SN feedback): **Mstar (and Mgas), never Mcdm**
- Mstar R²⊥ = 0.95 at depth 2 (`1.414 Ωₘ − 0.130 A_SN1 + 0.308 = 0`), and A_SN1
  reaches R²⊥≈1.0 inside deeper Mgas chains — but A_SN1 is **never selected in Mcdm**.
  SN feedback shapes stellar mass and gas, not the total-matter field.
- Source: [Mstar/linear/depth2 L12](Mstar/linear/depth_2/diagnostics/equations.txt#L12) (block L5);
  A_SN1 in gas e.g. [Mgas/linear/depth3 L80](Mgas/linear/depth_3/diagnostics/equations.txt#L80)
  (2nd tuple Ωₘ–A_SN1–Ωᵦ, block L73).

### 2c. AGN sector (χ_QSO, ε_bh): **Mcdm/Mgas, never Mstar**
- The mirror image of 2b: the BH-feedback parameters that dominate Mcdm never appear
  in Mstar. Good paired figure with 2b (SN sector vs AGN sector swap probes).
- Source: χ_QSO/ε_bh in Mcdm at [Mcdm/linear/depth2 L75](Mcdm/linear/depth_2/diagnostics/equations.txt#L75)
  and [L267](Mcdm/linear/depth_2/diagnostics/equations.txt#L267); absent from
  [Mstar/linear/depth2 summary](Mstar/linear/depth_2/summary.txt) (χ_QSO=theta_26, ε_bh=theta_24).

### 2d. σ₈–nₛ (S₈-like): shared but probe-dependent form
- Mcdm R²⊥=0.998 (quadratic in σ₈), Mgas R²⊥=0.997 (linear), Mstar: not selected.
  The one genuinely multiplicative direction — log-mode holds up here (R²⊥≈0.995–0.996)
  where it fails elsewhere.
- Sources: [Mcdm/linear/depth2 L201](Mcdm/linear/depth_2/diagnostics/equations.txt#L201) (block L194);
  [Mgas/linear/depth2 L72](Mgas/linear/depth_2/diagnostics/equations.txt#L72) (block L65);
  log-mode [Mcdm/log/depth2 L195](Mcdm/log/depth_2/diagnostics/equations.txt#L195).

---

## 4. Criterion 3 — orthogonal directions that a combined model should break

You have no combined posterior, so these are **recommendations** with predicted
outcomes. The three probes constrain Ωₘ along three near-independent directions:

- Mgas:  `Ωₘ − 8.87 Ωᵦ`      (baryon-fraction axis) — [L12](Mgas/linear/depth_2/diagnostics/equations.txt#L12)
- Mcdm:  `Ωₘ − 18.6 χ_QSO`   (AGN axis; also Ωₘ–ε_bh) — [L75](Mcdm/linear/depth_2/diagnostics/equations.txt#L75)
- Mstar: `Ωₘ − 0.092 A_SN1`  (SN axis; also Ωₘ–α_imf) — [L12](Mstar/linear/depth_2/diagnostics/equations.txt#L12)

(coefficients are the g₁/g₂ slope ratios from those three equation lines.)

Because the second parameter differs in each (Ωᵦ vs χ_QSO vs A_SN1, which are
mutually near-orthogonal nuisances), **a joint Mcdm+Mgas+Mstar posterior should
collapse all three 2-D degeneracies and pin Ωₘ**, while simultaneously constraining
Ωᵦ, χ_QSO, and A_SN1 that no single probe pins on its own.

**Recommended experiments (in priority order):**
1. **Mcdm + Mstar** — the cleanest test: AGN axis (χ_QSO) ⟂ SN axis (A_SN1), no shared
   nuisance, so both single-probe degeneracies should break. Best headline pair.
2. **Mcdm + Mgas** — Ωₘ–Ωᵦ vs Ωₘ–χ_QSO; predict Ωᵦ *and* χ_QSO both tighten.
3. **All three** — expect Ωₘ marginal to shrink toward the intersection of all axes;
   run the detector on the joint posterior and check these pairs' R²⊥ *drop* (the
   signature of a broken degeneracy).

---

## 5. Criterion 4 — reparameterizations to try (predicting degeneracies vs raw params)

Define degeneracy coordinates from the fits above and re-train the emulator/NPE to
predict them; the hypothesis is these are easier (lower loss / better calibration)
targets than the raw cosmological parameters:

- `d₁ = Ωₘ − 8.87 Ωᵦ`     (Mgas — baryon fraction; near-perfectly measured)
- `d₂ = Ωₘ − 18.6 χ_QSO`  (Mcdm — AGN axis)
- `d₃ = Ωₘ − 0.092 A_SN1` (Mstar — SN axis)
- `d₄ = σ₈ · f(nₛ)` from the S₈-like σ₈–nₛ relation (the one true power-law direction)

Since Mgas measures `d₁` to R²⊥=1.0000 but Ωᵦ alone is prior-dominated, predicting
`d₁` should be dramatically easier than predicting Ωᵦ — the clearest single
demonstration of criterion 4 available here.

---

## 6. How the pairs nest with depth (context, not headline)

Depth 3–6 confirm the pairs are the real structure and just accrete the next-most
informative nuisance:
- **Mgas**: Ωₘ–Ωᵦ → +ε_bh → +σ₈,nₛ → +f_edd, driving MI 0.74 → **1.06** at depth 6,
  R²⊥ staying 1.0000. The baryon-fraction axis is the spine.
  ([depth3 L12](Mgas/linear/depth_3/diagnostics/equations.txt#L12),
  [depth6 L12](Mgas/linear/depth_6/diagnostics/equations.txt#L12))
- **Mcdm**: Ωₘ–χ_QSO → +nₛ → +ρ_wind → +ε_r,ε_bh (MI → 0.53). AGN + wind feedback.
  ([depth3 L12](Mcdm/linear/depth_3/diagnostics/equations.txt#L12),
  [depth6 L12](Mcdm/linear/depth_6/diagnostics/equations.txt#L12))
- **Mstar**: Ωₘ–A_SN1 → +α_imf/nₛ → +ρ_wind,f_windE,Z_windE (MI → 0.22). Slowest to
  grow → stellar maps carry the least aggregated degeneracy, consistent with them
  being the noisiest cosmological probe.
  ([depth6 L12](Mstar/linear/depth_6/diagnostics/equations.txt#L12))

Deeper fits add complexity (quadratic/exp terms) for marginal R²⊥ gains — for the
paper, the **depth-2 pairs are the interpretable, quotable results**; cite depth-3
only where the third parameter is physically motivated (e.g. Ωₘ–Ωᵦ–ε_bh in gas).

---

## Caveats (for honesty in the write-up)
- **No marginal widths were measured**: `samples.npy`/`posterior.pkl` live on the
  cluster. "Poorly constrained in the marginal" rests on standard CAMELS single-field
  behavior (Ωₘ, σ₈ constrained; Ωᵦ, h, nₛ and extended-SB28 astro largely
  prior-dominated), not on numbers from these runs. To make criterion-1 claims
  quantitative, provide the samples and I'll compute marginal-vs-conditional widths.
- **"Passes through the truth" not yet checked**: you have the injected θ\* for the
  fiducial sim. Give me θ\* and I'll verify each surface passes through it (e.g. the
  Ωₘ–nₛ line at [Mcdm/linear/depth2 L12](Mcdm/linear/depth_2/diagnostics/equations.txt#L12),
  `−8.929 Ωₘ + 4.753 nₛ − 0.758 = 0`, has an intercept offset from the CV fiducial, so
  the observed sim may not be the (0.3, 0.962) fiducial — worth confirming).
- **A parameter "appearing" ≠ individually constrained**: MI selection means it is
  constrained *in combination*; high R²⊥ at large depth is partly the extra freedom of
  more terms. The depth-2 pairs are the safe quantitative claims.
- A few high-depth fits produced no consensus equation (log-mode Ωₘ–χ_QSO, some
  depth-4/5 tuples) — excluded rather than reported.
