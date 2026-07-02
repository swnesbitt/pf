# Plan — Vertical wind shear controls on convective intensity, organization, and stratiform area

**Goal.** Test, robustly and with proper confounder control, three hypotheses using the
TRMM/GPM precipitation-feature superdatabase joined to ERA5:

- **H1 (intensity).** For a *given CAPE*, stronger vertical wind shear raises the
  20/30/40-dBZ **echo-top heights** and lowers the **85/37-GHz PCT** (deeper, more
  ice-laden convection).
- **H2 (organization).** Stronger shear favors **convective organization** — higher
  P(MCS) and larger MCS area — at given CAPE.
- **H3 (stratiform).** Stronger shear increases **stratiform rain area** (and its
  fraction), at given CAPE.

Examined with **2-D histograms across controlling factors**, separately **land/ocean**,
**regionally**, and restricted to **|lat| < 40°**.

---

## 1. Data and variables (already in `pf_catalog.duckdb`)

Join `features` ⨝ `era5` on `(feature_id, mission)` (86.8 M TRMM rows carry shear; GPM too).

**Response variables** (`features`):
| hypothesis | responses |
|---|---|
| H1 intensity | `max_ht_20dbz`, `max_ht_30dbz`, `max_ht_40dbz` (m; use QC'd — see §3), `min_pct_85_89`, `min_pct_37` (K) |
| H2 organization | `is_mcs`, `feature_class`, `area_km2`, `conv_area_km2`, MCS rain fraction (`volrain_total` filtered) |
| H3 stratiform | `strat_area_km2`, `strat_area_frac`, `strat_rain_frac`, `volrain_strat` |

**Controlling / environment variables** (`era5`):
- **Shear** (primary predictor): `shear_6000m_*` (deep-layer, organization), `shear_3000m_*`
  (low-level), `shear_1000m_*` (cold-pool/inflow). Bulk |V(H) − V(10 m)|.
- **CAPE** (the thing to hold fixed): `*_cape_*`. **Use the ambient box value, not the
  centroid** (see §3): e.g. `p90_cape_2p50deg` or `mean_cape_5deg`.
- **Confounders to control**: `*_cin_*` (CIN), `*_tpr_*` (column water vapor / moisture →
  entrainment/dilution), `sst`/`skt` (surface, land/ocean proxy), latitude, `area_km2`
  (PCT and echo-top depend on feature size), local solar time.

**Filters (baseline sample):** `|centroid_lat| < 40`, finite CAPE & shear, and for the
*intensity* tests restrict to genuinely convective features (`max_ht_30dbz > 0` OR
`conv_area_km2 > 0`) so we compare storms, not drizzle. Land vs ocean via `frac_ocean ≥ 0.5`.

---

## 2. Core technique — conditional analysis in environment phase space

"For a given CAPE" = **stratify, don't just correlate.** The backbone is a 2-D composite:

> Bin features by (CAPE, shear); in each bin show a **robust statistic of the response**
> (median, and the p90/p99 for the *intensity tail*). Reading **along the shear axis at
> fixed CAPE row** isolates the shear effect.

This is the satellite-era analog of environment compositing (CFAD-in-environment-space)
and avoids the central confounder: **CAPE and shear are anticorrelated across regimes**
(high-shear/low-CAPE baroclinic vs. high-CAPE/low-shear tropical), so a raw shear–intensity
correlation is regime aliasing. Binning breaks that; regression/GAM quantifies it.

---

## 3. Key methodological decisions (and why)

1. **Ambient (box) CAPE, not centroid CAPE.** ERA5 CAPE *at the storm* is depleted by the
   ongoing convection (and ERA5 only partly "sees" the storm). The **surrounding-box upper
   percentile** (`p90_cape_2p50deg`, or `max_cape_5deg`) better represents the *pre-convective*
   environment the storm drew on. Report sensitivity to this choice.
2. **Deep-layer shear (0–6 km) is the organization variable** (RKW/Weisman-Rotunno line
   theory); 0–3 km speaks to updraft/cold-pool interaction. Run H1/H3 with all three layers
   but lead with `shear_6000m`.
3. **Control for free-tropospheric moisture (`tpr`).** Entrainment/dilution work (Hannah
   2017; Zhang 2009) shows undilute vs. dilute CAPE diverge with environmental humidity —
   so stratify or regress on moisture, else "CAPE" is ambiguous.
4. **Echo-top QC.** Use the Hirose-2023 QC fields (`echotop_qc_flags`,
   `max_ht_20dbz_censored`, `ray_obs_ceiling_m`) — drop censored/clutter-contaminated tops.
5. **Robust statistics.** Distributions are heavy-tailed → use **medians + IQR** for central
   tendency and **p90/p99** for the intensity tail (Zipser's "most intense" lives in the tail).
6. **Min-PCT depends on feature size** (bigger features more likely to contain a very cold
   pixel). Treat PCT either at fixed `area_km2` or report alongside size — don't read it as
   pure intensity.
7. **Sample independence.** Features in the same orbit/region/time are autocorrelated →
   significance via **spatial–temporal block bootstrap** (resample by orbit or by 5°×pentad
   block), not naive N.
8. **Sensor harmonization.** TRMM min-detectable ≈ 16 dBZ vs GPM ≈ 12 dBZ, and TRMM |lat|<37.
   Lead with **TRMM-only** for the intensity/echo-top tests (consistent sensitivity), use GPM
   as an independent replication; never pool raw echo-tops across sensors without noting it.

---

## 4. Proposed analyses

**A1 — CAPE–shear joint climate (setup / confounder map).** 2-D histogram of feature count
over (ambient CAPE, 0–6 km shear), land vs ocean, and the CAPE–shear correlation by region.
Establishes the sampling and the anticorrelation we must control for.

**A2 — H1 intensity composites (the headline).** For each response
(`max_ht_{20,30,40}dbz`, `min_pct_{85,37}`): heatmap of **median(response)** over
(CAPE × shear) bins, land and ocean panels. Then the **"fixed-CAPE" line plot**: response
vs. shear, one line per CAPE quartile — the direct hypothesis test (positive slope for
echo-tops, negative for PCT). Repeat for **p90 / p99** (extreme-convection tail).
Quantify ∂(echo-top)/∂(shear)|CAPE with bootstrapped CIs.

**A3 — H2 organization.** (a) **P(is_mcs)** heatmap over (CAPE × shear); (b) median
`area_km2` and `conv_area_km2` of convective features vs. shear at fixed CAPE; (c) MCS
**rain fraction** vs. shear|CAPE. Logistic regression P(MCS) ~ shear + CAPE + moisture +
land/ocean for the partial shear effect.

**A4 — H3 stratiform.** `strat_area_km2` and `strat_area_frac` vs. shear at fixed CAPE,
land/ocean — and **within MCSs only** (the stratiform region is an organized-system feature).
2-D composite + fixed-CAPE lines.

**A5 — Regional stratification.** Re-run A2–A4 composites for the basin boxes (Amazon,
Congo, Sahel, Maritime Continent, W/E Pacific, US Great Plains, SE S. America, Bay of
Bengal) to test robustness and regime dependence (continental high-shear MCS vs. maritime
weak-shear shallow).

**A6 — Multivariate isolation / robustness battery.**
- **Partial / multiple regression**: standardize predictors; report the **partial
  coefficient of shear** controlling CAPE, CIN, moisture, lat, land/ocean.
- **Quantile regression** (statsmodels) of echo-top **p90/p99** on shear|CAPE — shear may
  matter most for the tail.
- **GAM partial-dependence** (pyGAM): response ~ s(CAPE) + s(shear) + s(tpr) + s(lat);
  the **shear partial-dependence curve** is the cleanest "effect of shear holding others."
- **Moisture-tercile stratification** (dilution control).
- **Centroid-vs-box CAPE** and **0–6 vs 0–3 km shear** sensitivity.
- **Block bootstrap** CIs throughout; per-bin **N maps** so we never over-interpret thin bins.

**A7 — What explains the *extremes* of echo-top height and microwave Tb (variable isolation).**
Beyond the shear-at-fixed-CAPE tests, rank *which* environmental (and morphological) variables
control the **tail** of intensity — the Zipser "most intense storms" question.
- **Define extremes** (per land/ocean, |lat|<40): echo-top `max_ht_40dbz ≥ p99` (also 20/30-dBZ
  tops, and absolute cuts e.g. 40-dBZ ≥ ~14 km); coldest Tb `min_pct_37 ≲ 150 K`,
  `min_pct_85 ≲ 100 K` (Zipser extreme thresholds). QC the tops (sensor ceiling, §3.4).
- **(i) Composite environment of extremes vs. typical.** Box/ridge plots of ambient CAPE, CIN,
  `shear_{1,3,6}km`, `tpr`, SST for the top-1%/0.1% vs. the median feature — *what environment
  do the extremes inhabit* (robust, interpretable, Zipser-style).
- **(ii) Quantile regression** at τ = 0.90/0.99 of each response on the standardized predictor
  set — each coefficient is the marginal shift of the **tail** per variable, controlling others.
- **(iii) Gradient-boosted trees / random forest** predicting `P(extreme)` (class-weighted) and
  the continuous response, interpreted with **SHAP** (mean|SHAP| ranking, dependence plots for
  functional form, and **interaction values** to expose CAPE×shear synergy) + permutation
  importance as a cross-check. This is the modern "isolate the variables that explain extremes",
  capturing nonlinearity and interactions a linear fit misses.
- **(iv) Logistic regression** `P(extreme) ~ standardized predictors` → odds ratios as an
  interpretable baseline against the ML ranking.
- **Rigor:** train/test split by **time/region block** (autocorrelation → no leakage);
  precision–recall for the rare class; **group correlated predictors** (CAPE box radii, shear
  layers) so tree importance isn't diluted; report predictor correlation matrix; SHAP on held-out.
- **Output:** ranked variable-importance bars per response × land/ocean × region; SHAP dependence
  plots for the top drivers; explicit CAPE×shear interaction. Expect (per literature) moisture/CWV,
  instability, and shear to lead — and the land/ocean ranking to differ.

---

## 4B. Machine-learning & explainable-AI framework (cross-cutting, all hypotheses)

ML/XAI is **complementary to the physical composites (A1–A5), not a replacement** — its value
is handling **nonlinearity, many correlated predictors jointly, interactions (CAPE×shear), and
the tail**, then *explaining* the fitted relationship. Treat it as **associational** unless using
the causal-ML estimators below. Guiding rule: **convergent evidence** — an ML-flagged driver only
counts if it agrees with the binned composites and a simple regression.

**Models (glass-box first, then high-capacity):**
- **Explainable Boosting Machine (EBM, InterpretML)** — a glass-box GAM with auto interaction
  detection; its shape functions *are* the explanation (no post-hoc approximation). Lead model.
- **Gradient-boosted trees** (XGBoost / LightGBM / CatBoost) and **random forest** — high-capacity
  baselines; CatBoost handles the categorical region/land-ocean cleanly.
- **Tail / distributional models:** LightGBM **quantile objective** or **quantile regression
  forests** for echo-top p90/p99; **NGBoost** for the full predictive distribution. Directly serve
  the extreme-value question (A7).
- **Interpretable baselines:** elastic-net **logistic** (P(MCS), P(extreme)) and linear/quantile
  regression — standardized coefficients / odds ratios to anchor the ML ranking.
- (Tabular deep nets — TabNet/FT-Transformer — noted but not led; GBMs typically win on tabular.)

**Explainability (XAI):**
- **TreeSHAP** (exact, fast for trees): global `mean|SHAP|` ranking, **beeswarm** summary,
  **dependence plots** (functional form, e.g. echo-top rising-then-saturating in shear), and
  **SHAP interaction values** to quantify **CAPE×shear synergy** per hypothesis.
- **ALE (Accumulated Local Effects)** — preferred over PDP here because our predictors are
  **strongly correlated** (CAPE box radii; the three shear layers); ALE stays valid where PDP
  is biased. Pair with **ICE** curves to expose effect heterogeneity.
- **Friedman H-statistic** — quantifies how much of the response variance is interaction
  (test whether CAPE×shear is a real synergy vs. additive).
- **Permutation importance** as a cross-check, but **conditional/grouped** (group all CAPE-box
  and all shear-layer features) so collinearity doesn't dilute or double-count importance.

**Causal-ML — to actually "isolate" shear's effect (not just rank association):**
- **Double / debiased ML (EconML `LinearDML`, `CausalForestDML`)** — treat **shear as a
  continuous treatment**, partial out flexible ML nuisance models of the confounders
  (CAPE, CIN, moisture, SST, lat, region) → a **debiased partial effect of shear with valid CIs**.
  This is the rigorous ML version of "effect of shear at fixed CAPE+moisture+…".
- **Causal forest (heterogeneous effects)** — does shear's effect **vary with CAPE / region /
  land-ocean**? Estimates the conditional effect surface nonparametrically — literally H1's
  "for a given CAPE" as a continuous function.
- **Conditional mutual information / partial distance correlation** — nonparametric dependence
  of response on shear *controlling CAPE*, as a model-free sanity check.
- **Honesty:** all assume **unconfoundedness** (no unobserved common cause — e.g. large-scale
  forcing). State it; this is attribution-grade evidence, not proof of causation.

**Per-hypothesis use:** H1 → EBM/GBM regress echo-top & PCT, SHAP rank + ALE shear curve + DML
partial shear effect + quantile-GBM tail. H2 → classify P(MCS) + regress area, SHAP, DML
shear→organization. H3 → regress strat area/frac, SHAP, ALE. A7 → P(extreme) classifier +
quantile models + SHAP + composite environment.

**ML rigor (mandatory):** spatiotemporal **block** train/val/test split (no autocorrelation
leakage); **PR-AUC + calibration** for the rare extreme class (class weights/`scale_pos_weight`);
**group correlated predictors**; **bootstrap the model + SHAP** for importance CIs; always report
held-out performance and compare ML drivers against the A1–A5 composites.

**Stack (into `pf`):** `scikit-learn`, `xgboost`/`lightgbm`/`catboost`, `shap`, `interpret` (EBM),
`PyALE`/`alibi` (ALE), `econml` (DML/causal forest), `statsmodels` (quantile reg), `ngboost`.

---

## 5. Threats to validity (state in the notebook)
- **Snapshots, not lifecycles** — TRMM/GPM see a feature once; shear→organization is a
  *time-integrated* process. Interpret as statistical association; optionally condition on a
  lifecycle proxy (convective fraction / stratiform fraction).
- **ERA5 CAPE contamination** near active convection (→ box ambient, §3).
- **CAPE–shear regime aliasing** (→ binning + region split, A1/A5).
- **Sensor sensitivity & sampling** (→ TRMM-lead, |lat|<40, GPM replication, §3.8).
- **PCT–size dependence** (→ control area, §3.6).
- **Multiple comparisons** across many bins/regions (→ bootstrap CIs, effect sizes over p-values).

---

## 6. Literature grounding (techniques & precedent)
- **Intensity metrics & land/ocean contrast:** Zipser, Cecil, Liu, Nesbitt & Yorty (2006),
  *Where are the most intense thunderstorms on Earth?* BAMS — 40-dBZ/30-dBZ height & 37/85-GHz
  PCT as intensity proxies. https://www.researchgate.net/publication/237387508
- **MCS / shear / stratiform:** Houze (2004), *Mesoscale convective systems*, Rev. Geophys. —
  shear & stratiform-region development. https://atmos.uw.edu/MG/PDFs/ROG04_houze_MCS.pdf
- **Shear & MCS intensity (direct precedent):** Baidu et al. (2022), *Effects of vertical wind
  shear on intensities of MCSs over West/Central Africa*, Atmos. Sci. Lett.
  https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/asl.1094
- **Shear & organization, recent obs+ERA5:** Klein/“soil-moisture gradients strengthen MCSs by
  increasing wind shear” (2025), Nat. Geosci. https://www.nature.com/articles/s41561-025-01666-8
- **Entrainment/dilution → control moisture:** Hannah (2017), *Entrainment vs Dilution in
  Tropical Deep Convection*, JAS https://journals.ametsoc.org/view/journals/atsc/74/11/jas-d-16-0169.1.xml ;
  Zhang (2009), JGR https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JD010976
- **ERA5 convective environments / shear definitions & binning:** Taszarek et al. (2021),
  *Global climatology of convective environments from ERA5*, npj Clim. Atmos. Sci.
  https://www.nature.com/articles/s41612-021-00190-x
- **Shear-relative compositing technique:** Chen, Knaff & Marks (2006), TC rainfall shear
  asymmetry, MWR https://journals.ametsoc.org/view/journals/mwre/134/11/mwr3245.1.xml
- **Idealized basis:** RKW / Weisman–Rotunno deep-layer-shear line theory; LeMone et al.
  (1998, JAS) shear & convective-band organization. (canonical; cite from texts)
- **ML attribution of extremes (A7) & XAI/causal-ML framework (§4B):**
  Lundberg & Lee (2017) SHAP (method); Apley & Zhu (2020) **ALE plots** (correlated-feature
  effects) https://christophm.github.io/interpretable-ml-book/ale.html ; Nori et al. **EBM /
  InterpretML** (glass-box GAM); Chernozhukov et al. (2018) **Double ML**
  https://arxiv.org/pdf/1701.08687 ; Athey & Wager **causal forests** (EconML
  https://www.pywhy.org/EconML/ ); Arif & Massam (2025) *Estimating causal effects with ML: a
  guide for ecologists*, Methods Ecol. Evol. (observational-attribution caveats)
  https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.70191 ;
  Frontiers Env. Sci. (2022) RF+SHAP for MCS QPE
  https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2022.1057081/full ;
  *Modulation of MCSs over tropical oceans* (RF+SHAP; moisture/CWV/instability lead)
  https://arxiv.org/html/2604.21023

---

## 7. Notebook structure (DuckDB-aggregation; one query per panel, ~MB not GB)
1. Setup: connect catalog, `MISSION`, baseline filter, the `features ⨝ era5` view, helpers.
2. A1 CAPE–shear joint climate + correlation.
3. A2 intensity composites (5 responses × land/ocean) + fixed-CAPE lines + tail (p90/p99).
4. A3 organization (P(MCS), area, MCS rain frac).
5. A4 stratiform area.
6. A5 regional panels.
7. A6 regression / quantile-regression / GAM partial dependence + bootstrap CIs.
8. Synthesis table: signed, CI'd shear effect per hypothesis × (land/ocean) × region.

All binning/medians/percentiles done in SQL (`GROUP BY` + `quantile_cont`); regression/GAM
on the small per-feature sample pulled only for the convective subset (or on bin-summaries).

---

## 8. Decisions to confirm before building
- **Mission:** TRMM-only (faithful, consistent sensitivity) vs TRMM+GPM (more samples, longer)?
- **Ambient-CAPE definition:** `p90_cape_2p50deg` (recommended) vs `mean_cape_5deg` vs centroid?
- **Extra deps OK?** `statsmodels` (quantile regression), `pyGAM` (partial dependence), and
  `xgboost`/`lightgbm` + `shap` (A7 extreme-value attribution) — install into `pf`, or keep to
  binning + composites + bootstrap only?
- **Extreme thresholds:** percentile-based (p99, per land/ocean) vs absolute Zipser cuts
  (40-dBZ ≥ 14 km; PCT37 ≤ 150 K) — or report both?
- **Scope now:** start with A1–A4 (composites, the hypothesis core) and add A5/A6/A7
  (regional + regression/GAM + ML extreme attribution) after, or build the full set at once?
