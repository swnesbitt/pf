# Shear / CAPE hypothesis suite

Tests whether vertical wind shear (controlling for CAPE) deepens convection, organizes it
into larger MCSs, and grows stratiform area — plus which variables explain the *extremes* of
echo-top height and microwave Tb. Built on the PF superdatabase ⨝ ERA5. Design & literature:
`../shear_cape_hypotheses_PLAN.md`.

## Run order & kernels
| notebook | content | kernel |
|---|---|---|
| `00_data_setup` | analysis table sanity, CAPE–shear anticorrelation | **pf** |
| `01_cape_shear_climate` | A1 joint climate / confounder map | pf |
| `02_intensity_H1` | A2 echo-top & PCT vs shear at fixed CAPE (+ tail) | pf |
| `03_organization_H2` | A3 P(MCS), area, MCS rain fraction | pf |
| `04_stratiform_H3` | A4 stratiform area | pf |
| `05_regional` | A5 per-basin shear sensitivity | pf |
| `06_regression` | A6 quantile regression + GAM partial dependence | **pf_ml** |
| `07_ml_xai` | §4B EBM + GBM + SHAP + ALE | pf_ml |
| `08_causal_ml` | DML + causal forest (isolate shear) | pf_ml |
| `09_extremes_attribution` | A7 extreme-value drivers (SHAP / quantile / composite-env) | pf_ml |

`_shc.py` is the shared engine (connection, `fe` view, composite/fixed-CAPE/sample helpers,
regions, bin edges, extreme thresholds). Notebooks do `from _shc import *`.

## Compute
- 00–05: DuckDB aggregation over the materialized table — fast, low memory, `pf` kernel.
- 06–09: ML on a `load_sample()` subset; for heavy fits use the **`l40s`** GPU partition
  (XGBoost `device='cuda'`, LightGBM `device='gpu'`, CatBoost `task_type='GPU'`).
