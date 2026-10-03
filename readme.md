# AMME Model Selection and Prediction

This repository provides annual AMME model selection, prediction evaluation, and contribution analysis for South China Sea summer monsoon (SCSSM) onset pentads. The development period (1988-2015) compares window lengths N=4-9 and four selection criteria: CC, MAE, RMSE, and CE. The independent prediction period (2016-2025) uses the fixed MAE + N=8 configuration.

## Scripts

| Script | Purpose |
| --- | --- |
| `AMME_selection.py` | Core module for candidate screening, evaluation metrics, and annual model selection. |
| `predict_adaptive_train_N.py` | Compares window lengths and criteria during development, then generates independent predictions and complete selection records. |
| `revised_ce_eval.py` | Wraps the main CE workflow and provides a revised CE comparison using climatological baseline normalization. |
| `predict_oneyear_set.py` | Displays candidate model trajectories, the selected model, and predictions for a single year. |
| `fig1_scssmo_pdo.py` | Plots onset changes, PDO stages, interannual variability, and model prediction errors, and reports statistical tests. |
| `fig3_contribution_common_shap.py` | Plots annual SHAP contributions using a common background and baseline. |
| `fig4_contribution_common_period_relative.py` | Plots relative SHAP contributions by period and uncertainty boxes across candidate models. |
| `fig5_scssmo.py` | Reads existing onset pentads and ERA5 U850 data to plot filled zonal wind contours with an onset curve. |

## Data and Dependencies

- `y_pred_N500000.csv`: predictions from 500,000 models for 1979-2025.
- `precursor_factors_1979_2025.csv`: observed onset pentads (`scssm_onset`) and predictor variables.
- `models/`: selected models required for SHAP analysis; `fixed_candidate_attributions.csv`: candidate attribution data for the uncertainty boxes in Figure 4. Figure 3 does not include uncertainty boxes. Both Figure 3 and Figure 4 scripts save images only and do not export CSV files.

Figure 1 additionally requires ERA5 SST data and `lg_scssm.csv`; Figure 5 requires ERA5 U850 data. Data paths are configured in the scripts. Core dependencies are NumPy, pandas, and Matplotlib. Contribution and climate analyses also use SHAP, XGBoost, joblib, SciPy, statsmodels, xarray, eofs, and cmaps.

## CE Criterion

MAE and RMSE are min-max normalized across the retained candidate set, and CC is the signed Pearson correlation coefficient:

```text
CE = w_CC * (1 - CC)/2 + w_MAE * MAE_norm + w_RMSE * RMSE_norm
```

The model with the lowest CE is selected. Component weights are proportional to the inverse RMSE of the corresponding selectors over up to N preceding one-year forecasts and normalized to sum to one, with equal initial weights. The main workflow uses this CE definition; the revised CE comparison uses climatological baseline normalization.

## Usage and Outputs

Generate the selection records first, then run the prediction visualization or relevant figure scripts:

```bash
python predict_adaptive_train_N.py
python predict_oneyear_set.py --no-show --skip-overview
python fig1_scssmo_pdo.py --no-show
python fig3_contribution_common_shap.py
python fig4_contribution_common_period_relative.py
python fig5_scssmo.py --no-show
```

Selection results are saved in `selection_results/`:

- `development_N_select.csv`: annual absolute errors for each window length and criterion during development.
- `test_n8_predict.csv`: independent prediction results and selected model identifiers.
- `AMME_MAE_N8_1988_2025.csv`: complete selection records for 1988-2025 using fixed MAE + N=8, used by Figure 1 and SHAP analysis.

`ModelIndex` starts at 0, and `ModelID` starts at 1. The Figure 3 and Figure 4 scripts also read `test_n8_predict.csv` to verify independent prediction records. All figures are saved in `fig/`.
