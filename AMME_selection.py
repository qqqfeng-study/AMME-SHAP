import os
import math
import numpy as np
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "selection_results")

DEVELOPMENT_FILE = os.path.join(OUTPUT_DIR, "development_N_select.csv")
TEST_PREDICTION_FILE = os.path.join(OUTPUT_DIR, "test_n8_predict.csv")
FIXED_SELECTION_FILE = os.path.join(OUTPUT_DIR, "AMME_MAE_N8_1988_2025.csv")
FINAL_CRITERION = "MAE"
PRED_CSV = os.path.join(BASE_DIR, "y_pred_N500000.csv")
OBS_CSV = os.path.join(BASE_DIR, "precursor_factors_1979_2025.csv")
YEAR_START = 1979
YEAR_END = 2025
N_YEARS = YEAR_END - YEAR_START + 1
WINDOW_N = 8
DEVELOPMENT_START = 1988
DEVELOPMENT_END = 2015
TEST_START = 2016
TEST_END = 2025


def y2i(year):
    return int(year - YEAR_START)


def _minmax(values):
    values = np.asarray(values, dtype=np.float64)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    delta = vmax - vmin
    if (
        not np.isfinite(delta)
        or delta == 0
    ):
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / delta


def load_prediction_matrix(
    force_reload=False
):
    os.makedirs(CACHE_DIR, exist_ok=True)
    pred_name = os.path.splitext(os.path.basename(PRED_CSV))[0]
    cache_file = os.path.join(CACHE_DIR, f'{pred_name}_f64.npy')
    if (
        os.path.exists(cache_file)
        and not force_reload
    ):
        mat = np.load(cache_file)
        if mat.shape[1] != N_YEARS:
            raise ValueError(f'缓存预测矩阵列数错误：{mat.shape[1]}，预期 {N_YEARS}')
        return mat
    df = pd.read_csv(PRED_CSV)
    unnamed_cols = [
        col
        for col in df.columns
        if str(col).startswith('Unnamed')
    ]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    expected_year_cols = [
        str(year)
        for year in range(YEAR_START, YEAR_END + 1)
    ]
    current_cols = [
        str(col)
        for col in df.columns
    ]
    if all((year in current_cols for year in expected_year_cols)):
        rename_map = {
            col: str(col)
            for col in df.columns
        }
        df = df.rename(columns=rename_map)
        df = df[expected_year_cols]
    if df.shape[1] != N_YEARS:
        raise ValueError(
            "\n预测矩阵列数异常。\n"
            f"实际 shape = {df.shape}\n"
            f"要求年份数 = {N_YEARS}\n"
            f"年份范围 = {YEAR_START}–{YEAR_END}"
        )
    mat = df.to_numpy(dtype=np.float64)
    np.save(cache_file, mat)
    print(f'Prediction matrix: {mat.shape}')
    return mat


def load_obs():
    data = pd.read_csv(OBS_CSV, index_col=0)
    data.index = pd.to_numeric(data.index)
    data = data.drop(columns=['mse_850', 'vm_eof2', 'tp_feb'], errors='ignore')
    obs = (
        data.loc[YEAR_START:YEAR_END, 'scssm_onset'].to_numpy(dtype=np.float64)
    )
    if len(obs) != N_YEARS:
        raise ValueError(f'观测长度错误：{len(obs)}，预期 {N_YEARS}')
    return obs


def _window_metrics(
    pred_win,
    obs_win
):
    x = np.asarray(obs_win, dtype=np.float64)
    y = np.asarray(pred_win, dtype=np.float64)
    x_centered = (
        x - x.mean()
    )
    y_centered = (
        y - y.mean(axis=1, keepdims=True)
    )
    sx = np.sqrt(np.sum(x_centered ** 2))
    sy = np.sqrt(np.sum(y_centered ** 2, axis=1))
    denominator = (
        sx * sy
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        cc = (
            np.sum(y_centered * x_centered[None, :], axis=1) / denominator
        )
    cc = np.clip(cc, -1.0, 1.0)
    const_obs = np.allclose(x, x[0])
    const_pred = np.all(np.isclose(y, y[:, 0:1]), axis=1)
    if const_obs:
        cc[:] = np.nan
    cc[
        const_pred
    ] = np.nan
    mae = np.mean(np.abs(y - x[None, :]), axis=1)
    rmse = np.sqrt(np.mean((y - x[None, :]) ** 2, axis=1))
    return (cc, mae, rmse)


def select_one_year(
    pred_mat,
    obs,
    end_year,
    window_n=WINDOW_N,
    bias_history=None
):
    if bias_history is None:
        bias_history = []
    n_models = (
        pred_mat.shape[0]
    )
    ll = int(window_n)
    original_start_year = (
        end_year - ll
    )
    start_year = (
        original_start_year
    )
    allowed_lens = ll
    coverage_threshold = max(1, int(math.ceil(0.1 * n_models)))
    temp_start_year = (
        start_year
    )
    temp_allowed_lens = (
        allowed_lens
    )
    coverage_passed = False
    pred_g = None
    true_g = None
    while (
        temp_allowed_lens > 1
        and temp_start_year < end_year
    ):
        pred_g = pred_mat[:, y2i(temp_start_year)]
        true_g = obs[y2i(temp_start_year)]
        diff_g = (
            pred_g - true_g
        )
        close = np.isclose(pred_g, true_g, atol=0.5)
        if true_g > 28:
            coverage_mask = (
                close & (diff_g >= 0)
            )
        elif true_g < 28:
            coverage_mask = (
                close & (diff_g <= 0)
            )
        else:
            coverage_mask = close
        n_covered = int(coverage_mask.sum())
        if (
            n_covered >= coverage_threshold
        ):
            start_year = (
                temp_start_year
            )
            allowed_lens = (
                temp_allowed_lens
            )
            coverage_passed = True
            break
        temp_start_year += 1
        temp_allowed_lens -= 1
    if coverage_passed:
        filter_year = (
            original_start_year
        )
        errors = np.abs(pred_mat[:, y2i(filter_year)] - obs[y2i(filter_year)])
    else:
        errors = np.abs(pred_g - true_g)
    error_threshold = np.percentile(errors, 10)
    close_models = np.where(errors <= error_threshold)[0]
    if len(close_models) == 0:
        raise RuntimeError(f'{end_year}: 没有候选模型通过 10% 筛选。')
    start_idx = y2i(start_year)
    end_idx = y2i(end_year)
    pred_win = pred_mat[close_models, start_idx:end_idx]
    pred_win = pred_win[:, :allowed_lens]
    obs_win = obs[start_idx:end_idx]
    obs_win = obs_win[:allowed_lens]
    cc, mae, rmse = (
        _window_metrics(pred_win, obs_win)
    )
    cc_for_selection = np.where(np.isnan(cc), -np.inf, cc)
    local_idx_cc = int(np.argmax(cc_for_selection))
    idx_cc = int(close_models[local_idx_cc])
    local_idx_mae = int(np.argmin(mae))
    idx_mae = int(close_models[local_idx_mae])
    local_idx_rmse = int(np.argmin(rmse))
    idx_rmse = int(close_models[local_idx_rmse])
    previous = [
        item
        for item in bias_history
        if item["Year"] < end_year
    ]
    previous = sorted(previous, key=lambda item: item['Year'], reverse=True)[:ll]
    if len(previous) == 0:
        w_cc = 1.0 / 3.0
        w_mae = 1.0 / 3.0
        w_rmse = 1.0 / 3.0
        cc_for_ce = np.clip(np.where(np.isnan(cc), -1.0, cc), -1.0, 1.0)
        cc_loss = (1.0 - cc_for_ce) / 2.0
        ce_score = (
            w_cc * cc_loss + w_mae * _minmax(mae) + w_rmse * _minmax(rmse)
        )
        local_idx_ce = int(np.argmin(ce_score))
        idx_ce = int(close_models[local_idx_ce])
    else:
        hist_rmse_cc = math.sqrt(np.mean([item['SE_CC'] for item in previous]))
        hist_rmse_mae = math.sqrt(np.mean([item['SE_MAE'] for item in previous]))
        hist_rmse_rmse = math.sqrt(np.mean([item['SE_RMSE'] for item in previous]))
        eps = 1e-8
        inverse_errors = np.array(
            [
                1.0 / (hist_rmse_cc + eps),
                1.0 / (hist_rmse_mae + eps),
                1.0 / (hist_rmse_rmse + eps),
            ],
            dtype=np.float64
        )
        weights = (
            inverse_errors / inverse_errors.sum()
        )
        (
            w_cc,
            w_mae,
            w_rmse
        ) = weights
        cc_for_ce = np.where(np.isnan(cc), -1.0, cc)
        cc_for_ce = np.clip(cc_for_ce, -1.0, 1.0)
        cc_loss = (1.0 - cc_for_ce) / 2.0
        mae_norm = _minmax(mae)
        rmse_norm = _minmax(rmse)
        ce_score = (
            w_cc * cc_loss + w_mae * mae_norm + w_rmse * rmse_norm
        )
        local_idx_ce = int(np.argmin(ce_score))
        idx_ce = int(close_models[local_idx_ce])
    selected_ce_score = float(ce_score[local_idx_ce])
    col = y2i(end_year)
    true_value = float(obs[col])
    pred_cc = float(pred_mat[idx_cc, col])
    pred_mae = float(pred_mat[idx_mae, col])
    pred_rmse = float(pred_mat[idx_rmse, col])
    if idx_ce is None:
        pred_ce = np.nan
    else:
        pred_ce = float(pred_mat[idx_ce, col])
    return {
        "Year":
            int(end_year),
        "true":
            true_value,
        "window_n":
            int(ll),
        "original_start_year":
            int(original_start_year),
        "start_year":
            int(start_year),
        "allowed_lens":
            int(allowed_lens),
        "n_candidates":
            int(len(close_models)),
        "idx_cc":
            idx_cc,
        "idx_mae":
            idx_mae,
        "idx_rmse":
            idx_rmse,
        "idx_ce":
            idx_ce,
        "pred_cc":
            pred_cc,
        "pred_mae":
            pred_mae,
        "pred_rmse":
            pred_rmse,
        "pred_ce":
            pred_ce,
        "selected_ce_score":
            selected_ce_score,
        "w_cc":
            float(w_cc) if np.isfinite(w_cc) else np.nan,
        "w_mae":
            float(w_mae) if np.isfinite(w_mae) else np.nan,
        "w_rmse":
            float(w_rmse) if np.isfinite(w_rmse) else np.nan,
    }


def run_pipeline(
    pred_mat,
    obs,
    target_years,
    window_n=WINDOW_N
):
    target_years = sorted((int(year) for year in target_years))
    if len(target_years) == 0:
        raise ValueError('target_years 不能为空。')
    first_possible_year = (
        YEAR_START + window_n
    )
    last_target_year = max(target_years)
    bias_history = []
    results = []
    for year in range(first_possible_year, last_target_year + 1):
        result = select_one_year(
            pred_mat=pred_mat,
            obs=obs,
            end_year=year,
            window_n=window_n,
            bias_history=bias_history
        )
        bias_history.append({
            "Year":
                year,
            "SE_CC":
                (result['pred_cc'] - result['true']) ** 2,
            "SE_MAE":
                (result['pred_mae'] - result['true']) ** 2,
            "SE_RMSE":
                (result['pred_rmse'] - result['true']) ** 2,
        })
        if year in target_years:
            results.append(result)
    return (pd.DataFrame(results), bias_history)


def skill_metrics(
    df,
    pred_col
):
    valid = df.dropna(subset=['true', pred_col])
    if len(valid) == 0:
        return {'CC': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'n': 0}
    obs = valid['true'].to_numpy(dtype=np.float64)
    pred = valid[pred_col].to_numpy(dtype=np.float64)
    if (
        len(valid) < 2
        or np.allclose(obs, obs[0])
        or np.allclose(pred, pred[0])
    ):
        cc = np.nan
    else:
        cc = float(np.corrcoef(pred, obs)[0, 1])
    mae = float(np.mean(np.abs(pred - obs)))
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    return {'CC': cc, 'MAE': mae, 'RMSE': rmse, 'n': int(len(valid))}


def build_development_table(
    pred_mat,
    obs
):
    years = list(range(DEVELOPMENT_START, DEVELOPMENT_END + 1))
    df, _ = run_pipeline(
        pred_mat=pred_mat,
        obs=obs,
        target_years=years,
        window_n=WINDOW_N
    )
    table = pd.DataFrame({
        "Year":
            df["Year"],
        "OBS":
            df["true"],
        "Pred_CC":
            df["pred_cc"],
        "Pred_MAE":
            df["pred_mae"],
        "Pred_RMSE":
            df["pred_rmse"],
        "Pred_CE":
            df["pred_ce"],
    })
    skill_cc = skill_metrics(df, 'pred_cc')
    skill_mae = skill_metrics(df, 'pred_mae')
    skill_rmse = skill_metrics(df, 'pred_rmse')
    skill_ce = skill_metrics(df, 'pred_ce')
    summary_rows = pd.DataFrame([
        {
            "Year":
                "EVAL_CC",
            "OBS":
                np.nan,
            "Pred_CC":
                skill_cc["CC"],
            "Pred_MAE":
                skill_mae["CC"],
            "Pred_RMSE":
                skill_rmse["CC"],
            "Pred_CE":
                skill_ce["CC"],
        },
        {
            "Year":
                "EVAL_MAE",
            "OBS":
                np.nan,
            "Pred_CC":
                skill_cc["MAE"],
            "Pred_MAE":
                skill_mae["MAE"],
            "Pred_RMSE":
                skill_rmse["MAE"],
            "Pred_CE":
                skill_ce["MAE"],
        },
        {
            "Year":
                "EVAL_RMSE",
            "OBS":
                np.nan,
            "Pred_CC":
                skill_cc["RMSE"],
            "Pred_MAE":
                skill_mae["RMSE"],
            "Pred_RMSE":
                skill_rmse["RMSE"],
            "Pred_CE":
                skill_ce["RMSE"],
        },
    ])
    table = pd.concat([table, summary_rows], ignore_index=True)
    return table


def build_test_table(
    pred_mat,
    obs
):
    years = list(range(TEST_START, TEST_END + 1))
    df, _ = run_pipeline(
        pred_mat=pred_mat,
        obs=obs,
        target_years=years,
        window_n=WINDOW_N
    )
    table = pd.DataFrame({
        "Year":
            df["Year"],
        "OBS":
            df["true"],
        "Pred_MAE":
            df["pred_mae"],
    })
    skill = skill_metrics(df, 'pred_mae')
    summary_rows = pd.DataFrame([
        {'Year': 'EVAL_CC', 'OBS': np.nan, 'Pred_MAE': skill['CC']},
        {'Year': 'EVAL_MAE', 'OBS': np.nan, 'Pred_MAE': skill['MAE']},
        {'Year': 'EVAL_RMSE', 'OBS': np.nan, 'Pred_MAE': skill['RMSE']},
    ])
    table = pd.concat([table, summary_rows], ignore_index=True)
    return table


def main():
    from predict_adaptive_train_N import main as run_selection
    return run_selection()


if __name__ == "__main__":
    main()
