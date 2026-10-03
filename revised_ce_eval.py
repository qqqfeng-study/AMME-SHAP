import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
import AMME_selection as C
Config = Tuple[str, float, float]
AdaptiveWeightConfig = Tuple[str, int, float]


def run_historical_error_weighted_pipeline(
    pred_matrix: np.ndarray,
    observations: np.ndarray,
    target_years: Iterable[int],
    window_n: int,
) -> pd.DataFrame:
    frame, _ = C.run_pipeline(
        pred_matrix,
        observations,
        sorted({int(year) for year in target_years}),
        window_n=window_n,
    )
    return frame.rename(
        columns={'idx_ce': 'idx_revised_ce', 'pred_ce': 'pred_revised_ce'}
    )


def baseline_normalized_losses(
    mae: np.ndarray,
    rmse: np.ndarray,
    baseline_mae: float,
    baseline_rmse: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        raise ValueError("baseline_mae must be finite and greater than zero")
    if not np.isfinite(baseline_rmse) or baseline_rmse <= 0:
        raise ValueError("baseline_rmse must be finite and greater than zero")
    return (
        np.asarray(mae, dtype=float) / baseline_mae,
        np.asarray(rmse, dtype=float) / baseline_rmse,
    )


def revised_ce_score(
    correlation: np.ndarray,
    normalized_mae: np.ndarray,
    normalized_rmse: np.ndarray,
    lambda_corr: float = 0.5,
    alpha_mae: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not 0.0 <= lambda_corr <= 1.0:
        raise ValueError("lambda_corr must be in [0, 1]")
    if not 0.0 <= alpha_mae <= 1.0:
        raise ValueError("alpha_mae must be in [0, 1]")
    correlation = np.asarray(correlation, dtype=float)
    normalized_mae = np.asarray(normalized_mae, dtype=float)
    normalized_rmse = np.asarray(normalized_rmse, dtype=float)
    if not correlation.shape == normalized_mae.shape == normalized_rmse.shape:
        raise ValueError("all CE component arrays must have the same shape")
    bounded_correlation = np.clip(correlation, -1.0, 1.0)
    corr_component = lambda_corr * (1.0 - bounded_correlation) / 2.0
    mae_component = (1.0 - lambda_corr) * alpha_mae * normalized_mae
    rmse_component = (
        (1.0 - lambda_corr) * (1.0 - alpha_mae) * normalized_rmse
    )
    score = corr_component + mae_component + rmse_component
    score = np.where(np.isfinite(correlation), score, np.inf)
    return score, {
        "correlation": corr_component,
        "mae": mae_component,
        "rmse": rmse_component,
    }


def inverse_error_reliability_weights(
    squared_error_history: np.ndarray,
    lookback: int,
    epsilon: float = 1e-8,
) -> np.ndarray:
    history = np.asarray(squared_error_history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 3:
        raise ValueError("squared_error_history must have shape (n_years, 3)")
    if history.shape[0] == 0:
        raise ValueError("at least one historical error row is required")
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and greater than zero")
    if np.any(~np.isfinite(history)) or np.any(history < 0):
        raise ValueError("historical squared errors must be finite and nonnegative")
    recent = history[-min(lookback, history.shape[0]) :]
    rolling_rmse = np.sqrt(np.mean(recent, axis=0))
    inverse = 1.0 / (rolling_rmse + epsilon)
    return inverse / inverse.sum()


def _normalized_three_weights(weights: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (3,):
        raise ValueError("{} must contain exactly three weights".format(label))
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("{} must be finite and nonnegative".format(label))
    total = float(values.sum())
    if total <= 0:
        raise ValueError("{} must have a positive sum".format(label))
    return values / total


def shrink_reliability_weights(
    reliability_weights: np.ndarray,
    base_weights: np.ndarray = None,
    rho: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    reliability = _normalized_three_weights(reliability_weights, 'reliability_weights')
    base = _normalized_three_weights(
        np.array([0.5, 0.25, 0.25]) if base_weights is None else base_weights,
        "base_weights",
    )
    combined = (1.0 - rho) * base + rho * reliability
    return combined / combined.sum()


def weighted_ce_score(
    correlation: np.ndarray,
    normalized_mae: np.ndarray,
    normalized_rmse: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    component_weights = _normalized_three_weights(weights, "weights")
    correlation = np.asarray(correlation, dtype=float)
    normalized_mae = np.asarray(normalized_mae, dtype=float)
    normalized_rmse = np.asarray(normalized_rmse, dtype=float)
    if not correlation.shape == normalized_mae.shape == normalized_rmse.shape:
        raise ValueError("all CE component arrays must have the same shape")
    bounded_correlation = np.clip(correlation, -1.0, 1.0)
    corr_component = component_weights[0] * (1.0 - bounded_correlation) / 2.0
    mae_component = component_weights[1] * normalized_mae
    rmse_component = component_weights[2] * normalized_rmse
    score = corr_component + mae_component + rmse_component
    score = np.where(np.isfinite(correlation), score, np.inf)
    return score, {
        "correlation": corr_component,
        "mae": mae_component,
        "rmse": rmse_component,
    }


def _signed_window_metrics(
    pred_window: np.ndarray, obs_window: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_window = np.asarray(obs_window, dtype=float)
    pred_window = np.asarray(pred_window, dtype=float)
    obs_centered = obs_window - obs_window.mean()
    pred_centered = pred_window - pred_window.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(obs_centered**2)) * np.sqrt(
        np.sum(pred_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (
            np.sum(pred_centered * obs_centered[None, :], axis=1) / denominator
        )
    correlation = np.clip(correlation, -1.0, 1.0)
    constant_predictions = np.all(np.isclose(pred_window, pred_window[:, 0:1]), axis=1)
    if np.allclose(obs_window, obs_window[0]):
        correlation[:] = np.nan
    correlation[constant_predictions] = np.nan
    mae = np.mean(np.abs(pred_window - obs_window[None, :]), axis=1)
    rmse = np.sqrt(np.mean((pred_window - obs_window[None, :]) ** 2, axis=1))
    return correlation, mae, rmse


def _candidate_profile(
    pred_matrix: np.ndarray,
    observations: np.ndarray,
    end_year: int,
    window_n: int,
) -> Dict[str, object]:
    if window_n < 2:
        raise ValueError("window_n must be at least 2")
    n_models = pred_matrix.shape[0]
    start_year = end_year - window_n
    allowed_length = window_n
    threshold = max(1, int(math.ceil(0.1 * n_models)))
    temporary_start = start_year
    temporary_length = allowed_length
    broke = False
    prediction_at_gate = None
    truth_at_gate = None
    while temporary_length > 1 and temporary_start < end_year:
        prediction_at_gate = pred_matrix[:, C.y2i(temporary_start)]
        truth_at_gate = observations[C.y2i(temporary_start)]
        difference = prediction_at_gate - truth_at_gate
        close = np.isclose(prediction_at_gate, truth_at_gate, atol=0.5)
        if truth_at_gate > 28:
            gate_mask = close & (difference >= 0)
        elif truth_at_gate < 28:
            gate_mask = close & (difference <= 0)
        else:
            gate_mask = close
        if int(gate_mask.sum()) >= threshold:
            start_year = temporary_start
            allowed_length = temporary_length
            broke = True
            break
        temporary_start += 1
        temporary_length -= 1
    if broke:
        gate_year = end_year - window_n
        errors = np.abs(
            pred_matrix[:, C.y2i(gate_year)] - observations[C.y2i(gate_year)]
        )
    else:
        errors = np.abs(prediction_at_gate - truth_at_gate)
    candidate_indices = np.where(errors <= np.percentile(errors, 10))[0]
    window_columns = slice(C.y2i(start_year), C.y2i(end_year))
    pred_window = pred_matrix[candidate_indices][:, window_columns]
    pred_window = pred_window[:, :allowed_length]
    obs_window = observations[window_columns][:allowed_length]
    correlation, mae, rmse = _signed_window_metrics(pred_window, obs_window)
    climatology = np.full_like(obs_window, obs_window.mean(), dtype=float)
    baseline_mae = float(np.mean(np.abs(climatology - obs_window)))
    baseline_rmse = float(np.sqrt(np.mean((climatology - obs_window) ** 2)))
    normalized_mae, normalized_rmse = baseline_normalized_losses(
        mae, rmse, baseline_mae, baseline_rmse
    )
    return {
        "candidate_indices": candidate_indices,
        "start_year": start_year,
        "allowed_length": allowed_length,
        "correlation": correlation,
        "mae": mae,
        "rmse": rmse,
        "normalized_mae": normalized_mae,
        "normalized_rmse": normalized_rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
    }


def run_revised_pipeline_multi(
    pred_matrix: np.ndarray,
    observations: np.ndarray,
    target_years: Iterable[int],
    window_n: int,
    configs: Sequence[Config],
) -> Dict[str, pd.DataFrame]:
    if not configs:
        raise ValueError("at least one CE configuration is required")
    rows_by_config: Dict[str, List[Dict[str, object]]] = {
        name: [] for name, _, _ in configs
    }
    for year in target_years:
        if year - window_n < C.YEAR_START:
            continue
        profile = _candidate_profile(pred_matrix, observations, year, window_n)
        candidate_indices = profile["candidate_indices"]
        correlation = profile["correlation"]
        mae = profile["mae"]
        rmse = profile["rmse"]
        i_cc = int(np.argmax(np.where(np.isnan(correlation), -np.inf, correlation)))
        i_mae = int(np.argmin(mae))
        i_rmse = int(np.argmin(rmse))
        forecast_column = C.y2i(year)
        truth = float(observations[forecast_column])
        for name, lambda_corr, alpha_mae in configs:
            score, parts = revised_ce_score(
                correlation,
                profile["normalized_mae"],
                profile["normalized_rmse"],
                lambda_corr=lambda_corr,
                alpha_mae=alpha_mae,
            )
            i_ce = int(np.argmin(score))
            selected_index = int(candidate_indices[i_ce])
            rows_by_config[name].append(
                {
                    "Year": int(year),
                    "window_n": int(window_n),
                    "start_year": int(profile["start_year"]),
                    "allowed_length": int(profile["allowed_length"]),
                    "n_candidates": int(len(candidate_indices)),
                    "lambda_corr": float(lambda_corr),
                    "alpha_mae": float(alpha_mae),
                    "baseline_mae": float(profile["baseline_mae"]),
                    "baseline_rmse": float(profile["baseline_rmse"]),
                    "idx_cc": int(candidate_indices[i_cc]),
                    "idx_mae": int(candidate_indices[i_mae]),
                    "idx_rmse": int(candidate_indices[i_rmse]),
                    "idx_revised_ce": selected_index,
                    "pred_cc": float(
                        pred_matrix[int(candidate_indices[i_cc]), forecast_column]
                    ),
                    "pred_mae": float(
                        pred_matrix[int(candidate_indices[i_mae]), forecast_column]
                    ),
                    "pred_rmse": float(
                        pred_matrix[int(candidate_indices[i_rmse]), forecast_column]
                    ),
                    "pred_revised_ce": float(
                        pred_matrix[selected_index, forecast_column]
                    ),
                    "selected_window_r": float(correlation[i_ce]),
                    "selected_window_mae": float(mae[i_ce]),
                    "selected_window_rmse": float(rmse[i_ce]),
                    "selected_ce": float(score[i_ce]),
                    "selected_corr_contribution": float(parts['correlation'][i_ce]),
                    "selected_mae_contribution": float(parts["mae"][i_ce]),
                    "selected_rmse_contribution": float(parts["rmse"][i_ce]),
                    "true": truth,
                }
            )
    return {
        name: pd.DataFrame(rows) for name, rows in rows_by_config.items()
    }


def run_shrunk_pipeline_multi(
    pred_matrix: np.ndarray,
    observations: np.ndarray,
    target_years: Iterable[int],
    window_n: int,
    configs: Sequence[AdaptiveWeightConfig],
    hist_from: int = 1985,
    base_weights: np.ndarray = None,
) -> Dict[str, pd.DataFrame]:
    target_years = sorted(set(int(year) for year in target_years))
    if not target_years:
        raise ValueError("target_years must not be empty")
    if not configs:
        raise ValueError("at least one adaptive-weight configuration is required")
    names = [name for name, _, _ in configs]
    if len(names) != len(set(names)):
        raise ValueError("adaptive-weight configuration names must be unique")
    base = _normalized_three_weights(
        np.array([0.5, 0.25, 0.25]) if base_weights is None else base_weights,
        "base_weights",
    )
    rows_by_config: Dict[str, List[Dict[str, object]]] = {
        name: [] for name in names
    }
    squared_error_history: List[np.ndarray] = []
    target_set = set(target_years)
    for year in range(hist_from, max(target_years) + 1):
        if year - window_n < C.YEAR_START:
            continue
        profile = _candidate_profile(pred_matrix, observations, year, window_n)
        candidate_indices = profile["candidate_indices"]
        correlation = profile["correlation"]
        mae = profile["mae"]
        rmse = profile["rmse"]
        i_cc = int(np.argmax(np.where(np.isnan(correlation), -np.inf, correlation)))
        i_mae = int(np.argmin(mae))
        i_rmse = int(np.argmin(rmse))
        forecast_column = C.y2i(year)
        truth = float(observations[forecast_column])
        single_indices = np.array(
            [
                int(candidate_indices[i_cc]),
                int(candidate_indices[i_mae]),
                int(candidate_indices[i_rmse]),
            ],
            dtype=int,
        )
        single_predictions = pred_matrix[single_indices, forecast_column].astype(float)
        for name, lookback, rho in configs:
            if squared_error_history:
                reliability = inverse_error_reliability_weights(
                    np.vstack(squared_error_history), lookback=lookback
                )
            else:
                reliability = base
            weights = shrink_reliability_weights(
                reliability_weights=reliability,
                base_weights=base,
                rho=rho,
            )
            score, parts = weighted_ce_score(
                correlation,
                profile["normalized_mae"],
                profile["normalized_rmse"],
                weights=weights,
            )
            i_ce = int(np.argmin(score))
            selected_index = int(candidate_indices[i_ce])
            if year in target_set:
                rows_by_config[name].append(
                    {
                        "Year": int(year),
                        "window_n": int(window_n),
                        "lookback": int(lookback),
                        "rho": float(rho),
                        "w_cc": float(weights[0]),
                        "w_mae": float(weights[1]),
                        "w_rmse": float(weights[2]),
                        "idx_cc": int(single_indices[0]),
                        "idx_mae": int(single_indices[1]),
                        "idx_rmse": int(single_indices[2]),
                        "idx_revised_ce": selected_index,
                        "pred_cc": float(single_predictions[0]),
                        "pred_mae": float(single_predictions[1]),
                        "pred_rmse": float(single_predictions[2]),
                        "pred_revised_ce": float(
                            pred_matrix[selected_index, forecast_column]
                        ),
                        "selected_ce": float(score[i_ce]),
                        "selected_corr_contribution": float(parts['correlation'][i_ce]),
                        "selected_mae_contribution": float(parts["mae"][i_ce]),
                        "selected_rmse_contribution": float(parts["rmse"][i_ce]),
                        "true": truth,
                    }
                )
        squared_error_history.append((single_predictions - truth) ** 2)
    return {
        name: pd.DataFrame(rows) for name, rows in rows_by_config.items()
    }


def skill_metrics(df: pd.DataFrame, prediction_column: str) -> Dict[str, float]:
    clean = df.dropna(subset=[prediction_column])
    prediction = clean[prediction_column].to_numpy(dtype=float)
    truth = clean["true"].to_numpy(dtype=float)
    return {
        "CC": float(np.corrcoef(prediction, truth)[0, 1]),
        "MAE": float(np.mean(np.abs(prediction - truth))),
        "RMSE": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "n": int(len(clean)),
    }


def adaptive_window_forecast(
    results_by_window: Mapping[int, pd.DataFrame],
    prediction_column: str = "pred_revised_ce",
    evaluation_years: Iterable[int] = range(2016, 2026),
    lookback: int = 10,
) -> pd.DataFrame:
    windows = sorted(results_by_window)
    indexed = {n: results_by_window[n].set_index("Year") for n in windows}
    rows = []
    for year in evaluation_years:
        rmse_by_window = {}
        for window_n in windows:
            available = indexed[window_n]
            historical_years = [
                y
                for y in range(year - lookback, year)
                if y in available.index
            ]
            if len(historical_years) != lookback:
                continue
            subset = available.loc[historical_years]
            errors = (
                subset[prediction_column].to_numpy() - subset['true'].to_numpy()
            )
            rmse_by_window[window_n] = float(np.sqrt(np.mean(errors ** 2)))
        if not rmse_by_window:
            raise ValueError(
                'no window has a complete historical lookback for year ' + str(year)
            )
        best_window = min(rmse_by_window, key=rmse_by_window.get)
        selected = indexed[best_window].loc[year]
        rows.append(
            {
                "Year": int(year),
                "N_chosen": int(best_window),
                prediction_column: float(selected[prediction_column]),
                "true": float(selected["true"]),
                "hist_rmse_by_N": "; ".join(
                    "N{}:{:.3f}".format(n, value)
                    for n, value in sorted(rmse_by_window.items())
                ),
            }
        )
    return pd.DataFrame(rows)


def _rounded(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: (int(value) if key == "n" else round(float(value), 4))
        for key, value in metrics.items()
    }


def _default_configs(
    lambda_corr: float, alpha_mae: float, sensitivity: bool
) -> List[Config]:
    if not sensitivity:
        return [("revised_ce", lambda_corr, alpha_mae)]
    lambda_values = sorted({0.25, 0.5, 0.75, float(lambda_corr)})
    return [
        ("lambda_{:.2f}".format(value), value, alpha_mae)
        for value in lambda_values
    ]


def _adaptive_weight_configs(
    lookback: int, rho: float, sensitivity: bool
) -> List[AdaptiveWeightConfig]:
    requested = (int(lookback), float(rho))
    if not sensitivity:
        return [("shrunk_ce", requested[0], requested[1])]
    settings = {
        requested,
        (6, 0.5),
        (10, 0.0),
        (10, 0.25),
        (10, 0.5),
        (10, 0.75),
        (10, 1.0),
        (15, 0.5),
    }
    return [
        (
            "L{:02d}_rho{:.2f}".format(setting_lookback, setting_rho),
            setting_lookback,
            setting_rho,
        )
        for setting_lookback, setting_rho in sorted(settings)
    ]


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AMME composite selection scores.")
    parser.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="fixed")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lambda-corr", type=float, default=0.5)
    parser.add_argument("--alpha-mae", type=float, default=0.5)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument(
        "--weight-mode",
        choices=["fixed", "shrunk"],
        default="fixed",
        help="Fixed scientific weights or rolling inverse-error weights with shrinkage",
    )
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--weight-sensitivity", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path for the default revised-CE result",
    )
    args = parser.parse_args(argv)
    if args.lookback < 1:
        parser.error("--lookback must be at least 1")
    if not 0.0 <= args.rho <= 1.0:
        parser.error("--rho must be in [0, 1]")
    if args.weight_mode == "shrunk" and args.sensitivity:
        parser.error("--sensitivity applies only to --weight-mode fixed")
    pred_matrix = C.load_prediction_matrix()
    observations = C.load_obs()
    report: Dict[str, object] = {
        "formula": {
            "weight_mode": args.weight_mode,
            "lambda_corr": args.lambda_corr,
            "alpha_mae": args.alpha_mae,
            "lookback": args.lookback,
            "rho": args.rho,
            "base_weights": [0.5, 0.25, 0.25],
            "error_normalization": "historical_climatology_baseline",
            "correlation": "signed_pearson_r",
        }
    }
    default_frame = None
    if args.mode in {"fixed", "both"}:
        if args.weight_mode == "fixed":
            configs = _default_configs(
                args.lambda_corr, args.alpha_mae, args.sensitivity
            )
            fixed = run_revised_pipeline_multi(
                pred_matrix,
                observations,
                range(2016, 2026),
                window_n=args.window,
                configs=configs,
            )
            default_name = (
                "lambda_{:.2f}".format(args.lambda_corr)
                if args.sensitivity
                else "revised_ce"
            )
        else:
            adaptive_configs = _adaptive_weight_configs(
                args.lookback, args.rho, args.weight_sensitivity
            )
            fixed = run_shrunk_pipeline_multi(
                pred_matrix,
                observations,
                range(2016, 2026),
                window_n=args.window,
                configs=adaptive_configs,
                hist_from=1985,
            )
            default_name = (
                "L{:02d}_rho{:.2f}".format(args.lookback, args.rho)
                if args.weight_sensitivity
                else "shrunk_ce"
            )
        fixed_metrics = {}
        for name, frame in fixed.items():
            fixed_metrics[name] = _rounded(skill_metrics(frame, 'pred_revised_ce'))
        default_frame = fixed[default_name]
        fixed_metrics["single_cc"] = _rounded(skill_metrics(default_frame, 'pred_cc'))
        fixed_metrics["single_mae"] = _rounded(skill_metrics(default_frame, 'pred_mae'))
        fixed_metrics["single_rmse"] = _rounded(
            skill_metrics(default_frame, "pred_rmse")
        )
        old_frame, _ = C.run_pipeline(
            pred_matrix,
            observations,
            list(range(2016, 2026)),
            window_n=args.window,
        )
        fixed_metrics["old_ce"] = _rounded(C.skill_metrics(old_frame, 'pred_ce'))
        report["fixed_window_2016_2025"] = fixed_metrics
        if args.weight_mode == "shrunk":
            report["weight_summary_2016_2025"] = {
                name: {
                    "mean_w_cc": round(float(frame["w_cc"].mean()), 4),
                    "mean_w_mae": round(float(frame["w_mae"].mean()), 4),
                    "mean_w_rmse": round(float(frame["w_rmse"].mean()), 4),
                    "min_w_cc": round(float(frame["w_cc"].min()), 4),
                    "max_w_cc": round(float(frame["w_cc"].max()), 4),
                }
                for name, frame in fixed.items()
            }
        report["fixed_default_same_as"] = {
            "old_ce_years": int(
                np.sum(
                    np.isclose(
                        default_frame["pred_revised_ce"].to_numpy(),
                        old_frame["pred_ce"].to_numpy(),
                    )
                )
            ),
            "rmse_selector_years": int(
                np.sum(
                    default_frame["idx_revised_ce"].to_numpy()
                    == default_frame["idx_rmse"].to_numpy()
                )
            ),
            "mae_selector_years": int(
                np.sum(
                    default_frame["idx_revised_ce"].to_numpy()
                    == default_frame["idx_mae"].to_numpy()
                )
            ),
        }
    if args.mode in {"adaptive", "both"}:
        adaptive_source: Dict[int, pd.DataFrame] = {}
        for window_n in [4, 5, 6, 7, 8, 9]:
            print("Computing revised CE for N={} ...".format(window_n), flush=True)
            if args.weight_mode == "fixed":
                frames = run_revised_pipeline_multi(
                    pred_matrix,
                    observations,
                    range(1986, 2026),
                    window_n=window_n,
                    configs=[("revised_ce", args.lambda_corr, args.alpha_mae)],
                )
            else:
                frames = run_shrunk_pipeline_multi(
                    pred_matrix,
                    observations,
                    range(1986, 2026),
                    window_n=window_n,
                    configs=[("revised_ce", args.lookback, args.rho)],
                    hist_from=1985,
                )
            adaptive_source[window_n] = frames["revised_ce"]
        adaptive = adaptive_window_forecast(adaptive_source)
        default_frame = adaptive
        report["adaptive_window_2016_2025"] = _rounded(
            skill_metrics(adaptive, "pred_revised_ce")
        )
        report["adaptive_selected_windows"] = {
            str(int(year)): int(window)
            for year, window in zip(adaptive["Year"], adaptive["N_chosen"])
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        if default_frame is None:
            raise RuntimeError("no result frame was produced")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        default_frame.to_csv(args.output, index=False)
        print("Saved: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
