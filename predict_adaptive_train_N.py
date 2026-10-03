from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
import AMME_selection as C
import revised_ce_eval
YEAR_START = C.YEAR_START
YEAR_END = C.YEAR_END
DEFAULT_PROJECT_ROOT = Path(C.BASE_DIR)
DEFAULT_LIBRARY_NAME = Path(C.PRED_CSV).name
CANDIDATE_N = (4, 5, 6, 7, 8, 9)
CRITERIA = ("CC", "MAE", "RMSE", "CE")
FINAL_CRITERION = C.FINAL_CRITERION
FINAL_N = C.WINDOW_N
PREDICTION_COLUMNS = {
    "CC": "pred_cc",
    "MAE": "pred_mae",
    "RMSE": "pred_rmse",
    "CE": "pred_revised_ce",
}
VALIDATION_YEARS = tuple(range(C.DEVELOPMENT_START, C.DEVELOPMENT_END + 1))
TEST_YEARS = tuple(range(C.TEST_START, C.TEST_END + 1))


@dataclass(frozen=True)
class SelectedConfiguration:
    criterion: str
    window_n: int
    mean_abs_error: float


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def load_observations(path: Path) -> tuple[np.ndarray, dict[int, float]]:
    frame = pd.read_csv(path, index_col=0)
    if "scssm_onset" not in frame.columns:
        raise ValueError(f"{path} must contain a scssm_onset column")
    series = frame["scssm_onset"].copy()
    series.index = series.index.astype(int)
    expected_years = list(range(YEAR_START, YEAR_END + 1))
    missing = [year for year in expected_years if year not in series.index]
    if missing:
        raise ValueError(f"{path} is missing observation years {missing}")
    ordered = series.loc[expected_years].to_numpy(dtype=float)
    if not np.isfinite(ordered).all():
        raise ValueError("observations contain non-finite values")
    return ordered, {year: float(value) for year, value in zip(expected_years, ordered)}


def load_prediction_library(path: Path, max_models: int) -> np.ndarray:
    if max_models < 1:
        raise ValueError("max_models must be positive")
    frame = pd.read_csv(path, nrows=max_models)
    year_columns = [str(year) for year in range(YEAR_START, YEAR_END + 1)]
    missing = [column for column in year_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing prediction columns {missing}")
    if len(frame) < max_models:
        raise ValueError(
            f"{path} contains {len(frame):,} rows; {max_models:,} were requested"
        )
    matrix = frame.loc[:, year_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("prediction library contains non-finite values")
    return matrix


def run_selection_frames(
    prediction_matrix: np.ndarray,
    observation_array: np.ndarray,
    *,
    years: Sequence[int],
    candidate_n: Sequence[int],
) -> dict[int, pd.DataFrame]:
    requested_years = sorted({int(year) for year in years})
    frames = {}
    for window_n in sorted({int(value) for value in candidate_n}):
        frame = revised_ce_eval.run_historical_error_weighted_pipeline(
            prediction_matrix,
            observation_array,
            requested_years,
            window_n=window_n,
        )
        missing = sorted(set(requested_years) - set(frame["Year"]))
        if missing:
            raise ValueError(f"N={window_n} is missing hindcasts for {missing}")
        frames[window_n] = frame
    return frames


def build_hindcasts(
    frames: Mapping[int, pd.DataFrame],
) -> dict[str, dict[int, dict[int, float]]]:
    return {
        criterion: {
            window_n: {
                int(year): float(prediction)
                for year, prediction in zip(frame["Year"], frame[column])
            }
            for window_n, frame in frames.items()
        }
        for criterion, column in PREDICTION_COLUMNS.items()
    }


def run_single_library_hindcasts(
    prediction_matrix: np.ndarray,
    observation_array: np.ndarray,
    *,
    years: Sequence[int],
    candidate_n: Sequence[int],
) -> dict[str, dict[int, dict[int, float]]]:
    return build_hindcasts(
        run_selection_frames(
            prediction_matrix,
            observation_array,
            years=years,
            candidate_n=candidate_n,
        )
    )





def build_fixed_selection_table(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[:, ["Year", "idx_mae", "pred_mae", "true"]].rename(
        columns={
            "idx_mae": "ModelIndex",
            "pred_mae": "Prediction",
            "true": "Observed",
        }
    ).copy()
    selected["ModelIndex"] = selected["ModelIndex"].astype(int)
    selected.insert(2, "ModelID", selected["ModelIndex"] + 1)
    selected["Error"] = selected["Prediction"] - selected["Observed"]
    selected["AbsError"] = selected["Error"].abs()
    return selected.sort_values("Year").reset_index(drop=True)


def build_validation_error_table(
    hindcasts: Mapping[str, Mapping[int, Mapping[int, float]]],
    observations: Mapping[int, float],
    *,
    years: Sequence[int],
    candidate_n: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year in sorted({int(value) for value in years}):
        observed = float(observations[year])
        for window_n in sorted({int(value) for value in candidate_n}):
            row: dict[str, float | int] = {
                "Year": year,
                "Observed": observed,
                "N": window_n,
            }
            for criterion in CRITERIA:
                prediction = float(hindcasts[criterion][window_n][year])
                row[criterion] = abs(prediction - observed)
            rows.append(row)
    return pd.DataFrame(rows, columns=["Year", "Observed", "N", *CRITERIA])


def build_validation_metric_table(
    hindcasts: Mapping[str, Mapping[int, Mapping[int, float]]],
    observations: Mapping[int, float],
    *,
    years: Sequence[int],
    candidate_n: Sequence[int],
) -> pd.DataFrame:
    requested_years = sorted({int(year) for year in years})
    observation = np.array(
        [float(observations[year]) for year in requested_years],
        dtype=float,
    )
    rows: list[dict[str, float | int | str]] = []
    for criterion in CRITERIA:
        for window_n in sorted({int(value) for value in candidate_n}):
            prediction = np.array(
                [
                    float(hindcasts[criterion][window_n][year])
                    for year in requested_years
                ],
                dtype=float,
            )
            error = prediction - observation
            correlation = (
                float(np.corrcoef(prediction, observation)[0, 1])
                if np.std(prediction) > 0 and np.std(observation) > 0
                else float("nan")
            )
            rows.append(
                {
                    "criterion": criterion,
                    "N": window_n,
                    "CC": correlation,
                    "MAE": float(np.mean(np.abs(error))),
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                }
            )
    return pd.DataFrame(rows, columns=["criterion", "N", "CC", "MAE", "RMSE"])


def mean_error_grid(validation_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for criterion in CRITERIA:
        for window_n, group in validation_table.groupby("N", sort=True):
            rows.append(
                {
                    "criterion": criterion,
                    "N": int(window_n),
                    "mean_abs_error": float(group[criterion].mean()),
                }
            )
    return pd.DataFrame(rows)


def select_configuration(validation_table: pd.DataFrame) -> SelectedConfiguration:
    summary = mean_error_grid(validation_table)
    criterion_order = {criterion: index for index, criterion in enumerate(CRITERIA)}
    best = min(
        summary.to_dict("records"),
        key=lambda row: (
            float(row["mean_abs_error"]),
            int(row["N"]),
            criterion_order[str(row["criterion"])],
        ),
    )
    return SelectedConfiguration(
        criterion=str(best["criterion"]),
        window_n=int(best["N"]),
        mean_abs_error=float(best["mean_abs_error"]),
    )


def final_configuration(validation_table: pd.DataFrame) -> SelectedConfiguration:
    rows = validation_table.loc[validation_table["N"] == FINAL_N, FINAL_CRITERION]
    if rows.empty:
        raise ValueError(f"validation table does not contain N={FINAL_N}")
    return SelectedConfiguration(
        criterion=FINAL_CRITERION,
        window_n=FINAL_N,
        mean_abs_error=float(rows.mean()),
    )


def build_test_table(
    hindcasts: Mapping[str, Mapping[int, Mapping[int, float]]],
    observations: Mapping[int, float],
    *,
    years: Sequence[int],
    selected: SelectedConfiguration,
) -> pd.DataFrame:
    rows = []
    selected_predictions = hindcasts[selected.criterion][selected.window_n]
    for year in sorted({int(value) for value in years}):
        observed = float(observations[year])
        prediction = float(selected_predictions[year])
        error = prediction - observed
        rows.append(
            {
                "selected_criterion": selected.criterion,
                "selected_N": selected.window_n,
                "validation_mean_abs_error": selected.mean_abs_error,
                "Year": year,
                "Prediction": prediction,
                "Observed": observed,
                "Error": error,
                "AbsError": abs(error),
            }
        )
    return pd.DataFrame(rows)


def test_period_metrics(test_table: pd.DataFrame) -> dict[str, float]:
    prediction = test_table["Prediction"].to_numpy(dtype=float)
    observation = test_table["Observed"].to_numpy(dtype=float)
    correlation = (
        float(np.corrcoef(prediction, observation)[0, 1])
        if np.std(prediction) > 0 and np.std(observation) > 0
        else float("nan")
    )
    error = prediction - observation
    return {
        "CC": correlation,
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }


def format_annual_forecasts(test_table: pd.DataFrame) -> str:
    display_columns = ["Year", "Prediction", "Observed", "Error", "AbsError"]
    return test_table.loc[:, display_columns].round(3).to_string(index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AMME criteria and export fixed MAE + N=8 forecasts.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--max-models", type=int, default=500_000)
    parser.add_argument("--candidate-n", type=_parse_int_list, default=list(CANDIDATE_N))
    parser.add_argument(
        "--validation-years", type=_parse_int_list, default=list(VALIDATION_YEARS)
    )
    parser.add_argument("--test-years", type=_parse_int_list, default=list(TEST_YEARS))
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--test-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    library_path = args.library or project_root / DEFAULT_LIBRARY_NAME
    observation_path = args.observations or (
        project_root / "precursor_factors_1979_2025.csv"
    )
    if not library_path.exists():
        raise FileNotFoundError(library_path)
    if not observation_path.exists():
        raise FileNotFoundError(observation_path)
    if set(args.validation_years) & set(args.test_years):
        raise ValueError("validation and test years must not overlap")
    if max(args.validation_years) >= min(args.test_years):
        raise ValueError("all validation years must precede all test years")
    observation_array, observations = load_observations(observation_path)
    prediction_matrix = load_prediction_library(library_path, args.max_models)
    full_selection_years = range(C.DEVELOPMENT_START, C.TEST_END + 1)
    all_years = sorted(
        set(args.validation_years) | set(args.test_years) | set(full_selection_years)
    )
    selection_frames = run_selection_frames(
        prediction_matrix,
        observation_array,
        years=all_years,
        candidate_n=args.candidate_n,
    )
    hindcasts = build_hindcasts(selection_frames)
    validation_table = build_validation_error_table(
        hindcasts,
        observations,
        years=args.validation_years,
        candidate_n=args.candidate_n,
    )
    validation_metrics = build_validation_metric_table(
        hindcasts,
        observations,
        years=args.validation_years,
        candidate_n=args.candidate_n,
    )
    development_best = select_configuration(validation_table)
    selected = final_configuration(validation_table)
    test_table = build_test_table(
        hindcasts,
        observations,
        years=args.test_years,
        selected=selected,
    )
    model_indices = selection_frames[FINAL_N].set_index("Year")["idx_mae"]
    test_table["ModelIndex"] = test_table["Year"].map(model_indices).astype(int)
    test_table["ModelID"] = test_table["ModelIndex"] + 1
    test_table = test_table.loc[:, [
        "Year", "ModelIndex", "ModelID", "Prediction", "Observed", "Error",
        "AbsError", "selected_criterion", "selected_N", "validation_mean_abs_error",
    ]]
    metrics = test_period_metrics(test_table)
    output_dir = project_root / "selection_results"
    selection_output = args.selection_output or output_dir / Path(C.DEVELOPMENT_FILE).name
    test_output = args.test_output or output_dir / Path(C.TEST_PREDICTION_FILE).name
    fixed_selection_output = output_dir / Path(C.FIXED_SELECTION_FILE).name
    fixed_selection = build_fixed_selection_table(
        selection_frames[FINAL_N].loc[
            selection_frames[FINAL_N]["Year"].between(C.DEVELOPMENT_START, C.TEST_END)
        ]
    )
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    validation_table.to_csv(selection_output, index=False, float_format="%.2f")
    test_table.to_csv(test_output, index=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_selection.to_csv(fixed_selection_output, index=False)
    for metric in ("CC", "MAE", "RMSE"):
        summary = validation_metrics.pivot(
            index="N",
            columns="criterion",
            values=metric,
        )
        print(f"1988--2015 {metric} by criterion and N:")
        print(summary.loc[:, list(CRITERIA)].round(3).to_string())
    print(
        f"Development-period minimum: {development_best.criterion} "
        f"+ N={development_best.window_n}; "
        f"MAE={development_best.mean_abs_error:.3f}"
    )
    print(
        f"Frozen final configuration: {selected.criterion} + N={selected.window_n}; "
        f"validation MAE={selected.mean_abs_error:.3f}"
    )
    print(
        "2016--2025: "
        f"CC={metrics['CC']:.3f}, MAE={metrics['MAE']:.3f}, "
        f"RMSE={metrics['RMSE']:.3f}"
    )
    print("2016--2025 annual forecasts from the frozen configuration:")
    print(format_annual_forecasts(test_table))
    print(f"Wrote {selection_output}")
    print(f"Wrote {test_output}")
    print(f"Wrote {fixed_selection_output}")
    return 0


if __name__ == "__main__":
    main()
