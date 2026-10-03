# %%
"""Explain local MAE + N=8 forecasts with common-background TreeSHAP.

Inputs are the project selection CSVs, precursor data and numbered model
files under models/. SHAP values are computed in memory; only figures are saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
SELECTION_FILE = PROJECT_ROOT / "selection_results" / "AMME_MAE_N8_1988_2025.csv"
HELDOUT_FILE = PROJECT_ROOT / "selection_results" / "test_n8_predict.csv"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_FILE = PROJECT_ROOT / "precursor_factors_1979_2025.csv"
FIG_DIR = SCRIPT_DIR / "fig"


START_YEAR = 1988
END_YEAR = 2025
HELDOUT_START_YEAR = 2016
TRAIN_START = 1979
TRAIN_END = 2015
PREDICTOR_COLUMNS = ["stt", "enso", "sam", "ao_mar", "lst"]
FEATURE_NAMES = ["STT", "ENSO", "SAM", "AO", "LST"]
TARGET_COLUMN = "scssm_onset"

POSITIVE_SHAP_COLOR = "#fa930d"
NEGATIVE_SHAP_COLOR = "#1278f4"
LATER_ONSET_COLOR = POSITIVE_SHAP_COLOR
EARLIER_ONSET_COLOR = NEGATIVE_SHAP_COLOR
MEAN_ONSET = 28.25
PDO_TRANSITION_PERIODS = ((1998, 2004), (2014, 2017))
PDO_STAGE_COLORS = {
    "PDO(+)": "#fa930d",
    "PDO(T)": "gray",
    "PDO(-)": "#1278f4",
}
COMMON_BASELINE_YEAR = 2016
ADDITIVITY_TOLERANCE = 5e-5


@dataclass(frozen=True)
class AnnualShapResult:
    year: int
    model_index: int
    model_id: int
    sample: np.ndarray
    shap_values: np.ndarray
    absolute_shares: np.ndarray
    signed_shares: np.ndarray
    model_expected_value: float
    common_baseline: float
    baseline_shift: float
    prediction: float
    reconstructed: float
    additivity_residual: float
    iteration_limit: int


def build_pdo_stage_spans(
    start_year: int,
    end_year: int,
) -> list[tuple[str, int, int]]:
    """Return the PDO background stages used in Figure 3, clipped to Figure 4."""

    spans: list[tuple[str, int, int]] = []
    phase_start = start_year
    phase_label = "PDO(+)"
    for transition_start, transition_end in PDO_TRANSITION_PERIODS:
        if transition_end < start_year:
            phase_label = "PDO(-)"
            continue
        if transition_start > end_year:
            break
        if transition_start > phase_start:
            spans.append(
                (phase_label, phase_start, min(transition_start - 1, end_year))
            )
        clipped_start = max(transition_start, start_year)
        clipped_end = min(transition_end, end_year)
        if clipped_start <= clipped_end:
            spans.append(("PDO(T)", clipped_start, clipped_end))
        phase_start = max(phase_start, transition_end + 1)
        phase_label = "PDO(-)"
    if phase_start <= end_year:
        spans.append((phase_label, phase_start, end_year))
    return spans


def observed_onset_mean(observed: pd.Series) -> float:
    """Return the finite mean onset pentad of the displayed observations."""

    values = pd.to_numeric(observed, errors="raise").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("observed onset values must be finite and non-empty")
    return float(values.mean())


def _local_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"File paths must stay inside {PROJECT_ROOT}: {resolved}") from error
    return resolved


def load_heldout_predictions(path: Path = HELDOUT_FILE) -> pd.DataFrame:
    """Load the canonical MAE + N=8 predictions for 2016--2025."""

    path = _local_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"held-out prediction CSV not found: {path}")

    heldout = pd.read_csv(path)
    required = {
        "selected_criterion",
        "selected_N",
        "Year",
        "Prediction",
        "Observed",
        "Error",
        "AbsError",
    }
    missing_columns = sorted(required - set(heldout.columns))
    if missing_columns:
        raise ValueError(f"held-out CSV is missing columns: {missing_columns}")

    heldout["Year"] = pd.to_numeric(heldout["Year"], errors="raise").astype(int)
    heldout["selected_N"] = pd.to_numeric(
        heldout["selected_N"], errors="raise"
    ).astype(int)
    for column in ("Prediction", "Observed", "Error", "AbsError"):
        heldout[column] = pd.to_numeric(heldout[column], errors="raise").astype(float)

    heldout = heldout.sort_values("Year").reset_index(drop=True)
    expected_years = list(range(HELDOUT_START_YEAR, END_YEAR + 1))
    if heldout["Year"].tolist() != expected_years:
        raise ValueError("held-out CSV must contain exactly 2016--2025")
    if not heldout["selected_criterion"].eq("MAE").all():
        raise ValueError("held-out CSV contains a criterion other than MAE")
    if not heldout["selected_N"].eq(8).all():
        raise ValueError("held-out CSV contains a selected_N other than 8")
    return heldout


def load_mae_n8_selection(
    path: Path = SELECTION_FILE,
    heldout_path: Path = HELDOUT_FILE,
) -> pd.DataFrame:
    path = _local_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAE + N=8 selection CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "Year", "ModelIndex", "ModelID", "Prediction", "Observed", "Error", "AbsError",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"selection CSV is missing columns: {missing}")
    for column in ("Year", "ModelIndex", "ModelID"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("Prediction", "Observed", "Error", "AbsError"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame = frame.loc[frame["Year"].between(START_YEAR, END_YEAR)]
    frame = frame.sort_values("Year").reset_index(drop=True)
    if frame["Year"].tolist() != list(range(START_YEAR, END_YEAR + 1)):
        raise ValueError("selection CSV must contain exactly one row per year from 1988 to 2025")
    if (frame["ModelIndex"] < 0).any():
        raise ValueError("ModelIndex must be non-negative")
    if not np.array_equal(frame["ModelID"], frame["ModelIndex"] + 1):
        raise ValueError("ModelID must equal zero-based ModelIndex + 1")
    frame["ModelPrediction"] = frame["Prediction"]
    heldout = load_heldout_predictions(heldout_path).set_index("Year")
    fixed_test = frame.set_index("Year").loc[HELDOUT_START_YEAR:END_YEAR]
    if not np.array_equal(fixed_test["Observed"], heldout["Observed"]):
        raise ValueError("fixed-selection and held-out observations do not match")
    if not np.allclose(fixed_test["Prediction"], heldout["Prediction"], rtol=0.0, atol=5e-4):
        raise ValueError("fixed-selection and held-out predictions disagree")
    for column in ("ModelIndex", "ModelID"):
        if column in heldout and not np.array_equal(fixed_test[column], heldout[column]):
            raise ValueError(f"fixed-selection and held-out {column} values disagree")
    mask = frame["Year"].between(HELDOUT_START_YEAR, END_YEAR)
    for column in ("Prediction", "Observed", "Error", "AbsError"):
        frame.loc[mask, column] = frame.loc[mask, "Year"].map(heldout[column])
    return frame


def build_model_paths(
    selection: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
) -> list[Path]:
    model_dir = _local_path(model_dir)
    model_indices = selection["ModelIndex"].to_numpy(dtype=int)
    model_ids = selection["ModelID"].to_numpy(dtype=int)
    if (model_indices < 0).any() or not np.array_equal(model_ids, model_indices + 1):
        raise ValueError("ModelID must equal zero-based ModelIndex + 1")
    paths = [
        _local_path(model_dir / f"bst_model_seed676_model{int(model_id)}.pkl")
        for model_id in model_ids
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} selected model files are missing:\n{preview}")
    return paths


def load_standardized_data(
    path: Path = DATA_FILE,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Load predictors using the same 1979--2015 scaling as the new models."""

    path = _local_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"precursor CSV not found: {path}")

    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_numeric(data.index, errors="raise").astype(int)
    data = data.sort_index()
    required = PREDICTOR_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"precursor CSV is missing columns: {missing_columns}")

    expected_years = list(range(TRAIN_START, END_YEAR + 1))
    missing_years = sorted(set(expected_years) - set(data.index))
    if missing_years:
        raise ValueError(f"precursor CSV is missing years: {missing_years}")

    predictors = data.loc[expected_years, PREDICTOR_COLUMNS].astype(float)
    training = predictors.loc[TRAIN_START:TRAIN_END]
    training_mean = training.mean(axis=0)
    training_std = training.std(axis=0, ddof=0)
    if np.any(np.isclose(training_std.to_numpy(), 0.0)):
        bad = training_std.index[np.isclose(training_std.to_numpy(), 0.0)].tolist()
        raise ValueError(f"zero training-period standard deviation: {bad}")

    standardized = (predictors - training_mean) / training_std
    observations = data.loc[expected_years, TARGET_COLUMN].astype(float)
    if not np.isfinite(standardized.to_numpy()).all():
        raise ValueError("standardized predictors contain NaN or Inf")
    if not np.isfinite(observations.to_numpy()).all():
        raise ValueError("observations contain NaN or Inf")
    return standardized, list(FEATURE_NAMES), observations


def _plotting_modules():
    """Import the Python-only plotting stack and report environment blockers."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "The Python plotting environment requires a compatible matplotlib "
            "installation."
        ) from exc
    return plt


def _explanation_modules():
    try:
        import joblib
        import shap
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "Common-background TreeSHAP requires the mldl environment with "
            "compatible joblib, shap, and xgboost installations."
        ) from exc

    return joblib, shap, xgb


def _iteration_limit(model, booster) -> int:
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return int(model.best_iteration) + 1
    return int(booster.num_boosted_rounds())


def _scalar_expected_value(expected_value) -> float:
    values = np.asarray(expected_value, dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise RuntimeError(
            "regression TreeSHAP must provide one finite expected value"
        )
    return float(values[0])


def calculate_shap_results(
    selection: pd.DataFrame,
    model_paths: list[Path],
    standardized: pd.DataFrame,
    common_baseline_year: int = COMMON_BASELINE_YEAR,
) -> list[AnnualShapResult]:
    """Calculate annual TreeSHAP values against one 1979--2015 background.

    The common scalar baseline is the selected model's expected output in
    ``common_baseline_year``. For every other selected model, the difference
    between its own expectation and this scalar is kept as ``baseline_shift``:

    ``prediction = B* + baseline_shift + sum(feature_shap)``.

    This preserves exact local additivity without assigning model-baseline
    drift to any of the five physical predictors.
    """

    if len(selection) != len(model_paths):
        raise ValueError("selection and model_paths must have the same length")
    if common_baseline_year not in set(selection["Year"].astype(int)):
        raise ValueError(
            f"common baseline year {common_baseline_year} is absent from selection"
        )

    background = standardized.loc[
        TRAIN_START:TRAIN_END, PREDICTOR_COLUMNS
    ].to_numpy(dtype=float)
    if background.shape != (
        TRAIN_END - TRAIN_START + 1,
        len(PREDICTOR_COLUMNS),
    ):
        raise RuntimeError(
            "the common TreeSHAP background must contain all 1979--2015 rows"
        )
    if not np.isfinite(background).all():
        raise ValueError("the common TreeSHAP background contains NaN or Inf")

    joblib, shap, xgb = _explanation_modules()
    raw_results: list[dict[str, object]] = []
    for row, model_path in zip(selection.itertuples(index=False), model_paths):
        year = int(row.Year)
        sample = standardized.loc[[year], PREDICTOR_COLUMNS].to_numpy(dtype=float)
        model = joblib.load(_local_path(model_path))
        booster = model.get_booster()
        limit = _iteration_limit(model, booster)
        sliced_booster = booster[:limit]
        explainer = shap.TreeExplainer(
            sliced_booster,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",
            feature_names=FEATURE_NAMES,
        )
        shap_values = np.asarray(
            explainer.shap_values(sample, check_additivity=False),
            dtype=float,
        )
        if shap_values.ndim == 2:
            if shap_values.shape[0] != 1:
                raise RuntimeError(f"{year}: expected one SHAP sample row")
            shap_values = shap_values[0]
        if shap_values.shape != (len(PREDICTOR_COLUMNS),):
            raise RuntimeError(
                f"{year}: unexpected SHAP shape {shap_values.shape}"
            )
        absolute_total = float(np.abs(shap_values).sum())
        if np.isclose(absolute_total, 0.0):
            absolute_shares = np.zeros_like(shap_values)
            signed_shares = np.zeros_like(shap_values)
        else:
            absolute_shares = np.abs(shap_values) / absolute_total
            signed_shares = shap_values / absolute_total
        model_expected_value = _scalar_expected_value(explainer.expected_value)
        prediction = float(
            np.asarray(
                sliced_booster.predict(xgb.DMatrix(sample)), dtype=float
            ).reshape(-1)[0]
        )
        sklearn_prediction = float(
            np.asarray(model.predict(sample), dtype=float).reshape(-1)[0]
        )
        if not np.isclose(
            prediction,
            sklearn_prediction,
            rtol=0.0,
            atol=ADDITIVITY_TOLERANCE,
        ):
            raise RuntimeError(
                f"{year}: best-iteration Booster prediction does not match "
                f"model.predict; booster={prediction}, model={sklearn_prediction}"
            )
        local_reconstructed = float(model_expected_value + shap_values.sum())

        if not np.isclose(
            prediction,
            local_reconstructed,
            rtol=0.0,
            atol=ADDITIVITY_TOLERANCE,
        ):
            raise RuntimeError(
                f"{year}: common-background TreeSHAP additivity failed; "
                f"prediction={prediction}, reconstructed={local_reconstructed}"
            )
        expected_model_prediction = float(
            getattr(row, "ModelPrediction", row.Prediction)
        )
        if not np.isclose(
            prediction,
            expected_model_prediction,
            rtol=1e-5,
            atol=1e-4,
        ):
            raise RuntimeError(
                f"{year}: rebuilt model does not match the selection CSV; "
                f"ModelID={int(row.ModelID)}, model={prediction}, "
                f"CSV={expected_model_prediction}"
            )

        raw_results.append(
            {
                "year": year,
                "model_index": int(row.ModelIndex),
                "model_id": int(row.ModelID),
                "sample": sample[0].copy(),
                "shap_values": shap_values.copy(),
                "absolute_shares": absolute_shares.copy(),
                "signed_shares": signed_shares.copy(),
                "model_expected_value": model_expected_value,
                "prediction": prediction,
                "iteration_limit": limit,
            }
        )

    anchor_rows = [
        item for item in raw_results if int(item["year"]) == common_baseline_year
    ]
    if len(anchor_rows) != 1:
        raise RuntimeError(
            f"expected exactly one common-baseline row for {common_baseline_year}"
        )
    common_baseline = float(anchor_rows[0]["model_expected_value"])

    results: list[AnnualShapResult] = []
    for item in raw_results:
        model_expected_value = float(item["model_expected_value"])
        shap_values = np.asarray(item["shap_values"], dtype=float)
        prediction = float(item["prediction"])
        baseline_shift = model_expected_value - common_baseline
        reconstructed = float(
            common_baseline + baseline_shift + shap_values.sum()
        )
        additivity_residual = prediction - reconstructed
        if abs(additivity_residual) > ADDITIVITY_TOLERANCE:
            raise RuntimeError(
                f"{int(item['year'])}: common-baseline decomposition failed; "
                f"residual={additivity_residual:.3e}"
            )
        results.append(
            AnnualShapResult(
                year=int(item["year"]),
                model_index=int(item["model_index"]),
                model_id=int(item["model_id"]),
                sample=np.asarray(item["sample"], dtype=float),
                shap_values=shap_values,
                absolute_shares=np.asarray(item["absolute_shares"], dtype=float),
                signed_shares=np.asarray(item["signed_shares"], dtype=float),
                model_expected_value=model_expected_value,
                common_baseline=common_baseline,
                baseline_shift=baseline_shift,
                prediction=prediction,
                reconstructed=reconstructed,
                additivity_residual=additivity_residual,
                iteration_limit=int(item["iteration_limit"]),
            )
        )
    return results


def build_signed_table(
    results: list[AnnualShapResult],
    selection: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    records = []
    for result in results:
        record = {"year": result.year}
        record.update(dict(zip(feature_names, result.shap_values)))
        records.append(record)

    signed = pd.DataFrame(records).set_index("year").sort_index()
    predictions = selection.set_index("Year")[["Prediction", "Observed"]].rename(
        columns={"Prediction": "PRE", "Observed": "OBS"}
    )
    joined = signed.join(predictions, how="inner")
    expected_years = list(range(START_YEAR, END_YEAR + 1))
    if joined.index.tolist() != expected_years:
        raise RuntimeError("signed SHAP table is not aligned to 1988--2025")
    common_baselines = {float(result.common_baseline) for result in results}
    if len(common_baselines) != 1:
        raise RuntimeError("annual results do not share one common baseline")
    joined.attrs["common_baseline"] = common_baselines.pop()
    joined.attrs["background_start"] = TRAIN_START
    joined.attrs["background_end"] = TRAIN_END
    joined.attrs["common_baseline_year"] = COMMON_BASELINE_YEAR
    return joined


def load_plot_indices(years: pd.Index, path: Path = DATA_FILE) -> pd.DataFrame:
    """Align indices standardized over 1979--2015 to the displayed SHAP years."""

    indices, feature_names, _ = load_standardized_data(path)
    if indices.index.has_duplicates:
        raise ValueError("precursor indices contain duplicate years")
    missing_years = years.difference(indices.index).tolist()
    if missing_years:
        raise ValueError(f"precursor indices are missing years: {missing_years}")
    return indices.loc[years, PREDICTOR_COLUMNS].rename(
        columns=dict(zip(PREDICTOR_COLUMNS, feature_names))
    )


def plot_signed_lines(df_signed: pd.DataFrame, output_path: Path) -> None:
    """Plot SHAP bars and standardized indices with a common right-axis scale."""

    plt = _plotting_modules()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    years = df_signed.index
    indices = load_plot_indices(years)

    xg = df_signed["PRE"]
    scssm = df_signed["OBS"]
    xg_fit = np.polyval(np.polyfit(xg.index, xg.values, 3), xg.index)
    scssm_fit = np.polyval(np.polyfit(scssm.index, scssm.values, 3), scssm.index)

    plt.rcParams.update(
        {
            "axes.linewidth": 2,
            "font.size": 10,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "font.serif": "Calibri",
        }
    )

    n_rows = 6
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(6.4, 1.2 * n_rows),
        sharex=True,
        dpi=300,
    )
    fig.subplots_adjust(left=0.14, right=0.82, bottom=0.085, top=0.98, hspace=0.1)

    # common_baseline = float(df_signed.attrs["common_baseline"])
    # fig.text(
    #     0.98,
    #     0.995,
    #     (
    #         f"Interventional TreeSHAP background: {TRAIN_START}\N{EN DASH}{TRAIN_END}; "
    #         f"common $B^*$ ({COMMON_BASELINE_YEAR} anchor) = "
    #         f"{common_baseline:.2f} pentads"
    #     ),
    #     ha="right",
    #     va="top",
    #     fontsize=6.5,
    # )

    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        vals = df_signed[name]
        colors = [
            POSITIVE_SHAP_COLOR if value > 0 else NEGATIVE_SHAP_COLOR
            for value in vals
        ]
        ax.bar(
            years,
            vals.values,
            color=colors,
            width=1,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(years.min() - 1, years.max() + 1)
        ax.set_ylim(-3.5, 3.5)
        ax.set_yticks([-2.5, 0, 2.5])
        index_ax = ax.twinx()
        index_line, = index_ax.plot(
            years,
            indices[name].to_numpy(),
            color="#808080",
            linestyle="-",
            linewidth=2.2,
            alpha=0.35,
            label="Index",
            zorder=4,
        )
        # Match all left-axis tick positions, including the common zero line.
        index_ax.set_ylim(ax.get_ylim())
        index_ax.set_yticks(ax.get_yticks())
        index_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        # Use the same default tick font size and dimensions as the left axis.
        index_ax.tick_params(axis="y", colors="#808080")
        index_ax.grid(False)
        for spine in ("left", "top", "bottom"):
            index_ax.spines[spine].set_visible(False)
        index_ax.spines["right"].set_visible(True)
        index_ax.spines["right"].set_color("#808080")
        ax.spines["right"].set_visible(False)
        ax.text(
            0.01,
            0.92,
            name,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="top",
        )
        if i < n_rows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if i == 0:
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(True)
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
        else:
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.text(
            -0.18,
            1.038,
            chr(97 + i),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
        )
        if i == 0:
            index_ax.legend(
                handles=[
                    Patch(
                        facecolor=POSITIVE_SHAP_COLOR,
                        edgecolor="black",
                        label="+SHAP (later)",
                    ),
                    Patch(
                        facecolor=NEGATIVE_SHAP_COLOR,
                        edgecolor="black",
                        label="−SHAP (earlier)",
                    ),
                    index_line,
                ],
                loc="upper right",
                fontsize=5.5,
                frameon=False,
                ncol=3,
                columnspacing=0.8,
                handletextpad=0.35,
            )

    fig.text(
        0.06,
        0.60,
        "SHAP contribution (pentads)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.908,
        0.60,
        "Standardized index",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
        color="#808080",
    )

    ax_last = axes[-1]
    predicted = df_signed["PRE"]
    observed = df_signed["OBS"]
    mean_onset = observed_onset_mean(observed)
    predicted_colors = [
        LATER_ONSET_COLOR if value >= MEAN_ONSET else EARLIER_ONSET_COLOR
        for value in predicted
    ]
    ax_last.scatter(
        years,
        predicted.values,
        color=predicted_colors,
        s=38,
        alpha=0.6,
        marker="x",
        linewidths=1,
        zorder=5,
    )
    ax_last.plot(
        years,
        xg_fit,
        color="#868080",#POSITIVE_SHAP_COLOR,
        linewidth=2.5,
        linestyle="-",
        label="XG-MAE N=8 Fit (3rd Poly)",
        zorder=1,
        alpha=0.8,
    )
    observed_colors = [
        LATER_ONSET_COLOR if value >= MEAN_ONSET else EARLIER_ONSET_COLOR
        for value in observed
    ]
    ax_last.scatter(
        years,
        observed.values,
        color=observed_colors,
        s=38,
        alpha=0.6,
        marker="o",
        edgecolors="none",
        linewidths=1,
        zorder=4,
    )
    ax_last.plot(
        years,
        scssm_fit,
        color="#0B0B0B",#NEGATIVE_SHAP_COLOR,
        linewidth=2.5,
        linestyle="-",
        label="SCSSM Fit (3rd Poly)",
        zorder=1,
        alpha=0.8,
    )

    for label, start, end in build_pdo_stage_spans(START_YEAR, END_YEAR):
        color = PDO_STAGE_COLORS[label]
        ax_last.axvspan(
            start - 0.5,
            end + 0.5,
            color=color,
            alpha=0.2,
            zorder=0,
        )
        ax_last.text(
            (start + end) / 2,
            35.5,
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
        )
    ax_last.axhline(mean_onset, color="black", linewidth=0.8, ls="--")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label="Obs (late)",
            markerfacecolor=LATER_ONSET_COLOR,
            markeredgecolor=LATER_ONSET_COLOR,
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label="Obs (early)",
            markerfacecolor=EARLIER_ONSET_COLOR,
            markeredgecolor=EARLIER_ONSET_COLOR,
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color=LATER_ONSET_COLOR,
            label="Pred (late)",
            markersize=6,
            markeredgewidth=1.2,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color=EARLIER_ONSET_COLOR,
            label="Pred (early)",
            markersize=6,
            markeredgewidth=1.2,
        ),
        Line2D(
            [0],
            [0],
            color="#0B0B0B",#NEGATIVE_SHAP_COLOR,
            lw=2.5,
            label="Fit (Obs)",
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            color="#868080",#POSITIVE_SHAP_COLOR,
            lw=2.5,
            label="Fit (Pred)",
            alpha=0.8,
        ),
        Line2D([0], [0], color="black", lw=1, label="Mean", ls="--"),
    ]
    ax_last.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=5.5,
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.55, 1.18),
        columnspacing=0.65,
        handletextpad=0.35,
        handlelength=1.4,
    )
    ax_last.set_ylim(22, 37)
    ax_last.spines["top"].set_visible(False)
    ax_last.spines["bottom"].set_visible(True)
    ax_last.set_xlabel("Year", fontsize=8)
    ax_last_box = ax_last.get_position()
    fig.text(
        0.06,
        ax_last_box.y0 + ax_last_box.height / 2,
        "Onset pentad",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
    )
    ax_last.text(
        -0.18,
        1.038,
        "f",
        transform=ax_last.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
    )
    fig.savefig(_local_path(output_path), dpi=600, bbox_inches="tight", facecolor="white")
    plt.show()


def main() -> int:
    # Check the selected Python plotting runtime before loading 38 models.
    _plotting_modules()
    selection = load_mae_n8_selection()
    model_paths = build_model_paths(selection)
    standardized, feature_names, _ = load_standardized_data()
    results = calculate_shap_results(selection, model_paths, standardized)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    signed = build_signed_table(results, selection, feature_names)
    plot_signed_lines(
        signed,
        FIG_DIR / "figure3_annual_shap_contribution_bars.png",
    )
    return 0


if __name__ == "__main__":
    main()

# %%
