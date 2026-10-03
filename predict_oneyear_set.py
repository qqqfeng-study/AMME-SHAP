from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import AMME_selection as C
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
PROJECT_ROOT = Path(C.BASE_DIR)
LIBRARY_FILE = Path(C.PRED_CSV)
OBSERVATION_FILE = Path(C.OBS_CSV)
TEST_PREDICTION_FILE = Path(C.TEST_PREDICTION_FILE)

FIG_DIR = PROJECT_ROOT / "fig"
YEAR_START = C.YEAR_START
YEAR_END = C.YEAR_END
N_MODELS = 500_000
FORECAST_YEAR = 2025
FINAL_CRITERION = C.FINAL_CRITERION
FINAL_N = C.WINDOW_N
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 9


@dataclass(frozen=True)
class CandidateProfile:
    indices: np.ndarray
    start_year: int
    allowed_length: int


@dataclass(frozen=True)
class OneYearPlotData:
    end_year: int
    window_n: int
    selected_model_index: int
    candidate_indices: np.ndarray
    years: np.ndarray
    observations: np.ndarray
    selected_predictions: np.ndarray
    candidate_predictions: np.ndarray


def load_prediction_library(
    path: Path = LIBRARY_FILE,
    *,
    years: Iterable[int] = range(YEAR_START, YEAR_END + 1),
    expected_models: int = N_MODELS,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction library not found: {path}")
    requested_years = [int(year) for year in years]
    if not requested_years or len(requested_years) != len(set(requested_years)):
        raise ValueError("years must contain unique year values")
    if min(requested_years) < YEAR_START or max(requested_years) > YEAR_END:
        raise ValueError(f"years must be within {YEAR_START}-{YEAR_END}")
    predictions = pd.read_csv(path, usecols=[str(year) for year in requested_years])
    if len(predictions) != expected_models:
        raise ValueError(
            f"Expected {expected_models:,} models, found {len(predictions):,}"
        )
    predictions.columns = requested_years
    predictions = predictions.astype(np.float64)
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("Prediction library contains non-finite values")
    return predictions


def load_observations(path: Path = OBSERVATION_FILE) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Observation file not found: {path}")
    data = pd.read_csv(path, index_col=0)
    if "scssm_onset" not in data.columns:
        raise KeyError("Observation file has no 'scssm_onset' column")
    observations = data["scssm_onset"].copy()
    observations.index = observations.index.astype(int)
    needed = list(range(YEAR_START, YEAR_END + 1))
    missing = sorted(set(needed) - set(observations.index))
    if missing:
        raise ValueError(f"Observation years are missing: {missing}")
    observations = observations.loc[needed].astype(np.float64)
    if not np.isfinite(observations.to_numpy()).all():
        raise ValueError("Observations contain non-finite values")
    return observations


def load_fixed_selection(path: Path = TEST_PREDICTION_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. Run predict_adaptive_train_N.py first."
        )
    selection = pd.read_csv(path)
    required = {
        "Year", "ModelIndex", "ModelID", "Prediction", "Observed",
        "selected_criterion", "selected_N",
    }
    missing = required - set(selection.columns)
    if missing:
        raise KeyError(f"Prediction columns are missing: {sorted(missing)}")
    selection["Year"] = selection["Year"].astype(int)
    selection["ModelIndex"] = selection["ModelIndex"].astype(int)
    selection["ModelID"] = selection["ModelID"].astype(int)
    if not np.array_equal(selection["ModelID"], selection["ModelIndex"] + 1):
        raise ValueError("ModelID must equal zero-based ModelIndex + 1")
    if not selection["selected_criterion"].eq(FINAL_CRITERION).all():
        raise ValueError(f"Selection criterion must be {FINAL_CRITERION}")
    if not selection["selected_N"].eq(FINAL_N).all():
        raise ValueError(f"Selection window must be {FINAL_N}")
    return selection.sort_values("Year").reset_index(drop=True)


def nearest_candidate_profile(
    predictions: pd.DataFrame,
    observations: pd.Series,
    *,
    end_year: int,
    window_n: int,
) -> CandidateProfile:
    initial_start = end_year - window_n
    start_year = initial_start
    threshold = math.ceil(0.1 * len(predictions))
    allowed_length = window_n
    temporary_start = start_year
    temporary_length = allowed_length
    gate_prediction = None
    gate_observation = None
    coverage_found = False
    while temporary_length > 1 and temporary_start < end_year:
        if temporary_start not in predictions.columns:
            raise KeyError(f"Prediction column {temporary_start} is unavailable")
        if temporary_start not in observations.index:
            raise KeyError(f"Observation year {temporary_start} is unavailable")
        gate_prediction = predictions[temporary_start].to_numpy(dtype=np.float64)
        gate_observation = float(observations.loc[temporary_start])
        difference = gate_prediction - gate_observation
        close = np.isclose(gate_prediction, gate_observation, atol=0.5)
        if gate_observation > 28:
            gate_mask = close & (difference >= 0)
        elif gate_observation < 28:
            gate_mask = close & (difference <= 0)
        else:
            gate_mask = close
        if int(gate_mask.sum()) >= threshold:
            start_year = temporary_start
            allowed_length = temporary_length
            coverage_found = True
            break
        temporary_start += 1
        temporary_length -= 1
    if gate_prediction is None or gate_observation is None:
        raise ValueError("No valid coverage-gate year is available")
    if coverage_found:
        initial_prediction = predictions[initial_start].to_numpy(dtype=np.float64)
        initial_observation = float(observations.loc[initial_start])
        absolute_error = np.abs(initial_prediction - initial_observation)
    else:
        absolute_error = np.abs(gate_prediction - gate_observation)
    percentile_10 = np.percentile(absolute_error, 10)
    candidates = np.flatnonzero(absolute_error <= percentile_10)
    return CandidateProfile(candidates, start_year, allowed_length)


def prepare_one_year_plot_data(
    predictions: pd.DataFrame,
    observations: pd.Series,
    selection: pd.DataFrame,
    *,
    end_year: int = FORECAST_YEAR,
    window_n: int = FINAL_N,
) -> OneYearPlotData:
    selected_row = selection.loc[selection["Year"] == end_year]
    if len(selected_row) != 1:
        raise ValueError(f"Expected one fixed-selection row for {end_year}")
    selected_model_index = int(selected_row.iloc[0]["ModelIndex"])
    if selected_model_index not in predictions.index:
        raise IndexError(f"ModelIndex {selected_model_index} is outside the library")
    profile = nearest_candidate_profile(
        predictions,
        observations,
        end_year=end_year,
        window_n=window_n,
    )
    years = np.arange(profile.start_year, end_year + 1, dtype=int)
    missing_years = [year for year in years if year not in predictions.columns]
    if missing_years:
        raise KeyError(f"Prediction columns are missing: {missing_years}")
    if selected_model_index not in profile.indices:
        raise ValueError("Saved selected model is outside the reconstructed pool")
    selected_predictions = predictions.loc[selected_model_index, years].to_numpy(
        dtype=np.float64
    )
    saved_prediction = float(selected_row.iloc[0]["Prediction"])
    if not np.isclose(selected_predictions[-1], saved_prediction, atol=1e-10, rtol=0.0):
        raise ValueError("Selected model index does not reproduce saved prediction")
    return OneYearPlotData(
        end_year=end_year,
        window_n=window_n,
        selected_model_index=selected_model_index,
        candidate_indices=profile.indices,
        years=years,
        observations=observations.loc[years].to_numpy(dtype=np.float64),
        selected_predictions=selected_predictions,
        candidate_predictions=predictions.loc[profile.indices, years].to_numpy(
            dtype=np.float64
        ),
    )


def add_model_trajectories(
    ax,
    years: np.ndarray,
    predictions: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int = 2,
) -> None:
    for start in range(0, len(predictions), 5000):
        batch = predictions[start:start + 5000]
        segments = np.empty((len(batch), len(years), 2), dtype=np.float64)
        segments[:, :, 0] = years
        segments[:, :, 1] = batch
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        )


def plot_one_year_forecast(
    plot_data: OneYearPlotData,
    output_path: Path,
    *,
    show: bool = True,
) -> None:
    plt.rcParams.update(
        {
            "axes.linewidth": 2,
            "font.size": FONT_SIZE_TICK,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
        }
    )
    years = plot_data.years
    observations = plot_data.observations
    selected = plot_data.selected_predictions
    color_real = "#3E62AD"
    color_pred = "#FF8A00"
    color_gray = "#9c9a9a"
    color_forecast_year = "#C41E3A"
    alpha_base = 0.6
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=300, facecolor="#FFFFFF")
    ax.axvspan(years[0], years[-2], facecolor="#8ad1f5", alpha=0.15, zorder=-2)
    add_model_trajectories(
        ax, years, plot_data.candidate_predictions,
        color=color_gray, linewidth=1, alpha=0.15, zorder=1,
    )
    ax.plot(
        years[:-1], observations[:-1], color=color_real, linewidth=3.5,
        marker="s", markersize=8, alpha=alpha_base, label="Real", zorder=3,
    )
    ax.plot(
        years[-2:], observations[-2:], color=color_real, linewidth=3.5,
        linestyle="--", marker="s", markersize=8, alpha=alpha_base, zorder=3,
    )
    ax.plot(
        years[:-1], selected[:-1], color=color_pred, linewidth=3.5,
        marker="o", markersize=8, alpha=alpha_base, label="Selected", zorder=4,
    )
    ax.plot(
        years[-1:], selected[-1:], color=color_pred, linewidth=0, marker="o",
        markersize=8, markeredgecolor="black", markeredgewidth=1.5,
        markerfacecolor=color_pred, alpha=alpha_base, zorder=5, label="Forecast",
    )
    ax.plot(
        years[-2:], selected[-2:], color=color_pred, linewidth=2.5,
        linestyle="--", marker="o", markersize=8, markerfacecolor="none",
        markeredgecolor="none", alpha=alpha_base, zorder=3,
    )
    ax.axvline(
        x=plot_data.end_year, color=color_forecast_year, linestyle="--", linewidth=1.2,
        alpha=0.6, zorder=1,
    )
    ax.set_title(
        f"Prediction year: {plot_data.end_year}", loc="left", fontsize=FONT_SIZE_TITLE,
        fontweight="bold", pad=8,
    )
    ax.set_ylim(21, 37)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=FONT_SIZE_TICK)
    for year, label in zip(years, ax.get_xticklabels()):
        if year == plot_data.end_year:
            label.set_color(color_forecast_year)
    ax.set_ylabel("Onset pentad", fontsize=FONT_SIZE_LABEL)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color("black")
    ax.grid(alpha=0.25, ls="--", axis="x", zorder=0)
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(years)
    rel_labels = [
        f"(t-{plot_data.window_n - i})" if i < plot_data.window_n else "(t)"
        for i in range(len(years))
    ]
    secax.set_xticklabels(rel_labels, fontsize=FONT_SIZE_TICK, fontweight="bold")
    secax.spines["bottom"].set_position(("outward", 18))
    secax.spines["bottom"].set_visible(False)
    secax.tick_params(axis="x", length=0, pad=2)
    for i, label in enumerate(secax.get_xticklabels()):
        label.set_color("#1a1a1a" if i < plot_data.window_n else color_forecast_year)
    legend_elements = [
        Line2D([0], [0], color=color_real, lw=2.5, marker="s", markersize=6,
               alpha=alpha_base, linestyle="-", label="Observed"),
        Line2D([0], [0], color=color_pred, lw=2, marker="o", markersize=6,markeredgecolor="black", linestyle="--", label="Prediction"),
        Line2D([0], [0], color=color_pred, lw=2.5, marker="o", markersize=6,
               alpha=alpha_base, linestyle="-", label="Selected model"),
        Line2D([0], [0], color=color_gray, lw=1.5, alpha=0.3,
               linestyle="-", label="Retained candidates"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.72, 0.98),
        ncol=2, fontsize=FONT_SIZE_LEGEND, frameon=False, columnspacing=1.2,
    )
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_generated_models(
    predictions: pd.DataFrame,
    observations: pd.Series,
    *,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Calibri"]
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["font.size"] = 14
    mpl.rcParams["xtick.major.width"] = 2
    mpl.rcParams["ytick.major.width"] = 2
    years = np.asarray(predictions.columns, dtype=int)
    observed = observations.loc[years].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300, facecolor="white")
    add_model_trajectories(
        ax, years, predictions.to_numpy(dtype=np.float64),
        color="#415DA5", linewidth=mpl.rcParams["lines.linewidth"], alpha=0.1,
    )
    line1 = ax.plot(
        years, observed, label="OBS", color="#131313", lw=3, ls="-", alpha=0.9
    )[0]
    ax.axvspan(1979, 2015, color="#E9E0D6", alpha=0.3)
    ax.axvspan(2016, 2025, color="#4396DF", alpha=0.3)
    ax.grid(alpha=0.2, ls="--", zorder=0, color="#444040")
    ax.set_ylim(20, 38)
    ax.set_xlim(1978, 2026)
    ax.set_xticks(np.arange(1979, 2026, 5))
    yticks = np.arange(20, 39, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) if y % 2 == 0 else "" for y in yticks])
    ax.set_ylabel("Onset Pentad", fontsize=16)
    ax.set_xlabel("Year", fontsize=16)
    ax.text(1998.5, 21, "Train Model", fontsize=16, color="black", ha="center")
    ax.text(2019.5, 21, "Test Model", fontsize=16, color="black", ha="center")
    ml_model_legend = Line2D(
        [0], [0], color="#415DA5", lw=2, linestyle="-",
        label="Multi-Model", alpha=0.6,
    )
    handles = [line1, ml_model_legend]
    ax.legend(
        handles, [handle.get_label() for handle in handles], fontsize=10,
        loc="upper right", ncol=3, columnspacing=0.8,
    )
    plt.subplots_adjust(left=0.07, right=0.85, top=0.85, bottom=0.2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plot fixed MAE + N=8 forecasts.")
    parser.add_argument("--year", type=int, default=FORECAST_YEAR)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--skip-overview", action="store_true")
    args = parser.parse_args(argv)
    observations = load_observations()
    selection = load_fixed_selection()
    predictions = load_prediction_library()
    plot_data = prepare_one_year_plot_data(
        predictions, observations, selection, end_year=args.year
    )
    plot_one_year_forecast(
        plot_data,
        FIG_DIR / f"prediction_{args.year}.png",
        show=not args.no_show,
    )
    print(
        f"{args.year}: ModelIndex={plot_data.selected_model_index}, "
        f"Prediction={plot_data.selected_predictions[-1]:.6f}, "
        f"Candidates={len(plot_data.candidate_indices):,}"
    )
    if not args.skip_overview:
        plot_generated_models(
            predictions, observations,
            output_path=FIG_DIR / "generated_models.png", show=not args.no_show,
        )


if __name__ == "__main__":
    main()
