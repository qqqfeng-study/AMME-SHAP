#%%
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import f, linregress
import statsmodels.api as sm_api
import xarray as xr
from eofs.xarray import Eof


def resolve_script_directory() -> Path:
    """Locate this folder during script, VS Code cell, or Jupyter execution."""
    script_file = globals().get("__file__")
    if script_file:
        return Path(script_file).resolve().parent

    working_directory = Path.cwd().resolve()
    candidates = (
        working_directory,
        working_directory / "pdo_relationship",
        working_directory / "AMME" / "pdo_relationship",
    )
    for candidate in candidates:
        if (candidate / "pdo_fig_v1.py").is_file():
            return candidate
    raise RuntimeError(
        "Cannot locate pdo_relationship. Open the AMME project or its "
        "pdo_relationship folder before running this code interactively."
    )


HERE = resolve_script_directory()

WINDOW = 11
ALPHA = 0.10
MEAN_BLOCK_LENGTH = 4
DEFAULT_RESAMPLES = 199_999
DEFAULT_SEED = 20_260_912
PDO_TRANSITION_QUANTILE = 0.25
PDO_TRANSITION_SENSITIVITY_QUANTILES = (0.20, 0.25, 0.30)
PDO_MIN_TRANSITION_YEARS = 3
PDO_NOW_START_YEAR = 2020

RF_PREDICTIONS = np.array(
    [
        27.68, 27.32, 27.39, 31.01, 29.10, 29.49, 26.83, 29.17, 27.19,
        27.15, 31.34, 26.86, 26.71, 27.62, 27.78, 27.86, 28.22, 27.49,
        26.77, 28.18, 29.15, 29.08, 28.76, 26.16, 29.74, 30.02,
    ],
    dtype=float,
)[-10:]


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs and fixed analysis choices used in the final figure."""

    sst_file: Path = (
        HERE.parents[2]
        / "DATA/DATA/ERA5_native_SCSSM/SST"
        / "ERA5_ecmwf_SST_Y1940-2025_M01-12_monthly.nc"
    )
    factors_file: Path = HERE / "precursor_factors_1979_2025.csv"
    sm_file: Path = HERE.parent / "lg_scssm.csv"
    xg_file: Path = HERE / "selection_results" / "AMME_MAE_N8_1988_2025.csv"
    output_file: Path = HERE / "fig" / "pdo_fig_v1_revised.png"
    onset_start_year: int = 1979
    onset_end_year: int = 2025
    pdo_end_year: int = 2024
    eof_domain: tuple[float, float, float, float] = (120.0, 260.0, 20.0, 60.0)
    pdo_sign: int = -1
    filter_taps: int = 11
    filter_cutoff: float = 1 / 11
    filter_design_fs: float = 2.0
    latitude_chunk: int = 64
    f_resamples: int = DEFAULT_RESAMPLES
    f_seed: int = DEFAULT_SEED
    moderation_resamples: int = 9_999
    moderation_seed: int = 7_388


@dataclass
class FTestResult:
    """All values required for panel b and the reviewer response."""

    scan: pd.DataFrame
    pointwise_critical: float
    scan_critical: float
    lag1_acf: float

    @property
    def peak(self) -> pd.Series:
        return self.scan.loc[self.scan["F"].idxmax()]


@dataclass(frozen=True)
class PDOStageDefinition:
    """Objective near-zero PDO transitions and the separate Now stage."""

    transition_periods: tuple[tuple[int, int], ...]
    threshold: float
    threshold_quantile: float
    minimum_transition_years: int
    now_start: int


def validate_year_index(table: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return data with a unique, ordered integer-year index."""
    result = table.copy()
    years = pd.to_numeric(result.index, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(years).all() or not np.equal(years, years.astype(int)).all():
        raise ValueError("Year labels must be finite integers.")
    result.index = years.astype(int)
    if result.index.has_duplicates:
        raise ValueError("Duplicate years are not allowed.")
    return result.sort_index()


def load_factors(config: AnalysisConfig) -> pd.DataFrame:
    """Load and validate the annual predictor and onset table."""
    factors = validate_year_index(pd.read_csv(config.factors_file, index_col=0))
    expected = np.arange(config.onset_start_year, config.onset_end_year + 1)
    factors = factors.loc[config.onset_start_year : config.onset_end_year]
    if not np.array_equal(factors.index.to_numpy(), expected):
        raise ValueError("Expected one ordered factor row per year, 1979--2025.")
    required = ["lst", "scssm_onset"]
    if not set(required).issubset(factors.columns):
        raise ValueError(f"The factors file must contain {required}.")
    if not np.isfinite(factors[required].to_numpy(dtype=float)).all():
        raise ValueError("LST and SCSSM-onset values must all be finite.")
    return factors


def load_low_pass_pdo(config: AnalysisConfig) -> pd.Series:
    """Reproduce the author's ERA5 SST -> EOF1 -> low-pass PDO series."""
    if not config.sst_file.is_file():
        raise FileNotFoundError(f"Missing SST file: {config.sst_file}")
    if config.pdo_sign not in (-1, 1):
        raise ValueError("pdo_sign must be either -1 or 1.")

    annual_pieces = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Engine 'cfgrib' loading failed.*", category=RuntimeWarning
        )
        dataset = xr.open_dataset(config.sst_file, engine="netcdf4")

    with dataset:
        sst = dataset["sst"].drop_vars(["expver", "number"], errors="ignore")
        if "valid_time" in sst.dims:
            sst = sst.rename({"valid_time": "time"})
        sst = sst.sel(
            time=slice(
                f"{config.onset_start_year}-01-01",
                f"{config.pdo_end_year}-12-31",
            )
        )
        expected_months = pd.period_range(
            f"{config.onset_start_year}-01",
            f"{config.pdo_end_year}-12",
            freq="M",
        )
        actual_months = pd.DatetimeIndex(sst.time.values).to_period("M")
        if not actual_months.equals(expected_months):
            raise ValueError("The PDO period must contain 12 unique months per year.")

        for lower in range(0, sst.sizes["latitude"], config.latitude_chunk):
            block = sst.isel(
                latitude=slice(lower, lower + config.latitude_chunk)
            ).load()
            anomaly = block.groupby("time.month") - block.groupby("time.month").mean(
                "time"
            )
            annual_pieces.append(anomaly.resample(time="1Y").mean("time"))

    annual = xr.concat(annual_pieces, dim="latitude").sortby(
        ["longitude", "latitude"]
    )
    global_mean = annual.weighted(np.cos(np.deg2rad(annual.latitude))).mean(
        ("latitude", "longitude")
    )
    annual = annual - global_mean
    annual = annual.assign_coords(longitude=annual.longitude % 360).sortby("longitude")

    west, east, south, north = config.eof_domain
    north_pacific = annual.sel(
        longitude=slice(west, east), latitude=slice(south, north)
    )
    weights = np.sqrt(np.cos(np.deg2rad(north_pacific.latitude.values)))[:, None]
    solver = Eof(north_pacific, weights=weights, center=True)
    annual_pc = config.pdo_sign * solver.pcs(npcs=1, pcscaling=1).values[:, 0]

    offsets = np.arange(config.filter_taps) - (config.filter_taps - 1) / 2
    normalized_cutoff = config.filter_cutoff / (config.filter_design_fs / 2)
    lanczos_window = np.sinc(offsets / ((config.filter_taps - 1) / 2))
    coefficients = (
        normalized_cutoff
        * np.sinc(normalized_cutoff * offsets)
        * lanczos_window
    )
    coefficients /= coefficients.sum()
    low_pass_pc = filtfilt(coefficients, [1.0], annual_pc)
    years = pd.DatetimeIndex(north_pacific.time.values).year
    return pd.Series(low_pass_pc, index=years, name="pdo_low")


def contiguous_true_periods(mask: pd.Series) -> list[tuple[int, int]]:
    """Return inclusive year ranges for contiguous True values."""
    mask = validate_year_index(mask.astype(bool))
    periods: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for year, selected in mask.items():
        year = int(year)
        if selected and start is None:
            start = year
        if start is not None and previous is not None and year != previous + 1:
            periods.append((start, previous))
            start = year if selected else None
        if start is not None and not selected:
            periods.append((start, previous if previous is not None else year - 1))
            start = None
        previous = year
    if start is not None and previous is not None:
        periods.append((start, previous))
    return periods


def select_pdo_transition_periods(
    pdo: pd.Series,
    threshold: float,
    minimum_transition_years: int,
) -> tuple[tuple[int, int], ...]:
    """Return sustained near-zero intervals before the separate Now stage."""
    if threshold <= 0:
        raise ValueError("The PDO transition threshold must be positive.")
    if minimum_transition_years < 1:
        raise ValueError("PDO transition duration must be at least one year.")
    pdo = validate_year_index(pdo.astype(float))
    candidates = tuple(
        (start, end)
        for start, end in contiguous_true_periods(pdo.abs().le(threshold))
        if end - start + 1 >= minimum_transition_years
    )
    if not candidates:
        raise ValueError(
            "No sustained near-zero PDO transition satisfies the duration rule."
        )
    return candidates


def derive_pdo_stage_definition(
    pdo: pd.Series,
    *,
    now_start: int = PDO_NOW_START_YEAR,
    threshold_quantile: float = PDO_TRANSITION_QUANTILE,
    minimum_transition_years: int = PDO_MIN_TRANSITION_YEARS,
) -> PDOStageDefinition:
    """Derive sustained pre-Now near-zero transitions from the low-pass PDO."""
    if not 0 < threshold_quantile < 0.5:
        raise ValueError("PDO transition quantile must lie between 0 and 0.5.")
    historical = validate_year_index(pdo.astype(float)).loc[: now_start - 1]
    threshold = float(historical.abs().quantile(threshold_quantile))
    transition_periods = select_pdo_transition_periods(
        historical,
        threshold=threshold,
        minimum_transition_years=minimum_transition_years,
    )
    return PDOStageDefinition(
        transition_periods=transition_periods,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        minimum_transition_years=minimum_transition_years,
        now_start=now_start,
    )


def classify_pdo_stages(
    years: pd.Index | np.ndarray,
    definition: PDOStageDefinition,
) -> pd.Series:
    """Classify years as +PDO, transition, -PDO, or the separate Now stage."""
    years = np.asarray(years, dtype=int)
    labels = np.full(years.size, "negative", dtype=object)
    labels[years < definition.transition_periods[0][0]] = "positive"
    for start, end in definition.transition_periods:
        labels[(years >= start) & (years <= end)] = "transition"
    labels[years >= definition.now_start] = "now"
    return pd.Series(labels, index=years, name="phase")


def bandpass_interannual(series: pd.Series) -> pd.Series:
    """Return the author's 3--8-year Butterworth component."""
    nyquist = 0.5
    low_cut = (1.0 / 8.0) / nyquist
    high_cut = (1.0 / 3.0) / nyquist
    numerator, denominator = butter(4, [low_cut, high_cut], btype="band")
    values = filtfilt(numerator, denominator, series.to_numpy(dtype=float))
    return pd.Series(values, index=series.index, name="onset_bandpass")


def prepare_variability(onset: pd.Series) -> pd.DataFrame:
    """Prepare standardized onset and standardized interannual variability."""
    standardized = (onset - onset.mean()) / onset.std(ddof=1)
    signed_difference = standardized.diff()
    running_standard_deviation = standardized.rolling(
        WINDOW, center=True, min_periods=WINDOW
    ).std(ddof=1)
    return pd.DataFrame(
        {
            "onset": onset,
            "standardized_onset": standardized,
            "signed_difference": signed_difference,
            "absolute_difference": signed_difference.abs(),
            "running_standard_deviation": running_standard_deviation,
        }
    ).rename_axis("year")


def variance_ratios(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Calculate every complete subsequent/preceding sample-variance ratio."""
    values = np.asarray(values, dtype=float)
    if values.shape[-1] < 2 * window:
        raise ValueError("The series is too short for two complete windows.")
    windows = sliding_window_view(values, window, axis=-1)
    variances = np.var(windows - windows[..., :1], axis=-1, ddof=1)
    preceding = variances[..., :-window]
    subsequent = variances[..., window:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = subsequent / preceding
    return np.where((subsequent == 0) & (preceding == 0), 1.0, ratios)


def scan_f_test(standardized_onset: pd.Series) -> pd.DataFrame:
    """Compare adjacent 11-year samples of standardized onset at every split."""
    values = standardized_onset.to_numpy(dtype=float)
    years = standardized_onset.index.to_numpy(dtype=int)
    ratios = variance_ratios(values)
    split_positions = np.arange(WINDOW, len(values) - WINDOW + 1)
    rows = []
    for ratio, split in zip(ratios, split_positions):
        rows.append(
            {
                "display_year": int(years[split - 1]),
                "subsequent_start_year": int(years[split]),
                "preceding_years": f"{years[split - WINDOW]}--{years[split - 1]}",
                "subsequent_years": f"{years[split]}--{years[split + WINDOW - 1]}",
                "F": float(ratio),
                "nominal_upper_p": float(f.sf(ratio, WINDOW - 1, WINDOW - 1)),
            }
        )
    return pd.DataFrame(rows)


def stationary_bootstrap_indices(
    length: int,
    repetitions: int,
    mean_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate circular stationary-bootstrap indices with geometric blocks."""
    indices = np.empty((repetitions, length), dtype=np.int64)
    indices[:, 0] = rng.integers(length, size=repetitions)
    restart_probability = 1.0 / mean_block_length
    for position in range(1, length):
        restart = rng.random(repetitions) < restart_probability
        continued = (indices[:, position - 1] + 1) % length
        new_start = rng.integers(length, size=repetitions)
        indices[:, position] = np.where(restart, new_start, continued)
    return indices


def bootstrap_scan_correction(
    values: np.ndarray,
    observed_f: np.ndarray,
    repetitions: int,
    seed: int,
    batch_size: int = 2_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate upper-tail p values under dependence and a complete scan."""
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, MEAN_BLOCK_LENGTH, len(values)])
    )
    point_exceedances = np.zeros(len(observed_f), dtype=np.int64)
    scan_exceedances = np.zeros(len(observed_f), dtype=np.int64)
    null_maxima = np.empty(repetitions, dtype=float)
    completed = 0
    while completed < repetitions:
        batch = min(batch_size, repetitions - completed)
        indices = stationary_bootstrap_indices(
            len(values), batch, MEAN_BLOCK_LENGTH, rng
        )
        null_f = variance_ratios(values[indices])
        maxima = np.max(null_f, axis=1)
        null_maxima[completed : completed + batch] = maxima
        point_exceedances += np.sum(
            null_f >= observed_f[None, :] - 1e-12, axis=0
        )
        scan_exceedances += np.sum(
            maxima[:, None] >= observed_f[None, :] - 1e-12, axis=0
        )
        completed += batch
    point_p = (point_exceedances + 1) / (repetitions + 1)
    scan_p = (scan_exceedances + 1) / (repetitions + 1)
    critical = float(np.quantile(null_maxima, 1 - ALPHA, method="higher"))
    return point_p, scan_p, critical


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Return the conventional biased-denominator lag-1 sample ACF."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    return float(np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered))


def run_f_test(prepared: pd.DataFrame, config: AnalysisConfig) -> FTestResult:
    """Run the onset-variance F scan plus dependence and multiplicity correction."""
    standardized_onset = prepared["standardized_onset"]
    scan = scan_f_test(standardized_onset)
    point_p, scan_p, scan_critical = bootstrap_scan_correction(
        standardized_onset.to_numpy(),
        scan["F"].to_numpy(),
        repetitions=config.f_resamples,
        seed=config.f_seed,
    )
    scan["bootstrap_point_p"] = point_p
    scan["bootstrap_scan_p"] = scan_p
    return FTestResult(
        scan=scan,
        pointwise_critical=float(f.ppf(1 - ALPHA, WINDOW - 1, WINDOW - 1)),
        scan_critical=scan_critical,
        lag1_acf=lag1_autocorrelation(standardized_onset.to_numpy()),
    )


def fit_interaction(
    data: pd.DataFrame, hac_lags: int
) -> tuple[object, pd.DataFrame]:
    """Fit onset ~ LST + PDO + LST*PDO + time with HAC uncertainty."""
    predictors = data[["lst", "pdo_low"]]
    standardized = (predictors - predictors.mean()) / predictors.std(ddof=1)
    design = pd.DataFrame(
        {
            "constant": 1.0,
            "lst": standardized["lst"],
            "pdo": standardized["pdo_low"],
            "lst_x_pdo": standardized["lst"] * standardized["pdo_low"],
            "time_decades": (data.index - data.index.to_numpy().mean()) / 10,
        },
        index=data.index,
    )
    fit = sm_api.OLS(data["scssm_onset"], design).fit(
        cov_type="HAC",
        use_t=True,
        cov_kwds={"maxlags": hac_lags, "use_correction": True},
    )
    confidence = fit.conf_int()
    table = pd.DataFrame(
        {
            "estimate": fit.params,
            "se_hac": fit.bse,
            "p_hac": fit.pvalues,
            "ci95_low": confidence[0],
            "ci95_high": confidence[1],
        }
    )
    return fit, table


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson r, or NaN for an unidentifiable correlation."""
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_phase_correlation_difference(
    data: pd.DataFrame,
    phase_summary: pd.DataFrame,
    block_length: int,
    repetitions: int,
    seed: int,
    minimum_phase_n: int = 5,
) -> dict[str, float | int | str]:
    """Bootstrap r(+PDO)-r(-PDO) using paired moving blocks."""
    n = len(data)
    starts = np.random.default_rng(seed + block_length).integers(
        0,
        n - block_length + 1,
        size=(repetitions, int(np.ceil(n / block_length))),
    )
    indices = (
        starts[:, :, None] + np.arange(block_length)
    ).reshape(repetitions, -1)[:, :n]
    predictor = data["lst"].to_numpy()
    onset = data["scssm_onset"].to_numpy()
    phases = data["phase"].to_numpy()
    boot_correlations = []
    for label in ("positive", "negative"):
        mask = phases[indices] == label
        counts = mask.sum(axis=1)
        predictor_sample = predictor[indices]
        onset_sample = onset[indices]
        safe_counts = np.maximum(counts, 1)
        predictor_centered = predictor_sample - (
            (predictor_sample * mask).sum(axis=1) / safe_counts
        )[:, None]
        onset_centered = onset_sample - (
            (onset_sample * mask).sum(axis=1) / safe_counts
        )[:, None]
        numerator = (predictor_centered * onset_centered * mask).sum(axis=1)
        denominator = np.sqrt(
            (predictor_centered**2 * mask).sum(axis=1)
            * (onset_centered**2 * mask).sum(axis=1)
        )
        valid = (counts >= minimum_phase_n) & (denominator > 1e-12)
        values = np.full(repetitions, np.nan)
        np.divide(numerator, denominator, out=values, where=valid)
        boot_correlations.append(values)
    observed = (
        phase_summary.loc["positive", "r"]
        - phase_summary.loc["negative", "r"]
    )
    bootstrap = boot_correlations[0] - boot_correlations[1]
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    usable = (
        len(bootstrap) >= 0.9 * repetitions
        and phase_summary.loc[["positive", "negative"], "n"].min()
        >= minimum_phase_n
        and np.isfinite(observed)
    )
    if usable:
        low, high = 2 * observed - np.quantile(bootstrap, [0.975, 0.025])
        p_value = (
            1
            + np.count_nonzero(np.abs(bootstrap - observed) >= abs(observed))
        ) / (len(bootstrap) + 1)
    else:
        low = high = p_value = np.nan
    return {
        "block_years": block_length,
        "delta_r": observed,
        "ci95_low": low,
        "ci95_high": high,
        "p_bootstrap_approx": p_value,
        "valid_repeats": len(bootstrap),
        "total_repeats": repetitions,
        "status": "ok" if usable else "insufficient_phase_samples",
    }


def summarize_pdo_stages(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize all displayed stages; correlations use +PDO and -PDO only."""
    rows = []
    for label in ("positive", "transition", "negative", "now"):
        group = data.loc[data["phase"] == label]
        periods = contiguous_true_periods(
            pd.Series(True, index=group.index, dtype=bool)
        ) if len(group) else []
        rows.append(
            {
                "phase": label,
                "n": len(group),
                "periods": "; ".join(
                    f"{start}--{end}" for start, end in periods
                ),
                "onset_mean": group["scssm_onset"].mean(),
                "onset_sd": group["scssm_onset"].std(ddof=1),
                "r": correlation(
                    group["lst"].to_numpy(), group["scssm_onset"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("phase")


def run_pdo_moderation(
    factors: pd.DataFrame, pdo: pd.Series, config: AnalysisConfig
) -> dict[str, object]:
    """Run the core statistical checks formerly contained in pdo_corr.py."""
    data = factors[["lst", "scssm_onset"]].join(pdo, how="inner").copy()
    expected = np.arange(config.onset_start_year, config.pdo_end_year + 1)
    if not np.array_equal(data.index.to_numpy(), expected):
        raise ValueError("PDO moderation data must be continuous from 1979--2024.")
    stage_definition = derive_pdo_stage_definition(pdo)
    data["phase"] = classify_pdo_stages(data.index, stage_definition)
    phase_summary = summarize_pdo_stages(data)
    pre_now = data.loc[data.index < stage_definition.now_start]
    _, interaction = fit_interaction(pre_now, hac_lags=3)
    correlation_tests = pd.DataFrame(
        [
            bootstrap_phase_correlation_difference(
                data,
                phase_summary,
                block_length=block,
                repetitions=config.moderation_resamples,
                seed=config.moderation_seed,
            )
            for block in (3, 5, 7)
        ]
    )
    threshold_sensitivity_rows = []
    for quantile in PDO_TRANSITION_SENSITIVITY_QUANTILES:
        sensitivity_definition = derive_pdo_stage_definition(
            pdo,
            threshold_quantile=quantile,
        )
        sensitivity_data = data.copy()
        sensitivity_data["phase"] = classify_pdo_stages(
            sensitivity_data.index,
            sensitivity_definition,
        )
        sensitivity_summary = summarize_pdo_stages(sensitivity_data)
        row = bootstrap_phase_correlation_difference(
            sensitivity_data,
            sensitivity_summary,
            block_length=5,
            repetitions=config.moderation_resamples,
            seed=config.moderation_seed,
        )
        row.update(
            {
                "threshold_quantile": quantile,
                "threshold": sensitivity_definition.threshold,
                "transition_periods": "; ".join(
                    f"{start}--{end}"
                    for start, end in sensitivity_definition.transition_periods
                ),
            }
        )
        threshold_sensitivity_rows.append(row)
    sensitivity_rows = []
    for lag in (1, 3, 5):
        _, table = fit_interaction(data, hac_lags=lag)
        row = table.loc["lst_x_pdo"].to_dict()
        row.update(
            {
                "analysis": "include_now",
                "hac_lags": lag,
                "n": len(data),
            }
        )
        sensitivity_rows.append(row)
    interior = data.iloc[config.filter_taps - 1 : -(config.filter_taps - 1)]
    if len(interior) >= 15:
        _, table = fit_interaction(interior, hac_lags=3)
        row = table.loc["lst_x_pdo"].to_dict()
        row.update(
            {
                "analysis": "exclude_10_years_each_edge",
                "hac_lags": 3,
                "n": len(interior),
            }
        )
        sensitivity_rows.append(row)
    regime_only = data.loc[data["phase"].isin(["positive", "negative"])]
    _, table = fit_interaction(regime_only, hac_lags=3)
    row = table.loc["lst_x_pdo"].to_dict()
    row.update(
        {
            "analysis": "exclude_transition_and_now",
            "hac_lags": 3,
            "n": len(regime_only),
        }
    )
    sensitivity_rows.append(row)
    return {
        "data": data,
        "stage_definition": stage_definition,
        "phase_summary": phase_summary,
        "interaction": interaction,
        "correlation_tests": correlation_tests,
        "threshold_sensitivity": pd.DataFrame(threshold_sensitivity_rows),
        "sensitivity": pd.DataFrame(sensitivity_rows),
    }


def build_display_stage_spans(
    start_year: int,
    end_year: int,
    definition: PDOStageDefinition,
) -> list[tuple[str, int, int]]:
    """Return the contiguous PDO stages shown by the panel-a background."""
    spans: list[tuple[str, int, int]] = []
    phase_start = start_year
    phase_label = "PDO(+)"
    for transition_start, transition_end in definition.transition_periods:
        if transition_start > phase_start:
            spans.append((phase_label, phase_start, transition_start - 1))
        spans.append(("PDO(T)", transition_start, transition_end))
        phase_start = transition_end + 1
        phase_label = "PDO(-)"
    if phase_start <= end_year:
        spans.append((phase_label, phase_start, end_year))
    return spans


def fit_onset_segments(onset: pd.Series) -> list[dict[str, object]]:
    """Fit the three fixed onset segments used in pdo_fig_v1.py."""
    segments = []
    for start, end in ((1979, 2001), (2001, 2016), (2016, 2025)):
        observed = onset.loc[start:end]
        regression = linregress(np.arange(len(observed)), observed.to_numpy())
        fitted = pd.Series(
            regression.intercept + regression.slope * np.arange(len(observed)),
            index=observed.index,
        )
        segments.append(
            {
                "start": start,
                "end": end,
                "fitted": fitted,
                "mean": fitted.mean(),
                "slope": regression.slope,
                "p": regression.pvalue,
            }
        )
    return segments


def load_error_matrix(
    onset: pd.Series, config: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Load three prediction series and return their annual absolute errors."""
    years = np.arange(config.onset_start_year, config.onset_end_year + 1)
    sm_prediction = pd.read_csv(config.sm_file).squeeze("columns")
    if len(sm_prediction) != len(years):
        raise ValueError("SM-17 predictions must contain 47 values for 1979--2025.")
    sm_prediction.index = years
    xg_table = pd.read_csv(config.xg_file)
    xg_prediction = (
        xg_table.loc[
            xg_table["Year"].between(1988, config.onset_end_year),
            ["Year", "Prediction"],
        ]
        .set_index("Year")["Prediction"]
    )
    rf_prediction = pd.Series(RF_PREDICTIONS, index=np.arange(2016, 2026))
    observed = onset.reindex(years)
    errors = np.vstack(
        [
            (xg_prediction.reindex(years) - observed).abs().to_numpy(),
            (rf_prediction.reindex(years) - observed).abs().to_numpy(),
            (sm_prediction.reindex(years) - observed).abs().to_numpy(),
        ]
    )
    return years, errors


def significance_stars(p_value: float) -> str:
    """Use one star for 90%, two for 99%, and three for 99.9%."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_2(value: float) -> str:
    formatted = f"{value:.2f}"
    return "0" if formatted == "0.00" else formatted


def configure_figure_style() -> None:
    mpl.rcParams.update(
        {
            "axes.linewidth": 2,
            "font.size": 16,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
        }
    )


def plot_final_figure(
    pdo: pd.Series,
    stage_definition: PDOStageDefinition,
    prepared: pd.DataFrame,
    onset_bandpass: pd.Series,
    segments: list[dict[str, object]],
    f_result: FTestResult,
    error_years: np.ndarray,
    error_matrix: np.ndarray,
    output_file: Path,
    show: bool,
) -> None:
    """Create and save the final three-panel v1 figure exactly once."""
    configure_figure_style()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1, 0.8], "hspace": 0.1},
    )
    x_limits = (1978, 2026)
    ax_a = axes[0]
    ax_a_right = ax_a.twinx()
    phase_colors = np.where(pdo.to_numpy() > 0, "#fa930d", "#1278f4")
    ax_a.bar(
        pdo.index, pdo.to_numpy(), width=1, color=phase_colors,
        edgecolor="#2D2C2D", linewidth=1, align="center", alpha=0.8, zorder=2,
    )
    bandpass_color = plt.cm.tab20.colors[14]
    (bandpass_line,) = ax_a.plot(
        onset_bandpass.index, onset_bandpass, color=bandpass_color,
        linewidth=3, alpha=0.8,
    )
    ax_a.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_a.set_ylim(-2.3, 2.3)
    ax_a.set_ylabel("Standardized anomaly", fontsize=14.8)
    ax_a.yaxis.set_label_coords(-0.048, 0.5)
    (onset_line,) = ax_a_right.plot(
        prepared.index, prepared["onset"], color="#111011",
        linewidth=3.5, alpha=0.8,
    )
    text_offsets = [(-8.5, -7.95), (-4.5, -6.65), (0.0, -8.155)]
    for segment, (x_offset, y_offset) in zip(segments, text_offsets):
        fitted = segment["fitted"]
        ax_a_right.plot(
            fitted.index, fitted, color="#2B2C30", linewidth=3, alpha=0.8
        )
        ax_a_right.hlines(
            segment["mean"], fitted.index[0], fitted.index[-1],
            colors="#2B2C30", linestyles="--", linewidth=3, alpha=0.8, zorder=10,
        )
        ax_a_right.text(
            fitted.index[-10] + x_offset,
            fitted.iloc[-10] + y_offset,
            f"Mean={format_2(segment['mean'])}; "
            f"Slope={format_2(segment['slope'])}{significance_stars(segment['p'])}",
            color="#2B2C30", fontsize=12, va="center", alpha=0.8,
        )
    ax_a_right.set_yticks(np.arange(20, 37, 5))
    ax_a_right.set_ylim(20, 36)
    ax_a_right.set_ylabel("Onset pentad", fontsize=14.8)
    ax_a_right.yaxis.set_label_coords(1.05, 0.5)
    for axis, line in ((ax_a, bandpass_line), (ax_a_right, onset_line)):
        axis.tick_params(axis="y", which="both", colors=line.get_color())
        axis.yaxis.label.set_color(line.get_color())
    stage_colors = {"PDO(+)": "#fa930d", "PDO(T)": "gray", "PDO(-)": "#1278f4"}
    for stage, start, end in build_display_stage_spans(
        int(pdo.index.min()), int(prepared.index.max()), stage_definition
    ):
        ax_a.axvspan(
            start - 0.5, end + 0.5, color=stage_colors[stage], alpha=0.08, zorder=0
        )
    legend_elements = [
        Line2D([0], [0], color="#fa930d", lw=6, label="PDO(+)"),
        Line2D([0], [0], color="gray", lw=6, label="PDO(T)"),
        Line2D([0], [0], color="#1278f4", lw=6, label="PDO(-)"),
        Line2D([0], [0], color="#2B2C30", lw=4, label="Obs"),
        Line2D([0], [0], color="#2B2C30", lw=2, label="Trend"),
        Line2D([0], [0], color=bandpass_color, lw=2, label="3-8-yr bandpass"),
        Line2D([0], [0], color="#2B2C30", lw=2, ls="--", label="Mean"),
    ]
    ax_a.legend(
        handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 1.02),
        fontsize=10, frameon=False, ncol=7,
    )
    ax_a.set_xlim(*x_limits)
    ax_a.text(
        -0.088, 1.05, "a", transform=ax_a.transAxes, fontsize=22,
        fontweight="bold", va="top", ha="right",
    )

    ax_b = axes[1]
    ax_b_right = ax_b.twinx()
    variability_color = "#1365A9"
    absolute_difference = prepared["absolute_difference"].dropna()
    bars = ax_b.bar(
        absolute_difference.index, absolute_difference, color=variability_color,
        edgecolor="#2D2C2D", width=1, alpha=0.7,
    )
    (standard_deviation_line,) = ax_b.plot(
        prepared.index, prepared["running_standard_deviation"],
        color=variability_color, linewidth=3,alpha=0.7
    )
    ax_b.set_ylim(0, absolute_difference.max() * 1.2)
    ax_b.set_yticks(np.arange(0, absolute_difference.max() * 1.2, 1))
    ax_b.set_ylabel("Standardized variability", fontsize=14.8)
    ax_b.yaxis.set_label_coords(-0.048, 0.5)
    scan = f_result.scan
    (f_line,) = ax_b_right.plot(
        scan["display_year"], scan["F"], color="#dc4c36", linewidth=3, alpha=0.8
    )
    significance_line = ax_b_right.axhline(
        f_result.pointwise_critical, color="#dc4c36", linestyle="--",
        linewidth=2, alpha=0.5,
    )
    ax_b_right.set_ylim(0, scan["F"].max() * 1.2)
    ax_b_right.set_ylabel(r"$F$ statistic", fontsize=14.8)
    ax_b_right.yaxis.set_label_coords(1.05, 0.5)
    for axis, color in ((ax_b, variability_color), (ax_b_right, f_line.get_color())):
        axis.tick_params(axis="y", which="both", colors=color)
        axis.yaxis.label.set_color(color)
    ax_b.set_xlim(*x_limits)
    ax_b.text(
        -0.088, 1.05, "b", transform=ax_b.transAxes, fontsize=22,
        fontweight="bold", va="top", ha="right",
    )
    ax_b.legend(
        [bars,  standard_deviation_line, f_line, significance_line],
        [r"$|\Delta z|$",  "11-yr SD", "F-test", "90%"],
        loc="upper right", fontsize=10, frameon=False, ncol=5,
    )

    ax_c = axes[2]
    extent = [error_years[0] - 0.5, error_years[-1] + 0.5, 0, len(error_matrix)]
    error_boundaries = np.arange(0, 5.0 + 0.1, 0.1)
    error_cmap = mpl.colormaps["OrRd"].resampled(len(error_boundaries) - 1)
    error_norm = mpl.colors.BoundaryNorm(
        error_boundaries, ncolors=error_cmap.N, clip=True
    )
    image = ax_c.imshow(
        error_matrix, aspect="auto", cmap=error_cmap, norm=error_norm,
        extent=extent, origin="lower",
    )
    ax_c.set_yticks(np.arange(len(error_matrix)) + 0.5)
    ax_c.set_yticklabels(["AMME", "Hu-25", "SM-17"], fontsize=14)
    ax_c.set_ylim(0, len(error_matrix))
    ax_c.set_xlim(*x_limits)
    ax_c.set_xlabel("Year", fontsize=14.8)
    ax_c.set_xticks(np.arange(x_limits[0] + 1, x_limits[1] + 1, 5))
    ax_c.text(
        -0.088, 1.05, "c", transform=ax_c.transAxes, fontsize=22,
        fontweight="bold", va="top", ha="right",
    )
    colorbar_axis = figure.add_axes([0.152, 0.150, 0.088, 0.015])
    colorbar = figure.colorbar(
        image, cax=colorbar_axis, orientation="horizontal",
        boundaries=error_boundaries,
    )
    colorbar.set_ticks(np.arange(0, 6, 1))
    colorbar.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    colorbar.ax.tick_params(
        axis="x", which="major", labelsize=12, length=5, width=1.2
    )
    colorbar.ax.tick_params(axis="x", which="minor", length=2.5, width=0.8)
    colorbar.set_label("AE (pentads)", fontsize=12, labelpad=3)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(figure)


def report_f_test(prepared: pd.DataFrame, result: FTestResult, config: AnalysisConfig) -> None:
    """Print every definition and value needed for the F-test reviewer response."""
    difference = prepared["absolute_difference"].dropna()
    peak = result.peak
    significant = result.scan.loc[result.scan["bootstrap_scan_p"] <= ALPHA]
    print("\nSliding F-test audit")
    print(
        f"F-test source: standardized annual SCSSM onset, "
        f"{prepared.index[0]}--{prepared.index[-1]}, n={len(prepared)}"
    )
    print(
        f"Absolute differences: d_t=abs(z_t-z_(t-1)), n={len(difference)}, "
        f"years {difference.index[0]}--{difference.index[-1]}"
    )
    print(
        "F = variance(subsequent 11 standardized onset dates) / "
        "variance(preceding 11 standardized onset dates); df=(10, 10)"
    )
    print(
        f"Complete overlapping comparisons={len(result.scan)}; "
        "displayed at the final year of each preceding sample"
    )
    print(f"Upper-tail alpha={ALPHA:.2f}; pointwise critical F={result.pointwise_critical:.6f}")
    print(f"Lag-1 ACF of standardized onset={result.lag1_acf:.6f}")
    print(
        f"Stationary bootstrap of standardized onset: expected block length={MEAN_BLOCK_LENGTH}, "
        f"replicates={config.f_resamples}, seed={config.f_seed}"
    )
    print(
        "Repeated-test correction: maximum F over all overlapping positions "
        f"in every replicate; scan-wide critical F={result.scan_critical:.6f}"
    )
    print(
        f"Peak displayed year={int(peak['display_year'])}; "
        f"windows {peak['preceding_years']} vs {peak['subsequent_years']}"
    )
    print(
        f"Peak F={peak['F']:.6f}; nominal p={peak['nominal_upper_p']:.6f}; "
        f"bootstrap pointwise p={peak['bootstrap_point_p']:.6f}; "
        f"scan-corrected p={peak['bootstrap_scan_p']:.6f}"
    )
    print(
        "Scan-corrected significant displayed years (90%): "
        + ", ".join(str(int(year)) for year in significant["display_year"])
    )
    print(
        "Blue curve: centred 11-year sample standard deviation of z_t, "
        "S_T=sqrt(sum[(z_t-mean_T)^2]/10), t=T-5,...,T+5; "
        "dimensionless; valid years "
        f"{prepared['running_standard_deviation'].first_valid_index()}--"
        f"{prepared['running_standard_deviation'].last_valid_index()}"
    )


def report_pdo_moderation(result: dict[str, object]) -> None:
    """Print the PDO moderation statistics without creating a second figure."""
    stage = result["stage_definition"]
    print("\nPDO-LST moderation audit")
    print(
        "PDO stage definition: before the separate Now stage, tau is the "
        f"{100 * stage.threshold_quantile:.0f}th percentile of |low-pass PDO| "
        f"(tau={stage.threshold:.6f}); PDO(T) comprises contiguous near-zero "
        "intervals lasting at least "
        f"{stage.minimum_transition_years} years."
    )
    transition_text = ", ".join(
        f"{start}--{end}" for start, end in stage.transition_periods
    )
    print(
        f"Panel-a sequence: +PDO, PDO(T), -PDO, PDO(T), -PDO; "
        f"PDO(T)={transition_text}. For moderation summaries, "
        f"Now={stage.now_start} onward is retained as a separate stage and "
        "excluded from the +PDO versus -PDO correlation comparison."
    )
    print("Phase summary:")
    print(result["phase_summary"].to_string())
    print("\nPre-Now continuous LST x PDO interaction (1979--2019; HAC lags=3):")
    print(result["interaction"].loc[["lst_x_pdo"]].to_string())
    print("\nPhase-correlation difference sensitivity:")
    print(result["correlation_tests"].to_string(index=False))
    print("\nTransition-threshold sensitivity (paired block length=5 years):")
    print(result["threshold_sensitivity"].to_string(index=False))
    print("\nHAC and filter-edge sensitivity:")
    print(result["sensitivity"].to_string(index=False))


def validate_core_calculations() -> None:
    """Run quick deterministic checks for F indexing, blocks, and PDO stages."""
    toy = pd.Series(
        np.r_[np.arange(1.0, 12.0), 2 * np.arange(1.0, 12.0)],
        index=np.arange(2001, 2023),
    )
    toy_scan = scan_f_test(toy)
    assert len(toy_scan) == 1
    assert toy_scan.loc[0, "display_year"] == 2011
    assert toy_scan.loc[0, "preceding_years"] == "2001--2011"
    assert toy_scan.loc[0, "subsequent_years"] == "2012--2022"
    assert np.isclose(toy_scan.loc[0, "F"], 4.0)
    assert variance_ratios(np.full(22, 0.37))[0] == 1.0
    indices = stationary_bootstrap_indices(
        12, 100, 10**12, np.random.default_rng(4)
    )
    assert np.all(np.diff(indices, axis=1) % 12 == 1)
    toy_pdo = pd.Series(
        np.r_[np.repeat(0.5, 5), [0.10, 0.05, -0.05, -0.10], np.repeat(-0.5, 5)],
        index=np.arange(2000, 2014),
    )
    assert select_pdo_transition_periods(
        toy_pdo,
        threshold=0.15,
        minimum_transition_years=3,
    ) == ((2005, 2008),)
    toy_definition = PDOStageDefinition(
        transition_periods=((2005, 2008),),
        threshold=0.15,
        threshold_quantile=0.25,
        minimum_transition_years=3,
        now_start=2013,
    )
    toy_stages = classify_pdo_stages(np.arange(2000, 2014), toy_definition)
    assert toy_stages.loc[2004] == "positive"
    assert toy_stages.loc[2005] == "transition"
    assert toy_stages.loc[2009] == "negative"
    assert toy_stages.loc[2013] == "now"
    toy_years = np.arange(1979, 2026)
    toy_onset = pd.Series(25.0 + 0.2 * (toy_years - 1979), index=toy_years)
    toy_segments = fit_onset_segments(toy_onset)
    assert [(item["start"], item["end"]) for item in toy_segments] == [
        (1979, 2001), (2001, 2016), (2016, 2025)
    ]
    assert np.allclose([item["slope"] for item in toy_segments], 0.2)


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-show", action="store_true")
    # VS Code/Jupyter kernels add their own command-line arguments.  Ignoring
    # those kernel arguments keeps the script runnable in an Interactive Window,
    # while normal terminal execution still accepts this script's CLI options.
    if argv is None and ("ipykernel" in sys.modules or hasattr(sys, "ps1")):
        argv = []
    args = parser.parse_args(argv)
    if args.resamples < 9_999:
        parser.error("Use at least 9,999 F-test bootstrap replicates.")
    config = AnalysisConfig(f_resamples=args.resamples, f_seed=args.seed)
    validate_core_calculations()
    factors = load_factors(config)
    onset = factors["scssm_onset"].astype(float)
    pdo = load_low_pass_pdo(config)
    prepared = prepare_variability(onset)
    onset_bandpass = bandpass_interannual(prepared["standardized_onset"])
    f_result = run_f_test(prepared, config)
    moderation_result = run_pdo_moderation(factors, pdo, config)
    segments = fit_onset_segments(onset)
    error_years, error_matrix = load_error_matrix(onset, config)
    plot_final_figure(
        pdo, moderation_result["stage_definition"], prepared,
        onset_bandpass, segments, f_result,
        error_years, error_matrix, output_file=config.output_file,
        show=not args.no_show,
    )
    print("\nPanel-a fixed-period onset trends:")
    for segment in segments:
        print(
            f"{segment['start']}--{segment['end']}: "
            f"mean={segment['mean']:.6f}, slope={segment['slope']:.6f}, "
            f"p={segment['p']:.6f}"
        )
    report_f_test(prepared, f_result, config)
    report_pdo_moderation(moderation_result)
    print(f"\nSaved final figure: {config.output_file}")


if __name__ == "__main__":
    main()

# %%
