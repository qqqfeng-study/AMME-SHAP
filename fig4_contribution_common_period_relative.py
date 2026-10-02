# %%
"""Explain the fixed seed676 MAE + N=8 forecasts for 1988--2025.

The selected ``ModelIndex`` comes from the fixed-selection CSV and is resolved
to a rebuilt ``ModelID`` through the reconstruction report. The 2016--2025
values displayed as predictions come from the held-out test CSV. Feature
attributions use exact interventional TreeSHAP with the same standardized
1979--2015 background for every selected model. A single 2016-anchor baseline
``B*`` is used for reporting, while each model's baseline shift is retained as
a separate non-physical term so that local additivity is not altered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SELECTION_FILE = (
    PROJECT_ROOT
    / "add_sensitive"
    / "output"
    / "fixed_mae_n8"
    / "seed676_MAE_N8_1988_2025.csv"
)
HELDOUT_FILE = (
    PROJECT_ROOT
    / "add_sensitive"
    / "output"
    / "heldout_train_test"
    / "test_2016_2025.csv"
)
MODEL_DIR = PROJECT_ROOT / "DATA" / "rebuild_seed676" / "models"
MODEL_MANIFEST_FILE = (
    PROJECT_ROOT / "DATA" / "rebuild_seed676" / "rebuild_seed676_report.csv"
)
DATA_FILE = PROJECT_ROOT / "precursor_factors_1979_2025.csv"
FIG_DIR = SCRIPT_DIR / "fig"
FIVE_YEAR_UNCERTAINTY_FILE = (
    PROJECT_ROOT
    / "shap_effect"
    / "output"
    / "five_year_common_background"
    / "five_year_common_background_source_data.csv"
)
ANNUAL_COMMON_BACKGROUND_FILE = (
    PROJECT_ROOT / "shap_effect" / "output" / "main_attributions.csv"
)
FIXED_CANDIDATE_ATTRIBUTIONS_FILE = (
    PROJECT_ROOT / "shap_effect" / "output" / "fixed_candidate_attributions.csv"
)

START_YEAR = 1988
END_YEAR = 2025
HELDOUT_START_YEAR = 2016
TRAIN_START = 1979
TRAIN_END = 2015
PREDICTOR_COLUMNS = ["stt", "enso", "sam", "ao_mar", "lst"]
FEATURE_NAMES = ["STT", "ENSO", "SAM", "AO", "LST"]
FIVE_YEAR_FEATURE_ORDER = ["STT", "ENSO", "SAM", "AO", "LST"]
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
EXPECTED_ANCHOR_MODELS = 20
COMMON_BASELINE_YEAR = 2016
ADDITIVITY_TOLERANCE = 5e-5
LONG_PERIODS = ((1988, 2009), (2010, 2025))
# Width (inches on the rendered canvas) that panel a's bars are held at. Panel a
# is aligned to panel b's tick grid, which fixes its plot scale, so its bar
# footprint is solved from this target rather than being a fixed data-unit value.
PANEL_A_BAR_WIDTH_IN = 0.609


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


def load_model_manifest(path: Path = MODEL_MANIFEST_FILE) -> pd.DataFrame:
    """Load the authoritative ModelIndex-to-ModelID reconstruction mapping."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rebuilt-model report not found: {path}")

    manifest = pd.read_csv(path)
    required = {"BaseSeed", "ModelIndex", "ModelID", "Reproduced", "PklFile"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"model report is missing columns: {missing_columns}")

    for column in ("BaseSeed", "ModelIndex", "ModelID"):
        manifest[column] = pd.to_numeric(
            manifest[column], errors="raise"
        ).astype(int)
    if not manifest["BaseSeed"].eq(676).all():
        raise ValueError("model report contains a BaseSeed other than 676")

    if manifest["ModelIndex"].duplicated().any():
        duplicates = sorted(
            manifest.loc[
                manifest["ModelIndex"].duplicated(keep=False), "ModelIndex"
            ].unique()
        )
        raise ValueError(f"duplicate ModelIndex values in model report: {duplicates}")

    reproduced = manifest["Reproduced"]
    if reproduced.dtype != bool:
        reproduced = reproduced.astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    if reproduced.isna().any():
        raise ValueError("model report contains invalid Reproduced values")
    manifest["Reproduced"] = reproduced.astype(bool)
    return manifest.sort_values("ModelIndex").reset_index(drop=True)


def load_heldout_predictions(path: Path = HELDOUT_FILE) -> pd.DataFrame:
    """Load the canonical MAE + N=8 predictions for 2016--2025."""

    path = Path(path)
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
    manifest_path: Path = MODEL_MANIFEST_FILE,
) -> pd.DataFrame:
    """Load the fixed selection and reconcile its two prediction sources."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAE + N=8 selection CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {
        "Year",
        "ModelIndex",
        "ModelID",
        "Prediction",
        "Observed",
        "Error",
        "AbsError",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"selection CSV is missing columns: {missing_columns}")

    for column in ("Year", "ModelIndex", "ModelID"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("Prediction", "Observed", "Error", "AbsError"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)

    frame = (
        frame.loc[frame["Year"].between(START_YEAR, END_YEAR)]
        .sort_values("Year")
        .reset_index(drop=True)
    )
    expected_years = list(range(START_YEAR, END_YEAR + 1))
    if frame["Year"].tolist() != expected_years:
        missing_years = sorted(set(expected_years) - set(frame["Year"]))
        duplicate_years = sorted(
            frame.loc[frame["Year"].duplicated(keep=False), "Year"].unique()
        )
        raise ValueError(
            "selection CSV must contain exactly one row for every year "
            f"from {START_YEAR} to {END_YEAR}; "
            f"missing={missing_years}, duplicates={duplicate_years}"
        )
    manifest = load_model_manifest(manifest_path)
    model_id_by_index = manifest.set_index("ModelIndex")["ModelID"]
    mapped_model_ids = frame["ModelIndex"].map(model_id_by_index)
    if mapped_model_ids.isna().any():
        missing_indices = frame.loc[mapped_model_ids.isna(), "ModelIndex"].tolist()
        raise ValueError(
            f"ModelIndex values missing from model report: {missing_indices}"
        )
    mapped_model_ids = mapped_model_ids.astype(int)
    if not np.array_equal(frame["ModelID"].to_numpy(), mapped_model_ids.to_numpy()):
        raise ValueError("selection CSV ModelID does not match the model report")
    frame["ModelID"] = mapped_model_ids

    failed = frame.loc[
        ~frame["ModelIndex"].map(
            manifest.set_index("ModelIndex")["Reproduced"]
        ),
        "ModelIndex",
    ].tolist()
    if failed:
        raise ValueError(f"selected models were not successfully rebuilt: {failed}")

    frame["ModelPrediction"] = frame["Prediction"]
    heldout = load_heldout_predictions(heldout_path).set_index("Year")
    fixed_test = frame.set_index("Year").loc[HELDOUT_START_YEAR:END_YEAR]
    if not np.array_equal(
        fixed_test["Observed"].to_numpy(), heldout["Observed"].to_numpy()
    ):
        raise ValueError("fixed-selection and held-out observations do not match")
    if not np.allclose(
        fixed_test["Prediction"].to_numpy(),
        heldout["Prediction"].to_numpy(),
        rtol=0.0,
        atol=5e-4,
    ):
        raise ValueError("fixed-selection and held-out predictions disagree")

    heldout_rows = frame["Year"].between(HELDOUT_START_YEAR, END_YEAR)
    for column in ("Prediction", "Observed", "Error", "AbsError"):
        frame.loc[heldout_rows, column] = frame.loc[heldout_rows, "Year"].map(
            heldout[column]
        )
    return frame


def build_model_paths(
    selection: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    manifest_path: Path = MODEL_MANIFEST_FILE,
) -> list[Path]:
    """Resolve model files from ModelIndex through the reconstruction report."""

    model_dir = Path(model_dir)
    manifest = load_model_manifest(manifest_path)
    manifest_by_index = manifest.set_index("ModelIndex")
    model_rows = manifest_by_index.reindex(selection["ModelIndex"].to_numpy())
    if model_rows["ModelID"].isna().any():
        missing_indices = selection.loc[
            model_rows["ModelID"].isna().to_numpy(), "ModelIndex"
        ].tolist()
        raise ValueError(
            f"ModelIndex values missing from model report: {missing_indices}"
        )
    model_ids = model_rows["ModelID"].astype(int).reset_index(drop=True)
    if "ModelID" in selection and not np.array_equal(
        selection["ModelID"].to_numpy(dtype=int), model_ids.to_numpy()
    ):
        raise ValueError("selection ModelID does not match ModelIndex mapping")

    paths = []
    for model_id, pkl_file in zip(model_ids, model_rows["PklFile"]):
        filename = PurePosixPath(str(pkl_file).replace("\\", "/")).name
        expected_filename = f"bst_model_seed676_model{int(model_id)}.pkl"
        if filename != expected_filename:
            raise ValueError(
                "model report filename does not match its ModelID: "
                f"{filename!r} != {expected_filename!r}"
            )
        paths.append(model_dir / filename)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} selected rebuilt model files are missing:\n{preview}"
        )
    return paths


def load_standardized_data(
    path: Path = DATA_FILE,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Load predictors using the same 1979--2015 scaling as the new models."""

    path = Path(path)
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
        model = joblib.load(model_path)
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


def validate_annual_adaptive_reference(
    results: list[AnnualShapResult],
    feature_names: list[str],
    path: Path = ANNUAL_COMMON_BACKGROUND_FILE,
) -> None:
    """Require Figure 4 values to match the published adaptive trajectory."""

    required = {
        "scenario",
        "year",
        "feature",
        "shap_value",
        "common_baseline",
    }
    reference = pd.read_csv(path)
    missing_columns = sorted(required - set(reference.columns))
    if missing_columns:
        raise ValueError(
            f"annual common-background table is missing columns: {missing_columns}"
        )
    reference = reference.loc[
        reference["scenario"].eq("annual_switched"),
        ["year", "feature", "shap_value", "common_baseline"],
    ].copy()
    if reference.duplicated(["year", "feature"]).any():
        raise ValueError("adaptive annual reference has duplicate year-feature rows")

    expected = reference.pivot(
        index="year", columns="feature", values="shap_value"
    ).sort_index()
    actual = pd.DataFrame(
        {
            result.year: dict(zip(feature_names, result.shap_values))
            for result in results
        }
    ).T.sort_index()
    actual.index.name = "year"
    actual = actual.reindex(columns=feature_names)
    expected = expected.reindex(index=actual.index, columns=feature_names)
    if expected.isna().any().any():
        raise ValueError("adaptive annual reference is incomplete for Figure 4")

    differences = np.abs(
        actual.to_numpy(dtype=float) - expected.to_numpy(dtype=float)
    )
    maximum_difference = float(np.max(differences, initial=0.0))
    if maximum_difference > 5e-8:
        location = np.unravel_index(np.argmax(differences), differences.shape)
        year = int(actual.index[location[0]])
        feature = str(actual.columns[location[1]])
        raise ValueError(
            "Figure 4 does not match the common-background adaptive trajectory; "
            f"max difference={maximum_difference:.3e} at {year}, {feature}"
        )

    reference_baselines = pd.to_numeric(
        reference["common_baseline"], errors="raise"
    ).to_numpy(dtype=float)
    result_baselines = np.asarray(
        [result.common_baseline for result in results], dtype=float
    )
    if not np.allclose(
        reference_baselines,
        result_baselines[0],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Figure 4 and its adaptive reference use different common baselines"
        )


def plot_signed_lines(df_signed: pd.DataFrame, output_path: Path) -> None:
    """Plot annual SHAP contributions in the original compact six-row style."""

    plt = _plotting_modules()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.085, top=0.98, hspace=0.1)
    years = df_signed.index

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
            -0.148,
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
            ax.legend(
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
                ],
                loc="upper right",
                fontsize=5.5,
                frameon=False,
                ncol=2,
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
        bbox_to_anchor=(0.638, 1.18),
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
        -0.148,
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
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.show()


def mean_absolute_shap_by_periods(
    df_signed: pd.DataFrame,
    periods: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    """Average absolute annual SHAP contributions over inclusive periods."""

    rows = []
    for start_year, end_year in periods:
        if start_year > end_year:
            raise ValueError(f"invalid period: {start_year}-{end_year}")
        window = df_signed.loc[start_year:end_year, FEATURE_NAMES]
        expected_years = list(range(start_year, end_year + 1))
        if window.index.astype(int).tolist() != expected_years:
            raise ValueError(
                f"SHAP table does not fully cover {start_year}-{end_year}"
            )
        values = window.abs().mean()
        values["year_range"] = f"{start_year}-{end_year}"
        rows.append(values)
    result = pd.DataFrame(rows).set_index("year_range")[FIVE_YEAR_FEATURE_ORDER]
    result.attrs.update(df_signed.attrs)
    return result


def relative_contribution_percent(df_mean_abs: pd.DataFrame) -> pd.DataFrame:
    """Normalize each period's mean absolute SHAP contributions to 100%."""

    values = df_mean_abs.astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("mean absolute SHAP values must be finite and non-negative")
    totals = values.sum(axis=1)
    if (totals <= 0).any():
        invalid_periods = totals.index[totals <= 0].astype(str).tolist()
        raise ValueError(
            "relative contributions require a positive total for every period: "
            f"{invalid_periods}"
        )
    result = values.div(totals, axis=0).mul(100.0)
    if not np.allclose(
        result.sum(axis=1).to_numpy(dtype=float),
        100.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("relative SHAP contributions do not sum to 100%")
    result.attrs.update(df_mean_abs.attrs)
    return result


def mean_annual_relative_contribution_by_periods(
    df_signed: pd.DataFrame,
    periods: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    """Average annual relative absolute SHAP contributions over each period."""

    annual_relative = relative_contribution_percent(
        df_signed.loc[:, FIVE_YEAR_FEATURE_ORDER].abs()
    )
    rows = []
    for start_year, end_year in periods:
        if start_year > end_year:
            raise ValueError(f"invalid period: {start_year}-{end_year}")
        window = annual_relative.loc[start_year:end_year, FIVE_YEAR_FEATURE_ORDER]
        expected_years = list(range(start_year, end_year + 1))
        if window.index.astype(int).tolist() != expected_years:
            raise ValueError(
                f"annual relative SHAP table does not fully cover "
                f"{start_year}-{end_year}"
            )
        values = window.mean(axis=0)
        values["year_range"] = f"{start_year}-{end_year}"
        rows.append(values)
    result = pd.DataFrame(rows).set_index("year_range")[FIVE_YEAR_FEATURE_ORDER]
    result.attrs.update(df_signed.attrs)
    if not np.allclose(
        result.sum(axis=1).to_numpy(dtype=float),
        100.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("period means of annual relative SHAP do not sum to 100%")
    return result


def five_year_mean_absolute_shap(df_signed: pd.DataFrame) -> pd.DataFrame:
    periods = tuple(
        (start_year, min(start_year + 4, END_YEAR))
        for start_year in range(START_YEAR, END_YEAR + 1, 5)
    )
    return mean_absolute_shap_by_periods(df_signed, periods)


def load_five_year_uncertainty(
    path: Path = FIVE_YEAR_UNCERTAINTY_FILE,
) -> pd.DataFrame:
    """Load validated model-choice variability from the fixed 2016 anchor."""

    required = {
        "feature",
        "period_start",
        "period_end",
        "candidate_n_models",
        "candidate_q025_mean_abs_shap",
        "candidate_q25_mean_abs_shap",
        "candidate_median_mean_abs_shap",
        "candidate_q75_mean_abs_shap",
        "candidate_q975_mean_abs_shap",
        "adaptive_mean_abs_shap",
        "adaptive_common_baseline",
    }
    table = pd.read_csv(path)
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise ValueError(
            f"five-year uncertainty table is missing columns: {missing_columns}"
        )

    table = table.loc[table["feature"].isin(FIVE_YEAR_FEATURE_ORDER)].copy()
    table["year_range"] = (
        table["period_start"].astype(int).astype(str)
        + "-"
        + table["period_end"].astype(int).astype(str)
    )
    if table.duplicated(["year_range", "feature"]).any():
        raise ValueError("five-year uncertainty table has duplicate period-feature rows")
    if not table["candidate_n_models"].astype(int).eq(EXPECTED_ANCHOR_MODELS).all():
        raise ValueError(
            f"each uncertainty box must contain {EXPECTED_ANCHOR_MODELS} models"
        )

    adaptive_baselines = pd.to_numeric(
        table["adaptive_common_baseline"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(adaptive_baselines).all():
        raise ValueError("adaptive common baselines must be finite")
    if not np.allclose(
        adaptive_baselines,
        adaptive_baselines[0],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("uncertainty table does not use one common baseline")

    quantile_columns = [
        "candidate_q025_mean_abs_shap",
        "candidate_q25_mean_abs_shap",
        "candidate_median_mean_abs_shap",
        "candidate_q75_mean_abs_shap",
        "candidate_q975_mean_abs_shap",
    ]
    quantiles = table[quantile_columns].to_numpy(dtype=float)
    if not np.isfinite(quantiles).all() or (quantiles < 0).any():
        raise ValueError("five-year uncertainty quantiles must be finite and non-negative")
    if (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("five-year uncertainty quantiles are not monotonically ordered")

    return table.set_index(["year_range", "feature"]).sort_index()


def summarize_candidate_relative_uncertainty_by_periods(
    periods: tuple[tuple[int, int], ...],
    path: Path = FIXED_CANDIDATE_ATTRIBUTIONS_FILE,
) -> pd.DataFrame:
    """Summarize variability after annual normalization and period averaging."""

    required = {
        "scenario",
        "anchor_year",
        "year",
        "model_index",
        "feature",
        "abs_shap",
        "common_baseline",
    }
    table = pd.read_csv(path)
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise ValueError(
            "fixed-candidate attribution table is missing columns: "
            f"{missing_columns}"
        )
    table = table.loc[
        table["scenario"].eq("fixed_candidate_ensemble_2016")
        & table["anchor_year"].eq(COMMON_BASELINE_YEAR)
        & table["feature"].isin(FIVE_YEAR_FEATURE_ORDER)
    ].copy()
    for column in ("anchor_year", "year", "model_index"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(int)
    for column in ("abs_shap", "common_baseline"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
    if table.duplicated(["year", "model_index", "feature"]).any():
        raise ValueError("fixed-candidate attribution rows are not unique")
    if table["model_index"].nunique() != EXPECTED_ANCHOR_MODELS:
        raise ValueError(
            f"expected {EXPECTED_ANCHOR_MODELS} fixed 2016-anchor models"
        )
    baselines = table["common_baseline"].to_numpy(dtype=float)
    if not np.isfinite(baselines).all() or not np.allclose(
        baselines, baselines[0], rtol=0.0, atol=1e-10
    ):
        raise ValueError("fixed candidates do not share one common baseline")

    records = []
    for start_year, end_year in periods:
        period = table.loc[table["year"].between(start_year, end_year)].copy()
        expected_years = set(range(start_year, end_year + 1))
        if set(period["year"].unique()) != expected_years:
            raise ValueError(
                f"fixed-candidate table does not fully cover {start_year}-{end_year}"
            )
        annual_abs = period.pivot(
            index=["model_index", "year"],
            columns="feature",
            values="abs_shap",
        ).reindex(columns=FIVE_YEAR_FEATURE_ORDER)
        if annual_abs.isna().any().any():
            raise ValueError(
                f"{start_year}-{end_year} candidate model-years do not contain "
                "all features"
            )
        annual_relative = relative_contribution_percent(annual_abs)
        relative_per_model = annual_relative.groupby(level="model_index").mean()
        if not np.allclose(
            relative_per_model.sum(axis=1).to_numpy(dtype=float),
            100.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"{start_year}-{end_year} candidate period means do not sum to 100%"
            )
        label = f"{start_year}-{end_year}"
        for feature in FIVE_YEAR_FEATURE_ORDER:
            values = relative_per_model[feature].to_numpy(dtype=float)
            if values.size != EXPECTED_ANCHOR_MODELS:
                raise ValueError(
                    f"{label}, {feature} does not contain "
                    f"{EXPECTED_ANCHOR_MODELS} candidate models"
                )
            q025, q25, median, q75, q975 = np.quantile(
                values, [0.025, 0.25, 0.5, 0.75, 0.975]
            )
            records.append(
                {
                    "year_range": label,
                    "feature": feature,
                    "candidate_n_models": EXPECTED_ANCHOR_MODELS,
                    "candidate_q025_relative_percent": q025,
                    "candidate_q25_relative_percent": q25,
                    "candidate_median_relative_percent": median,
                    "candidate_q75_relative_percent": q75,
                    "candidate_q975_relative_percent": q975,
                    "adaptive_common_baseline": float(baselines[0]),
                }
            )
    return pd.DataFrame.from_records(records).set_index(
        ["year_range", "feature"]
    ).sort_index()


def plot_five_year_bars(
    df_mean_5yr: pd.DataFrame,
    uncertainty: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot five-year SHAP bars in the original wide grouped-bar style."""

    plt = _plotting_modules()
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["font.size"] = 14
    mpl.rcParams["xtick.major.width"] = 2
    mpl.rcParams["ytick.major.width"] = 2

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=300)
    n_groups = len(df_mean_5yr.index)
    n_bars = len(df_mean_5yr.columns)
    x = np.arange(n_groups)
    bar_width = 0.7 / n_bars
    morandi_colors = plt.cm.tab20.colors
    # 因子配色固定，与 running_corr_v2.py 保持一致（配色不随排序变化）：
    # ENSO=tab20[0], STT=tab20[1], SAM=tab20[2], LST=tab20[3], AO=tab20[4]
    feature_colors = {
        "ENSO": morandi_colors[0],
        "STT": morandi_colors[1],
        "SAM": morandi_colors[2],
        "LST": morandi_colors[3],
        "AO": morandi_colors[4],
    }
    expected_index = pd.MultiIndex.from_product(
        [df_mean_5yr.index.astype(str), df_mean_5yr.columns.astype(str)],
        names=["year_range", "feature"],
    )
    missing_rows = expected_index.difference(uncertainty.index)
    if len(missing_rows):
        raise ValueError(
            "five-year uncertainty table is missing period-feature rows: "
            f"{missing_rows.tolist()}"
        )

    plotted_baseline = float(df_mean_5yr.attrs["common_baseline"])
    uncertainty_baseline = float(
        uncertainty["adaptive_common_baseline"].iloc[0]
    )
    if not np.isclose(
        plotted_baseline,
        uncertainty_baseline,
        rtol=0.0,
        atol=5e-8,
    ):
        raise ValueError(
            "Figure 5 bars and uncertainty boxes use different common baselines"
        )

    box_edge_color = "#868088"
    box_width = bar_width * 0.58
    cap_width = box_width * 0.72

    for i, column in enumerate(df_mean_5yr.columns):
        centers = x + i * bar_width
        ax.bar(
            centers,
            df_mean_5yr[column],
            width=bar_width,
            label=column,
            color=feature_colors[column],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=2,
        )
        for center, period in zip(centers, df_mean_5yr.index.astype(str)):
            row = uncertainty.loc[(period, column)]
            bar_value = float(df_mean_5yr.loc[period, column])
            reference_value = float(row["adaptive_mean_abs_shap"])
            if not np.isclose(
                bar_value,
                reference_value,
                rtol=0.0,
                atol=5e-8,
            ):
                raise ValueError(
                    "Figure 5 bar does not match the common-background adaptive "
                    f"value for {period}, {column}: "
                    f"{bar_value} != {reference_value}"
                )
            q025 = float(row["candidate_q025_mean_abs_shap"])
            q25 = float(row["candidate_q25_mean_abs_shap"])
            median = float(row["candidate_median_mean_abs_shap"])
            q75 = float(row["candidate_q75_mean_abs_shap"])
            q975 = float(row["candidate_q975_mean_abs_shap"])
            ax.vlines(
                center, q025, q975, color=box_edge_color, linewidth=1.0, zorder=4
            )
            ax.hlines(
                [q025, q975],
                center - cap_width / 2,
                center + cap_width / 2,
                color=box_edge_color,
                linewidth=1.0,
                zorder=4,
            )
            ax.add_patch(
                Rectangle(
                    (center - box_width / 2, q25),
                    box_width,
                    q75 - q25,
                    facecolor="white",
                    edgecolor=box_edge_color,
                    linewidth=1.0,
                    alpha=0.82,
                    zorder=5,
                )
            )
            ax.hlines(
                median,
                center - box_width / 2,
                center + box_width / 2,
                color=box_edge_color,
                linewidth=1.4,
                zorder=6,
            )

    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    period_labels = [label.replace("-", "\N{EN DASH}") for label in df_mean_5yr.index]
    period_labels[-1] += "*"
    ax.set_xticklabels(period_labels, rotation=0)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Mean absolute SHAP\ncontribution (pentads)")
    maximum_whisker = float(
        uncertainty.loc[
            expected_index, "candidate_q975_mean_abs_shap"
        ].max()
    )
    y_max = max(2.0, np.ceil(maximum_whisker * 2) / 2)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time Segments")
    ax.set_yticks(np.arange(0, y_max + 0.1, 0.5))
    feature_legend = ax.legend(
        loc="upper right",
        borderaxespad=0.0,
        fontsize=14,
        frameon=False,
        ncol=5,
    )
    ax.add_artist(feature_legend)
    uncertainty_handle = Line2D(
        [0],
        [0],
        color=box_edge_color,
        linewidth=1.0,
        marker="s",
        markerfacecolor="white",
        markeredgecolor=box_edge_color,
        markersize=7,
        label="2016-anchor ensemble",
    )
    ax.legend(
        handles=[uncertainty_handle],
        loc="upper left",
        borderaxespad=0.0,
        fontsize=10,
        frameon=False,
    )
    # fig.text(
    #     0.01,
    #     0.012,
    #     (
    #         "Boxes: median and IQR; whiskers: 2.5th\N{EN DASH}97.5th percentiles "
    #         f"across {EXPECTED_ANCHOR_MODELS} fixed 2016-anchor models "
    #         "(model-choice variability, not sampling uncertainty).\n"
    #         f"All values use the {TRAIN_START}\N{EN DASH}{TRAIN_END} background "
    #         f"and common $B^*$ = {plotted_baseline:.2f} pentads; "
    #         "*2023\N{EN DASH}2025 spans 3 years."
    #     ),
    #     fontsize=7.5,
    #     ha="left",
    #     va="bottom",
    # )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_two_period_bars(
    df_mean_long: pd.DataFrame,
    long_uncertainty: pd.DataFrame,
    df_mean_5yr: pd.DataFrame,
    five_year_uncertainty: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot broad-period and five-year relative attribution summaries."""

    plt = _plotting_modules()
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["font.size"] = 14
    mpl.rcParams["xtick.major.width"] = 2
    mpl.rcParams["ytick.major.width"] = 2

    # Layout geometry is declared up front because the panel-a alignment below
    # needs to know the width the axes will finally occupy on the canvas.
    figure_width, figure_height = 12.0, 6.8
    panel_left, panel_right = 0.09, 0.99
    axes_bottom, axes_top = 0.07, 0.98
    panel_gap = 0.15

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, figure_height),
        dpi=300,
        sharey=True,
        gridspec_kw={"hspace": panel_gap},
    )
    morandi_colors = plt.cm.tab20.colors
    feature_colors = {
        "ENSO": morandi_colors[0],
        "STT": morandi_colors[1],
        "SAM": morandi_colors[2],
        "LST": morandi_colors[3],
        "AO": morandi_colors[4],
    }
    box_edge_color = "#868088"

    def draw_panel(
        ax,
        means: pd.DataFrame,
        uncertainty: pd.DataFrame,
        panel_label: str,
        *,
        mark_partial_last: bool,
        show_feature_labels: bool,
        show_xlabel: bool,
        group_footprint: float = 0.7,
    ) -> float:
        n_groups = len(means.index)
        n_bars = len(means.columns)
        x = np.arange(n_groups)
        # ``group_footprint`` is the x-extent one cluster of five bars occupies.
        # Splitting it across the bars is what keeps the bars flush against one
        # another: a narrower footprint narrows every bar in the cluster without
        # ever opening a gap inside it.
        bar_width = group_footprint / n_bars
        box_width = bar_width * 0.58
        cap_width = box_width * 0.72
        expected_index = pd.MultiIndex.from_product(
            [means.index.astype(str), means.columns.astype(str)],
            names=["year_range", "feature"],
        )
        missing_rows = expected_index.difference(uncertainty.index)
        if len(missing_rows):
            raise ValueError(
                f"panel {panel_label} uncertainty is missing rows: "
                f"{missing_rows.tolist()}"
            )
        plotted_baseline = float(means.attrs["common_baseline"])
        uncertainty_baseline = float(
            uncertainty.loc[expected_index, "adaptive_common_baseline"].iloc[0]
        )
        if not np.isclose(
            plotted_baseline,
            uncertainty_baseline,
            rtol=0.0,
            atol=5e-8,
        ):
            raise ValueError(
                f"panel {panel_label} bars and boxes use different baselines"
            )

        for i, column in enumerate(means.columns):
            centers = x + i * bar_width
            ax.bar(
                centers,
                means[column],
                width=bar_width,
                label=column if show_feature_labels else None,
                color=feature_colors[column],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.7,
                zorder=2,
            )
            for center, period in zip(centers, means.index.astype(str)):
                row = uncertainty.loc[(period, column)]
                q025 = float(row["candidate_q025_relative_percent"])
                q25 = float(row["candidate_q25_relative_percent"])
                median = float(row["candidate_median_relative_percent"])
                q75 = float(row["candidate_q75_relative_percent"])
                q975 = float(row["candidate_q975_relative_percent"])
                ax.vlines(
                    center,
                    q025,
                    q975,
                    color=box_edge_color,
                    linewidth=1.0,
                    zorder=4,
                )
                ax.hlines(
                    [q025, q975],
                    center - cap_width / 2,
                    center + cap_width / 2,
                    color=box_edge_color,
                    linewidth=1.0,
                    zorder=4,
                )
                ax.add_patch(
                    mpl.patches.Rectangle(
                        (center - box_width / 2, q25),
                        box_width,
                        q75 - q25,
                        facecolor="white",
                        edgecolor=box_edge_color,
                        linewidth=1.0,
                        alpha=0.82,
                        zorder=5,
                    )
                )
                ax.hlines(
                    median,
                    center - box_width / 2,
                    center + box_width / 2,
                    color=box_edge_color,
                    linewidth=1.4,
                    zorder=6,
                )

        ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
        labels = [label.replace("-", "\N{EN DASH}") for label in means.index]
        if mark_partial_last:
            labels[-1] += "*"
        ax.set_xticklabels(labels, rotation=0)
        ax.axhline(0, color="black", linewidth=0.9)
        if show_xlabel:
            ax.set_xlabel("Time segments")
        ax.text(
            -0.1,
            1.02,
            panel_label,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="bottom",
            ha="right",
            clip_on=False,
        )
        return float(
            uncertainty.loc[
                expected_index, "candidate_q975_relative_percent"
            ].max()
        )

    # Panel b is drawn first: its five-year tick grid is the ruler panel a is
    # then aligned to, so the two panels read against one shared x axis.
    five_year_maximum = draw_panel(
        axes[1],
        df_mean_5yr,
        five_year_uncertainty,
        "b",
        mark_partial_last=True,
        show_feature_labels=False,
        show_xlabel=True,
    )

    # Pull panel a's two clusters towards the middle by pinning each of them over
    # the midpoint of the two panel-b bins that straddle that period's own time
    # centre: 1988-2009 (centre 1998.5) between the 1993-1997 and 1998-2002
    # ticks, 2010-2025 (centre 2017.5) between the 2013-2017 and 2018-2022
    # ticks. The two midpoints sit symmetrically about the panel centre, so the
    # clusters stay centred while the 3.7 in hole between them closes.
    b_ticks = axes[1].get_xticks()
    b_low, b_high = axes[1].get_xlim()
    alignment_targets = (
        (b_ticks[1] + b_ticks[2]) / 2.0,
        (b_ticks[5] + b_ticks[6]) / 2.0,
    )
    alignment_fractions = tuple(
        (value - b_low) / (b_high - b_low) for value in alignment_targets
    )
    # Panel a keeps its clusters exactly one data unit apart, so those two
    # fractions determine its x limits; those limits in turn fix its plot scale,
    # and the bar footprint is solved from that scale so the bars keep the width
    # they were slimmed to.
    panel_a_span = 1.0 / (alignment_fractions[1] - alignment_fractions[0])
    panel_width_in = (panel_right - panel_left) * figure_width
    panel_a_bar_width = (
        PANEL_A_BAR_WIDTH_IN * panel_a_span / panel_width_in
    )
    panel_a_x_low = 2.0 * panel_a_bar_width - alignment_fractions[0] * panel_a_span

    long_maximum = draw_panel(
        axes[0],
        df_mean_long,
        long_uncertainty,
        "a",
        mark_partial_last=False,
        show_feature_labels=True,
        show_xlabel=False,
        # Panel a carries only two clusters, so an unscaled footprint would leave
        # each of its bars far wider than panel b's. The footprint is derived
        # from the alignment above so the five bars of a cluster stay flush
        # against one another while keeping PANEL_A_BAR_WIDTH_IN.
        group_footprint=panel_a_bar_width * len(df_mean_long.columns),
    )
    axes[0].set_xlim(panel_a_x_low, panel_a_x_low + panel_a_span)
    y_max = max(40.0, np.ceil(max(long_maximum, five_year_maximum) / 10) * 10)
    for ax in axes:
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 0.1, 10))

    feature_legend = axes[0].legend(
        loc="upper right",
        borderaxespad=0.0,
        fontsize=14,
        frameon=False,
        ncol=5,
    )
    axes[0].add_artist(feature_legend)
    # The "2016-anchor ensemble" swatch that used to sit in the upper-left
    # corner of panel a has been dropped: the white boxes are already
    # introduced in the caption, so the extra key only duplicated it while
    # leaving an empty hole in the sparse two-period panel.
    fig.supylabel(
        "Relative mean absolute SHAP contribution (%)",
        # Pulled in from the figure edge so the label sits closer to the tick
        # numbers; the panel letters occupy the outermost strip.
        x=0.035,
        fontsize=14,
    )
    fig.subplots_adjust(
        left=panel_left,
        right=panel_right,
        bottom=axes_bottom,
        top=axes_top,
        hspace=panel_gap,
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.show()


def main() -> int:
    # Check the selected Python plotting runtime before loading 38 models.
    _plotting_modules()
    selection = load_mae_n8_selection()
    model_paths = build_model_paths(selection)
    standardized, feature_names, _ = load_standardized_data()
    results = calculate_shap_results(selection, model_paths, standardized)
    validate_annual_adaptive_reference(results, feature_names)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    signed = build_signed_table(results, selection, feature_names)
    five_year_periods = tuple(
        (start_year, min(start_year + 4, END_YEAR))
        for start_year in range(START_YEAR, END_YEAR + 1, 5)
    )
    long_relative = mean_annual_relative_contribution_by_periods(
        signed,
        LONG_PERIODS,
    )
    five_year_relative = mean_annual_relative_contribution_by_periods(
        signed,
        five_year_periods,
    )
    plot_two_period_bars(
        long_relative,
        summarize_candidate_relative_uncertainty_by_periods(LONG_PERIODS),
        five_year_relative,
        summarize_candidate_relative_uncertainty_by_periods(five_year_periods),
        FIG_DIR / "figure4_relative_shap_by_period.png",
    )
    return 0


if __name__ == "__main__":
    main()

# %%
