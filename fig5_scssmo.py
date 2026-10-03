from __future__ import annotations

import argparse
from pathlib import Path

import cmaps
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Ellipse

BASE_DIR = Path(__file__).resolve().parent
ONSET_CSV = BASE_DIR / "precursor_factors_1979_2025.csv"
U850_FILE = (
    BASE_DIR.parent / "DATA" / "DATA" / "ERA5_native_SCSSM" / "U_daliy"
    / "ERA5_ecmwf_U_Y1940-2025_M04-06_daily.nc"
)
OUTPUT_FILE = BASE_DIR / "fig" / "scssm_onset_pentad_ERA5_u850.png"
YEARS = np.arange(1979, 2026)
PLOT_DATES = pd.date_range("2025-04-26", "2025-06-30", freq="D")


def build_colormap():
    cmap_design = mpl.colormaps.get_cmap(cmaps.BlueWhiteOrangeRed)
    warm = mcolors.LinearSegmentedColormap.from_list(
        "warm", cmap_design(np.linspace(0.53, 0.9, 256))[:250], N=255
    )
    cold = mcolors.LinearSegmentedColormap.from_list(
        "cold", cmap_design(np.linspace(0.1, 0.47, 256))[:250], N=255
    )
    colors = np.vstack((cold(np.linspace(0, 1, 256)), warm(np.linspace(0, 1, 256))))
    return mcolors.LinearSegmentedColormap.from_list("cold_warm", colors)


cold_warm = build_colormap()


def load_onset_values(path: Path = ONSET_CSV) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_numeric(frame.index).astype(int)
    if not frame.index.is_unique:
        raise ValueError("Observation years must be unique")
    if "scssm_onset" not in frame.columns:
        raise ValueError("The observation CSV must contain scssm_onset")
    scssm_onset_values = pd.to_numeric(frame["scssm_onset"]).reindex(YEARS)
    if not np.isfinite(scssm_onset_values.to_numpy(dtype=float)).all():
        raise ValueError("Onset observations must be available for every year from 1979 to 2025")
    return scssm_onset_values


def load_wind(path: Path = U850_FILE) -> tuple[xr.DataArray, pd.DataFrame]:
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        u850 = dataset["u"].sel(
            valid_time=slice("1979-01-01", "2025-06-30"),
            latitude=slice(15, 5),
            longitude=slice(110, 120),
        )
        dates = pd.DatetimeIndex(u850["valid_time"].values)
        mask = (
            ((dates.month == 4) & (dates.day >= 26))
            | (dates.month == 5)
            | (dates.month == 6)
        )
        u850 = u850.isel(valid_time=np.flatnonzero(mask)).load()
    dates = pd.DatetimeIndex(u850["valid_time"].values)
    scssm = u850.mean(dim=("latitude", "longitude"))
    daily = pd.DataFrame({
        "Year": dates.year,
        "Date": dates.strftime("%m-%d"),
        "u850": scssm.to_numpy(),
    })
    scssm_each_year = daily.pivot(index="Year", columns="Date", values="u850")
    scssm_each_year = scssm_each_year.reindex(
        index=YEARS, columns=PLOT_DATES.strftime("%m-%d")
    )
    if not np.isfinite(scssm_each_year.to_numpy(dtype=float)).all():
        raise ValueError("Daily U850 must cover April 26 to June 30 in every year from 1979 to 2025")
    return u850, scssm_each_year


def plot_scssm_onset(
    scssm_each_year: pd.DataFrame,
    scssm_onset_values: pd.Series,
    output_path: Path = OUTPUT_FILE,
    *,
    show: bool = True,
) -> None:
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=600)
    years = scssm_each_year.index.to_numpy(dtype=int)
    days_idx = np.arange(scssm_each_year.shape[1])
    c1 = ax1.contourf(years, days_idx, scssm_each_year.T, levels=np.arange(-12, 13, 1), cmap=cold_warm, extend='both')
    ax1.contour(years, days_idx, scssm_each_year.T, levels=[0], colors="#B9B8B8", linewidths=1, linestyles='--')
    class TripleEllipseLegendHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            x = width / 2
            y = height / 2
            e1 = Ellipse((x, y), width, height,
                         facecolor=cold_warm(0.15), edgecolor='black', linewidth=0.5, transform=trans)
            e2 = Ellipse((x, y), width * 0.66, height * 0.66,
                         facecolor=cold_warm(0.5), edgecolor='black', linewidth=0.5, transform=trans,ls ='--')
            e3 = Ellipse((x, y), width * 0.33, height * 0.33,
                         facecolor=cold_warm(0.85), edgecolor='black', linewidth=0.5, transform=trans,ls ='--')
            return [e1, e2, e3]
    u850_legend = Ellipse((0, 0), 1, 1, facecolor='white', edgecolor='black', label='U850')
    cax = fig.add_axes([0.955, 0.12, 0.015, 0.6])
    cb = fig.colorbar(c1, cax=cax, aspect=20, shrink=0.8, extendrect='both', drawedges=True, extendfrac='auto')
    cb.ax.set_ylabel(r'Zonal wind (m s$^{-1}$)', fontsize=14)
    cb.ax.tick_params(labelsize=14)
    left_ticks = np.arange(0, 66, 5)
    dates = pd.date_range(start='2025-04-26', end='2025-06-30', freq='5D')
    day_labels = dates.strftime('%b %d').tolist()
    ax1.set_xlabel('Year', fontsize=16)
    ax1.set_ylabel('Date', fontsize=16)
    ax1.set_yticks(left_ticks)
    ax1.set_yticklabels(day_labels, fontsize=14)
    ax1.set_ylim(0, days_idx[-1])
    ax1.tick_params(axis='y', labelsize=14)
    ax2 = ax1.twinx()
    right_ticks = np.arange(2.5, 63.5, 5)
    right_labels = np.arange(24, 37)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel('Pentad', fontsize=16)
    ax2.set_yticks(right_ticks)
    ax2.set_yticklabels(right_labels, fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    era5_mapped = np.interp(scssm_onset_values, right_labels, right_ticks)
    line_onset = ax2.plot(years, era5_mapped, label='onset', marker='o', c="#424040", lw=3, markersize=8, alpha=0.8)[0]
    ax1.set_xticks(np.arange(1979, 2026, 3))
    ax1.set_xticklabels(np.arange(1979, 2026, 3), fontsize=14)
    ax1.set_xlim(1979, 2025)
    ax1.grid(True, linestyle='--', alpha=0.25, lw=1, color="#8D8787")
    legend_elements = [u850_legend, line_onset]
    ax1.legend(handles=legend_elements,
               handler_map={Ellipse: TripleEllipseLegendHandler()},
               loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1.02))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=600)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot U850 and observed SCSSM onset pentads.")
    parser.add_argument("--u850", type=Path, default=U850_FILE)
    parser.add_argument("--observations", type=Path, default=ONSET_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    scssm_onset_values = load_onset_values(args.observations)
    u850, scssm_each_year = load_wind(args.u850)
    plot_scssm_onset(
        scssm_each_year,
        scssm_onset_values,
        args.output,
        show=not args.no_show,
    )
    print(f"scssm_each_year: {scssm_each_year.shape}")
    print(f"Saved: {args.output}")
    return u850, scssm_each_year, scssm_onset_values


if __name__ == "__main__":
    u850, scssm_each_year, scssm_onset_values = main()
