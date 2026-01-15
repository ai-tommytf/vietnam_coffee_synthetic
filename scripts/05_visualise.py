"""
Step 8: Create Production Plots

JTBD: Generate publication-quality visualisations for the conference booth.

Plots:
1. Time series: Precipitation, temperature vs climatology
2. Anomaly maps: 2024 drought severity by province
3. Index summary: GDD, CDD, extreme heat trends

Output: /Users/tommylees/github/vietnam_coffee_synthetic/artefacts/weather_risk/
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Configuration
AGG_PATH = Path(
    "/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr"
)
CLIM_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr"
)
INDICES_PATH = Path(
    "/Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr"
)
ANOMALIES_PATH = Path(
    "/Users/tommylees/data/weather/processed/anomalies/vnm_adm1_anomalies_2020_2025.zarr"
)
BOUNDS_PATH = Path("/Users/tommylees/data/weather/boundaries/vnm_adm1.parquet")
OUTPUT_DIR = Path(
    "/Users/tommylees/github/vietnam_coffee_synthetic/artefacts/weather_risk"
)

# Coffee provinces in Central Highlands
COFFEE_PROVINCES = ["gia lai", "kon tum", "lam ??ng", "??k l?k", "??k nong"]

# Clean province name mapping
PROVINCE_DISPLAY = {
    "gia lai": "Gia Lai",
    "kon tum": "Kon Tum",
    "lam ??ng": "Lam Dong",
    "??k l?k": "Dak Lak",
    "??k nong": "Dak Nong",
}

# Plot styling - production quality with clean spines
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLOURS = {
    "actual": "#2E86AB",  # Blue
    "climatology": "#444444",  # Dark grey
    "anomaly_pos": "#E94F37",  # Red
    "anomaly_neg": "#1B4965",  # Dark blue
    "uncertainty": "#CCCCCC",
}


def get_coffee_region_ids(ds: xr.Dataset, gdf: gpd.GeoDataFrame) -> list:
    """Get geoid values for coffee-growing provinces."""
    coffee_ids = []
    for geoid in ds.geoid.values:
        if geoid in gdf["geoid"].values:
            name = gdf[gdf["geoid"] == geoid]["geoname"].iloc[0]
            if name in COFFEE_PROVINCES:
                coffee_ids.append(geoid)
    return coffee_ids


def plot_time_series_with_climatology(
    ds_agg: xr.Dataset,
    ds_clim: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
) -> None:
    """Plot time series of temperature and precipitation vs climatology."""
    print("Creating time series plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Select 2020-2025 and coffee regions
    ds = ds_agg.sel(time=slice("2020-01-01", "2025-12-31"))
    ds_coffee = ds.sel(geoid=coffee_ids).mean(dim="geoid")

    # Temperature plot
    ax1 = axes[0]
    time = pd.to_datetime(ds_coffee.time.values)
    tas = ds_coffee["tas"].values

    # 30-day rolling mean
    tas_smooth = pd.Series(tas).rolling(30, center=True).mean()

    ax1.plot(
        time,
        tas_smooth,
        color=COLOURS["actual"],
        linewidth=1.5,
        label="Actual (30-day mean)",
    )

    # Add climatology band
    doy = pd.to_datetime(ds_coffee.time.values).dayofyear
    clim_mean = ds_clim["tas_mean"].sel(geoid=coffee_ids).mean(dim="geoid").values
    clim_std = ds_clim["tas_std"].sel(geoid=coffee_ids).mean(dim="geoid").values

    # Map climatology to actual dates
    clim_tas = [clim_mean[d - 1] for d in doy]
    clim_std_vals = [clim_std[d - 1] for d in doy]

    ax1.fill_between(
        time,
        np.array(clim_tas) - np.array(clim_std_vals),
        np.array(clim_tas) + np.array(clim_std_vals),
        alpha=0.3,
        color=COLOURS["uncertainty"],
        label="Climatology (1991-2020) ±1σ",
    )
    ax1.plot(
        time,
        clim_tas,
        color=COLOURS["climatology"],
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Central Highlands Coffee Regions - Temperature")
    ax1.legend(loc="upper right")

    # Precipitation plot
    ax2 = axes[1]
    pr = ds_coffee["pr"].values

    # 30-day rolling sum
    pr_smooth = pd.Series(pr).rolling(30, center=True).sum()

    ax2.plot(
        time,
        pr_smooth,
        color=COLOURS["actual"],
        linewidth=1.5,
        label="Actual (30-day sum)",
    )

    # Add climatology
    clim_pr = ds_clim["pr_mean"].sel(geoid=coffee_ids).mean(dim="geoid").values
    clim_pr_mapped = [clim_pr[d - 1] * 30 for d in doy]  # Convert daily to 30-day

    ax2.plot(
        time,
        clim_pr_mapped,
        color=COLOURS["climatology"],
        linewidth=1,
        linestyle="--",
        alpha=0.7,
        label="Climatology",
    )

    ax2.set_ylabel("Precipitation (mm/30 days)")
    ax2.set_xlabel("Date")
    ax2.set_title("Central Highlands Coffee Regions - Precipitation")
    ax2.legend(loc="upper right")

    # Add event annotations
    events = {
        "2023-04": "Drought\nonset",
        "2024-03": "El Niño\npeak",
        "2024-10": "La Niña\nfloods",
    }

    for date_str, label in events.items():
        try:
            date = pd.to_datetime(date_str + "-15")
            for ax in axes:
                ax.axvline(date, color="#888888", linestyle=":", alpha=0.5)
            ax2.annotate(
                label,
                xy=(date, ax2.get_ylim()[1] * 0.9),
                ha="center",
                fontsize=8,
                color="#666666",
            )
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_monthly_anomalies(
    ds_anomalies: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
) -> None:
    """Plot monthly anomaly heatmap."""
    print("Creating monthly anomalies plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Select coffee regions
    ds = ds_anomalies.sel(geoid=coffee_ids).mean(dim="geoid")

    for idx, (var, ax, cmap, label) in enumerate(
        [
            ("tas", axes[0], "RdBu_r", "Temperature Anomaly (°C)"),
            ("pr", axes[1], "BrBG", "Precipitation Anomaly (mm/day)"),
        ]
    ):
        if var not in ds:
            continue

        # Convert to monthly
        monthly = ds[var].resample(time="ME").mean()

        # Pivot to year x month
        df = monthly.to_dataframe().reset_index()
        df["year"] = pd.to_datetime(df["time"]).dt.year
        df["month"] = pd.to_datetime(df["time"]).dt.month

        pivot = df.pivot(index="year", columns="month", values=var)

        # Plot heatmap
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(12))
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_ylabel("Year")
        ax.set_title(label)

        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.suptitle(
        "Central Highlands Coffee Regions - Monthly Anomalies",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_index_dashboard(
    ds_indices: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
) -> None:
    """Create a dashboard of climate indices."""
    print("Creating index dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select coffee regions
    ds = ds_indices.sel(geoid=coffee_ids).mean(dim="geoid")

    # GDD cumulative
    ax1 = axes[0, 0]
    if "gdd" in ds:
        # Annual cumulative GDD
        gdd_annual = ds["gdd"].resample(time="YE").sum()
        years = pd.to_datetime(gdd_annual.time.values).year
        ax1.bar(years, gdd_annual.values, color=COLOURS["actual"])
        ax1.axhline(
            gdd_annual.mean(),
            color=COLOURS["climatology"],
            linestyle="--",
            label="Mean",
        )
        ax1.set_ylabel("GDD (°C·days)")
        ax1.set_title("Annual Growing Degree Days")
        ax1.legend()

    # EDD cumulative
    ax2 = axes[0, 1]
    if "edd" in ds:
        edd_annual = ds["edd"].resample(time="YE").sum()
        years = pd.to_datetime(edd_annual.time.values).year
        colours = [
            COLOURS["anomaly_pos"] if v > edd_annual.mean() else COLOURS["actual"]
            for v in edd_annual.values
        ]
        ax2.bar(years, edd_annual.values, color=colours)
        ax2.axhline(
            edd_annual.mean(),
            color=COLOURS["climatology"],
            linestyle="--",
            label="Mean",
        )
        ax2.set_ylabel("EDD (°C·days)")
        ax2.set_title("Annual Extreme Degree Days (>30°C)")
        ax2.legend()

    # Consecutive dry days
    ax3 = axes[1, 0]
    if "dry_day" in ds:
        # Monthly dry day count
        dry_monthly = ds["dry_day"].resample(time="ME").sum()
        time = pd.to_datetime(dry_monthly.time.values)
        ax3.fill_between(
            time, 0, dry_monthly.values, color=COLOURS["anomaly_neg"], alpha=0.7
        )
        ax3.set_ylabel("Dry days per month")
        ax3.set_title("Dry Days (<1mm precipitation)")

    # Soil moisture
    ax4 = axes[1, 1]
    if "swvl_mean" in ds:
        swvl = ds["swvl_mean"]
        time = pd.to_datetime(swvl.time.values)
        swvl_smooth = pd.Series(swvl.values).rolling(30, center=True).mean()
        ax4.plot(time, swvl_smooth, color=COLOURS["actual"], linewidth=1.5)
        ax4.axhline(
            swvl.mean(), color=COLOURS["climatology"], linestyle="--", label="Mean"
        )
        ax4.set_ylabel("Soil Water (m³/m³)")
        ax4.set_title("Soil Moisture (30-day mean)")
        ax4.legend()

    plt.suptitle(
        "Central Highlands Coffee Regions - Climate Indices (2020-2025)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_annual_comparison(
    ds_agg: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
) -> None:
    """Compare annual precipitation and temperature."""
    print("Creating annual comparison plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Select coffee regions
    ds = ds_agg.sel(geoid=coffee_ids).mean(dim="geoid")

    # Annual means
    tas_annual = ds["tas"].resample(time="YE").mean()
    pr_annual = ds["pr"].resample(time="YE").sum()

    years = pd.to_datetime(tas_annual.time.values).year

    # Scatter plot with year labels
    scatter = ax.scatter(
        pr_annual.values,
        tas_annual.values,
        c=years,
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidths=0.5,
    )

    # Add year labels
    for i, year in enumerate(years):
        ax.annotate(
            str(year),
            (pr_annual.values[i], tas_annual.values[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    # Highlight recent years
    recent_mask = years >= 2020
    ax.scatter(
        pr_annual.values[recent_mask],
        tas_annual.values[recent_mask],
        facecolors="none",
        edgecolors=COLOURS["anomaly_pos"],
        s=150,
        linewidths=2,
        label="2020-2025",
    )

    ax.set_xlabel("Annual Precipitation (mm)")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.set_title("Central Highlands - Annual Climate Space (1980-2025)")
    ax.legend()

    plt.colorbar(scatter, ax=ax, label="Year")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - VISUALISATION")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    ds_agg = xr.open_zarr(AGG_PATH)
    gdf = gpd.read_parquet(BOUNDS_PATH)

    # Get coffee region IDs
    coffee_ids = get_coffee_region_ids(ds_agg, gdf)
    print(f"Found {len(coffee_ids)} coffee region(s)")

    if len(coffee_ids) == 0:
        print("WARNING: No coffee regions found, using all regions")
        coffee_ids = list(ds_agg.geoid.values)

    # Load all datasets
    ds_agg = ds_agg.load()

    # Create time series plot
    if CLIM_PATH.exists():
        ds_clim = xr.open_zarr(CLIM_PATH).load()
        plot_time_series_with_climatology(
            ds_agg,
            ds_clim,
            coffee_ids,
            OUTPUT_DIR / "02_time_series_vs_climatology.png",
        )
    else:
        print(f"Skipping time series plot - climatology not found at {CLIM_PATH}")

    # Create anomaly plot
    if ANOMALIES_PATH.exists():
        ds_anomalies = xr.open_zarr(ANOMALIES_PATH).load()
        plot_monthly_anomalies(
            ds_anomalies, coffee_ids, OUTPUT_DIR / "03_monthly_anomalies.png"
        )
    else:
        print(f"Skipping anomaly plot - anomalies not found at {ANOMALIES_PATH}")

    # Create index dashboard
    if INDICES_PATH.exists():
        ds_indices = xr.open_zarr(INDICES_PATH).load()
        plot_index_dashboard(
            ds_indices, coffee_ids, OUTPUT_DIR / "04_index_dashboard.png"
        )
    else:
        print(f"Skipping index dashboard - indices not found at {INDICES_PATH}")

    # Create annual comparison
    plot_annual_comparison(ds_agg, coffee_ids, OUTPUT_DIR / "05_annual_comparison.png")

    print("\n" + "=" * 60)
    print("VISUALISATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
