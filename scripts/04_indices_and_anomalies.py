"""
Step 5-7: Compute Indices and Anomalies

JTBD:
- Compute climate indices (GDD, EDD, SPI, CDD) for normals and recent data
- Calculate anomalies and z-scores relative to climatology

Input:
- Climatology: /Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr
- Recent data: /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr

Output:
- /Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr
- /Users/tommylees/data/weather/processed/anomalies/vnm_adm1_anomalies_2020_2025.zarr
"""

from pathlib import Path

import numpy as np
import xarray as xr

# Configuration
AGG_PATH = Path(
    "/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr"
)
CLIM_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr"
)
INDICES_OUTPUT = Path(
    "/Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr"
)
ANOMALIES_OUTPUT = Path(
    "/Users/tommylees/data/weather/processed/anomalies/vnm_adm1_anomalies_2020_2025.zarr"
)

# Index parameters
GDD_BASE = 10.0  # Base temperature for Growing Degree Days (coffee)
EDD_THRESHOLD = 30.0  # Threshold for Extreme Degree Days
CDD_PRECIP_THRESHOLD = 1.0  # mm/day threshold for dry day


def compute_gdd(tas: xr.DataArray, base: float = 10.0) -> xr.DataArray:
    """Compute Growing Degree Days."""
    gdd = tas - base
    gdd = gdd.where(gdd > 0, 0)  # Clip negative values to 0
    return gdd


def compute_edd(tasmax: xr.DataArray, threshold: float = 30.0) -> xr.DataArray:
    """Compute Extreme Degree Days (heat stress)."""
    edd = tasmax - threshold
    edd = edd.where(edd > 0, 0)  # Clip negative values to 0
    return edd


def compute_cdd(pr: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    """Compute consecutive dry days indicator (1 if dry, 0 if wet)."""
    return (pr < threshold).astype(float)


def compute_daily_indices(ds: xr.Dataset) -> xr.Dataset:
    """Compute all daily indices from weather data."""
    print("Computing daily indices...")

    # Drop non-numeric variables
    numeric_vars = [v for v in ds.data_vars if ds[v].dtype in [np.float32, np.float64]]
    ds = ds[numeric_vars]

    indices = xr.Dataset()

    # Growing Degree Days (GDD)
    if "tas" in ds:
        print("  - GDD (base 10C)")
        indices["gdd"] = compute_gdd(ds["tas"], GDD_BASE)
        indices["gdd"].attrs["long_name"] = f"Growing Degree Days (base {GDD_BASE}C)"
        indices["gdd"].attrs["units"] = "degC"

    # Extreme Degree Days (EDD)
    if "tasmax" in ds:
        print("  - EDD (threshold 30C)")
        indices["edd"] = compute_edd(ds["tasmax"], EDD_THRESHOLD)
        indices["edd"].attrs["long_name"] = f"Extreme Degree Days (>{EDD_THRESHOLD}C)"
        indices["edd"].attrs["units"] = "degC"

    # Dry day indicator
    if "pr" in ds:
        print("  - Dry day indicator")
        indices["dry_day"] = compute_cdd(ds["pr"], CDD_PRECIP_THRESHOLD)
        indices["dry_day"].attrs["long_name"] = (
            f"Dry Day (precipitation < {CDD_PRECIP_THRESHOLD}mm)"
        )
        indices["dry_day"].attrs["units"] = "1"

    # Precipitation (copy as-is for anomaly calculation)
    if "pr" in ds:
        indices["pr"] = ds["pr"]

    # Temperature (copy for anomaly calculation)
    if "tas" in ds:
        indices["tas"] = ds["tas"]

    # Soil moisture (mean of all layers)
    swvl_vars = [v for v in ds.data_vars if v.startswith("swvl")]
    if swvl_vars:
        print("  - Soil moisture (mean of layers)")
        swvl_data = xr.concat([ds[v] for v in swvl_vars], dim="layer")
        indices["swvl_mean"] = swvl_data.mean(dim="layer")
        indices["swvl_mean"].attrs["long_name"] = "Mean Volumetric Soil Water"
        indices["swvl_mean"].attrs["units"] = "m3/m3"

    return indices


def compute_anomalies(
    ds_actual: xr.Dataset,
    ds_clim: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute anomalies and z-scores relative to climatology."""
    print("\nComputing anomalies...")

    # Add day-of-year to actual data
    ds_actual = ds_actual.assign_coords(dayofyear=ds_actual.time.dt.dayofyear)

    anomalies = xr.Dataset()
    z_scores = xr.Dataset()

    # Variables to process
    vars_to_process = ["tas", "pr", "gdd", "edd", "swvl_mean"]

    for var in vars_to_process:
        if var not in ds_actual:
            continue

        mean_var = f"{var}_mean"
        std_var = f"{var}_std"

        if mean_var not in ds_clim or std_var not in ds_clim:
            # If no climatology, skip
            print(f"  Skipping {var} (no climatology)")
            continue

        print(f"  - {var}")

        # Get climatology values for each day
        clim_mean = ds_clim[mean_var]
        clim_std = ds_clim[std_var]

        # Compute anomalies by day-of-year
        anom_data = []
        zscore_data = []

        for doy in range(1, 367):
            # Select days matching this DOY
            mask = ds_actual.dayofyear == doy

            if not mask.any():
                continue

            actual_doy = ds_actual[var].where(mask, drop=True)
            mean_doy = clim_mean.sel(dayofyear=doy)
            std_doy = clim_std.sel(dayofyear=doy)

            # Compute anomaly
            anom = actual_doy - mean_doy
            anom_data.append(anom)

            # Compute z-score (handle zero std)
            std_safe = std_doy.where(std_doy > 0.001, 0.001)
            zscore = anom / std_safe
            zscore_data.append(zscore)

        # Concatenate all DOYs
        anomalies[var] = xr.concat(anom_data, dim="time").sortby("time")
        z_scores[f"{var}_zscore"] = xr.concat(zscore_data, dim="time").sortby("time")

        # Add attributes
        anomalies[var].attrs["long_name"] = f"{var} anomaly"
        z_scores[f"{var}_zscore"].attrs["long_name"] = (
            f"{var} standardised anomaly (z-score)"
        )

    # Flag extreme events
    print("  - Flagging extreme events (|z| > 2)")
    if "tas_zscore" in z_scores:
        z_scores["extreme_hot"] = z_scores["tas_zscore"] > 2
        z_scores["extreme_cold"] = z_scores["tas_zscore"] < -2
    if "pr_zscore" in z_scores:
        z_scores["extreme_dry"] = z_scores["pr_zscore"] < -2
        z_scores["extreme_wet"] = z_scores["pr_zscore"] > 2

    return anomalies, z_scores


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - INDICES AND ANOMALIES")
    print("=" * 60)

    # Load data
    print(f"\nLoading aggregated data from: {AGG_PATH}")
    ds_agg = xr.open_zarr(AGG_PATH)

    print(f"Loading climatology from: {CLIM_PATH}")
    ds_clim = xr.open_zarr(CLIM_PATH)

    # Select recent period (2020-2025)
    print("\nSelecting 2020-2025 period...")
    ds_recent = ds_agg.sel(time=slice("2020-01-01", "2025-12-31"))
    print(f"Recent data: {len(ds_recent.time)} days")

    # Load into memory
    print("Loading data into memory...")
    ds_recent = ds_recent.load()
    ds_clim = ds_clim.load()

    # Compute indices
    print("\n" + "-" * 40)
    ds_indices = compute_daily_indices(ds_recent)

    # Save indices
    INDICES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if INDICES_OUTPUT.exists():
        import shutil

        shutil.rmtree(INDICES_OUTPUT)

    # Ensure geoid is string type for zarr compatibility
    ds_indices = ds_indices.assign_coords(geoid=ds_indices.geoid.astype(str))

    print(f"\nSaving indices to: {INDICES_OUTPUT}")
    ds_indices.to_zarr(INDICES_OUTPUT, mode="w")

    # Compute anomalies
    print("\n" + "-" * 40)
    ds_anom, ds_zscore = compute_anomalies(ds_indices, ds_clim)

    # Merge anomalies and z-scores
    ds_anomalies = xr.merge([ds_anom, ds_zscore])

    # Save anomalies
    ANOMALIES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if ANOMALIES_OUTPUT.exists():
        import shutil

        shutil.rmtree(ANOMALIES_OUTPUT)

    print(f"\nSaving anomalies to: {ANOMALIES_OUTPUT}")
    ds_anomalies.to_zarr(ANOMALIES_OUTPUT, mode="w")

    # Summary
    print("\n" + "=" * 60)
    print("INDICES AND ANOMALIES COMPLETE")
    print("=" * 60)
    print(f"\nIndices output: {INDICES_OUTPUT}")
    print(f"  Variables: {list(ds_indices.data_vars)}")
    print(f"\nAnomalies output: {ANOMALIES_OUTPUT}")
    print(f"  Variables: {list(ds_anomalies.data_vars)}")


if __name__ == "__main__":
    main()
