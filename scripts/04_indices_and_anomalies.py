"""
Step 5-7: Compute Indices and Anomalies

JTBD:
- Compute climate indices (GDD, EDD, dry days) using tf-data-ml-utils
- Calculate anomalies and z-scores relative to climatology

Input:
- Climatology: /Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr
- Trend params: /Users/tommylees/data/weather/processed/climatology/vnm_adm1_trend_params.zarr
- Weather data: /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr

Output:
- /Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr
- /Users/tommylees/data/weather/processed/anomalies/vnm_adm1_anomalies_2020_2025.zarr
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tf_data_ml_utils.weather.stages.climatology import query_climatology
from tf_data_ml_utils.weather.stages.climatology.io import load_trend_params
from tf_data_ml_utils.weather.stages.indices import compute
from tf_data_ml_utils.weather.stages.indices.primitives import _edd_daily

# Configuration
AGG_PATH = Path(
    "/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr"
)
CLIM_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr"
)
TREND_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_trend_params.zarr"
)
INDICES_OUTPUT = Path(
    "/Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr"
)
INDICES_FULL_OUTPUT = Path(
    "/Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_1980_2025.zarr"
)
ANOMALIES_OUTPUT = Path(
    "/Users/tommylees/data/weather/processed/anomalies/vnm_adm1_anomalies_2020_2025.zarr"
)

# Index parameters for coffee
GDD_BASE = 10.0  # Base temperature for Growing Degree Days (coffee)
GDD_UPPER = 29.0  # Upper threshold for GDD (coffee optimal range)
EDD_THRESHOLD = 30.0  # Threshold for Extreme Degree Days (heat stress)
DRY_DAY_THRESHOLD = 1.0  # mm/day threshold for dry day


def compute_daily_indices(ds: xr.Dataset) -> xr.Dataset:
    """Compute all daily indices from weather data using tf-data-ml-utils.

    Uses the compute() function from tf_data_ml_utils.weather.stages.indices
    for standardised index computation.
    """
    print("Computing daily indices using tf-data-ml-utils...")

    indices = xr.Dataset()

    # Variable mapping: canonical name -> dataset variable name
    # Since our data already uses canonical names, mapping is identity
    var_mapping_tas = {"tas": "tas"}
    var_mapping_tasmax = {"tas": "tasmax"}  # Use tasmax for EDD
    var_mapping_pr = {"pr": "pr"}

    # Growing Degree Days (GDD) - using tas
    if "tas" in ds:
        print(f"  - GDD (base {GDD_BASE}C, upper {GDD_UPPER}C)")
        gdd = compute(
            ds,
            index="gdd",
            var_mapping=var_mapping_tas,
            params={"thresh": GDD_BASE, "upper": GDD_UPPER},
        )
        indices["gdd"] = gdd
        indices["gdd"].attrs["long_name"] = f"Growing Degree Days (base {GDD_BASE}C)"
        indices["gdd"].attrs["units"] = "degC"

    # Extreme Degree Days (EDD) - using tasmax for heat stress
    # Note: tf-data-ml-utils edd uses tas, but for heat stress we want tasmax
    # So we use the primitive directly with tasmax
    if "tasmax" in ds:
        print(f"  - EDD (threshold {EDD_THRESHOLD}C) using tasmax")
        edd = _edd_daily(ds["tasmax"], thresh=EDD_THRESHOLD)
        indices["edd"] = edd
        indices["edd"].attrs["long_name"] = f"Extreme Degree Days (>{EDD_THRESHOLD}C)"
        indices["edd"].attrs["units"] = "degC"

    # Dry day indicator - using pr
    if "pr" in ds:
        print(f"  - Dry day indicator (threshold {DRY_DAY_THRESHOLD}mm)")
        dry_days = compute(
            ds,
            index="dry_days",
            var_mapping=var_mapping_pr,
            params={"thresh": DRY_DAY_THRESHOLD},
        )
        indices["dry_day"] = dry_days
        indices["dry_day"].attrs["long_name"] = (
            f"Dry Day (precipitation < {DRY_DAY_THRESHOLD}mm)"
        )
        indices["dry_day"].attrs["units"] = "1"

    # Copy raw variables for anomaly calculation
    if "pr" in ds:
        indices["pr"] = ds["pr"]

    if "tas" in ds:
        indices["tas"] = ds["tas"]

    # Soil moisture - mean of all layers
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
    polys: xr.Dataset | None = None,
    transforms: xr.Dataset | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute anomalies and z-scores relative to climatology.

    Uses query_climatology from tf-data-ml-utils to get climatology values
    for each date, then computes anomalies and standardised anomalies (z-scores).
    """
    print("\nComputing anomalies using tf-data-ml-utils...")

    anomalies = xr.Dataset()
    z_scores = xr.Dataset()

    # Get time coordinates
    times = pd.DatetimeIndex(ds_actual.time.values)

    # Query climatology for actual dates
    # This handles day-of-year lookup and optional retrending
    print("  Querying climatology for actual dates...")
    should_retrend = polys is not None and transforms is not None
    clim_for_dates = query_climatology(
        ds_clim,
        valid_time=times,
        polys=polys,
        transforms=transforms,
        retrend=should_retrend,
        time_dim="time",
    )

    # Variables to process - those with mean/std in climatology
    # The climatology has a 'statistic' dimension with 'loc' (mean) and 'scale' (std)
    vars_to_process = ["tas", "pr", "gdd", "edd", "swvl_mean"]

    for var in vars_to_process:
        if var not in ds_actual:
            continue

        if var not in clim_for_dates:
            print(f"  Skipping {var} (no climatology)")
            continue

        print(f"  - {var}")

        # Get climatology mean and std for this variable
        # The climatology dataset has 'statistic' dimension with 'loc' and 'scale'
        try:
            clim_mean = clim_for_dates[var].sel(statistic="loc")
            clim_std = clim_for_dates[var].sel(statistic="scale")
        except (KeyError, ValueError):
            # Fallback for different climatology structure
            print(f"    Warning: Could not get loc/scale for {var}, skipping")
            continue

        # Align dimensions
        actual = ds_actual[var]

        # Compute anomaly
        anom = actual - clim_mean
        anomalies[var] = anom
        anomalies[var].attrs["long_name"] = f"{var} anomaly"

        # Compute z-score (handle zero std)
        std_safe = clim_std.where(clim_std > 0.001, 0.001)
        zscore = anom / std_safe
        z_scores[f"{var}_zscore"] = zscore
        z_scores[f"{var}_zscore"].attrs["long_name"] = (
            f"{var} standardised anomaly (z-score)"
        )

    # Flag extreme events (|z| > 2)
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
    print("Using tf-data-ml-utils indices module")
    print("=" * 60)

    # Load data
    print(f"\nLoading aggregated data from: {AGG_PATH}")
    ds_agg = xr.open_zarr(AGG_PATH)

    print(f"Loading climatology from: {CLIM_PATH}")
    ds_clim = xr.open_zarr(CLIM_PATH)

    # Load trend parameters if available (reconstructs polys and transforms)
    polys = None
    transforms = None
    if TREND_PATH.exists():
        print(f"Loading trend parameters from: {TREND_PATH}")
        polys, transforms = load_trend_params(TREND_PATH)

    # Drop non-numeric variables from full dataset
    numeric_vars = [v for v in ds_agg.data_vars if ds_agg[v].dtype.kind == "f"]
    ds_numeric = ds_agg[numeric_vars]

    # Load into memory
    print("Loading data into memory...")
    ds_numeric = ds_numeric.load()
    ds_clim = ds_clim.load()

    import shutil

    # =========================================================================
    # FULL PERIOD INDICES (1980-2025) - for climatology comparison plots
    # =========================================================================
    print("\n" + "-" * 40)
    print("Computing indices for FULL period (1980-2025)...")
    print(f"Full data: {len(ds_numeric.time)} days")
    print(f"Variables: {list(ds_numeric.data_vars)}")

    ds_indices_full = compute_daily_indices(ds_numeric)

    # Save full indices
    INDICES_FULL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if INDICES_FULL_OUTPUT.exists():
        shutil.rmtree(INDICES_FULL_OUTPUT)

    # Ensure geoid is string type for zarr compatibility
    ds_indices_full = ds_indices_full.assign_coords(geoid=ds_indices_full.geoid.astype(str))

    print(f"\nSaving full indices to: {INDICES_FULL_OUTPUT}")
    ds_indices_full.to_zarr(INDICES_FULL_OUTPUT, mode="w")

    # =========================================================================
    # RECENT PERIOD INDICES (2020-2025) - for anomaly calculations
    # =========================================================================
    print("\n" + "-" * 40)
    print("Computing indices for RECENT period (2020-2025)...")
    ds_recent = ds_numeric.sel(time=slice("2020-01-01", "2025-12-31"))
    print(f"Recent data: {len(ds_recent.time)} days")

    ds_indices = compute_daily_indices(ds_recent)

    # Save recent indices
    INDICES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if INDICES_OUTPUT.exists():
        shutil.rmtree(INDICES_OUTPUT)

    ds_indices = ds_indices.assign_coords(geoid=ds_indices.geoid.astype(str))

    print(f"\nSaving recent indices to: {INDICES_OUTPUT}")
    ds_indices.to_zarr(INDICES_OUTPUT, mode="w")

    # Compute anomalies
    print("\n" + "-" * 40)
    ds_anom, ds_zscore = compute_anomalies(ds_indices, ds_clim, polys, transforms)

    # Merge anomalies and z-scores
    ds_anomalies = xr.merge([ds_anom, ds_zscore])

    # Save anomalies
    ANOMALIES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if ANOMALIES_OUTPUT.exists():
        shutil.rmtree(ANOMALIES_OUTPUT)

    print(f"\nSaving anomalies to: {ANOMALIES_OUTPUT}")
    ds_anomalies.to_zarr(ANOMALIES_OUTPUT, mode="w")

    # Summary
    print("\n" + "=" * 60)
    print("INDICES AND ANOMALIES COMPLETE")
    print("=" * 60)
    print(f"\nFull indices (1980-2025): {INDICES_FULL_OUTPUT}")
    print(f"  Variables: {list(ds_indices_full.data_vars)}")
    print(f"\nRecent indices (2020-2025): {INDICES_OUTPUT}")
    print(f"  Variables: {list(ds_indices.data_vars)}")
    print(f"\nAnomalies output: {ANOMALIES_OUTPUT}")
    print(f"  Variables: {list(ds_anomalies.data_vars)}")


if __name__ == "__main__":
    main()
