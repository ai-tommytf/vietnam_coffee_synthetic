"""
Step 1-2: Inspect Raw Zarr Data and Standardise

JTBD:
- Understand the structure, dimensions, and quality of raw ERA5 data
- Convert to CF conventions with canonical variable names and units

Input: /Users/tommylees/data/weather/raw/vnm_1980_2025.zarr
Output: /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr

Uses tf-data-ml-utils standardise stage for CF-compliant standardisation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tf_data_ml_utils.weather.stages.standardise import (
    CanonicalConfig,
    DimensionConfig,
    standardise,
)

# Configuration
RAW_PATH = Path("/Users/tommylees/data/weather/raw/vnm_1980_2025.zarr")
INTERIM_PATH = Path("/Users/tommylees/data/weather/interim/vnm_1980_2025.zarr")
OUTPUT_DIR = Path(
    "/Users/tommylees/github/vietnam_coffee_synthetic/artefacts/weather_risk"
)

# Path to canonical variables YAML in tf-data-ml-utils
CANONICAL_CONFIG_PATH = Path(
    "/Users/tommylees/github/tf-data-ml-utils/tf_data_ml_utils/weather/stages/standardise/canonical_variables.yaml"
)


def inspect_raw_data(ds: xr.Dataset) -> dict:
    """Inspect raw data and return summary statistics."""
    print("=" * 60)
    print("RAW DATA INSPECTION")
    print("=" * 60)

    print("\n--- Dataset Structure ---")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Variables: {list(ds.data_vars)}")

    print("\n--- Time Range ---")
    time_min = ds.time.values.min()
    time_max = ds.time.values.max()
    print(f"Start: {time_min}")
    print(f"End: {time_max}")
    print(f"Total days: {len(ds.time)}")

    print("\n--- Spatial Extent ---")
    lat_min, lat_max = float(ds.latitude.values.min()), float(ds.latitude.values.max())
    lon_min, lon_max = (
        float(ds.longitude.values.min()),
        float(ds.longitude.values.max()),
    )
    print(f"Latitude: {lat_min:.2f} to {lat_max:.2f}")
    print(f"Longitude: {lon_min:.2f} to {lon_max:.2f}")
    print(
        f"Grid size: {len(ds.latitude)} x {len(ds.longitude)} = {len(ds.latitude) * len(ds.longitude)} cells"
    )

    print("\n--- Variable Summary ---")
    summary = {}
    for var in ds.data_vars:
        # Sample first timestep for quick stats
        sample = ds[var].isel(time=0).values
        nan_count = np.isnan(sample).sum()
        total = sample.size
        nan_pct = 100 * nan_count / total
        summary[var] = {
            "dtype": str(ds[var].dtype),
            "nan_pct_sample": nan_pct,
            "units": ds[var].attrs.get("units", "unknown"),
        }
        print(
            f"  {var}: {ds[var].dtype}, NaN: {nan_pct:.1f}%, units: {summary[var]['units']}"
        )

    return {
        "time_range": (str(time_min), str(time_max)),
        "lat_range": (lat_min, lat_max),
        "lon_range": (lon_min, lon_max),
        "variables": summary,
    }


def mask_missing_data(ds: xr.Dataset) -> xr.Dataset:
    """Apply consistent NaN masking across all variables.

    Uses temperature (tas) as the reference variable since it correctly has NaN
    where data is missing. This ensures variables like precipitation and
    evaporation don't show artificial zeros where there's actually no data.
    """
    print("\n--- Applying Consistent NaN Mask ---")

    # Use tas as reference - it has NaN where data is truly missing
    if "tas" not in ds:
        print("  WARNING: tas not found, skipping NaN masking")
        return ds

    # Create mask: True where tas is valid (not NaN), False where missing
    tas_valid = ~np.isnan(ds["tas"])

    # Find the last timestep where tas has any valid data
    has_valid_data = tas_valid.any(dim=["latitude", "longitude"])
    last_valid_idx = int(has_valid_data.values[::-1].argmax())
    if last_valid_idx > 0:
        last_valid_idx = len(has_valid_data) - last_valid_idx - 1
    else:
        if has_valid_data.values[-1]:
            last_valid_idx = len(has_valid_data) - 1
        else:
            last_valid_idx = int(has_valid_data.values.argmin()) - 1

    last_valid_time = ds.time.values[last_valid_idx]
    print(f"  Reference variable: tas")
    print(f"  Last valid timestep: {last_valid_time}")

    # Apply mask to all variables: where tas is NaN, set other vars to NaN too
    variables_masked = []
    for var in ds.data_vars:
        if var == "tas":
            continue

        sample_end = ds[var].isel(time=-1).values
        nan_pct_end = 100 * np.isnan(sample_end).sum() / sample_end.size

        if nan_pct_end < 100:
            ds[var] = ds[var].where(tas_valid)
            variables_masked.append(var)

    if variables_masked:
        print(f"  Masked variables: {', '.join(variables_masked)}")
    else:
        print("  All variables already have consistent NaN patterns")

    return ds


def create_spatial_plot(ds: xr.Dataset, output_path: Path) -> None:
    """Create a simple map of temperature to verify spatial coverage."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Temperature
    ds["tas"].isel(time=0).plot(ax=axes[0], cmap="RdYlBu_r")
    axes[0].set_title("Temperature (day 1)")

    # Precipitation
    ds["pr"].isel(time=0).plot(ax=axes[1], cmap="Blues")
    axes[1].set_title("Precipitation (day 1)")

    # Soil moisture layer 1
    ds["swvl1"].isel(time=0).plot(ax=axes[2], cmap="YlGnBu")
    axes[2].set_title("Soil Water Layer 1 (day 1)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved spatial plot to: {output_path}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - INSPECT AND STANDARDISE")
    print("=" * 60)

    # Load raw data
    print(f"\nLoading raw data from: {RAW_PATH}")
    ds = xr.open_zarr(RAW_PATH)

    # Inspect
    inspect_raw_data(ds)

    # Standardise using tf-data-ml-utils
    print("\n" + "=" * 60)
    print("STANDARDISATION (using tf-data-ml-utils)")
    print("=" * 60)

    # Load canonical config
    canonical_cfg = CanonicalConfig.from_yaml(CANONICAL_CONFIG_PATH)
    dim_cfg = DimensionConfig(
        spatial_dims=["latitude", "longitude"],
        temporal_dims=["time"],
    )

    print(f"\nUsing canonical config from: {CANONICAL_CONFIG_PATH}")
    print(f"Dimension config: spatial={dim_cfg.spatial_dims}, temporal={dim_cfg.temporal_dims}")

    # Standardise
    ds = standardise(ds, canonical_cfg, dim_cfg)

    # Apply additional NaN masking (project-specific)
    ds = mask_missing_data(ds)

    # Save
    print("\n--- Saving Standardised Data ---")
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)

    if INTERIM_PATH.exists():
        import shutil

        shutil.rmtree(INTERIM_PATH)
        print(f"  Removed existing: {INTERIM_PATH}")

    ds.to_zarr(INTERIM_PATH, mode="w", safe_chunks=False)
    print(f"  Saved to: {INTERIM_PATH}")

    # Create verification plot
    create_spatial_plot(ds, OUTPUT_DIR / "01_spatial_verification.png")

    # Final summary
    print("\n" + "=" * 60)
    print("STANDARDISATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {INTERIM_PATH}")
    print("\nVariables:")
    for var in ds.data_vars:
        units = ds[var].attrs.get("units", "unknown")
        print(f"  {var}: {units}")


if __name__ == "__main__":
    main()
