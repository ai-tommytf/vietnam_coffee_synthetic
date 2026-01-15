"""
Step 4: Compute Climatologies

JTBD: Calculate day-of-year climatologies (mean, std) from baseline period.

Baseline period: 1991-2020 (WMO standard 30-year normal)

Input: /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr
Output: /Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr
"""

from pathlib import Path

import numpy as np
import xarray as xr

# Configuration
INPUT_PATH = Path(
    "/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr"
)
OUTPUT_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr"
)

BASELINE_START = 1991
BASELINE_END = 2020
WINDOW_SIZE = 31  # Rolling window for smoothing


def compute_doy_climatology(
    ds: xr.Dataset,
    baseline_start: int,
    baseline_end: int,
    window_size: int = 31,
) -> xr.Dataset:
    """Compute day-of-year climatology with rolling window smoothing."""
    print(f"Computing climatology for {baseline_start}-{baseline_end}")
    print(f"Using {window_size}-day rolling window")

    # Drop non-numeric variables
    numeric_vars = [v for v in ds.data_vars if ds[v].dtype in [np.float32, np.float64]]
    ds = ds[numeric_vars]
    print(f"Processing variables: {list(ds.data_vars)}")

    # Select baseline period
    ds_baseline = ds.sel(time=slice(f"{baseline_start}-01-01", f"{baseline_end}-12-31"))
    print(f"Baseline data: {len(ds_baseline.time)} days")

    # Add day-of-year coordinate
    ds_baseline = ds_baseline.assign_coords(dayofyear=ds_baseline.time.dt.dayofyear)

    # Compute mean and std for each day of year
    print("Computing day-of-year statistics...")

    # Group by day of year
    doy_mean = ds_baseline.groupby("dayofyear").mean(dim="time")
    doy_std = ds_baseline.groupby("dayofyear").std(dim="time")

    # Apply rolling window smoothing
    print("Applying rolling window smoothing...")
    half_window = window_size // 2

    # Smooth each variable
    smoothed_mean_data = {}
    smoothed_std_data = {}

    for var in doy_mean.data_vars:
        # Get values (shape: dayofyear, geoid)
        mean_vals = doy_mean[var].values
        std_vals = doy_std[var].values

        # Pad with wrap-around for circular smoothing
        mean_padded = np.concatenate(
            [mean_vals[-half_window:], mean_vals, mean_vals[:half_window]], axis=0
        )
        std_padded = np.concatenate(
            [std_vals[-half_window:], std_vals, std_vals[:half_window]], axis=0
        )

        # Apply uniform filter (rolling mean)
        from scipy.ndimage import uniform_filter1d

        mean_smoothed = uniform_filter1d(
            mean_padded, size=window_size, axis=0, mode="nearest"
        )
        std_smoothed = uniform_filter1d(
            std_padded, size=window_size, axis=0, mode="nearest"
        )

        # Trim padding
        smoothed_mean_data[var] = mean_smoothed[half_window:-half_window]
        smoothed_std_data[var] = std_smoothed[half_window:-half_window]

    # Create output dataset
    output_ds = xr.Dataset(
        {
            **{
                f"{var}_mean": (["dayofyear", "geoid"], data)
                for var, data in smoothed_mean_data.items()
            },
            **{
                f"{var}_std": (["dayofyear", "geoid"], data)
                for var, data in smoothed_std_data.items()
            },
        },
        coords={
            "dayofyear": np.arange(1, 367),  # 1-366 for leap years
            "geoid": doy_mean.geoid.values,
        },
    )

    # Add metadata
    output_ds.attrs["baseline_start"] = baseline_start
    output_ds.attrs["baseline_end"] = baseline_end
    output_ds.attrs["window_size"] = window_size
    output_ds.attrs["description"] = (
        f"Day-of-year climatology from {baseline_start}-{baseline_end}"
    )

    return output_ds


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - CLIMATOLOGY")
    print("=" * 60)

    # Load aggregated data
    print(f"\nLoading aggregated data from: {INPUT_PATH}")
    ds = xr.open_zarr(INPUT_PATH)
    print(f"Data range: {ds.time.values.min()} to {ds.time.values.max()}")
    print(f"Regions: {len(ds.geoid)}")
    print(f"Variables: {list(ds.data_vars)}")

    # Load into memory
    print("\nLoading data into memory...")
    ds = ds.load()

    # Compute climatology
    print("\n" + "-" * 40)
    ds_clim = compute_doy_climatology(
        ds,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        window_size=WINDOW_SIZE,
    )

    # Save output
    print("\n" + "-" * 40)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        import shutil

        shutil.rmtree(OUTPUT_PATH)

    print(f"Saving climatology to: {OUTPUT_PATH}")
    ds_clim.to_zarr(OUTPUT_PATH, mode="w")

    print("\n" + "=" * 60)
    print("CLIMATOLOGY COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Variables: {list(ds_clim.data_vars)}")
    print(f"Shape: {dict(ds_clim.sizes)}")


if __name__ == "__main__":
    main()
