"""
Step 4: Compute Climatologies

JTBD: Calculate day-of-year climatologies from baseline period using tf-data-ml-utils.

Baseline period: 1991-2020 (WMO standard 30-year normal)

Input: /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr
Output: /Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr
"""

from pathlib import Path

import xarray as xr

from tf_data_ml_utils.weather.stages.climatology import (
    ClimatologyConfig,
    ClimatologyOutput,
    Distribution,
    compute_climatology,
)
from tf_data_ml_utils.weather.stages.climatology.config import (
    DetrendConfig,
    SmoothingConfig,
    VariableConfig,
)

# Configuration
INPUT_PATH = Path(
    "/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr"
)
OUTPUT_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr"
)
TREND_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_trend_params.zarr"
)

BASELINE_START = 1991
BASELINE_END = 2020
WINDOW_SIZE = 31  # Rolling window for smoothing (must be odd)


def create_climatology_config() -> ClimatologyConfig:
    """Create climatology configuration for Vietnam weather data."""
    return ClimatologyConfig(
        baseline_years=(BASELINE_START, BASELINE_END),
        window_size=WINDOW_SIZE,
        time_dim="time",
        # Detrending: enable for temperature variables to remove climate trend
        detrend=DetrendConfig(
            enabled=True,
            variables=["tas", "tasmin", "tasmax"],
            polyorder=1,
            baseline="1980-01-01",
        ),
        # Fourier smoothing for seasonal cycle
        smoothing=SmoothingConfig(
            enabled=True,
            n_bases=3,
            period=365.25,
        ),
        # Per-variable distribution configuration
        variables={
            "tas": VariableConfig(distribution=Distribution.NORM),
            "tasmin": VariableConfig(distribution=Distribution.NORM),
            "tasmax": VariableConfig(distribution=Distribution.NORM),
            "pr": VariableConfig(
                distribution=Distribution.ZI_GAMMA,
                filter_positive=True,
                fit_kwargs={"floc": 0},
            ),
            "evspsbl": VariableConfig(
                distribution=Distribution.ZI_GAMMA,
                filter_positive=False,  # Evaporation can be negative in ERA5
                fit_kwargs={"floc": 0},
            ),
            "swvl1": VariableConfig(
                distribution=Distribution.ZI_GAMMA,
                filter_positive=True,
                fit_kwargs={"floc": 0},
            ),
            "swvl2": VariableConfig(
                distribution=Distribution.ZI_GAMMA,
                filter_positive=True,
                fit_kwargs={"floc": 0},
            ),
            "swvl3": VariableConfig(
                distribution=Distribution.ZI_GAMMA,
                filter_positive=True,
                fit_kwargs={"floc": 0},
            ),
            "swvl4": VariableConfig(
                distribution=Distribution.ZI_GAMMA,
                filter_positive=True,
                fit_kwargs={"floc": 0},
            ),
        },
    )


def save_climatology_output(
    output: ClimatologyOutput, clim_path: Path, trend_path: Path
) -> None:
    """Save climatology output to zarr files."""
    import shutil

    from tf_data_ml_utils.weather.stages.climatology.io import (
        get_baseline,
        polys_to_coeffs,
    )

    # Save climatology
    clim_path.parent.mkdir(parents=True, exist_ok=True)
    if clim_path.exists():
        shutil.rmtree(clim_path)

    print(f"Saving climatology to: {clim_path}")
    output.climatology.to_zarr(clim_path, mode="w")

    # Save trend parameters if available
    # Convert poly1d objects to numeric coefficients for serialisation
    if output.trend_params is not None and len(output.trend_params.data_vars) > 0:
        if trend_path.exists():
            shutil.rmtree(trend_path)
        print(f"Saving trend parameters to: {trend_path}")

        # Convert poly1d to coefficients
        coeffs = polys_to_coeffs(output.trend_params.compute())

        # Get baseline from transforms
        if output.transforms is not None:
            baseline = get_baseline(output.transforms.compute())
        else:
            baseline = "1980-01-01"

        coeffs.attrs["baseline"] = baseline
        coeffs.attrs["polyorder"] = 1
        coeffs.attrs["slope_units"] = "per_day"

        # Ensure geoid is string type for zarr compatibility
        if "geoid" in coeffs.coords:
            coeffs = coeffs.assign_coords(geoid=coeffs.geoid.astype(str))

        coeffs.to_zarr(trend_path, mode="w")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - CLIMATOLOGY")
    print("Using tf-data-ml-utils climatology module")
    print("=" * 60)

    # Load aggregated data
    print(f"\nLoading aggregated data from: {INPUT_PATH}")
    ds = xr.open_zarr(INPUT_PATH)
    print(f"Data range: {ds.time.values.min()} to {ds.time.values.max()}")
    print(f"Regions: {len(ds.geoid)}")
    print(f"Variables: {list(ds.data_vars)}")

    # Drop non-numeric variables (region_name)
    numeric_vars = [v for v in ds.data_vars if ds[v].dtype.kind == "f"]
    ds = ds[numeric_vars]
    print(f"Numeric variables for climatology: {list(ds.data_vars)}")

    # Load into memory for faster processing
    print("\nLoading data into memory...")
    ds = ds.load()

    # Create configuration
    print("\n" + "-" * 40)
    cfg = create_climatology_config()
    print(f"Baseline period: {cfg.baseline_years[0]}-{cfg.baseline_years[1]}")
    print(f"Window size: {cfg.window_size} days")
    print(f"Detrending enabled: {cfg.detrend.enabled}")
    print(f"Smoothing enabled: {cfg.smoothing.enabled}")

    # Compute climatology using tf-data-ml-utils
    print("\nComputing climatology...")
    output: ClimatologyOutput = compute_climatology(ds, cfg)

    # Save outputs
    print("\n" + "-" * 40)
    save_climatology_output(output, OUTPUT_PATH, TREND_PATH)

    # Summary
    print("\n" + "=" * 60)
    print("CLIMATOLOGY COMPLETE")
    print("=" * 60)
    print(f"\nClimatology output: {OUTPUT_PATH}")
    print(f"  Variables: {list(output.climatology.data_vars)}")
    print(f"  Dimensions: {dict(output.climatology.sizes)}")
    if output.trend_params is not None and len(output.trend_params.data_vars) > 0:
        print(f"\nTrend parameters: {TREND_PATH}")
        print(f"  Variables: {list(output.trend_params.data_vars)}")

    # Show metadata
    if output.metadata:
        print("\nMetadata:")
        for key, value in output.metadata.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
