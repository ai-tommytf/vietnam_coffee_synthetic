"""
Step 1-2: Inspect Raw Zarr Data and Standardise

JTBD:
- Understand the structure, dimensions, and quality of raw ERA5 data
- Convert to CF conventions with canonical variable names and units

Input: /Users/tommylees/data/weather/raw/vnm_1980_2025.zarr
Output: /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Configuration
RAW_PATH = Path("/Users/tommylees/data/weather/raw/vnm_1980_2025.zarr")
INTERIM_PATH = Path("/Users/tommylees/data/weather/interim/vnm_1980_2025.zarr")
OUTPUT_DIR = Path(
    "/Users/tommylees/github/vietnam_coffee_synthetic/artefacts/weather_risk"
)


# Variable mapping: ERA5 names -> canonical CF names
VARIABLE_MAP = {
    "2m_temperature": "tas",
    "2m_temperature_max": "tasmax",
    "2m_temperature_min": "tasmin",
    "total_precipitation": "pr",
    "evaporation": "evspsbl",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
}


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


def standardise_variable_names(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables to CF canonical names."""
    print("\n--- Standardising Variable Names ---")
    rename_dict = {}
    for old_name, new_name in VARIABLE_MAP.items():
        if old_name in ds.data_vars:
            rename_dict[old_name] = new_name
            print(f"  {old_name} -> {new_name}")

    return ds.rename(rename_dict)


def convert_units(ds: xr.Dataset) -> xr.Dataset:
    """Convert units to standard values (temperature K->C, precip m->mm)."""
    print("\n--- Converting Units ---")

    # Temperature: Kelvin to Celsius
    for var in ["tas", "tasmax", "tasmin"]:
        if var in ds:
            # Check if already in Celsius (values < 100 suggest Celsius)
            sample_mean = float(ds[var].isel(time=0).mean().values)
            if sample_mean > 100:
                print(f"  {var}: K -> C (subtracting 273.15)")
                ds[var] = ds[var] - 273.15
                ds[var].attrs["units"] = "degC"
            else:
                print(f"  {var}: already in C (sample mean: {sample_mean:.1f})")
                ds[var].attrs["units"] = "degC"

    # Precipitation: m/day to mm/day
    if "pr" in ds:
        # ERA5 precipitation is in m, convert to mm
        sample_mean = float(ds["pr"].isel(time=0).mean().values)
        if sample_mean < 1:  # Values < 1 suggest metres
            print("  pr: m -> mm (multiplying by 1000)")
            ds["pr"] = ds["pr"] * 1000
            ds["pr"].attrs["units"] = "mm/day"
        else:
            print(f"  pr: already in mm (sample mean: {sample_mean:.1f})")
            ds["pr"].attrs["units"] = "mm/day"

    # Evaporation: m/day to mm/day (and typically negative in ERA5)
    if "evspsbl" in ds:
        sample_mean = float(ds["evspsbl"].isel(time=0).mean().values)
        if abs(sample_mean) < 1:
            print("  evspsbl: m -> mm and sign flip")
            ds["evspsbl"] = -ds["evspsbl"] * 1000  # ERA5 evap is negative
            ds["evspsbl"].attrs["units"] = "mm/day"
        else:
            print("  evspsbl: checking sign only")
            if sample_mean < 0:
                ds["evspsbl"] = -ds["evspsbl"]
            ds["evspsbl"].attrs["units"] = "mm/day"

    return ds


def add_cf_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Add CF-compliant attributes to variables."""
    cf_attrs = {
        "tas": {
            "long_name": "Near-Surface Air Temperature",
            "standard_name": "air_temperature",
        },
        "tasmax": {
            "long_name": "Daily Maximum Near-Surface Air Temperature",
            "standard_name": "air_temperature",
        },
        "tasmin": {
            "long_name": "Daily Minimum Near-Surface Air Temperature",
            "standard_name": "air_temperature",
        },
        "pr": {"long_name": "Precipitation", "standard_name": "precipitation_flux"},
        "evspsbl": {
            "long_name": "Evaporation",
            "standard_name": "water_evapotranspiration_flux",
        },
        "swvl1": {"long_name": "Volumetric Soil Water Layer 1 (0-7cm)"},
        "swvl2": {"long_name": "Volumetric Soil Water Layer 2 (7-28cm)"},
        "swvl3": {"long_name": "Volumetric Soil Water Layer 3 (28-100cm)"},
        "swvl4": {"long_name": "Volumetric Soil Water Layer 4 (100-289cm)"},
    }

    for var, attrs in cf_attrs.items():
        if var in ds:
            ds[var].attrs.update(attrs)

    # Add global attributes
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["source"] = "ERA5 reanalysis"
    ds.attrs["institution"] = "ECMWF"
    ds.attrs["processing"] = "Standardised for Vietnam coffee risk analysis"

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

    # Standardise
    print("\n" + "=" * 60)
    print("STANDARDISATION")
    print("=" * 60)

    ds = standardise_variable_names(ds)
    ds = convert_units(ds)
    ds = add_cf_attrs(ds)

    # Save
    print("\n--- Saving Standardised Data ---")
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing zarr if present
    if INTERIM_PATH.exists():
        import shutil

        shutil.rmtree(INTERIM_PATH)
        print(f"  Removed existing: {INTERIM_PATH}")

    # Write with safe_chunks=False to handle chunk alignment
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
