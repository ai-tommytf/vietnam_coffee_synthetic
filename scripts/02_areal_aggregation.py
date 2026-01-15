"""
Step 3: Areal Aggregation

JTBD: Aggregate gridded data to administrative boundaries (ADM0, ADM1, ADM2)
      using tf-data-ml-utils combine_reduce function.

Input:
- Weather: /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr
- Boundaries: /Users/tommylees/data/raw/boundaries/all_geoboundaries_processed.parquet

Output:
- /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm0_1980_2025.zarr
- /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr
- /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm2_1980_2025.zarr
"""

from pathlib import Path

import geopandas as gpd
import xarray as xr

from tf_data_ml_utils.weather.stages.areal_aggregation import combine_reduce

# Configuration
INTERIM_PATH = Path("/Users/tommylees/data/weather/interim/vnm_1980_2025.zarr")
BOUNDARIES_PATH = Path(
    "/Users/tommylees/data/raw/boundaries/all_geoboundaries_processed.parquet"
)
OUTPUT_DIR = Path("/Users/tommylees/data/weather/processed/areal_aggregation")
BOUNDS_OUTPUT_DIR = Path("/Users/tommylees/data/weather/boundaries")


def extract_vietnam_boundaries() -> dict[str, gpd.GeoDataFrame]:
    """Extract Vietnam boundaries at all ADM levels."""
    print("=" * 60)
    print("EXTRACTING VIETNAM BOUNDARIES")
    print("=" * 60)

    # Load all boundaries
    print(f"\nLoading boundaries from: {BOUNDARIES_PATH}")
    gdf = gpd.read_parquet(BOUNDARIES_PATH)

    # Filter to Vietnam
    vnm = gdf[gdf["shapegroup"] == "VNM"].copy()
    print(f"Found {len(vnm)} Vietnam regions total")

    # Separate by ADM level
    boundaries = {}
    for level in ["ADM0", "ADM1", "ADM2"]:
        subset = vnm[vnm["shapetype"] == level].copy()
        boundaries[level] = subset
        print(f"  {level}: {len(subset)} regions")

    # Save boundaries
    BOUNDS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for level, gdf_level in boundaries.items():
        output_path = BOUNDS_OUTPUT_DIR / f"vnm_{level.lower()}.parquet"
        gdf_level.to_parquet(output_path)
        print(f"  Saved: {output_path}")

    return boundaries


def aggregate_to_regions(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    id_column: str = "geoid",
) -> xr.Dataset:
    """Aggregate gridded data to regions using tf-data-ml-utils combine_reduce.

    This uses a MapReduce approach:
    1. COMBINE: Rasterize vector geometries onto grid
    2. REDUCE: Aggregate grid cells to regional means
    """
    print(f"  Aggregating to {len(gdf)} regions using combine_reduce...")

    # Use tf-data-ml-utils combine_reduce function
    # It handles rasterisation, area-weighted means, and centroid fallback
    result = combine_reduce(
        ds=ds,
        gdf=gdf,
        id_column=id_column,
        lat_dim="latitude",
        lon_dim="longitude",
    )

    return result


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - AREAL AGGREGATION")
    print("Using tf-data-ml-utils combine_reduce")
    print("=" * 60)

    # Extract boundaries
    boundaries = extract_vietnam_boundaries()

    # Load standardised data
    print(f"\nLoading standardised data from: {INTERIM_PATH}")
    ds = xr.open_zarr(INTERIM_PATH)

    # Load all data into memory for faster processing
    print("Loading data into memory...")
    ds = ds.load()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each ADM level
    for level in ["ADM0", "ADM1", "ADM2"]:
        print(f"\n{'=' * 60}")
        print(f"AGGREGATING TO {level}")
        print("=" * 60)

        gdf = boundaries[level]
        output_path = OUTPUT_DIR / f"vnm_{level.lower()}_1980_2025.zarr"

        # Aggregate using tf-data-ml-utils
        ds_agg = aggregate_to_regions(ds, gdf)

        # Rename dimension from geoid to geoid if needed (for consistency)
        if "geoid" in ds_agg.dims:
            pass  # Already correct
        elif "geo_id" in ds_agg.dims:
            ds_agg = ds_agg.rename({"geo_id": "geoid"})

        # Add region names as auxiliary coordinate
        names = gdf.set_index("geoid")["geoname"].to_dict()
        ds_agg["region_name"] = (
            "geoid",
            [names.get(gid, "") for gid in ds_agg.geoid.values],
        )

        # Save
        print(f"  Saving to: {output_path}")
        if output_path.exists():
            import shutil

            shutil.rmtree(output_path)

        ds_agg.to_zarr(output_path, mode="w")
        print(f"  Done: {len(ds_agg.geoid)} regions, {len(ds_agg.time)} timesteps")

    print("\n" + "=" * 60)
    print("AREAL AGGREGATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
