"""
Step 3: Areal Aggregation

JTBD: Aggregate gridded data to administrative boundaries (ADM0, ADM1, ADM2).

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
import numpy as np
import xarray as xr
from tqdm import tqdm

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


def create_weight_mask(
    ds: xr.Dataset,
    geometry: gpd.GeoSeries,
) -> xr.DataArray:
    """Create a weight mask for a single geometry."""
    # Get grid coordinates
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Check which points are inside the geometry
    from shapely.geometry import Point

    mask = np.zeros_like(lon_grid, dtype=np.float32)
    geom = geometry.iloc[0]

    for i in range(len(lats)):
        for j in range(len(lons)):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            if geom.contains(point):
                mask[i, j] = 1.0

    # Normalise weights (sum to 1)
    if mask.sum() > 0:
        mask = mask / mask.sum()

    return xr.DataArray(
        mask,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
    )


def aggregate_to_regions(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    id_column: str = "geoid",
) -> xr.Dataset:
    """Aggregate gridded data to regions using area-weighted mean."""
    # Get unique region IDs
    region_ids = gdf[id_column].unique()
    print(f"  Aggregating to {len(region_ids)} regions...")

    # Prepare output arrays
    n_time = len(ds.time)
    n_regions = len(region_ids)

    # Initialise output data structure
    output_data = {
        var: np.zeros((n_time, n_regions), dtype=np.float32) for var in ds.data_vars
    }

    # Process each region
    for idx, region_id in enumerate(tqdm(region_ids, desc="  Processing regions")):
        # Get geometry for this region
        region_geom = gdf[gdf[id_column] == region_id]

        # Create weight mask
        weights = create_weight_mask(ds, region_geom.geometry)

        # Skip if no grid cells in region
        if weights.sum() == 0:
            # Use nearest cell instead
            centroid = region_geom.geometry.iloc[0].centroid
            nearest_lat = ds.latitude.sel(latitude=centroid.y, method="nearest")
            nearest_lon = ds.longitude.sel(longitude=centroid.x, method="nearest")

            for var in ds.data_vars:
                values = ds[var].sel(latitude=nearest_lat, longitude=nearest_lon).values
                output_data[var][:, idx] = values
        else:
            # Weighted mean for each variable
            for var in ds.data_vars:
                # Load data for this variable
                data = ds[var].values  # shape: (time, lat, lon)

                # Apply weighted mean
                weighted_sum = (data * weights.values).sum(axis=(1, 2))
                output_data[var][:, idx] = weighted_sum

    # Create output dataset
    output_ds = xr.Dataset(
        {var: (["time", "geoid"], data) for var, data in output_data.items()},
        coords={
            "time": ds.time.values,
            "geoid": region_ids,
        },
    )

    # Copy variable attributes
    for var in ds.data_vars:
        output_ds[var].attrs = ds[var].attrs.copy()

    return output_ds


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("VIETNAM WEATHER DATA - AREAL AGGREGATION")
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

        # Aggregate
        ds_agg = aggregate_to_regions(ds, gdf)

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
