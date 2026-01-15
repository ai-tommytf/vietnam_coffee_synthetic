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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - enables rio accessor
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, zoom
from shapely.geometry import mapping

from tf_data_ml_utils.weather.stages.climatology import (
    Distribution,
    compute_statistics,
    load_trend_params,
    query_climatology,
)

# Configuration
RAW_GRID_PATH = Path("/Users/tommylees/data/weather/raw/vnm_1980_2025.zarr")
AGG_PATH = Path(
    "/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr"
)
CLIM_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr"
)
TREND_PATH = Path(
    "/Users/tommylees/data/weather/processed/climatology/vnm_adm1_trend_params.zarr"
)
INDICES_PATH = Path(
    "/Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr"
)
INDICES_FULL_PATH = Path(
    "/Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_1980_2025.zarr"
)
ANOMALIES_PATH = Path(
    "/Users/tommylees/data/weather/processed/anomalies/vnm_adm1_anomalies_2020_2025.zarr"
)
BOUNDS_ADM0_PATH = Path("/Users/tommylees/data/weather/boundaries/vnm_adm0.parquet")
BOUNDS_PATH = Path("/Users/tommylees/data/weather/boundaries/vnm_adm1.parquet")
OUTPUT_DIR = Path(
    "/Users/tommylees/github/vietnam_coffee_synthetic/artefacts/weather_risk"
)

# Distribution configuration for derived statistics
DIST_CONFIG = {
    "tas": Distribution.NORM,
    "tasmin": Distribution.NORM,
    "tasmax": Distribution.NORM,
    "pr": Distribution.ZI_GAMMA,
    "evspsbl": Distribution.ZI_GAMMA,
    "swvl1": Distribution.ZI_GAMMA,
    "swvl2": Distribution.ZI_GAMMA,
    "swvl3": Distribution.ZI_GAMMA,
    "swvl4": Distribution.ZI_GAMMA,
}

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


def plot_coffee_regions_map(
    gdf_adm0: gpd.GeoDataFrame,
    gdf_adm1: gpd.GeoDataFrame,
    coffee_ids: list,
    output_path: Path,
) -> None:
    """Plot map showing coffee regions used for index calculations.

    Parameters
    ----------
    gdf_adm0 : gpd.GeoDataFrame
        Country boundary
    gdf_adm1 : gpd.GeoDataFrame
        Province boundaries with geoid column
    coffee_ids : list
        List of geoid values for coffee regions
    output_path : Path
        Path to save the output figure
    """
    print("Creating coffee regions map...")

    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot all provinces in light grey
    gdf_adm1.plot(
        ax=ax,
        facecolor="#f0f0f0",
        edgecolor="grey",
        linewidth=0.5,
    )

    # Highlight coffee regions
    coffee_gdf = gdf_adm1[gdf_adm1["geoid"].isin(coffee_ids)]
    coffee_gdf.plot(
        ax=ax,
        facecolor=COLOURS["actual"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
    )

    # Add labels for coffee provinces
    for idx, row in coffee_gdf.iterrows():
        centroid = row.geometry.centroid
        # Get display name
        raw_name = row["geoname"]
        display_name = PROVINCE_DISPLAY.get(raw_name, raw_name.title())
        ax.annotate(
            display_name,
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground="black")
            ],
        )

    # Add country boundary
    gdf_adm0.boundary.plot(ax=ax, color="black", linewidth=2)

    # Formatting
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"Central Highlands Coffee Regions\n({len(coffee_ids)} provinces used for index calculations)",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOURS["actual"], edgecolor="black", alpha=0.7,
              label="Coffee Growing Regions"),
        Patch(facecolor="#f0f0f0", edgecolor="grey",
              label="Other Provinces"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def mask_to_boundary(
    ds: xr.Dataset, gdf: gpd.GeoDataFrame, crs: str = "EPSG:4326"
) -> xr.Dataset:
    """Mask gridded data to polygon boundary, setting cells outside to NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded data with latitude/longitude coordinates
    gdf : gpd.GeoDataFrame
        Boundary polygon(s) to mask to
    crs : str
        Coordinate reference system

    Returns
    -------
    xr.Dataset
        Data with cells outside boundary set to NaN
    """
    # Set CRS and spatial dimensions for rioxarray
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

    # Clip to boundary - cells outside become NaN
    return ds.rio.clip(gdf.geometry, gdf.crs, drop=False)


def plot_gridded_weather_map(
    ds_grid: xr.Dataset,
    gdf_adm0: gpd.GeoDataFrame,
    gdf_adm1: gpd.GeoDataFrame,
    output_path: Path,
    date: str = "2024-03-15",
) -> None:
    """Plot gridded weather data cropped to Vietnam boundary.

    Shows temperature and precipitation maps for a single date with
    grid cells outside Vietnam boundary set to NaN.
    """
    print("Creating gridded weather map...")

    # Select single date
    ds = ds_grid.sel(time=date, method="nearest")

    # Mask to Vietnam ADM0 boundary
    ds_masked = mask_to_boundary(ds, gdf_adm0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    # Temperature map
    ax1 = axes[0]
    temp = ds_masked["2m_temperature"]
    # Convert from Kelvin to Celsius if needed
    if np.nanmean(temp.values) > 100:  # Likely in Kelvin
        temp = temp - 273.15
    im1 = temp.plot(
        ax=ax1,
        cmap="RdYlBu_r",
        add_colorbar=True,
        cbar_kwargs={"label": "Temperature (°C)", "shrink": 0.7},
    )
    gdf_adm0.boundary.plot(ax=ax1, color="black", linewidth=1.5)
    gdf_adm1.boundary.plot(ax=ax1, color="grey", linewidth=0.5, alpha=0.7)
    ax1.set_title(f"Temperature - {date}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Precipitation map
    ax2 = axes[1]
    precip = ds_masked["total_precipitation"]
    # Convert from m to mm
    precip_mm = precip * 1000
    im2 = precip_mm.plot(
        ax=ax2,
        cmap="Blues",
        add_colorbar=True,
        cbar_kwargs={"label": "Precipitation (mm/day)", "shrink": 0.7},
        vmin=0,
    )
    gdf_adm0.boundary.plot(ax=ax2, color="black", linewidth=1.5)
    gdf_adm1.boundary.plot(ax=ax2, color="grey", linewidth=0.5, alpha=0.7)
    ax2.set_title(f"Precipitation - {date}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    plt.suptitle(
        "Vietnam ERA5 Weather Data",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def interpolate_to_high_res(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    target_res_km: float = 1.0,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate gridded data to higher resolution using zoom and gaussian smoothing.

    Parameters
    ----------
    data : np.ndarray
        2D array of gridded data (lat x lon)
    lat : np.ndarray
        1D array of latitude values
    lon : np.ndarray
        1D array of longitude values
    target_res_km : float
        Target resolution in km (approximately)
    sigma : float
        Gaussian filter sigma for smoothing

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Interpolated data, new lat array, new lon array
    """
    # Current resolution (ERA5 is ~25km at equator, ~0.25 degrees)
    current_res_deg = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.25
    # 1 degree ~ 111 km at equator
    current_res_km = current_res_deg * 111

    # Calculate zoom factor
    zoom_factor = current_res_km / target_res_km

    # Handle NaN values by interpolating them first
    mask = np.isnan(data)
    data_filled = data.copy()
    if mask.any():
        # Simple fill with nearest valid value for interpolation
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        data_filled = data_filled[tuple(indices)]

    # Zoom to higher resolution
    data_highres = zoom(data_filled, zoom_factor, order=3)  # cubic interpolation

    # Apply gaussian smoothing
    data_highres = gaussian_filter(data_highres, sigma=sigma)

    # Create new coordinate arrays
    lat_highres = np.linspace(lat.min(), lat.max(), data_highres.shape[0])
    lon_highres = np.linspace(lon.min(), lon.max(), data_highres.shape[1])

    # Re-apply mask at high resolution
    if mask.any():
        mask_highres = zoom(mask.astype(float), zoom_factor, order=0) > 0.5
        data_highres[mask_highres] = np.nan

    return data_highres, lat_highres, lon_highres


def create_highres_mask(
    gdf: gpd.GeoDataFrame, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Create a boolean mask for high-res grid from polygon boundary."""
    from shapely.geometry import Point

    # Create meshgrid of coordinates
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get the union of all geometries
    boundary = gdf.union_all()

    # Check each point - vectorized approach
    mask = np.zeros(lon_grid.shape, dtype=bool)
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            mask[i, j] = boundary.contains(Point(lon_grid[i, j], lat_grid[i, j]))

    return mask


def plot_gridded_weather_map_highres(
    ds_grid: xr.Dataset,
    gdf_adm0: gpd.GeoDataFrame,
    gdf_adm1: gpd.GeoDataFrame,
    output_path: Path,
    date: str = "2024-03-15",
    target_res_km: float = 1.0,
) -> None:
    """Plot high-resolution interpolated weather data cropped to Vietnam boundary.

    Uses scipy zoom and gaussian filter to downscale from ~25km to ~1km resolution.
    Axes are hidden for a cleaner presentation.
    """
    print("Creating high-resolution gridded weather map...")

    # Select single date
    ds = ds_grid.sel(time=date, method="nearest")

    # Get coordinate arrays (before masking, for full interpolation)
    lat = ds.latitude.values
    lon = ds.longitude.values

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    # Temperature map
    ax1 = axes[0]
    temp = ds["2m_temperature"].values
    # Convert from Kelvin to Celsius if needed
    if np.nanmean(temp) > 100:
        temp = temp - 273.15

    # Interpolate to high resolution (without mask first for smooth interpolation)
    temp_hr, lat_hr, lon_hr = interpolate_to_high_res(
        temp, lat, lon, target_res_km=target_res_km, sigma=2.0
    )

    # Create high-res mask using rasterio clip approach
    # Build a temporary high-res dataset to clip
    temp_da = xr.DataArray(
        temp_hr,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_hr, "longitude": lon_hr},
    )
    temp_da = temp_da.rio.write_crs("EPSG:4326")
    temp_da = temp_da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    temp_clipped = temp_da.rio.clip(gdf_adm0.geometry, gdf_adm0.crs, drop=False)

    im1 = ax1.pcolormesh(
        lon_hr, lat_hr, temp_clipped.values,
        cmap="RdYlBu_r",
        shading="auto",
    )
    plt.colorbar(im1, ax=ax1, label="Temperature (°C)", shrink=0.7)
    gdf_adm0.boundary.plot(ax=ax1, color="black", linewidth=1.5)
    gdf_adm1.boundary.plot(ax=ax1, color="grey", linewidth=0.5, alpha=0.7)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Precipitation map
    ax2 = axes[1]
    precip = ds["total_precipitation"].values
    # Convert from m to mm
    precip_mm = precip * 1000

    # Interpolate to high resolution
    precip_hr, lat_hr, lon_hr = interpolate_to_high_res(
        precip_mm, lat, lon, target_res_km=target_res_km, sigma=2.0
    )

    # Clip to boundary
    precip_da = xr.DataArray(
        precip_hr,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_hr, "longitude": lon_hr},
    )
    precip_da = precip_da.rio.write_crs("EPSG:4326")
    precip_da = precip_da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    precip_clipped = precip_da.rio.clip(gdf_adm0.geometry, gdf_adm0.crs, drop=False)

    im2 = ax2.pcolormesh(
        lon_hr, lat_hr, precip_clipped.values,
        cmap="Blues",
        shading="auto",
        vmin=0,
    )
    plt.colorbar(im2, ax=ax2, label="Precipitation (mm/day)", shrink=0.7)
    gdf_adm0.boundary.plot(ax=ax2, color="black", linewidth=1.5)
    gdf_adm1.boundary.plot(ax=ax2, color="grey", linewidth=0.5, alpha=0.7)
    ax2.set_aspect("equal")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_time_series_with_climatology(
    ds_agg: xr.Dataset,
    ds_clim: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
    polys: xr.Dataset | None = None,
    transforms: xr.Dataset | None = None,
) -> None:
    """Plot time series of temperature and precipitation vs climatology.

    Uses query_climatology with retrend to get temperature back to original scale,
    and compute_statistics to get proper derived statistics (mean, sigma bounds).
    Applies 30-day rolling to climatology to match rolling actuals.
    """
    print("Creating time series plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Select 2020-2025 and coffee regions
    ds = ds_agg.sel(time=slice("2020-01-01", "2025-12-31"))
    ds_coffee = ds.sel(geoid=coffee_ids).mean(dim="geoid")

    time = pd.to_datetime(ds_coffee.time.values)

    # Query climatology with retrend to get back to original temperature scale
    # This adds the trend back to detrended temperature variables
    clim_queried = query_climatology(
        ds_clim.sel(geoid=coffee_ids),
        time,
        polys=polys.sel(geoid=coffee_ids) if polys is not None else None,
        transforms=transforms.sel(geoid=coffee_ids) if transforms is not None else None,
        retrend=polys is not None,
    )

    # Compute derived statistics (mean, sigma_lower, sigma_upper) from distribution params
    clim_stats = compute_statistics(
        clim_queried,
        statistics=["mean", "sigma_lower", "sigma_upper"],
        dist_config=DIST_CONFIG,
    )

    # Average over coffee regions
    clim_coffee = clim_stats.mean(dim="geoid")

    # Temperature plot
    ax1 = axes[0]
    tas = ds_coffee["tas"].values

    # 30-day rolling mean for actuals
    tas_smooth = pd.Series(tas).rolling(30, center=True).mean()

    ax1.plot(
        time,
        tas_smooth,
        color=COLOURS["actual"],
        linewidth=1.5,
        label="Actual (30-day mean)",
    )

    # 30-day rolling for climatology to match
    clim_tas_mean = pd.Series(
        clim_coffee["tas"].sel(statistic="mean").values
    ).rolling(30, center=True).mean()
    clim_tas_lower = pd.Series(
        clim_coffee["tas"].sel(statistic="sigma_lower").values
    ).rolling(30, center=True).mean()
    clim_tas_upper = pd.Series(
        clim_coffee["tas"].sel(statistic="sigma_upper").values
    ).rolling(30, center=True).mean()

    ax1.fill_between(
        time,
        clim_tas_lower,
        clim_tas_upper,
        alpha=0.3,
        color=COLOURS["uncertainty"],
        label="Climatology (1991-2020) ±1σ",
    )
    ax1.plot(
        time,
        clim_tas_mean,
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

    # Mask precipitation where temperature is NaN (data ends)
    pr = np.where(np.isnan(tas), np.nan, pr)

    # 30-day rolling sum for actuals
    pr_smooth = pd.Series(pr).rolling(30, center=True).sum()

    ax2.plot(
        time,
        pr_smooth,
        color=COLOURS["actual"],
        linewidth=1.5,
        label="Actual (30-day sum)",
    )

    # 30-day rolling sum for climatology (cumsum over rolling window)
    clim_pr_mean = pd.Series(
        clim_coffee["pr"].sel(statistic="mean").values
    ).rolling(30, center=True).sum()
    clim_pr_lower = pd.Series(
        clim_coffee["pr"].sel(statistic="sigma_lower").values
    ).rolling(30, center=True).sum()
    clim_pr_upper = pd.Series(
        clim_coffee["pr"].sel(statistic="sigma_upper").values
    ).rolling(30, center=True).sum()

    # Clip lower bound at 0 for precipitation
    clim_pr_lower = clim_pr_lower.clip(lower=0)

    ax2.fill_between(
        time,
        clim_pr_lower,
        clim_pr_upper,
        alpha=0.3,
        color=COLOURS["uncertainty"],
        label="Climatology (1991-2020) ±1σ",
    )
    ax2.plot(
        time,
        clim_pr_mean,
        color=COLOURS["climatology"],
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )

    ax2.set_ylabel("Precipitation (mm/30 days)")
    ax2.set_xlabel("Date")
    ax2.set_title("Central Highlands Coffee Regions - Precipitation")
    ax2.legend(loc="upper right")

    # Add event annotations
    # Based on NOAA MEI.v2 index: El Niño peaked Nov-Dec 2023 (MEI=1.13),
    # La Niña emerged May-Jun 2024 (MEI=-0.24), intensified through late 2024
    events = {
        "2023-12": "El Niño\npeak",
        "2024-05": "Drought\npeak",
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


def plot_time_series_with_climatology_minimal(
    ds_agg: xr.Dataset,
    ds_clim: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
    polys: xr.Dataset | None = None,
    transforms: xr.Dataset | None = None,
) -> None:
    """Plot time series without titles and x-tick labels - clean version for presentations.

    Same as plot_time_series_with_climatology but with:
    - No titles
    - No x-tick labels
    - Only y-axis labels kept
    """
    print("Creating minimal time series plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Select 2020-2025 and coffee regions
    ds = ds_agg.sel(time=slice("2020-01-01", "2025-12-31"))
    ds_coffee = ds.sel(geoid=coffee_ids).mean(dim="geoid")

    time = pd.to_datetime(ds_coffee.time.values)

    # Query climatology with retrend to get back to original temperature scale
    clim_queried = query_climatology(
        ds_clim.sel(geoid=coffee_ids),
        time,
        polys=polys.sel(geoid=coffee_ids) if polys is not None else None,
        transforms=transforms.sel(geoid=coffee_ids) if transforms is not None else None,
        retrend=polys is not None,
    )

    # Compute derived statistics from distribution params
    clim_stats = compute_statistics(
        clim_queried,
        statistics=["mean", "sigma_lower", "sigma_upper"],
        dist_config=DIST_CONFIG,
    )

    # Average over coffee regions
    clim_coffee = clim_stats.mean(dim="geoid")

    # Temperature plot
    ax1 = axes[0]
    tas = ds_coffee["tas"].values

    # 30-day rolling mean for actuals
    tas_smooth = pd.Series(tas).rolling(30, center=True).mean()

    ax1.plot(
        time,
        tas_smooth,
        color=COLOURS["actual"],
        linewidth=1.5,
        label="Actual (30-day mean)",
    )

    # 30-day rolling for climatology
    clim_tas_mean = pd.Series(
        clim_coffee["tas"].sel(statistic="mean").values
    ).rolling(30, center=True).mean()
    clim_tas_lower = pd.Series(
        clim_coffee["tas"].sel(statistic="sigma_lower").values
    ).rolling(30, center=True).mean()
    clim_tas_upper = pd.Series(
        clim_coffee["tas"].sel(statistic="sigma_upper").values
    ).rolling(30, center=True).mean()

    ax1.fill_between(
        time,
        clim_tas_lower,
        clim_tas_upper,
        alpha=0.3,
        color=COLOURS["uncertainty"],
        label="Climatology (1991-2020) ±1σ",
    )
    ax1.plot(
        time,
        clim_tas_mean,
        color=COLOURS["climatology"],
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )

    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis="y", labelleft=False)

    # Precipitation plot
    ax2 = axes[1]
    pr = ds_coffee["pr"].values

    # Mask precipitation where temperature is NaN
    pr = np.where(np.isnan(tas), np.nan, pr)

    # 30-day rolling sum for actuals
    pr_smooth = pd.Series(pr).rolling(30, center=True).sum()

    ax2.plot(
        time,
        pr_smooth,
        color=COLOURS["actual"],
        linewidth=1.5,
        label="Actual (30-day sum)",
    )

    # 30-day rolling sum for climatology
    clim_pr_mean = pd.Series(
        clim_coffee["pr"].sel(statistic="mean").values
    ).rolling(30, center=True).sum()
    clim_pr_lower = pd.Series(
        clim_coffee["pr"].sel(statistic="sigma_lower").values
    ).rolling(30, center=True).sum()
    clim_pr_upper = pd.Series(
        clim_coffee["pr"].sel(statistic="sigma_upper").values
    ).rolling(30, center=True).sum()

    # Clip lower bound at 0 for precipitation
    clim_pr_lower = clim_pr_lower.clip(lower=0)

    ax2.fill_between(
        time,
        clim_pr_lower,
        clim_pr_upper,
        alpha=0.3,
        color=COLOURS["uncertainty"],
        label="Climatology (1991-2020) ±1σ",
    )
    ax2.plot(
        time,
        clim_pr_mean,
        color=COLOURS["climatology"],
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )

    ax2.set_ylabel("Precipitation (mm/30 days)")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis="y", labelleft=False)

    # Add event annotations
    events = {
        "2023-12": "El Niño\npeak",
        "2024-05": "Drought\npeak",
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


def plot_dayofyear_indices(
    ds_indices: xr.Dataset,
    coffee_ids: list,
    output_path: Path,
    current_season: str = "2024-2025",
) -> None:
    """Plot cumulative indices by day of year with climatology and historical years.

    Creates plots showing:
    - X-axis: Day of year (1-365)
    - Y-axis: Cumulative index value
    - Current season: Bold C0 coloured line
    - Climatology: Grey dashed line with ±1σ fill
    - Historical years: Thin grey lines at alpha=0.5

    Uses the FULL historical record (1980-2025) from pre-computed indices
    to compute climatology.

    Parameters
    ----------
    ds_indices : xr.Dataset
        Pre-computed indices dataset (1980-2025) with gdd, edd, dry_day, pr
    coffee_ids : list
        List of geoid values for coffee regions
    output_path : Path
        Path to save the output figure
    current_season : str
        Season to highlight as current (format: "YYYY-YYYY", e.g. "2024-2025")
    """
    print("Creating day-of-year indices plot...")

    # Select coffee regions and average
    ds = ds_indices.sel(geoid=coffee_ids).mean(dim="geoid")

    # Define indices to plot with their display configuration
    indices_config = [
        ("gdd", "Growing Degree Days (°C·days)", "cumsum"),
        ("edd", "Extreme Degree Days (°C·days)", "cumsum"),
        ("dry_day", "Dry Days (count)", "cumsum"),
        ("pr", "Precipitation (mm)", "cumsum"),
    ]

    # Filter to only available indices
    indices_config = [(v, l, a) for v, l, a in indices_config if v in ds.data_vars]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Parse current season years
    current_start_year, _ = map(int, current_season.split("-"))

    for idx, (var, ylabel, agg_type) in enumerate(indices_config):
        ax = axes[idx]

        # Build dataframe with year and dayofyear
        df = ds[var].to_dataframe().reset_index()
        df["date"] = pd.to_datetime(df["time"])
        df["year"] = df["date"].dt.year
        df["dayofyear"] = df["date"].dt.dayofyear

        # Define coffee season (Oct-Sep) - assign season year based on Oct start
        # Season 2024-2025 starts Oct 2024, ends Sep 2025
        df["season_year"] = df.apply(
            lambda r: r["year"] if r["date"].month >= 10 else r["year"] - 1, axis=1
        )

        # Get unique seasons
        seasons = sorted(df["season_year"].unique())

        # Storage for climatology calculation
        all_season_data = []

        # Plot each season
        for season in seasons:
            # Select data for this season (Oct of season year to Sep of next year)
            season_start = pd.Timestamp(f"{season}-10-01")
            season_end = pd.Timestamp(f"{season + 1}-09-30")
            mask = (df["date"] >= season_start) & (df["date"] <= season_end)
            season_df = df[mask].copy()

            if len(season_df) == 0:
                continue

            # Create season day (1 = Oct 1, etc.)
            season_df["season_day"] = (
                season_df["date"] - season_start
            ).dt.days + 1

            # Sort by season day and compute cumulative sum
            season_df = season_df.sort_values("season_day")
            if agg_type == "cumsum":
                season_df["cumulative"] = season_df[var].cumsum()
            else:
                season_df["cumulative"] = season_df[var]

            # Check if season is complete (>= 360 valid days)
            valid_days = season_df["cumulative"].notna().sum()
            is_complete = valid_days >= 360

            # Store for climatology (exclude current season AND incomplete seasons)
            if season != current_start_year and is_complete:
                all_season_data.append(season_df[["season_day", "cumulative"]])

            # Determine styling based on whether this is current season
            is_current = season == current_start_year

            if is_current:
                ax.plot(
                    season_df["season_day"],
                    season_df["cumulative"],
                    color="C0",
                    linewidth=2,
                    label=f"{season}-{season + 1}",
                    zorder=10,
                )
            elif is_complete:
                # Only plot complete historical seasons
                ax.plot(
                    season_df["season_day"],
                    season_df["cumulative"],
                    color="grey",
                    linewidth=0.5,
                    alpha=0.5,
                    linestyle="-",
                    zorder=1,
                )

        # Compute climatology from historical seasons
        if all_season_data:
            clim_df = pd.concat(all_season_data, ignore_index=True)
            clim_grouped = clim_df.groupby("season_day")["cumulative"]
            clim_mean = clim_grouped.mean()
            clim_std = clim_grouped.std()

            season_days = clim_mean.index

            # Plot climatology mean line
            ax.plot(
                season_days,
                clim_mean,
                color="grey",
                linewidth=1.5,
                linestyle="--",
                label="Climatology",
                zorder=5,
            )

            # Plot ±1σ uncertainty band
            ax.fill_between(
                season_days,
                clim_mean - clim_std,
                clim_mean + clim_std,
                color="grey",
                alpha=0.3,
                zorder=2,
            )

        # Formatting
        ax.set_xlabel("Day of Season (Oct 1 = Day 1)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Cumulative {var.upper()} by Season")
        ax.legend(loc="upper left")

        # Add month labels on x-axis
        month_starts = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 335]
        month_labels = [
            "Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
            "Apr", "May", "Jun", "Jul", "Aug", "Sep"
        ]
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_xlim(1, 365)

    # Count historical seasons used (from last variable's seasons list)
    n_historical = len(seasons) - 1 if seasons else 0

    plt.suptitle(
        f"Central Highlands Coffee Regions - Seasonal Index Accumulation\n"
        f"Current Season: {current_season} (blue) vs Historical ({n_historical} seasons, grey)",
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
    gdf_adm0 = gpd.read_parquet(BOUNDS_ADM0_PATH)

    # Get coffee region IDs
    coffee_ids = get_coffee_region_ids(ds_agg, gdf)
    print(f"Found {len(coffee_ids)} coffee region(s)")

    if len(coffee_ids) == 0:
        print("WARNING: No coffee regions found, using all regions")
        coffee_ids = list(ds_agg.geoid.values)

    # Create coffee regions map (00)
    plot_coffee_regions_map(
        gdf_adm0,
        gdf,
        coffee_ids,
        OUTPUT_DIR / "00_coffee_regions_map.png",
    )

    # Create gridded weather map (01b)
    if RAW_GRID_PATH.exists():
        ds_grid = xr.open_zarr(RAW_GRID_PATH)
        plot_gridded_weather_map(
            ds_grid,
            gdf_adm0,
            gdf,
            OUTPUT_DIR / "01b_gridded_weather_map.png",
            date="2024-03-15",  # El Niño peak date
        )
        # Create high-resolution version (01c)
        plot_gridded_weather_map_highres(
            ds_grid,
            gdf_adm0,
            gdf,
            OUTPUT_DIR / "01c_gridded_weather_map_highres.png",
            date="2024-03-15",
            target_res_km=5.0,  # ~5km resolution (1km would be very slow)
        )
    else:
        print(f"Skipping gridded map - raw data not found at {RAW_GRID_PATH}")

    # Load all datasets
    ds_agg = ds_agg.load()

    # Create time series plot
    if CLIM_PATH.exists():
        ds_clim = xr.open_zarr(CLIM_PATH).load()

        # Load trend parameters for retrending (converts detrended temps back to original)
        polys, transforms = None, None
        if TREND_PATH.exists():
            print(f"Loading trend parameters from: {TREND_PATH}")
            polys, transforms = load_trend_params(TREND_PATH)

        plot_time_series_with_climatology(
            ds_agg,
            ds_clim,
            coffee_ids,
            OUTPUT_DIR / "02_time_series_vs_climatology.png",
            polys=polys,
            transforms=transforms,
        )
        # Minimal version without titles and x-tick labels
        plot_time_series_with_climatology_minimal(
            ds_agg,
            ds_clim,
            coffee_ids,
            OUTPUT_DIR / "02b_time_series_minimal.png",
            polys=polys,
            transforms=transforms,
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

    # Create day-of-year seasonal indices plot (uses full 1980-2025 indices)
    if INDICES_FULL_PATH.exists():
        ds_indices_full = xr.open_zarr(INDICES_FULL_PATH).load()
        plot_dayofyear_indices(
            ds_indices_full,
            coffee_ids,
            OUTPUT_DIR / "07_dayofyear_indices.png",
            current_season="2024-2025",
        )
    else:
        print(f"Skipping day-of-year plot - full indices not found at {INDICES_FULL_PATH}")

    # Create annual comparison
    plot_annual_comparison(ds_agg, coffee_ids, OUTPUT_DIR / "05_annual_comparison.png")

    print("\n" + "=" * 60)
    print("VISUALISATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
