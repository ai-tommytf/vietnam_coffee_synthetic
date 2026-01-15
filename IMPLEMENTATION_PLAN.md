# Vietnam Weather Risk Analysis - Implementation Plan

**Status**: ✅ COMPLETE - All 5 scripts implemented and tested.

**Objective**: Process Vietnam ERA5 weather data to compute climatologies, anomalies, and indices for coffee yield risk analysis.

## Completed Steps

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 1-2 | `01_inspect_and_standardise.py` | ✅ | `interim/vnm_1980_2025.zarr` |
| 3 | `02_areal_aggregation.py` | ✅ | `processed/areal_aggregation/vnm_adm{0,1,2}_1980_2025.zarr` |
| 4 | `03_climatology.py` | ✅ | `processed/climatology/vnm_adm1_climatology.zarr` |
| 5-7 | `04_indices_and_anomalies.py` | ✅ | `processed/indices/`, `processed/anomalies/` |
| 8 | `05_visualise.py` | ✅ | `artefacts/weather_risk/*.png` |

## Generated Visualisations

1. `01_spatial_verification.png` - Spatial coverage check
2. `02_time_series_vs_climatology.png` - Temperature and precipitation vs normal
3. `03_monthly_anomalies.png` - Monthly anomaly heatmaps
4. `04_index_dashboard.png` - GDD, EDD, dry days, soil moisture
5. `05_annual_comparison.png` - Annual climate space plot

---

**Data Sources**:
- Weather: `/Users/tommylees/data/weather/raw/vnm_1980_2025.zarr`
- Boundaries: `/Users/tommylees/data/raw/boundaries/all_geoboundaries_processed.parquet`
- Tools: `~/github/tf-data-ml-utils` (install with `uv pip install ".[weather]"`)

**Variables Available**: `2m_temperature`, `2m_temperature_max`, `2m_temperature_min`, `evaporation`, `total_precipitation`, `volumetric_soil_water_layer_1-4`

---

## Code Organisation

**Structure**: A set of **5 minimal scripts** that run sequentially, importing from `tf-data-ml-utils` modules.

```
scripts/
├── 01_inspect_and_standardise.py   # Steps 1-2: Inspect raw data, standardise
├── 02_areal_aggregation.py         # Step 3: Aggregate to ADM0/1/2 boundaries
├── 03_climatology.py               # Step 4: Compute day-of-year climatologies
├── 04_indices_and_anomalies.py     # Steps 5-7: Compute indices, anomalies
└── 05_visualise.py                 # Step 8: Production plots
```

**Code Standards**:
- **Type checking**: Use `ty` (NOT mypy) for all type annotations
- **Linting/Formatting**: Use `ruff` for linting and formatting
- **Imports**: Import from `tf_data_ml_utils.weather.stages.*` modules - **NON-NEGOTIABLE**
- **Minimal**: Each script should be <200 lines, single responsibility
- **Runnable**: Each script runs independently via `uv run python scripts/0X_*.py`

**⚠️ CRITICAL: All scripts MUST use `tf-data-ml-utils/weather` code**:
- Do NOT reimplement climatology computation - use `tf_data_ml_utils.weather.stages.climatology`
- Do NOT reimplement index calculation - use `tf_data_ml_utils.weather.stages.indices`
- Do NOT reimplement standardisation - use `tf_data_ml_utils.weather.stages.standardise`
- Use helper functions like `stack_by_doy`, `cumsum_doy` from the weather module

**Run order**:
```bash
uv run python scripts/01_inspect_and_standardise.py
uv run python scripts/02_areal_aggregation.py
uv run python scripts/03_climatology.py
uv run python scripts/04_indices_and_anomalies.py
uv run python scripts/05_visualise.py
```

**Quality checks** (run after each script):
```bash
uv run ty check scripts/
uv run ruff check scripts/ --fix
uv run ruff format scripts/
```

---

## Step 1: Inspect Raw Zarr Data

**JTBD**: Understand the structure, dimensions, and quality of the raw ERA5 data before processing.

**Actions**:
1. Open the zarr store and print dataset structure (dims, coords, data_vars)
2. Check time range (expected: 1980-2025 based on metadata)
3. Check spatial extent (lat/lon bounds for Vietnam)
4. Check for missing values (NaN counts per variable)
5. Check units and attributes for each variable
6. Create a simple map plot of one timestep to verify spatial coverage

**Commands**:
```bash
cd ~/github/tf-data-ml-utils
uv run python -c "
import xarray as xr
ds = xr.open_zarr('/Users/tommylees/data/weather/raw/vnm_1980_2025.zarr')
print(ds)
print('Time range:', ds.time.values.min(), 'to', ds.time.values.max())
print('Lat range:', ds.latitude.values.min(), 'to', ds.latitude.values.max())
print('Lon range:', ds.longitude.values.min(), 'to', ds.longitude.values.max())
"
```

**Output**: Summary report of data structure, quality, and coverage.

---

## Step 2: Standardise Data

**JTBD**: Convert raw ERA5 data to CF conventions with canonical variable names and units.

**Actions**:
1. Use `weather-standardise` CLI or the `standardise.process()` function
2. Map ERA5 variable names to canonical names:
   - `2m_temperature` → `tas` (K → °C or keep K)
   - `2m_temperature_max` → `tasmax`
   - `2m_temperature_min` → `tasmin`
   - `total_precipitation` → `pr` (m/day → mm/day)
   - `evaporation` → `evspsbl`
   - `volumetric_soil_water_layer_*` → `mrsos` (or keep separate)
3. Ensure dimensions are named `latitude`, `longitude`, `time`
4. Save to `/Users/tommylees/data/weather/interim/vnm_1980_2025.zarr`

**Commands**:
```bash
weather-standardise \
  -i /Users/tommylees/data/weather/raw/vnm_1980_2025.zarr \
  -o /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr \
  -s "latitude,longitude"
```

**Output**: Standardised zarr at `interim/vnm_1980_2025.zarr`

---

## Step 3: Areal Aggregation

**JTBD**: Aggregate gridded data to administrative boundaries (ADM0, ADM1, ADM2).

**Sub-steps**:

### 3a: Extract Vietnam Boundaries

**Actions**:
1. Load geoboundaries parquet
2. Filter to Vietnam (`shapegroup == 'VNM'`)
3. Save separate files for each ADM level:
   - `vnm_adm0.parquet` (1 region - national)
   - `vnm_adm1.parquet` (64 regions - provinces)
   - `vnm_adm2.parquet` (705 regions - districts)

**Commands**:
```python
import geopandas as gpd
gdf = gpd.read_parquet('/Users/tommylees/data/raw/boundaries/all_geoboundaries_processed.parquet')
vnm = gdf[gdf['shapegroup'] == 'VNM']
for level in ['ADM0', 'ADM1', 'ADM2']:
    subset = vnm[vnm['shapetype'] == level]
    subset.to_parquet(f'/Users/tommylees/data/weather/boundaries/vnm_{level.lower()}.parquet')
```

### 3b: Run Areal Aggregation

**Actions**:
1. Use `weather-areal` CLI for each ADM level
2. Aggregate using area-weighted mean
3. Output zarr files with `geoid` dimension instead of lat/lon

**Commands**:
```bash
# National level (ADM0)
weather-areal \
  -i /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr \
  -s /Users/tommylees/data/weather/boundaries/vnm_adm0.parquet \
  -o /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm0_1980_2025.zarr \
  --id-column geoid

# Provincial level (ADM1)
weather-areal \
  -i /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr \
  -s /Users/tommylees/data/weather/boundaries/vnm_adm1.parquet \
  -o /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr \
  --id-column geoid

# District level (ADM2)
weather-areal \
  -i /Users/tommylees/data/weather/interim/vnm_1980_2025.zarr \
  -s /Users/tommylees/data/weather/boundaries/vnm_adm2.parquet \
  -o /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm2_1980_2025.zarr \
  --id-column geoid
```

**Output**: Three zarr files with dimensions `(time, geoid)` for each ADM level.

---

## Step 4: Compute Climatologies

**JTBD**: Calculate day-of-year climatologies (mean, std) from baseline period for "normal" reference.

**Actions**:
1. Use `weather-climatology` CLI
2. Set baseline period: 1991-2020 (WMO standard 30-year normal)
3. Compute day-of-year statistics with 31-day rolling window
4. Apply Fourier smoothing (default 4 bases)
5. Optionally detrend temperature variables

**Commands**:
```bash
# For ADM1 (provincial level - most useful for coffee regions)
weather-climatology \
  -i /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr \
  -o /Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr \
  --baseline-start 1991 \
  --baseline-end 2020 \
  --window-size 31
```

**Output**: Climatology zarr with dimensions `(dayofyear, geoid)` containing `mean` and `std` for each variable.

---

## Step 5: Compute Climate Indices (Normals)

**JTBD**: Calculate derived indices (GDD, EDD, SPI, etc.) that represent "normal" conditions.

**Actions**:
1. Create a features config YAML for Vietnam coffee-relevant indices:
   - `gdd_base10`: Growing degree days (base 10°C)
   - `edd_30`: Extreme degree days (threshold 30°C)
   - `precip_total`: Total precipitation
   - `cdd`: Consecutive dry days
   - `spi_30`: Standardised Precipitation Index (30-day)
2. Use `weather-indices` CLI to compute indices from climatology
3. Aggregate to monthly/seasonal periods if needed

**Config** (`configs/vietnam_coffee_indices.yaml`):
```yaml
variable_mapping:
  tas: "tas"
  pr: "pr"
  tasmax: "tasmax"
  tasmin: "tasmin"

features:
  - name: gdd_base10
    signal:
      signal_type: gdd
      params:
        t_base: 10.0
    reducer: sum

  - name: edd_30
    signal:
      signal_type: edd
      params:
        t_threshold: 30.0
    reducer: sum

  - name: precip_total
    signal:
      signal_type: precip
    reducer: sum

  - name: cdd_max
    signal:
      signal_type: cdd
    reducer: max
```

**Commands**:
```bash
weather-indices compute-indices \
  /Users/tommylees/data/weather/processed/climatology/vnm_adm1_climatology.zarr \
  --spatial-id vnm_adm1 \
  --config configs/vietnam_coffee_indices.yaml \
  -o /Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_normal.zarr
```

**Output**: Indices zarr with "normal" expected values.

---

## Step 6: Compute Actual Values (2020-2025)

**JTBD**: Calculate the same indices for recent historical years to compare against normals.

**Actions**:
1. Subset aggregated data to 2020-2025
2. Compute daily indices (same config as Step 5)
3. Aggregate to monthly/annual periods
4. Store as separate zarr with `time` dimension preserved

**Commands**:
```bash
# First subset to 2020-2025
uv run python -c "
import xarray as xr
ds = xr.open_zarr('/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_1980_2025.zarr')
ds_recent = ds.sel(time=slice('2020', '2025'))
ds_recent.to_zarr('/Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_2020_2025.zarr', mode='w')
"

# Then compute indices
weather-indices compute-indices \
  /Users/tommylees/data/weather/processed/areal_aggregation/vnm_adm1_2020_2025.zarr \
  --spatial-id vnm_adm1 \
  --config configs/vietnam_coffee_indices.yaml \
  -o /Users/tommylees/data/weather/processed/indices/vnm_adm1_indices_2020_2025.zarr
```

**Output**: Indices zarr with actual 2020-2025 values.

---

## Step 7: Compute Anomalies

**JTBD**: Calculate departures from normal to identify extreme conditions.

**Actions**:
1. Load climatology (normals) and actual values
2. Compute anomalies: `anomaly = actual - climatology_mean`
3. Compute standardised anomalies: `z_score = (actual - mean) / std`
4. Flag extreme events (|z| > 2)
5. Save anomaly dataset

**Script** (`scripts/compute_anomalies.py`):
```python
import xarray as xr

# Load data
clim = xr.open_zarr('.../climatology/vnm_adm1_climatology.zarr')
actual = xr.open_zarr('.../areal_aggregation/vnm_adm1_2020_2025.zarr')

# Align by day-of-year
actual_doy = actual.groupby('time.dayofyear')

# Compute anomalies
anomaly = actual_doy - clim['mean']
z_score = anomaly / clim['std']

# Flag extremes
extreme_hot = z_score['tas'] > 2
extreme_dry = z_score['pr'] < -2

# Save
anomaly.to_zarr('.../anomalies/vnm_adm1_anomalies_2020_2025.zarr')
```

**Output**: Anomaly zarr with `anomaly` and `z_score` for each variable.

---

## Step 8: Create Production Plots

**JTBD**: Generate publication-quality visualisations for the conference booth.

**Status**: ✅ COMPLETE

### Implementation Notes

1. **Production plot styling** - ✅ Implemented via `plt.rcParams` with `axes.spines.top: False` and `axes.spines.right: False`
2. **tf_data_ml_utils imports** - Not required for visualisation script (it loads pre-computed data, doesn't compute indices/climatology)
3. **Data handling** - Visualisation script loads pre-processed zarr data and creates plots

### Sub-steps

### 8a: Time Series Plots

**Actions**:
1. Plot precipitation, temperature, evaporation for Central Highlands provinces
2. Show actual values vs climatology (with uncertainty bands)
3. Highlight 2023/24 drought and 2025 floods
4. **Production styling**:
   - `ax.spines["right"].set_visible(False)`
   - `ax.spines["top"].set_visible(False)`
5. **Truncate NaN data** - Find last valid timestamp and slice data to end there

**Variables to plot**:
- Daily precipitation with 30-day rolling mean
- Daily temperature (min/mean/max) with climatology band
- Cumulative precipitation vs normal
- Soil moisture anomalies

### 8b: Anomaly Maps

**Actions**:
1. Create choropleth maps of Vietnam showing anomalies by province
2. Focus on coffee-relevant provinces (Dak Lak, Lam Dong, Gia Lai, Dak Nong, Kon Tum)
3. Show drought intensity (2024) and flood intensity (2025)

### 8c: Index Summary Dashboard (Day-of-Year Format)

**Actions**:
1. **Use `stack_by_doy` pattern** from `tf_data_ml_utils.weather.scripts.climo_index`:
   ```python
   def stack_by_doy(da: xr.DataArray) -> xr.DataArray:
       """Reshape (time) -> (year, dayofyear)."""
       da_coords = da.assign_coords({
           "dayofyear": da["time"].dt.dayofyear,
           "year": da["time"].dt.year,
       })
       return da_coords.set_index({"time": ["year", "dayofyear"]}).unstack("time")
   ```
2. **X-axis**: Day of year (1-365)
3. **Y-axis**: Index value (GDD, EDD, CDD, etc.)
4. **Climatology bands**: Show mean ± 1 std as shaded region
5. **Multiple years overlaid**: Historical years in grey, current year highlighted
6. **Cumulative indices**: Use `cumsum_doy` for accumulated metrics

**Example plot structure**:
```python
# Climatology band (mean ± std)
ax.fill_between(dayofyear, clim_mean - clim_std, clim_mean + clim_std,
                alpha=0.2, color="C0", label="Climatology ±1σ")
ax.plot(dayofyear, clim_mean, color="C0", lw=2, label="Climatology Mean")

# Historical years (grey)
for year in historical_years:
    ax.plot(dayofyear, data.sel(year=year), color="grey", alpha=0.3, lw=0.5)

# Current year (highlighted)
ax.plot(dayofyear, data.sel(year=current_year), color="C1", lw=2, label=f"{current_year}")

# Production styling
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
```

### 8d: Data Truncation

**JTBD**: Remove NaN data that appears after the last ingest date.

**Actions**:
1. Find the last valid (non-NaN) timestamp in the data
2. Slice all datasets to end at this timestamp before plotting
3. Do not display extrapolated or missing future data

```python
def find_last_valid_timestamp(da: xr.DataArray) -> pd.Timestamp:
    """Find the last timestamp with valid (non-NaN) data."""
    valid_mask = ~np.isnan(da.values)
    last_valid_idx = np.where(valid_mask.any(axis=tuple(range(1, valid_mask.ndim))))[0][-1]
    return pd.Timestamp(da.time.values[last_valid_idx])

# Usage
last_valid = find_last_valid_timestamp(ds["tas"])
ds = ds.sel(time=slice(None, last_valid))
```

**Output Location**: `/Users/tommylees/github/vietnam_coffee_synthetic/artefacts/weather_risk/`

---

## Summary: Pipeline Stages

| Step | Stage | Input | Output | CLI/Script |
|------|-------|-------|--------|------------|
| 1 | Inspect | Raw zarr | Report | Python script |
| 2 | Standardise | Raw zarr | Interim zarr | `weather-standardise` |
| 3 | Areal Aggregation | Interim zarr + boundaries | Processed zarr (×3) | `weather-areal` |
| 4 | Climatology | Processed zarr | Climatology zarr | `weather-climatology` |
| 5 | Indices (Normal) | Climatology zarr | Indices zarr (normal) | `weather-indices` |
| 6 | Indices (Actual) | Processed zarr (2020-25) | Indices zarr (actual) | `weather-indices` |
| 7 | Anomalies | Climatology + Actual | Anomalies zarr | Python script |
| 8 | Visualisation | All outputs | PNG plots | Python/matplotlib |

---

## Prerequisites

```bash
# Install tf-data-ml-utils with weather dependencies
cd ~/github/tf-data-ml-utils
uv pip install -e ".[weather]"

# Verify installation
weather-pipeline --help
weather-standardise --help
weather-areal --help
weather-climatology --help
weather-indices --help
```

---

## Key Outputs for Conference Booth

1. **Time series**: Vietnam Central Highlands precip/temp vs climatology (2020-2025)
2. **Anomaly map**: 2024 drought severity by province
3. **Risk indices**: GDD, EDD, SPI trends for coffee regions
4. **Event timeline**: Overlay of weather extremes with yield impacts
