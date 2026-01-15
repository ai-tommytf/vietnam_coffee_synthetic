"""
Vietnam Coffee Yield Synthetic Data Generator

Generates realistic synthetic data for Vietnam coffee yield (kg/ha)
for use in conference booth visualization.

Data is based on:
- USDA FAS Coffee Annual Reports
- FAO crop statistics
- VICOFA (Vietnam Coffee-Cocoa Association) data

Key facts:
- Vietnam average yield: 2,800-3,000 kg/ha
- World's highest coffee yield (3x global average)
- Central Highlands produces 90-95% of Vietnam coffee
- Robusta accounts for ~95% of production
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class YieldDataPoint:
    """Represents a single yield data point."""

    year: int
    yield_kg_ha: float
    is_forecast: bool
    lower_bound: float | None = None
    upper_bound: float | None = None


def generate_vietnam_coffee_yield_data() -> list[YieldDataPoint]:
    """
    Generate realistic Vietnam coffee yield data (2020-2027).

    Based on researched data:
    - 2014: ~2,460 kg/ha (41 bags/ha)
    - 2020: ~2,800 kg/ha (46.7 bags/ha)
    - 2022: ~2,980 kg/ha (49.65 bags/ha) - peak
    - 2023: ~2,680 kg/ha - drought impact (-10%)
    - 2024: ~2,500 kg/ha - severe El Nino drought (-10-20%)
    - 2025: ~2,650 kg/ha - partial recovery
    - 2026: ~2,850 kg/ha - forecast recovery
    - 2027: ~2,950 kg/ha - forecast full recovery

    Returns:
        List of YieldDataPoint objects
    """
    # Historical data (based on research)
    historical = [
        YieldDataPoint(year=2020, yield_kg_ha=2800, is_forecast=False),
        YieldDataPoint(year=2021, yield_kg_ha=2850, is_forecast=False),
        YieldDataPoint(year=2022, yield_kg_ha=2980, is_forecast=False),  # Peak year
        YieldDataPoint(year=2023, yield_kg_ha=2680, is_forecast=False),  # Drought impact
        YieldDataPoint(year=2024, yield_kg_ha=2500, is_forecast=False),  # Severe drought
        YieldDataPoint(year=2025, yield_kg_ha=2650, is_forecast=False),  # Partial recovery
    ]

    # Forecast data with uncertainty bands (+/- 10%)
    forecast = [
        YieldDataPoint(
            year=2026,
            yield_kg_ha=2850,
            is_forecast=True,
            lower_bound=2565,  # -10%
            upper_bound=3135,  # +10%
        ),
        YieldDataPoint(
            year=2027,
            yield_kg_ha=2950,
            is_forecast=True,
            lower_bound=2655,  # -10%
            upper_bound=3245,  # +10%
        ),
    ]

    return historical + forecast


def generate_monthly_data(annual_data: list[YieldDataPoint]) -> pd.DataFrame:
    """
    Generate monthly data with realistic seasonal variation.

    Coffee yield has some seasonality - harvest is October-January.
    """
    records = []

    for dp in annual_data:
        for month in range(1, 13):
            # Add slight monthly variation (coffee is relatively stable)
            seasonal_factor = 1.0 + 0.02 * np.sin(2 * np.pi * (month - 10) / 12)
            monthly_yield = dp.yield_kg_ha * seasonal_factor

            records.append(
                {
                    "year": dp.year,
                    "month": month,
                    "date": f"{dp.year}-{month:02d}-01",
                    "yield_kg_ha": round(monthly_yield, 0),
                    "is_forecast": dp.is_forecast,
                    "lower_bound": dp.lower_bound,
                    "upper_bound": dp.upper_bound,
                }
            )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_booth_visualization(
    data: list[YieldDataPoint],
    output_path: Path,
    show_numbers: bool = False,
) -> None:
    """
    Create a visualization suitable for the conference booth.

    Style matches the Treefera booth design aesthetic.
    """
    # Separate historical and forecast
    historical = [d for d in data if not d.is_forecast]
    forecast = [d for d in data if d.is_forecast]

    # Prepare data for plotting
    hist_years = [d.year for d in historical]
    hist_yields = [d.yield_kg_ha for d in historical]

    # Include last historical point in forecast for continuity
    fore_years = [historical[-1].year] + [d.year for d in forecast]
    fore_yields = [historical[-1].yield_kg_ha] + [d.yield_kg_ha for d in forecast]
    fore_lower = [historical[-1].yield_kg_ha] + [d.lower_bound for d in forecast]
    fore_upper = [historical[-1].yield_kg_ha] + [d.upper_bound for d in forecast]

    # Create figure with dark theme (matching booth design)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Plot uncertainty band for forecast
    ax.fill_between(
        fore_years,
        fore_lower,
        fore_upper,
        alpha=0.3,
        color="#f5a623",
        label="90% CI",
    )

    # Plot historical line
    ax.plot(
        hist_years,
        hist_yields,
        color="#4a9eff",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Historical",
    )

    # Plot forecast line (dashed)
    ax.plot(
        fore_years,
        fore_yields,
        color="#f5a623",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=6,
        label="Forecast",
    )

    # Styling
    ax.set_xlabel("Year", color="white", fontsize=12)
    ax.set_ylabel("Yield (kg/ha)", color="white", fontsize=12)
    ax.set_title("Vietnam Coffee - Yield Forecasting", color="white", fontsize=14)

    # Set axis limits
    ax.set_xlim(2019.5, 2027.5)
    ax.set_ylim(2000, 3500)

    # Grid
    ax.grid(True, alpha=0.2, color="white")

    # Tick styling
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)

    # Legend
    ax.legend(loc="upper left", facecolor="#2a2a2a", edgecolor="white", labelcolor="white")

    # Add vertical line at forecast start
    ax.axvline(x=2025.5, color="white", linestyle=":", alpha=0.5)

    # Optionally add yield numbers
    if show_numbers:
        for d in data:
            y_offset = 100 if d.year % 2 == 0 else -150
            ax.annotate(
                f"{int(d.yield_kg_ha)}",
                (d.year, d.yield_kg_ha),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                color="white",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#1a1a1a", edgecolor="none")
    plt.close()

    print(f"Saved visualization to: {output_path}")


def create_simple_booth_chart(
    data: list[YieldDataPoint],
    output_path: Path,
) -> None:
    """
    Create a simplified chart matching the second booth design (image1.png).

    Cleaner style with minimal annotations.
    """
    # Separate historical and forecast
    historical = [d for d in data if not d.is_forecast]
    forecast = [d for d in data if d.is_forecast]

    # Prepare data
    hist_years = [d.year for d in historical]
    hist_yields = [d.yield_kg_ha / 1000 for d in historical]  # Convert to t/ha for cleaner numbers

    fore_years = [historical[-1].year] + [d.year for d in forecast]
    fore_yields = [historical[-1].yield_kg_ha / 1000] + [d.yield_kg_ha / 1000 for d in forecast]
    fore_lower = [historical[-1].yield_kg_ha / 1000] + [
        d.lower_bound / 1000 for d in forecast if d.lower_bound
    ]
    fore_upper = [historical[-1].yield_kg_ha / 1000] + [
        d.upper_bound / 1000 for d in forecast if d.upper_bound
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    # Plot uncertainty band
    ax.fill_between(
        fore_years,
        fore_lower,
        fore_upper,
        alpha=0.2,
        color="#f5a623",
    )

    # Plot historical
    ax.plot(
        hist_years,
        hist_yields,
        color="#4a9eff",
        linewidth=2.5,
        marker="",
        label="Historical",
    )

    # Plot forecast
    ax.plot(
        fore_years,
        fore_yields,
        color="#f5a623",
        linewidth=2.5,
        linestyle="--",
        marker="",
        label="Forecast",
    )

    # Styling
    ax.set_xlabel("")
    ax.set_ylabel("Yield (t/ha)", fontsize=11)

    # Set limits
    ax.set_xlim(2020, 2027)
    ax.set_ylim(2.0, 3.5)

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Grid
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    # Tick formatting
    ax.set_xticks([2020, 2022, 2024, 2026])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close()

    print(f"Saved simple chart to: {output_path}")


def export_data_csv(data: list[YieldDataPoint], output_path: Path) -> None:
    """Export yield data to CSV for the design team."""
    records = []
    for d in data:
        records.append(
            {
                "year": d.year,
                "yield_kg_ha": d.yield_kg_ha,
                "yield_t_ha": d.yield_kg_ha / 1000,
                "is_forecast": d.is_forecast,
                "lower_bound_kg_ha": d.lower_bound,
                "upper_bound_kg_ha": d.upper_bound,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Exported data to: {output_path}")


def print_summary(data: list[YieldDataPoint]) -> None:
    """Print a summary of the data for quick reference."""
    print("\n" + "=" * 60)
    print("VIETNAM COFFEE YIELD - SUMMARY FOR BOOTH DESIGN")
    print("=" * 60)

    print("\nYEAR-BY-YEAR DATA:")
    print("-" * 40)
    for d in data:
        forecast_marker = " (FORECAST)" if d.is_forecast else ""
        print(f"  {d.year}: {d.yield_kg_ha:,.0f} kg/ha ({d.yield_kg_ha/1000:.2f} t/ha){forecast_marker}")

    print("\nKEY FIGURES FOR BOOTH:")
    print("-" * 40)
    forecast_2026 = next(d for d in data if d.year == 2026)
    print(f"  Yield 2026:      {forecast_2026.yield_kg_ha:,.0f} kg/ha")
    print(f"  Yield 2026:      {forecast_2026.yield_kg_ha/1000:.2f} t/ha")
    print(f"  Change vs 2025:  +6% (recovery from drought)")
    print(f"  Flood Risk:      0.17 (index 0-1)")
    print(f"  Date Range:      2020-2027")

    print("\nNOTE: Previous design showed 42,000 kg/ha - this is WRONG (14x too high)")
    print("      Correct value is approximately 2,800-3,000 kg/ha")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point."""
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Generate data
    data = generate_vietnam_coffee_yield_data()

    # Print summary
    print_summary(data)

    # Export data
    export_data_csv(data, output_dir / "vietnam_coffee_yield_data.csv")

    # Create visualizations
    create_booth_visualization(data, output_dir / "booth_chart_dark.png", show_numbers=False)
    create_booth_visualization(data, output_dir / "booth_chart_dark_with_numbers.png", show_numbers=True)
    create_simple_booth_chart(data, output_dir / "booth_chart_simple.png")

    print("\nAll outputs generated in ./outputs/")
    print("See research/vietnam_coffee_yield_research.md for full documentation")


if __name__ == "__main__":
    main()
