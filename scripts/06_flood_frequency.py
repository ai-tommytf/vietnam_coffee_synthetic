#!/usr/bin/env python3
"""
Vietnam Flood Hazard Analysis Dashboard

This script creates a static dashboard visualising Vietnam's flood hazard
based on multiple data sources and the ThinkHazard methodology.

Data Sources:
1. ThinkHazard (GFDRR/FATHOM) - Hazard classification model
2. EM-DAT - International Disaster Database (1980-2024)
3. ERIA Research Paper - Government official flood data (1990-2010)
4. FloodList / ReliefWeb - Recent event documentation

ThinkHazard Methodology:
- Uses FATHOM global flood model at 30m resolution
- Classifies hazard based on return period thresholds:
  * HIGH: 10-year return period (damaging floods expected within 10 years)
  * MEDIUM: 50-year return period
  * LOW: 1000-year return period
- Damaging intensity threshold: 0.5m flood depth
- Area threshold: 1% of admin unit flooded to damaging depth
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FloodEvent:
    """A documented flood event in Vietnam."""

    year: int
    month: Optional[str]
    deaths: int
    damage_usd_millions: Optional[float]
    affected_people: Optional[int]
    region: str
    event_name: str
    source: str


# Documented major flood events from multiple sources
DOCUMENTED_EVENTS: list[FloodEvent] = [
    # 1990s
    FloodEvent(1996, "Oct", 217, None, None, "Central Vietnam", "1996 Floods", "ERIA/EM-DAT"),
    FloodEvent(1999, "Oct-Nov", 595, 170, 1_000_000, "Central Vietnam", "1999 Vietnamese Floods (worst in 40 years)", "EM-DAT/ERIA"),
    FloodEvent(1999, "Dec", 132, None, None, "Central Vietnam", "December 1999 Floods", "EM-DAT"),
    # 2000s
    FloodEvent(2000, "Sep", 539, 210, 2_000_000, "Mekong Delta", "2000 Mekong Floods (historic levels)", "EM-DAT/ERIA"),
    FloodEvent(2004, "Nov", 40, None, 170_000, "Central Provinces", "2004 Floods", "Facts & Details"),
    FloodEvent(2005, "Oct-Dec", 57, None, None, "Mekong Delta/Central", "2005 Floods", "Facts & Details"),
    FloodEvent(2007, "Nov", 332, None, None, "Central Vietnam", "2007 Floods", "Facts & Details"),
    FloodEvent(2008, "Jul", 550, 700, None, "Red River/Northern", "2008 Floods", "Facts & Details/ERIA"),
    # 2010s
    FloodEvent(2010, "Oct", 112, 138, 200_000, "Central Vietnam", "2010 Central Floods", "Facts & Details"),
    FloodEvent(2011, "Dec", 55, 70, None, "Central Vietnam", "2011 Floods", "EM-DAT"),
    FloodEvent(2013, "Nov", 42, None, 400_000, "Central Provinces", "2013 Floods", "Facts & Details"),
    FloodEvent(2015, "Jul", 23, 60, None, "Quang Ninh", "2015 Quang Ninh Flood", "EM-DAT"),
    FloodEvent(2016, "Oct", 49, None, None, "Central Vietnam", "2016 Floods", "EM-DAT"),
    FloodEvent(2017, "Aug", 386, 2650, None, "Multiple Regions", "2017 Floods (worst in decade)", "EM-DAT"),
    # 2020s
    FloodEvent(2020, "Oct-Nov", 243, 1520, 7_700_000, "Central Vietnam", "2020 Central Vietnam Floods", "VNDMA/ReliefWeb"),
    FloodEvent(2024, "Sep", 519, 3200, 168_000, "Northern Vietnam", "Typhoon Yagi Floods (strongest in decades)", "OCHA/ReliefWeb"),
]


@dataclass
class DataSource:
    """A data source for flood frequency estimates."""

    name: str
    period: str
    flood_count: int
    years: int
    methodology: str
    url: str

    @property
    def annual_rate(self) -> float:
        return self.flood_count / self.years


DATA_SOURCES = [
    DataSource(
        "ERIA Research Paper",
        "1990-2010",
        74,
        20,
        "Government official flood records",
        "https://www.eria.org/ERIA-DP-2013-11.pdf",
    ),
    DataSource(
        "EM-DAT Database",
        "1980-2024",
        110,
        44,
        "International disaster database (CRED)",
        "https://www.emdat.be/",
    ),
    DataSource(
        "ThinkHazard/FATHOM",
        "Modelled",
        0,  # Model-based, not event count
        0,
        "Stochastic flood model (10,000 year simulation)",
        "https://thinkhazard.org/en/report/264-vietnam/FL",
    ),
]


# ThinkHazard classification thresholds
THINKHAZARD_LEVELS = {
    "HIGH": {"return_period": 10, "description": "Damaging floods expected at least once in 10 years", "color": "#e74c3c"},
    "MEDIUM": {"return_period": 50, "description": "Damaging floods expected once in 50 years", "color": "#f39c12"},
    "LOW": {"return_period": 1000, "description": "Rare flood events", "color": "#27ae60"},
    "VERY LOW": {"return_period": float("inf"), "description": "Negligible flood risk", "color": "#3498db"},
}


def _despine(ax):
    """Remove top and right spines from axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_flood_dashboard():
    """Create comprehensive flood hazard dashboard."""

    output_dir = Path(__file__).parent.parent / "artefacts" / "weather_risk"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Vietnam Flood Hazard Analysis: Data & Methodology",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, left=0.06, right=0.94, top=0.92, bottom=0.05)

    # 1. ThinkHazard Classification Explanation (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_thinkhazard_classification(ax1)

    # 2. Historical Events Timeline (top middle + right)
    ax2 = fig.add_subplot(gs[0, 1:])
    _plot_events_timeline(ax2)

    # 3. Deaths by Event (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_deaths_by_event(ax3)

    # 4. Data Sources Comparison (middle middle)
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_data_sources(ax4)

    # 5. FATHOM Model Methodology (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    _plot_fathom_methodology(ax5)

    # 6. Return Period Explanation (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    _plot_return_period_explanation(ax6)

    # 7. Economic Impact (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    _plot_economic_impact(ax7)

    # 8. Key Statistics Summary (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    _plot_key_statistics(ax8)

    output_path = output_dir / "06_flood_frequency_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Dashboard saved: {output_path}")
    return output_path


def _plot_thinkhazard_classification(ax):
    """Plot ThinkHazard hazard level classification."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("ThinkHazard Classification", fontsize=11, fontweight="bold", pad=10)

    # Vietnam status box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, 7), 9, 2.5, boxstyle="round,pad=0.1",
        facecolor="#e74c3c", edgecolor="black", linewidth=2, alpha=0.9
    ))
    ax.text(5, 8.6, "VIETNAM: HIGH HAZARD", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(5, 7.6, "Damaging floods expected at least", ha="center", va="center",
            fontsize=9, color="white")
    ax.text(5, 7.1, "once in the next 10 years", ha="center", va="center",
            fontsize=9, color="white")

    # Classification levels
    levels = [
        ("HIGH", 10, "#e74c3c", 5.5),
        ("MEDIUM", 50, "#f39c12", 4.0),
        ("LOW", 1000, "#27ae60", 2.5),
        ("VERY LOW", "∞", "#3498db", 1.0),
    ]

    for name, rp, color, y in levels:
        ax.add_patch(mpatches.Rectangle((0.5, y - 0.4), 2.5, 0.7,
                                         facecolor=color, edgecolor="black", alpha=0.8))
        ax.text(1.75, y, name, ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        ax.text(4, y, f"{rp}-yr return period", ha="left", va="center", fontsize=8)

    ax.text(5, 0.3, "Source: gfdrr.github.io/thinkhazardmethods", ha="center",
            fontsize=7, style="italic", color="gray")


def _plot_events_timeline(ax):
    """Plot documented flood events timeline."""
    ax.set_title("Documented Major Flood Events (1996-2024)", fontsize=11, fontweight="bold")

    years = [e.year for e in DOCUMENTED_EVENTS]
    deaths = [e.deaths for e in DOCUMENTED_EVENTS]
    names = [e.event_name.split("(")[0].strip()[:20] for e in DOCUMENTED_EVENTS]

    # Color by severity
    colors = []
    for d in deaths:
        if d >= 400:
            colors.append("#e74c3c")  # Severe
        elif d >= 100:
            colors.append("#f39c12")  # Moderate
        else:
            colors.append("#3498db")  # Lower

    bars = ax.bar(range(len(years)), deaths, color=colors, edgecolor="black", alpha=0.8)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"{y}" for y in years], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Deaths", fontsize=10)
    ax.set_xlabel("Year", fontsize=10)

    # Annotate major events
    for i, (bar, name, d) in enumerate(zip(bars, names, deaths)):
        if d >= 200:
            ax.annotate(f"{name}\n({d})", xy=(i, d), xytext=(i, d + 50),
                       fontsize=6, ha="center", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label="Severe (≥400 deaths)"),
        mpatches.Patch(facecolor="#f39c12", label="Moderate (100-399)"),
        mpatches.Patch(facecolor="#3498db", label="Lower (<100)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    _despine(ax)


def _plot_deaths_by_event(ax):
    """Plot cumulative deaths and top events."""
    ax.set_title("Top 5 Deadliest Flood Events", fontsize=11, fontweight="bold")

    # Sort by deaths
    sorted_events = sorted(DOCUMENTED_EVENTS, key=lambda x: x.deaths, reverse=True)[:5]

    names = [f"{e.year}: {e.event_name[:25]}" for e in sorted_events]
    deaths = [e.deaths for e in sorted_events]

    bars = ax.barh(range(len(names)), deaths, color="#e74c3c", edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Deaths", fontsize=10)
    ax.invert_yaxis()

    for bar, d in zip(bars, deaths):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{d:,}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(deaths) * 1.2)
    ax.grid(axis="x", alpha=0.3)
    _despine(ax)


def _plot_data_sources(ax):
    """Plot comparison of data sources."""
    ax.set_title("Data Source Comparison", fontsize=11, fontweight="bold")

    sources = [s for s in DATA_SOURCES if s.years > 0]
    names = [s.name for s in sources]
    rates = [s.annual_rate for s in sources]

    bars = ax.bar(names, rates, color=["#2ecc71", "#3498db"], edgecolor="black", alpha=0.8)
    ax.set_ylabel("Floods per Year", fontsize=10)

    for bar, rate, src in zip(bars, rates, sources):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rate:.1f}/yr\n({src.flood_count} events\n{src.period})",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(rates) * 1.5)

    # Add consensus estimate line
    consensus = np.mean(rates)
    ax.axhline(y=consensus, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Consensus: {consensus:.1f}/yr")
    ax.legend(loc="upper right", fontsize=8)
    _despine(ax)


def _plot_fathom_methodology(ax):
    """Explain FATHOM model methodology."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("FATHOM Model Methodology", fontsize=11, fontweight="bold", pad=10)

    methodology_text = """FATHOM Global Flood Model:

• Resolution: 30m globally
• Simulation: 10,000 years of
  stochastic flood events
• Return periods: 2 to 1000+ years
• Flood types: Fluvial, Pluvial, Coastal

Classification Criteria:
• Damaging depth: >0.5m
• Area threshold: >1% of admin unit
  flooded to damaging depth

Data Source:
• FABDEM+ terrain model
• Peer-reviewed in Nature journals
• Used by World Bank, GFDRR

Note: Dataset not publicly available
due to licensing restrictions."""
    ax.text(0.05, 0.95, methodology_text, fontsize=8, va="top",
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))


def _plot_return_period_explanation(ax):
    """Explain return period concept."""
    ax.set_title("Return Period Explained", fontsize=11, fontweight="bold")

    return_periods = [2, 5, 10, 25, 50, 100]
    annual_probs = [1 / rp * 100 for rp in return_periods]

    bars = ax.bar([f"{rp}-yr" for rp in return_periods], annual_probs,
                  color="#3498db", edgecolor="black", alpha=0.8)

    # Vietnam actual (HIGH hazard = 10-year return)
    ax.axhline(y=10, color="#e74c3c", linewidth=2.5, linestyle="--",
               label="Vietnam HIGH threshold (10%)")

    ax.set_ylabel("Annual Exceedance Prob. (%)", fontsize=9)
    ax.set_xlabel("Return Period", fontsize=9)

    for bar, prob in zip(bars, annual_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{prob:.0f}%", ha="center", fontsize=8)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 60)

    # Add explanation
    ax.text(0.02, 0.02,
            "Return period ≠ 'happens every N years'\n"
            "10-yr flood = 10% chance any given year",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    _despine(ax)


def _plot_economic_impact(ax):
    """Plot economic impact data."""
    ax.set_title("Economic Impact (Major Events)", fontsize=11, fontweight="bold")

    events_with_damage = [e for e in DOCUMENTED_EVENTS if e.damage_usd_millions]
    events_with_damage = sorted(events_with_damage, key=lambda x: x.damage_usd_millions or 0, reverse=True)[:6]

    names = [f"{e.year}" for e in events_with_damage]
    damages = [e.damage_usd_millions for e in events_with_damage]

    bars = ax.bar(names, damages, color="#9b59b6", edgecolor="black", alpha=0.8)
    ax.set_ylabel("Damage (USD millions)", fontsize=10)
    ax.set_xlabel("Year", fontsize=10)

    for bar, d in zip(bars, damages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"${d:,.0f}M", ha="center", fontsize=8, fontweight="bold")

    ax.set_ylim(0, max(damages) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # Add context
    ax.text(0.98, 0.98,
            "Floods = 97% of\nannual hazard losses",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="#ffe6e6", alpha=0.9))
    _despine(ax)


def _plot_key_statistics(ax):
    """Summary statistics panel."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Key Statistics Summary", fontsize=11, fontweight="bold", pad=10)

    # Calculate stats
    total_deaths = sum(e.deaths for e in DOCUMENTED_EVENTS)
    avg_deaths_per_event = total_deaths / len(DOCUMENTED_EVENTS)
    total_damage = sum(e.damage_usd_millions for e in DOCUMENTED_EVENTS if e.damage_usd_millions)

    stats_text = f"""
FLOOD FREQUENCY:
  • 3.7 significant floods/year (ERIA)
  • 2.5 significant floods/year (EM-DAT)
  • Consensus: ~3 floods/year

IMPACT (Documented 1996-2024):
  • Total deaths: {total_deaths:,}
  • Events recorded: {len(DOCUMENTED_EVENTS)}
  • Avg deaths/event: {avg_deaths_per_event:.0f}
  • Total damage: ${total_damage:,.0f}M USD

VULNERABILITY:
  • 46% of population at flood risk
  • 70% live in coastal/delta areas
  • Risk index: 9.9/10 (2024)

CLIMATE OUTLOOK:
  • Medium confidence in increased
    heavy precipitation frequency
  • Hazard level may increase
"""
    ax.text(0.05, 0.95, stats_text.strip(), fontsize=9, va="top",
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4e8", alpha=0.9))

    # Data sources
    ax.text(0.5, 0.02,
            "Sources: ThinkHazard, EM-DAT, ERIA, ReliefWeb, OCHA",
            transform=ax.transAxes, fontsize=7, ha="center", style="italic", color="gray")


def main():
    """Generate the flood hazard dashboard."""
    print("=" * 70)
    print("VIETNAM FLOOD HAZARD ANALYSIS DASHBOARD")
    print("=" * 70)
    print()

    print("Data Sources:")
    print("-" * 50)
    for src in DATA_SOURCES:
        print(f"  • {src.name}")
        print(f"    Period: {src.period}")
        print(f"    Method: {src.methodology}")
        print(f"    URL: {src.url}")
        print()

    print("Documented Events:")
    print("-" * 50)
    print(f"  Total events: {len(DOCUMENTED_EVENTS)}")
    print(f"  Total deaths: {sum(e.deaths for e in DOCUMENTED_EVENTS):,}")
    print(f"  Date range: {min(e.year for e in DOCUMENTED_EVENTS)}-{max(e.year for e in DOCUMENTED_EVENTS)}")
    print()

    print("ThinkHazard Classification for Vietnam:")
    print("-" * 50)
    print("  Level: HIGH")
    print("  Meaning: Damaging floods expected at least once in 10 years")
    print("  Model: FATHOM Global Flood (FL-GLOBAL-FATHOM)")
    print("  Resolution: 30m globally")
    print("  Intensity threshold: 0.5m flood depth")
    print()

    output_path = create_flood_dashboard()

    print("=" * 70)
    print(f"Dashboard generated: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
