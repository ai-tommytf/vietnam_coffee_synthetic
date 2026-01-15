"""
Vietnam Coffee Growing Regions - Production Quality Maps

Creates publication-ready maps showing Vietnam's key coffee producing regions
in the Central Highlands (Tay Nguyen).

Key coffee provinces:
- Dak Lak (Đắk Lắk) - Largest producer
- Lam Dong (Lâm Đồng) - High-altitude arabica
- Dak Nong (Đắk Nông)
- Gia Lai
- Kon Tum

Together these 5 provinces account for ~92% of Vietnam's coffee cultivation area
and ~90% of national output.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("/Users/tommylees/github/vietnam_coffee_synthetic/artefacts")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colour palette - professional, print-ready
COLOURS = {
    "vietnam_base": "#E8E8E8",  # Light grey for non-coffee regions
    "coffee_primary": "#2D5016",  # Dark green for main coffee provinces
    "coffee_secondary": "#4A7C23",  # Medium green for secondary
    "dak_lak": "#1B3D0F",  # Darkest green for Dak Lak (largest producer)
    "highlight": "#8B4513",  # Brown accent for coffee
    "water": "#B8D4E8",  # Light blue for surrounding water
    "border": "#404040",  # Dark grey for borders
    "text": "#1A1A1A",  # Near black for text
    "background": "#FAFAFA",  # Off-white background
}

# Coffee province production data (2025 estimates)
COFFEE_DATA = {
    "gia lai": {"production_pct": 18, "area_ha": 98000, "yield_kg_ha": 2750},
    "kon tum": {"production_pct": 8, "area_ha": 26000, "yield_kg_ha": 2600},
    "lam ??ng": {"production_pct": 22, "area_ha": 176000, "yield_kg_ha": 2900},
    "??k l?k": {"production_pct": 35, "area_ha": 210000, "yield_kg_ha": 2850},
    "??k nong": {"production_pct": 12, "area_ha": 136000, "yield_kg_ha": 2700},
}

# Clean province name mapping
PROVINCE_DISPLAY_NAMES = {
    "gia lai": "Gia Lai",
    "kon tum": "Kon Tum",
    "lam ??ng": "Lâm Đồng",
    "??k l?k": "Đắk Lắk",
    "??k nong": "Đắk Nông",
    "s?n la": "Sơn La",
}


def load_vietnam_boundaries():
    """Load Vietnam administrative boundaries from geoboundaries."""
    gdf = gpd.read_parquet(
        "/Users/tommylees/data/raw/boundaries/all_geoboundaries_processed.parquet"
    )
    vnm = gdf[gdf["shapegroup"] == "VNM"].copy()
    return vnm


def identify_coffee_provinces(adm1_gdf):
    """Identify coffee-growing provinces in the Central Highlands."""
    # Central Highlands coffee provinces (geoname patterns)
    central_highlands = ["gia lai", "kon tum", "lam ??ng", "??k l?k", "??k nong"]
    # Northern arabica province
    northern_arabica = ["s?n la"]

    adm1_gdf["is_coffee_central"] = adm1_gdf["geoname"].isin(central_highlands)
    adm1_gdf["is_coffee_northern"] = adm1_gdf["geoname"].isin(northern_arabica)
    adm1_gdf["is_coffee"] = (
        adm1_gdf["is_coffee_central"] | adm1_gdf["is_coffee_northern"]
    )
    adm1_gdf["is_dak_lak"] = adm1_gdf["geoname"] == "??k l?k"

    return adm1_gdf


def create_vietnam_overview_map(vnm_gdf):
    """Create an overview map of Vietnam highlighting coffee regions."""
    adm0 = vnm_gdf[vnm_gdf["shapetype"] == "ADM0"]
    adm1 = vnm_gdf[vnm_gdf["shapetype"] == "ADM1"].copy()
    adm1 = identify_coffee_provinces(adm1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 14), facecolor=COLOURS["background"])
    ax.set_facecolor(COLOURS["water"])

    # Plot all provinces first (base layer)
    non_coffee = adm1[~adm1["is_coffee"]]
    non_coffee.plot(
        ax=ax,
        color=COLOURS["vietnam_base"],
        edgecolor=COLOURS["border"],
        linewidth=0.3,
    )

    # Plot coffee provinces (excluding Dak Lak)
    coffee_not_dak_lak = adm1[adm1["is_coffee"] & ~adm1["is_dak_lak"]]
    coffee_not_dak_lak.plot(
        ax=ax,
        color=COLOURS["coffee_primary"],
        edgecolor=COLOURS["border"],
        linewidth=0.5,
    )

    # Plot Dak Lak separately (largest producer)
    dak_lak = adm1[adm1["is_dak_lak"]]
    dak_lak.plot(
        ax=ax, color=COLOURS["dak_lak"], edgecolor=COLOURS["border"], linewidth=0.5
    )

    # Add province labels for coffee regions
    coffee_provinces = adm1[adm1["is_coffee"]]
    for idx, row in coffee_provinces.iterrows():
        centroid = row.geometry.centroid
        display_name = PROVINCE_DISPLAY_NAMES.get(row["geoname"], row["geoname"])
        ax.annotate(
            display_name,
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )

    # Add major cities
    cities = {
        "Hanoi": (105.85, 21.03),
        "Ho Chi Minh City": (106.63, 10.82),
        "Da Nang": (108.21, 16.05),
        "Buon Ma Thuot": (108.05, 12.67),  # Coffee capital
    }

    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, "o", color=COLOURS["highlight"], markersize=6, zorder=5)
        ax.annotate(
            city,
            xy=(lon, lat),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=COLOURS["text"],
        )

    # Title and labels
    ax.set_title(
        "Vietnam Coffee Growing Regions",
        fontsize=16,
        fontweight="bold",
        color=COLOURS["text"],
        pad=20,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor=COLOURS["dak_lak"],
            edgecolor=COLOURS["border"],
            label="Đắk Lắk (35% of production)",
        ),
        mpatches.Patch(
            facecolor=COLOURS["coffee_primary"],
            edgecolor=COLOURS["border"],
            label="Other Central Highlands (57%)",
        ),
        mpatches.Patch(
            facecolor=COLOURS["vietnam_base"],
            edgecolor=COLOURS["border"],
            label="Other provinces",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOURS["highlight"],
            markersize=8,
            label="Major cities",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor=COLOURS["border"],
        fontsize=8,
    )

    # Add annotation box
    info_text = (
        "Central Highlands (Tây Nguyên)\n"
        "• 92% of Vietnam's coffee area\n"
        "• 90% of national production\n"
        "• 95% Robusta, 5% Arabica\n"
        "• Average yield: 2,850 kg/ha"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9)
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    ax.set_axis_off()

    # Set bounds with padding
    bounds = adm0.total_bounds
    ax.set_xlim(bounds[0] - 0.5, bounds[2] + 0.5)
    ax.set_ylim(bounds[1] - 0.5, bounds[3] + 0.5)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "vietnam_coffee_overview.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor=COLOURS["background"]
    )
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_central_highlands_detail_map(vnm_gdf):
    """Create a detailed map of the Central Highlands coffee region."""
    adm1 = vnm_gdf[vnm_gdf["shapetype"] == "ADM1"].copy()
    adm1 = identify_coffee_provinces(adm1)

    # Get Central Highlands provinces and neighbours
    central_highlands_names = ["gia lai", "kon tum", "lam ??ng", "??k l?k", "??k nong"]
    central_highlands = adm1[adm1["geoname"].isin(central_highlands_names)]

    # Get bounding box of Central Highlands and expand
    bounds = central_highlands.total_bounds
    buffer = 1.0
    xlim = (bounds[0] - buffer, bounds[2] + buffer)
    ylim = (bounds[1] - buffer, bounds[3] + buffer)

    # Filter to provinces in view
    adm1_filtered = adm1.cx[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor=COLOURS["background"])
    ax.set_facecolor(COLOURS["water"])

    # Plot non-coffee provinces in view
    non_coffee = adm1_filtered[~adm1_filtered["is_coffee"]]
    non_coffee.plot(
        ax=ax,
        color=COLOURS["vietnam_base"],
        edgecolor=COLOURS["border"],
        linewidth=0.5,
    )

    # Create colour map based on production percentage
    def get_colour_by_production(geoname):
        if geoname in COFFEE_DATA:
            pct = COFFEE_DATA[geoname]["production_pct"]
            # Interpolate green shade based on production
            intensity = pct / 35  # Normalise to max (Dak Lak)
            r = int(45 * (1 - intensity * 0.6))
            g = int(124 * (0.4 + intensity * 0.6))
            b = int(35 * (1 - intensity * 0.6))
            return f"#{r:02x}{g:02x}{b:02x}"
        return COLOURS["coffee_primary"]

    # Plot each coffee province with production-based colouring
    for idx, row in central_highlands.iterrows():
        colour = get_colour_by_production(row["geoname"])
        gpd.GeoDataFrame([row], crs=adm1.crs).plot(
            ax=ax, color=colour, edgecolor=COLOURS["border"], linewidth=1
        )

    # Add labels with production data
    for idx, row in central_highlands.iterrows():
        centroid = row.geometry.centroid
        display_name = PROVINCE_DISPLAY_NAMES.get(row["geoname"], row["geoname"])

        if row["geoname"] in COFFEE_DATA:
            data = COFFEE_DATA[row["geoname"]]
            label = f"{display_name}\n{data['production_pct']}% production\n{data['area_ha']:,} ha"
        else:
            label = display_name

        ax.annotate(
            label,
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )

    # Add Buon Ma Thuot marker
    ax.plot(108.05, 12.67, "*", color=COLOURS["highlight"], markersize=15, zorder=5)
    ax.annotate(
        "Buôn Ma Thuột\n(Coffee Capital)",
        xy=(108.05, 12.67),
        xytext=(15, -15),
        textcoords="offset points",
        fontsize=8,
        color=COLOURS["text"],
        fontweight="bold",
    )

    # Title
    ax.set_title(
        "Central Highlands Coffee Region (Tây Nguyên)\nVietnam's Coffee Heartland",
        fontsize=14,
        fontweight="bold",
        color=COLOURS["text"],
        pad=20,
    )

    # Colour bar / legend for production
    from matplotlib.colors import LinearSegmentedColormap

    # Create custom colourbar
    cmap = LinearSegmentedColormap.from_list(
        "coffee", [COLOURS["coffee_secondary"], COLOURS["dak_lak"]]
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=8, vmax=35))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.3, aspect=15, pad=0.02)
    cbar.set_label("Share of National Production (%)", fontsize=9)

    # Info box
    info_text = (
        "Key Statistics (2025/26)\n"
        "─────────────────────\n"
        "Total Area: ~646,000 ha\n"
        "Production: 30.8M bags\n"
        "Avg Yield: 2,850 kg/ha\n"
        "Variety: 95% Robusta"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95)
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
        family="monospace",
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_axis_off()

    plt.tight_layout()
    output_path = OUTPUT_DIR / "central_highlands_coffee_detail.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor=COLOURS["background"]
    )
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_production_comparison_chart():
    """Create a horizontal bar chart comparing coffee provinces."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLOURS["background"])

    provinces = ["Đắk Lắk", "Lâm Đồng", "Gia Lai", "Đắk Nông", "Kon Tum"]
    production_pct = [35, 22, 18, 12, 8]
    area_ha = [210000, 176000, 98000, 136000, 26000]
    yields = [2850, 2900, 2750, 2700, 2600]

    # Create gradient colours
    colours = [
        "#1B3D0F",
        "#2D5016",
        "#3D6A1E",
        "#4A7C23",
        "#5A8F2E",
    ]

    bars = ax.barh(
        provinces, production_pct, color=colours, edgecolor=COLOURS["border"]
    )

    # Add value labels
    for i, (bar, pct, area, yld) in enumerate(
        zip(bars, production_pct, area_ha, yields)
    ):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct}%  |  {area:,} ha  |  {yld} kg/ha",
            va="center",
            fontsize=9,
            color=COLOURS["text"],
        )

    ax.set_xlabel("Share of National Production (%)", fontsize=11)
    ax.set_title(
        "Vietnam Coffee Production by Province (2025/26)",
        fontsize=14,
        fontweight="bold",
        color=COLOURS["text"],
        pad=15,
    )

    ax.set_xlim(0, 50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add note
    ax.text(
        0.98,
        0.02,
        "Central Highlands total: 95% of Vietnam's coffee\nData: USDA FAS 2025",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        color="#666666",
    )

    plt.tight_layout()
    output_path = OUTPUT_DIR / "coffee_production_by_province.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor=COLOURS["background"]
    )
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_yield_timeline():
    """Create a timeline chart showing yield trends."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor=COLOURS["background"])

    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    yields = [2800, 2850, 2980, 2680, 2500, 2650, 2850, 2950]

    # Plot historical data (up to and including 2025)
    hist_years = years[:6]  # Include 2025 in historical line
    hist_yields = yields[:6]
    ax.plot(
        hist_years,
        hist_yields,
        "o-",
        color=COLOURS["coffee_primary"],
        linewidth=2,
        markersize=8,
        label="Historical",
    )

    # Highlight current year (2025)
    ax.plot(
        [2025],
        [2650],
        "o",
        color=COLOURS["highlight"],
        markersize=12,
        label="Current (2025)",
        zorder=5,
    )

    # Plot forecast with dashed line - connect from 2025 to remove gap
    forecast_years = [2025] + years[6:]  # Start from 2025 to connect
    forecast_yields = [yields[5]] + yields[6:]  # Include 2025 value
    ax.plot(
        forecast_years,
        forecast_yields,
        "o--",
        color=COLOURS["coffee_secondary"],
        linewidth=2,
        markersize=8,
        label="Forecast",
    )

    # Add uncertainty band for forecast (only for future years)
    ax.fill_between(
        years[6:],
        [y * 0.9 for y in yields[6:]],
        [y * 1.1 for y in yields[6:]],
        alpha=0.2,
        color=COLOURS["coffee_secondary"],
        label="±10% uncertainty",
    )

    # Add vertical line to mark forecast start
    ax.axvline(x=2025, color="#999999", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        2025.1,
        3350,
        "Forecast →",
        fontsize=9,
        color="#666666",
        fontweight="bold",
        va="top",
    )

    # Add event annotations - El Niño drought and La Niña floods
    events = {
        2024: ("El Niño\ndrought", "below"),
        2025: ("La Niña\nfloods", "below"),
    }

    for year, (text, pos) in events.items():
        y_val = yields[years.index(year)]
        y_offset = -180
        ax.annotate(
            text,
            xy=(year, y_val),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=COLOURS["text"],
            arrowprops=dict(arrowstyle="->", color=COLOURS["border"], lw=0.8),
        )

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Yield (kg/ha)", fontsize=11)
    ax.set_title(
        "Vietnam Coffee Yield Trends (2020-2027)",
        fontsize=14,
        fontweight="bold",
        color=COLOURS["text"],
        pad=15,
    )

    ax.set_ylim(2000, 3500)
    ax.set_xlim(2019.5, 2027.5)
    ax.axhline(y=2780, color="#CCCCCC", linestyle=":", label="5-year average")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove y-axis tick labels (per JH feedback) but keep the axis label
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)

    # Add context note
    ax.text(
        0.98,
        0.02,
        "Vietnam has world's highest coffee yields (~2.8 t/ha)\n"
        "Approx. 3.4× global average | Data: USDA FAS, FAO",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        color="#666666",
    )

    plt.tight_layout()
    output_path = OUTPUT_DIR / "coffee_yield_timeline.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor=COLOURS["background"]
    )
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_infographic_summary():
    """Create a summary infographic with key statistics."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLOURS["background"])

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # Load data
    vnm_gdf = load_vietnam_boundaries()
    adm1 = vnm_gdf[vnm_gdf["shapetype"] == "ADM1"].copy()
    adm1 = identify_coffee_provinces(adm1)

    # Panel 1: Mini map
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLOURS["water"])

    non_coffee = adm1[~adm1["is_coffee"]]
    non_coffee.plot(
        ax=ax1,
        color=COLOURS["vietnam_base"],
        edgecolor=COLOURS["border"],
        linewidth=0.2,
    )
    coffee = adm1[adm1["is_coffee"]]
    coffee.plot(
        ax=ax1,
        color=COLOURS["coffee_primary"],
        edgecolor=COLOURS["border"],
        linewidth=0.3,
    )
    ax1.set_axis_off()
    ax1.set_title("Coffee Regions", fontsize=11, fontweight="bold")

    # Panel 2: Key stats
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    stats = [
        ("Global Rank", "#2 Producer"),
        ("World Share", "~20%"),
        ("Variety", "95% Robusta"),
        ("Yield", "2,850 kg/ha"),
        ("Production", "30.8M bags"),
        ("Export Value", "$4.2B (2023)"),
    ]

    for i, (label, value) in enumerate(stats):
        y = 9 - i * 1.5
        ax2.text(0.5, y, label, fontsize=10, color="#666666")
        ax2.text(6, y, value, fontsize=11, fontweight="bold", color=COLOURS["text"])

    ax2.set_title("Key Statistics (2025/26)", fontsize=11, fontweight="bold")

    # Panel 3: Province breakdown
    ax3 = fig.add_subplot(gs[0, 2])
    provinces = ["Đắk Lắk", "Lâm Đồng", "Gia Lai", "Đắk Nông", "Kon Tum"]
    sizes = [35, 22, 18, 12, 8]
    colours_pie = ["#1B3D0F", "#2D5016", "#3D6A1E", "#4A7C23", "#5A8F2E"]
    explode = (0.05, 0, 0, 0, 0)

    wedges, texts, autotexts = ax3.pie(
        sizes,
        explode=explode,
        labels=provinces,
        autopct="%1.0f%%",
        colors=colours_pie,
        startangle=90,
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax3.set_title("Production by Province", fontsize=11, fontweight="bold")

    # Panel 4: Yield comparison
    ax4 = fig.add_subplot(gs[1, 0])
    countries = ["Vietnam", "Brazil", "Indonesia", "World Avg"]
    yields_compare = [2850, 1620, 590, 850]
    colours_bar = [COLOURS["coffee_primary"], "#8B4513", "#CD853F", "#A9A9A9"]

    bars = ax4.bar(
        countries, yields_compare, color=colours_bar, edgecolor=COLOURS["border"]
    )
    ax4.set_ylabel("Yield (kg/ha)")
    ax4.set_title("Yield Comparison", fontsize=11, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    for bar, y in zip(bars, yields_compare):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{y:,}",
            ha="center",
            fontsize=9,
        )

    # Panel 5: Timeline mini
    ax5 = fig.add_subplot(gs[1, 1:])
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    yields_time = [2800, 2850, 2980, 2680, 2500, 2650, 2850, 2950]

    # Historical line (up to and including 2025)
    ax5.plot(
        years[:6], yields_time[:6], "o-", color=COLOURS["coffee_primary"], linewidth=2
    )
    # Highlight current year
    ax5.plot([2025], [2650], "o", color=COLOURS["highlight"], markersize=10, zorder=5)
    # Forecast - connect from 2025 to remove gap
    ax5.plot(
        [2025] + years[6:],
        [yields_time[5]] + yields_time[6:],
        "o--",
        color=COLOURS["coffee_secondary"],
        linewidth=2,
    )
    ax5.fill_between(
        years[6:],
        [y * 0.9 for y in yields_time[6:]],
        [y * 1.1 for y in yields_time[6:]],
        alpha=0.2,
        color=COLOURS["coffee_secondary"],
    )

    # Vertical line to mark forecast start
    ax5.axvline(x=2025, color="#999999", linestyle="--", linewidth=1, alpha=0.7)
    ax5.text(2025.1, 3200, "Forecast →", fontsize=8, color="#666666", fontweight="bold")

    ax5.axhline(y=2780, color="#CCCCCC", linestyle=":", alpha=0.7)
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Yield (kg/ha)")
    ax5.set_title("Yield Trend & Forecast", fontsize=11, fontweight="bold")
    ax5.set_ylim(2200, 3300)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Remove y-axis tick labels (per JH feedback)
    ax5.set_yticklabels([])
    ax5.tick_params(axis="y", length=0)

    # Add annotations - El Niño drought and La Niña floods
    ax5.annotate(
        "El Niño\ndrought",
        xy=(2024, 2500),
        xytext=(2023.5, 2300),
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.5),
    )
    ax5.annotate(
        "La Niña\nfloods",
        xy=(2025, 2650),
        xytext=(2024.3, 2850),
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.5),
    )

    # Main title
    fig.suptitle(
        "Vietnam Coffee Industry Overview",
        fontsize=18,
        fontweight="bold",
        color=COLOURS["text"],
        y=0.98,
    )

    # Footer
    fig.text(
        0.5,
        0.01,
        "Data Sources: USDA FAS (2025), FAO, ICO | Central Highlands (Tây Nguyên) accounts for 92% of production",
        ha="center",
        fontsize=9,
        style="italic",
        color="#666666",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = OUTPUT_DIR / "vietnam_coffee_infographic.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor=COLOURS["background"]
    )
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate all production-quality maps and visualisations."""
    print("Loading Vietnam boundary data...")
    vnm_gdf = load_vietnam_boundaries()

    print("\nGenerating maps and visualisations...")
    create_vietnam_overview_map(vnm_gdf)
    create_central_highlands_detail_map(vnm_gdf)
    create_production_comparison_chart()
    create_yield_timeline()
    create_infographic_summary()

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
