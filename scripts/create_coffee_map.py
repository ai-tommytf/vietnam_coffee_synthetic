"""
Create a map visualising coffee probability in Vietnam.

Uses the vietnam_coffee_probability.tif data with green colour scheme
to highlight areas with high coffee probability.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap


def create_coffee_probability_map(
    tif_path: Path,
    output_path: Path,
    title: str = "Vietnam Coffee Growing Regions",
) -> None:
    """
    Create a map showing coffee probability in Vietnam.

    Uses a green colour palette where darker green indicates
    higher probability of coffee cultivation.
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds

    # Replace -inf/nan with 0 for visualisation
    data = np.where(np.isfinite(data), data, 0)

    # Clip to valid probability range
    data = np.clip(data, 0, 1)

    # Create custom green colormap (white to dark green)
    colors = [
        "#f7fcf5",  # Very light green (almost white) - no coffee
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#41ab5d",
        "#238b45",
        "#006d2c",
        "#00441b",  # Dark green - high probability
    ]
    green_cmap = LinearSegmentedColormap.from_list("coffee_green", colors)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12), facecolor="white")

    # Plot the raster
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap=green_cmap,
        vmin=0,
        vmax=1,
    )

    # Add colourbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Coffee Probability")
    cbar.ax.tick_params(labelsize=10)

    # Styling
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", color="gray")

    # Set aspect ratio to equal for geographic accuracy
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()

    print(f"Saved coffee probability map to: {output_path}")


def create_coffee_probability_map_dark(
    tif_path: Path,
    output_path: Path,
    title: str = "Vietnam Coffee Growing Regions",
) -> None:
    """
    Create a dark-themed map matching the booth aesthetic.
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds

    # Replace -inf/nan with 0
    data = np.where(np.isfinite(data), data, 0)
    data = np.clip(data, 0, 1)

    # Dark theme green colormap
    colors = [
        "#1a1a1a",  # Dark background - no coffee
        "#0d2818",
        "#1a4024",
        "#265930",
        "#32723c",
        "#3e8c48",
        "#4aa654",
        "#56c060",
        "#62da6c",  # Bright green - high probability
    ]
    green_cmap = LinearSegmentedColormap.from_list("coffee_green_dark", colors)

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(10, 12), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Plot the raster
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap=green_cmap,
        vmin=0,
        vmax=1,
    )

    # Add colourbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Coffee Probability")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    cbar.outline.set_edgecolor("white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Styling
    ax.set_xlabel("Longitude", fontsize=11, color="white")
    ax.set_ylabel("Latitude", fontsize=11, color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color="white")

    # Grid
    ax.grid(True, alpha=0.2, linestyle="--", color="white")

    # Tick styling
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)

    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#1a1a1a", bbox_inches="tight")
    plt.close()

    print(f"Saved dark-themed coffee probability map to: {output_path}")


def main() -> None:
    """Generate coffee probability maps."""
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    tif_path = data_dir / "vietnam_coffee_probability.tif"

    # Create both light and dark themed maps
    create_coffee_probability_map(
        tif_path,
        output_dir / "vietnam_coffee_map.png",
        title="Vietnam Coffee Growing Regions",
    )

    create_coffee_probability_map_dark(
        tif_path,
        output_dir / "vietnam_coffee_map_dark.png",
        title="Vietnam Coffee Growing Regions",
    )


if __name__ == "__main__":
    main()
