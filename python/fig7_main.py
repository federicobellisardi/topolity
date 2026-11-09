"""Main-text Fig. 7 (Barcelona densification) redrawn from the saved selection.

Reads the selected cells written by ``multi_city_settlement.py`` (no recomputation)
and draws the figure with ``make_combined_figure``; the basemap zoom is chosen for
print resolution.  Output: paper_v2/images/fig7.{pdf,png}.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

import multi_city_settlement as mcs

CITY = "barcelone"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "terrain_aware_densification" / CITY
PAPER_IMAGES = ROOT / "paper_v2" / "images"


def main() -> None:
    cells = gpd.read_file(RESULTS / f"{CITY}_selected_favorable_unfavorable_cells.gpkg")
    selected = (pd.read_csv(RESULTS / f"{CITY}_selected_favorable_unfavorable_cells.csv", sep=";")
                .sort_values("rank_total").reset_index(drop=True))
    with tempfile.TemporaryDirectory() as tmp:
        mcs.make_combined_figure(CITY, cells[["cell_id", "geometry"]], selected, Path(tmp))
        for ext in ("pdf", "png"):
            shutil.copy(Path(tmp) / f"{CITY}_fig7_densification.{ext}", PAPER_IMAGES / f"fig7.{ext}")
    print(f"written {PAPER_IMAGES / 'fig7.pdf'}")


if __name__ == "__main__":
    main()
