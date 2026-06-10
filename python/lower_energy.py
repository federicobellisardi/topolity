#!/usr/bin/env python3
"""Analyze fine-grid configurations with energy lower than the original graph."""


from __future__ import annotations

import math
import pickle
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.affinity import affine_transform
from shapely.geometry import LineString




BASE_DIR = Path("/home/fbellisardi/code/topolity/data/data_processed")
CSV_RELATIVE_PATH = Path("graphs_fine_grid/fine_grid_gravitational_work.csv")

OUTPUT_DIR = Path("/home/fbellisardi/code/topolity/output/lower_energy_analysis")
BOXPLOT_DIR = OUTPUT_DIR / "boxplots"
NETWORK_PANEL_DIR = OUTPUT_DIR / "network_panels"

ENERGY_COL = "total_energy_on_3d_path_J"

SAVE = True
SHOW = False
FAST_CHECK = False

SKIP_CITIES = {"amsterdam"}

MAX_EDGES = 6000
MIN_EDGE_LENGTH_M = 50
N_COLS = 3
FIGSIZE_PER_PANEL = 4.6

ORIGINAL_COLOR = "black"
LOWER_COLOR = "#E63946"

PALETTE = {
    "rotation": "#fdae61",
    "translation": "#2b83ba",
    "scale": "#2ca25f",
    "original": "#E63946",
}

HIGHWAY_PRIORITY = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "motorway_link": 6,
    "trunk_link": 7,
    "primary_link": 8,
    "secondary_link": 9,
    "tertiary_link": 10,
    "residential": 20,
    "unclassified": 21,
    "service": 30,
}




def classify_transformation_type(value) -> str:
    v = str(value).lower()

    if "original" in v:
        return "original"
    if "rot" in v or "rotation" in v:
        return "rotation"
    if "trans" in v or "translation" in v:
        return "translation"
    if "scale" in v:
        return "scale"

    return "other"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_original_graph(city_dir: Path, fine_dir: Path) -> Path | None:
    candidates = [
        fine_dir / "graph_original.pkl",
        city_dir / "graphs" / "graph_original.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def fit_affine_from_graphs(G0, G, max_nodes: int = 5000) -> list[float]:
    rows = []

    for node_id, d0 in G0.nodes(data=True):
        if node_id not in G.nodes:
            continue

        d1 = G.nodes[node_id]

        if not {"x", "y"}.issubset(d0) or not {"x", "y"}.issubset(d1):
            continue

        rows.append([
            float(d0["x"]),
            float(d0["y"]),
            float(d1["x"]),
            float(d1["y"]),
        ])

    if len(rows) < 3:
        return [1, 0, 0, 1, 0, 0]

    arr = np.asarray(rows, dtype=float)

    if len(arr) > max_nodes:
        rng = np.random.default_rng(42)
        arr = arr[rng.choice(len(arr), size=max_nodes, replace=False)]

    x0 = arr[:, 0]
    y0 = arr[:, 1]
    x1 = arr[:, 2]
    y1 = arr[:, 3]

    design = np.column_stack([x0, y0, np.ones_like(x0)])

    coef_x, *_ = np.linalg.lstsq(design, x1, rcond=None)
    coef_y, *_ = np.linalg.lstsq(design, y1, rcond=None)

    a, b, xoff = coef_x
    d, e, yoff = coef_y

    return [a, b, d, e, xoff, yoff]


def graph_to_edges_gdf(
    G0,
    G=None,
    max_edges: int = MAX_EDGES,
    min_length_m: float = MIN_EDGE_LENGTH_M,
) -> gpd.GeoDataFrame:
    if G is None:
        G = G0

    matrix = fit_affine_from_graphs(G0, G)
    rows = []

    for u, v, k, data in G0.edges(keys=True, data=True):
        length = float(data.get("length", 0.0))

        if length < min_length_m:
            continue

        highway = data.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0]

        if data.get("geometry") is not None:
            geom0 = data["geometry"]
        else:
            if u not in G0.nodes or v not in G0.nodes:
                continue

            du = G0.nodes[u]
            dv = G0.nodes[v]

            if not {"x", "y"}.issubset(du) or not {"x", "y"}.issubset(dv):
                continue

            geom0 = LineString([
                (float(du["x"]), float(du["y"])),
                (float(dv["x"]), float(dv["y"])),
            ])

        rows.append({
            "u": u,
            "v": v,
            "length": length,
            "highway": str(highway),
            "geometry": affine_transform(geom0, matrix),
        })

    if not rows:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["priority"] = gdf["highway"].map(HIGHWAY_PRIORITY).fillna(50)
    gdf = gdf.sort_values(["priority", "length"], ascending=[True, False])

    if len(gdf) > max_edges:
        gdf = gdf.head(max_edges)

    return gdf.to_crs("EPSG:3857")


def prepare_city_dataframe(csv_path: Path, city: str) -> tuple[pd.DataFrame | None, float | None, str]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, None, f"read_error: {e}"

    if df.empty:
        return df, None, "empty_csv"

    if ENERGY_COL not in df.columns or "filename" not in df.columns:
        return df, None, f"missing_column: {ENERGY_COL}"

    df[ENERGY_COL] = pd.to_numeric(df[ENERGY_COL], errors="coerce")
    df = df.dropna(subset=[ENERGY_COL]).copy()

    original = df[df["filename"].astype(str).eq("graph_original.pkl")]

    if original.empty:
        return df, None, "missing_original"

    original_energy = float(original[ENERGY_COL].iloc[0])

    df["city"] = city
    df["energy_plot"] = df[ENERGY_COL]
    df["original_energy"] = original_energy
    df["delta_energy"] = df["energy_plot"] - original_energy
    df["delta_energy_pct"] = 100 * df["delta_energy"] / original_energy
    df["is_lower_energy"] = (
        (df["filename"].astype(str) != "graph_original.pkl")
        & (df["energy_plot"] < original_energy)
    )

    type_source = df["variant_type"] if "variant_type" in df.columns else df["filename"]
    df["transformation_type"] = type_source.apply(classify_transformation_type)

    return df, original_energy, "ok"




def plot_city_boxplot(city: str, df: pd.DataFrame) -> None:
    df_plot = df[df["transformation_type"].isin(["rotation", "translation", "scale"])].copy()

    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=df_plot,
        x="transformation_type",
        y="delta_energy_pct",
        hue="transformation_type",
        order=["rotation", "translation", "scale"],
        palette=PALETTE,
        showfliers=False,
        linewidth=1.5,
        dodge=False,
        legend=False,
        ax=ax,
    )

    ax.axhline(0, color=PALETTE["original"], linewidth=2.5, zorder=3)

    ymax = max(abs(df_plot["delta_energy_pct"].max()), 1)
    ax.text(
        x=2,
        y=0.01 * ymax,
        s="original",
        color=PALETTE["original"],
        fontsize=14,
        ha="right",
        va="bottom",
    )

    ax.set_title(city, fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\Delta E_{3D}$ (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if SAVE:
        out_png = BOXPLOT_DIR / f"{city}_lower_energy_boxplot.png"
        out_pdf = BOXPLOT_DIR / f"{city}_lower_energy_boxplot.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"  saved boxplot: {out_png}")

    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def build_valid_network_configs(
    lower: pd.DataFrame,
    fine_dir: Path,
    G0,
) -> list[dict]:
    valid_configs = []

    for _, row in lower.iterrows():
        filename = str(row["filename"])
        pkl_path = fine_dir / filename

        if not pkl_path.exists():
            print(f"  skip: missing pickle {filename}")
            continue

        G = load_pickle(pkl_path)
        lower_edges = graph_to_edges_gdf(G0, G)

        if lower_edges.empty:
            print(f"  skip: empty network {filename}")
            continue

        valid_configs.append({
            "filename": filename,
            "delta_pct": float(row["delta_energy_pct"]),
            "edges": lower_edges,
        })

    return valid_configs


def plot_city_network_panels(
    city: str,
    valid_configs: list[dict],
    original_edges: gpd.GeoDataFrame,
) -> bool:
    if not valid_configs:
        return False

    n = len(valid_configs)
    ncols = min(N_COLS, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(FIGSIZE_PER_PANEL * ncols, FIGSIZE_PER_PANEL * nrows),
        squeeze=False,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.93,
        wspace=0.02,
        hspace=0.10,
    )

    fig.suptitle(city.capitalize(), fontsize=28, x=0.01, y=0.985, ha="left")

    xmin, ymin, xmax, ymax = original_edges.total_bounds
    pad_x = 0.03 * (xmax - xmin)
    pad_y = 0.03 * (ymax - ymin)

    for ax, item in zip(axes.ravel(), valid_configs):
        original_edges.plot(
            ax=ax,
            linewidth=0.65,
            alpha=0.42,
            color=ORIGINAL_COLOR,
            label="original",
            zorder=2,
        )

        item["edges"].plot(
            ax=ax,
            linewidth=0.75,
            alpha=0.62,
            color=LOWER_COLOR,
            label="lower energy",
            zorder=3,
        )

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal")

        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            attribution_size=3,
            reset_extent=False,
        )

        short_name = item["filename"].replace("graph_", "").replace(".pkl", "")
        ax.set_title(
            f"\nΔE={item['delta_pct']:.3f}%",
            fontsize=8,
            pad=2,
        )

        ax.set_axis_off()
        ax.legend(
            frameon=False,
            loc="upper right",
            fontsize=5,
            handlelength=1.2,
            borderaxespad=0.1,
        )

    for ax in axes.ravel()[len(valid_configs):]:
        ax.set_axis_off()

    if SAVE:
        out_png = NETWORK_PANEL_DIR / f"{city}_lower_energy_network_panels.png"
        out_pdf = NETWORK_PANEL_DIR / f"{city}_lower_energy_network_panels.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
        print(f"  saved panels: {out_png}")

    if SHOW:
        plt.show()
    else:
        plt.close(fig)

    return True




def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BOXPLOT_DIR.mkdir(parents=True, exist_ok=True)
    NETWORK_PANEL_DIR.mkdir(parents=True, exist_ok=True)

    city_summary_rows = []
    lower_config_rows = []
    all_config_rows = []
    error_rows = []
    panel_summary_rows = []

    for csv_path in sorted(BASE_DIR.glob(f"*/{CSV_RELATIVE_PATH}")):
        city = csv_path.parts[-3]
        fine_dir = csv_path.parent
        city_dir = BASE_DIR / city

        df, original_energy, status = prepare_city_dataframe(csv_path, city)

        if df is None:
            error_rows.append({"city": city, "csv_path": str(csv_path), "status": status})
            continue

        if status != "ok":
            city_summary_rows.append({
                "city": city,
                "number_configurations": len(df),
                "configuration_with_lower_energy": 0,
                "filename": "",
                "status": status,
            })
            error_rows.append({"city": city, "csv_path": str(csv_path), "status": status})
            continue

        lower = df[df["is_lower_energy"]].copy()
        lower = lower.sort_values(ENERGY_COL, ascending=True)

        city_summary_rows.append({
            "city": city,
            "number_configurations": len(df) - 1,
            "configuration_with_lower_energy": len(lower),
            "filename": ", ".join(lower["filename"].astype(str).tolist()),
            "status": "ok",
        })

        all_config_rows.extend(df.to_dict("records"))

        if lower.empty:
            continue

        print(f"\n{city}: {len(lower)} lower-energy configurations")

        lower_config_rows.extend(lower.to_dict("records"))

        plot_city_boxplot(city, df)

        if city in SKIP_CITIES:
            panel_summary_rows.append({
                "city": city,
                "lower_energy_rows": len(lower),
                "plotted_configurations": 0,
                "plotted": False,
                "status": "skipped_city",
            })
            continue

        original_pkl = find_original_graph(city_dir, fine_dir)

        if original_pkl is None:
            panel_summary_rows.append({
                "city": city,
                "lower_energy_rows": len(lower),
                "plotted_configurations": 0,
                "plotted": False,
                "status": "missing_original_graph",
            })
            continue

        G0 = load_pickle(original_pkl)
        original_edges = graph_to_edges_gdf(G0, G0)

        valid_configs = build_valid_network_configs(
            lower=lower,
            fine_dir=fine_dir,
            G0=G0,
        )

        plotted = plot_city_network_panels(
            city=city,
            valid_configs=valid_configs,
            original_edges=original_edges,
        )

        panel_summary_rows.append({
            "city": city,
            "lower_energy_rows": len(lower),
            "plotted_configurations": len(valid_configs),
            "plotted": plotted,
            "status": "ok" if plotted else "no_valid_configs",
        })

    city_summary = pd.DataFrame(city_summary_rows).sort_values(
        ["configuration_with_lower_energy", "number_configurations"],
        ascending=[False, False],
    ).reset_index(drop=True)

    lower_configs = pd.DataFrame(lower_config_rows)
    all_configs = pd.DataFrame(all_config_rows)
    errors = pd.DataFrame(error_rows)
    panel_summary = pd.DataFrame(panel_summary_rows)

    city_summary.to_csv(OUTPUT_DIR / "lower_energy_city_summary.csv", index=False)
    lower_configs.to_csv(OUTPUT_DIR / "lower_energy_configurations.csv", index=False)
    all_configs.to_csv(OUTPUT_DIR / "all_configurations_energy_diagnostics.csv", index=False)
    errors.to_csv(OUTPUT_DIR / "lower_energy_read_errors.csv", index=False)
    panel_summary.to_csv(OUTPUT_DIR / "lower_energy_network_panels_summary.csv", index=False)

    print("\nSaved CSV files:")
    print(f"  {OUTPUT_DIR / 'lower_energy_city_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'lower_energy_configurations.csv'}")
    print(f"  {OUTPUT_DIR / 'all_configurations_energy_diagnostics.csv'}")
    print(f"  {OUTPUT_DIR / 'lower_energy_read_errors.csv'}")
    print(f"  {OUTPUT_DIR / 'lower_energy_network_panels_summary.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()