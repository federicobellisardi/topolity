#!/usr/bin/env python3
"""Fine-grid gravitational work analysis across all cities."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
from shapely.geometry import LineString



DATA_ROOT = Path("/home/fbellisardi/code/topolity/data/data_processed")
OUT_ROOT = Path("/home/fbellisardi/code/topolity/notebooks/output")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MIN_TRANSLATION_M = 200

MAKE_OVERLAYS = True
MAX_OVERLAYS_PER_CITY = None  # es. 5, oppure None per tutte

USE_EXISTING_SUMMARY_IF_AVAILABLE = True

ENERGY_COL = "total_energy_on_3d_path_J"
VERTICAL_COL = "vertical_energy_on_3d_path_J"
HORIZONTAL_COL = "horizontal_energy_3d_on_3d_path_J"



def safe_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_city_name(x: str) -> str:
    return str(x).replace("_", " ").title()


def fix_negative_zero(x: float, tol: float = 1e-12) -> float:
    return 0.0 if abs(x) < tol else x


def choose_unit(values) -> tuple[float, str]:
    """
    Values are assumed to be in percentage points.

    Examples:
    - 1.0       -> 1 %
    - 0.1       -> 1 ‰
    - 0.001     -> 10 ppm
    - 0.000001  -> 10 ppb
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 1.0, "%"

    max_abs = np.nanmax(np.abs(values))

    if max_abs >= 0.1:
        return 1.0, "%"
    elif max_abs >= 0.001:
        return 10.0, "‰"
    elif max_abs >= 1e-6:
        return 10000.0, "ppm"
    else:
        return 1e7, "ppb"
        return 1.0, "%"

def decimals_for(values) -> int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 3

    m = np.nanmax(np.abs(values))

    if m >= 100:
        return 1
    elif m >= 10:
        return 2
    elif m >= 1:
        return 3
    elif m >= 0.1:
        return 4
    else:
        return 5

def make_formatter(decimals: int, unit: str):
    def _fmt(x, pos=None):
        x = fix_negative_zero(x)
        return f"{x:.{decimals}f}{unit}"

    return FuncFormatter(_fmt)


def get_cities(data_root: Path) -> list[str]:
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])



def _get_node_latlon(nodedata):
    if not isinstance(nodedata, dict):
        return None

    if "lat" in nodedata and "lon" in nodedata:
        return float(nodedata["lat"]), float(nodedata["lon"])

    if "y" in nodedata and "x" in nodedata:
        return float(nodedata["y"]), float(nodedata["x"])

    if "latitude" in nodedata and "longitude" in nodedata:
        return float(nodedata["latitude"]), float(nodedata["longitude"])

    return None


def graph_to_gdf_edges(G, prefer_nodes: bool = False) -> gpd.GeoDataFrame:
    records = []

    iterator = G.edges(data=True, keys=True) if getattr(G, "is_multigraph", lambda: False)() else G.edges(data=True)

    for item in iterator:
        if len(item) == 4:
            u, v, _k, data = item
        else:
            u, v, data = item

        if not isinstance(data, dict):
            continue

        line = None
        geom = data.get("geometry", None)

        if (
            not prefer_nodes
            and geom is not None
            and getattr(geom, "geom_type", None) == "LineString"
        ):
            line = geom
        else:
            nu = G.nodes[u] if hasattr(G, "nodes") else G.get(u, {})
            nv = G.nodes[v] if hasattr(G, "nodes") else G.get(v, {})

            pu = _get_node_latlon(nu)
            pv = _get_node_latlon(nv)

            if pu is None or pv is None:
                if geom is not None and getattr(geom, "geom_type", None) == "LineString":
                    line = geom
                else:
                    continue
            else:
                line = LineString([(pu[1], pu[0]), (pv[1], pv[0])])

        records.append({"geometry": line})

    if not records:
        raise RuntimeError("Unable to build edge geometries")

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def format_lonlat_axes_3857(ax):
    """
    Avoids set_ticklabels warning by using FuncFormatter instead of manual labels.
    """
    tr = Transformer.from_crs(3857, 4326, always_xy=True)

    def fmt_x(x, pos=None):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cy = 0.5 * (ylim[0] + ylim[1])
        lon, _ = tr.transform(x, cy)
        return f"{lon:.3f}"

    def fmt_y(y, pos=None):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx = 0.5 * (xlim[0] + xlim[1])
        _, lat = tr.transform(cx, y)
        return f"{lat:.3f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.tick_params(axis="x", rotation=45)


def make_overlay_image(
    orig_pkl: Path,
    trans_pkl: Path,
    out_png: Path,
    out_pdf: Path,
    pad_frac: float = 0.03,
    basemap: str = "CartoDB.PositronNoLabels",
    zoom: int | None = None,
    trans_from_nodes: bool = True,
    figsize: tuple[float, float] = (10, 10),
):
    with open(orig_pkl, "rb") as f:
        G_orig = pickle.load(f)

    with open(trans_pkl, "rb") as f:
        G_trans = pickle.load(f)

    gdf_o_wgs = graph_to_gdf_edges(G_orig, prefer_nodes=False)
    gdf_t_wgs = graph_to_gdf_edges(G_trans, prefer_nodes=trans_from_nodes)

    gdf_o = gdf_o_wgs.to_crs(3857)
    gdf_t = gdf_t_wgs.to_crs(3857)

    bounds = np.array([gdf_o.total_bounds, gdf_t.total_bounds])

    minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
    maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

    pad_x = (maxx - minx) * pad_frac
    pad_y = (maxy - miny) * pad_frac

    content_width = (maxx - minx) + 2 * pad_x
    content_height = (maxy - miny) + 2 * pad_y

    target_aspect = figsize[0] / figsize[1]
    current_aspect = content_width / content_height

    if current_aspect < target_aspect:
        target_width = content_height * target_aspect
        pad_x += (target_width - content_width) / 2
    else:
        target_height = content_width / target_aspect
        pad_y += (target_height - content_height) / 2

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")

    try:
        provider = ctx.providers
        for part in basemap.split("."):
            provider = getattr(provider, part)

        kwargs = {
            "ax": ax,
            "crs": gdf_o.crs,
            "source": provider,
            "alpha": 1.0,
            "attribution": False,
            "zorder": 0,
        }

        if isinstance(zoom, int):
            kwargs["zoom"] = zoom

        ctx.add_basemap(**kwargs)

    except Exception as e:
        print(f"[info] Basemap skipped: {e}")

    original_color = "#ef8a62"
    translated_color = "#67a9cf"

    gdf_o.plot(
        ax=ax,
        color=original_color,
        linewidth=0.7,
        alpha=0.95,
        zorder=2,
        rasterized=True,
    )

    gdf_t.plot(
        ax=ax,
        color=translated_color,
        linewidth=0.9,
        alpha=0.45,
        zorder=3,
        rasterized=True,
    )

    format_lonlat_axes_3857(ax)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    legend_elems = [
        Line2D([0], [0], color=original_color, lw=2.2, label="Original network"),
        Line2D([0], [0], color=translated_color, lw=2.2, label="Transformed network"),
    ]

    ax.legend(handles=legend_elems, frameon=True, loc="lower right")

    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"[ok] Saved overlay: {out_png}")



def process_city(city: str) -> dict:
    csv_path = DATA_ROOT / city / "graphs_fine_grid" / "fine_grid_gravitational_work.csv"
    graphs_dir = DATA_ROOT / city / "graphs_fine_grid"

    city_out = OUT_ROOT / city
    city_out.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[{city}] CSV mancante: {csv_path}")
        return {"city": city, "status": "no_csv"}

    df = pd.read_csv(csv_path)

    required_cols = [
        "variant_type",
        ENERGY_COL,
        VERTICAL_COL,
        HORIZONTAL_COL,
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"[{city}] colonne mancanti: {missing}")
        return {
            "city": city,
            "status": "missing_columns",
            "missing_columns": ",".join(missing),
        }

    df["energy_J"] = safe_numeric(df[ENERGY_COL])
    df = df[df["energy_J"].notna()].copy()

    orig = df[df["variant_type"].astype(str) == "original"]

    if orig.empty:
        print(f"[{city}] original non trovato")
        return {"city": city, "status": "no_original"}

    E0 = float(orig.iloc[0]["energy_J"])

    if not np.isfinite(E0) or E0 == 0:
        print(f"[{city}] energia originale non valida: {E0}")
        return {"city": city, "status": "bad_original_energy"}

    df["energy_diff_J"] = E0 - df["energy_J"]
    df["pct_diff"] = df["energy_diff_J"] / E0 * 100.0

    df["vertical_comp_J"] = safe_numeric(df[VERTICAL_COL])
    df["horizontal_comp_J"] = safe_numeric(df[HORIZONTAL_COL])

    orig_vert = safe_numeric(orig[VERTICAL_COL]).iloc[0]
    orig_hor = safe_numeric(orig[HORIZONTAL_COL]).iloc[0]

    df["delta_vertical_J"] = orig_vert - df["vertical_comp_J"]
    df["delta_horizontal_J"] = orig_hor - df["horizontal_comp_J"]

    df["pct_delta_vertical"] = df["delta_vertical_J"] / E0 * 100.0
    df["pct_delta_horizontal"] = df["delta_horizontal_J"] / E0 * 100.0

    df["reconstructed_pct_diff"] = (
        df["pct_delta_vertical"].fillna(0.0)
        + df["pct_delta_horizontal"].fillna(0.0)
    )

    df["component_residual_pct"] = df["pct_diff"] - df["reconstructed_pct_diff"]

    better = df[df["energy_J"] < E0].copy()

    if "parameter" in better.columns:
        param_abs = safe_numeric(better["parameter"]).abs().fillna(0)
    else:
        param_abs = pd.Series(0, index=better.index)

    is_trans = better["variant_type"].astype(str).str.startswith("translation_")
    small_mask = is_trans & (param_abs < MIN_TRANSLATION_M)

    better_f = better[~small_mask].copy()
    n_removed = int(small_mask.sum())

    df.to_csv(city_out / "all_results.csv", index=False)
    better.to_csv(city_out / "better_than_original.csv", index=False)
    better_f.to_csv(city_out / "better_than_original_filtered.csv", index=False)

    collected = []

    if MAKE_OVERLAYS and not better_f.empty:
        orig_pkl = graphs_dir / "graph_original.pkl"

        if not orig_pkl.exists():
            warnings.warn(f"[{city}] pickle original mancante: {orig_pkl}")
        else:
            rows = better_f.copy()

            if MAX_OVERLAYS_PER_CITY is not None:
                rows = rows.sort_values("pct_diff", ascending=False).head(MAX_OVERLAYS_PER_CITY)

            for _, row in rows.iterrows():
                fname = row.get("filename", None)

                if not isinstance(fname, str):
                    warnings.warn(f"[{city}] filename mancante in una riga -- skip")
                    continue

                pkl = graphs_dir / fname

                if not pkl.exists():
                    warnings.warn(f"[{city}] pickle mancante {pkl} -- skip")
                    continue

                png_out = city_out / f"{fname}.png"
                pdf_out = city_out / f"{fname}.pdf"

                try:
                    make_overlay_image(
                        orig_pkl=orig_pkl,
                        trans_pkl=pkl,
                        out_png=png_out,
                        out_pdf=pdf_out,
                        basemap="CartoDB.PositronNoLabels",
                        figsize=(10, 10),
                    )

                    collected.append(
                        {
                            "filename": fname,
                            "pkl": str(pkl),
                            "png": str(png_out),
                            "pdf": str(pdf_out),
                        }
                    )

                except Exception as e:
                    warnings.warn(f"[{city}] errore overlay per {pkl}: {e}")
                    continue

    with open(city_out / "collected_graphs_metadata.json", "w") as f:
        json.dump(collected, f, indent=2)

    record = {
        "city": city,
        "status": "ok",
        "energy_col": ENERGY_COL,
        "vertical_col": VERTICAL_COL,
        "horizontal_col": HORIZONTAL_COL,
        "E0_J": E0,
        "n_total": len(df),
        "n_better": len(better),
        "n_better_filtered": len(better_f),
        "n_removed_translation_small": n_removed,
        "mean_pct_total": better_f["pct_diff"].mean(),
        "mean_pct_vertical": better_f["pct_delta_vertical"].mean(),
        "mean_pct_horizontal": better_f["pct_delta_horizontal"].mean(),
        "mean_residual_pct": better_f["component_residual_pct"].mean(),
    }

    print(
        f"[{city}] total={len(df)} "
        f"better={len(better)} "
        f"retained={len(better_f)} "
        f"removed={n_removed} "
        f"mean_total={record['mean_pct_total']:.6g}%"
    )

    return record



def plot_aggregate_results(summary_fp: Path):
    sns.set_theme(style="whitegrid", context="talk")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
        }
    )

    df_sum = pd.read_csv(summary_fp)

    if "status" in df_sum.columns:
        df_sum = df_sum[df_sum["status"] == "ok"].copy()

    df_sum = df_sum.dropna(subset=["mean_pct_total"])

    if df_sum.empty:
        print("Nessun dato valido per i grafici aggregati.")
        return

    df_sum["city_label"] = df_sum["city"].apply(clean_city_name)

    for c in ["mean_pct_total", "mean_pct_vertical", "mean_pct_horizontal"]:
        if c not in df_sum.columns:
            df_sum[c] = np.nan
        df_sum[c] = pd.to_numeric(df_sum[c], errors="coerce")

    df_sum = df_sum.sort_values("mean_pct_total", ascending=True)

    scale, unit = choose_unit(
        df_sum[["mean_pct_total", "mean_pct_vertical", "mean_pct_horizontal"]]
        .to_numpy()
        .ravel()
    )

    df_plot = df_sum.copy()
    df_plot["total"] = df_plot["mean_pct_total"] * scale
    df_plot["vertical"] = df_plot["mean_pct_vertical"] * scale
    df_plot["horizontal"] = df_plot["mean_pct_horizontal"] * scale
    
    print("\n[diagnostic] aggregate values after scaling")
    print("unit:", unit)
    print(df_plot[["city_label", "total", "vertical", "horizontal"]])

    all_vals = df_plot[["total", "vertical", "horizontal"]].to_numpy().ravel()
    decimals = decimals_for(all_vals)
    formatter = make_formatter(decimals, unit)

    finite_vals = all_vals[np.isfinite(all_vals)]
    max_abs = max(np.nanmax(np.abs(finite_vals)), 1e-9)
    pad = max_abs * 0.15

    # --------------------------------------------------------
    # 1) Mean total change per city
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(df_plot))))

    sns.barplot(
        data=df_plot,
        x="total",
        y="city_label",
        hue="city_label",
        palette=sns.color_palette("mako_r", len(df_plot)),
        edgecolor="black",
        linewidth=1.2,
        legend=False,
        ax=ax,
    )

    ax.set_xlim(
        min(df_plot["total"].min(), 0.0) - pad,
        max(df_plot["total"].max(), 0.0) + pad,
    )

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel(f"Mean energy reduction ({unit})")
    ax.set_title("Mean energy reduction per city", weight="bold")
    # ax.set_ylabel("City")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=35)

    for p, val in zip(ax.patches, df_plot["total"].values):
        val = fix_negative_zero(val)
        y = p.get_y() + p.get_height() / 2

        if val < 0:
            x = val + pad * 0.12
            ha = "left"
        else:
            x = val - pad * 0.12
            ha = "right"

        ax.text(
            x,
            y,
            f"{val:.{decimals}f}{unit}",
            va="center",
            ha=ha,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )

    fig.tight_layout()
    out1 = OUT_ROOT / "analysis_all_cities_mean_change.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 2) Vertical vs horizontal grouped bars
    # --------------------------------------------------------
    comp_long = df_plot.melt(
        id_vars=["city_label"],
        value_vars=["vertical", "horizontal"],
        var_name="component",
        value_name="value",
    )

    comp_long["component"] = comp_long["component"].map(
        {
            "vertical": "Vertical",
            "horizontal": "Horizontal",
        }
    )

    fig, ax = plt.subplots(figsize=(11, max(5, 0.65 * len(df_plot))))

    sns.barplot(
        data=comp_long,
        x="value",
        y="city_label",
        hue="component",
        orient="h",
        palette={"Vertical": "#d95f02", "Horizontal": "#1b9e77"},
        edgecolor="black",
        linewidth=1.1,
        ax=ax,
    )

    comp_max = max(np.nanmax(np.abs(comp_long["value"])), 1e-9)
    comp_pad = comp_max * 0.25

    ax.set_xlim(
        min(comp_long["value"].min(), 0.0) - comp_pad,
        max(comp_long["value"].max(), 0.0) + comp_pad,
    )

    ax.axvline(0, color="black", linewidth=1.1)
    ax.set_xlabel(f"Contribution to energy reduction ({unit})")
    ax.set_ylabel("City")
    ax.set_title("Vertical vs horizontal contribution to energy reduction", weight="bold")
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="", loc="best")

    for container in ax.containers:
        for bar in container:
            val = fix_negative_zero(bar.get_width())
            y = bar.get_y() + bar.get_height() / 2

            if val >= 0:
                x = val + comp_pad * 0.04
                ha = "left"
            else:
                x = val - comp_pad * 0.04
                ha = "right"

            ax.text(
                x,
                y,
                f"{val:.{decimals}f}{unit}",
                va="center",
                ha=ha,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
            )

    fig.tight_layout()
    out2 = OUT_ROOT / "analysis_all_cities_components_grouped.png"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 3) Scatter: vertical contribution vs total change
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    size_col = "n_better_filtered" if "n_better_filtered" in df_plot.columns else None

    if size_col is not None:
        sns.scatterplot(
            data=df_plot,
            x="vertical",
            y="total",
            size=size_col,
            sizes=(80, 350),
            legend=False,
            color="#2b8cbe",
            edgecolor="black",
            linewidth=1.1,
            ax=ax,
        )
    else:
        sns.scatterplot(
            data=df_plot,
            x="vertical",
            y="total",
            s=150,
            legend=False,
            color="#2b8cbe",
            edgecolor="black",
            linewidth=1.1,
            ax=ax,
        )

    xpad = max(np.ptp(df_plot["vertical"]), max_abs * 0.1, 1e-9) * 0.20
    ypad = max(np.ptp(df_plot["total"]), max_abs * 0.1, 1e-9) * 0.20

    ax.set_xlim(df_plot["vertical"].min() - xpad, df_plot["vertical"].max() + xpad)
    ax.set_ylim(df_plot["total"].min() - ypad, df_plot["total"].max() + ypad)

    for _, r in df_plot.iterrows():
        ax.annotate(
            r["city_label"],
            xy=(r["vertical"], r["total"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=10,
            weight="semibold",
        )

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.6)

    ax.set_xlabel(f"Mean vertical contribution ({unit})")
    ax.set_ylabel(f"Mean total energy reduction ({unit})")
    ax.set_title("Vertical contribution vs total energy reduction", weight="bold")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    out3 = OUT_ROOT / "analysis_all_cities_scatter.png"
    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)

    print("Saved aggregate plots:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)



def main():
    print("DATA_ROOT =", DATA_ROOT)
    print("OUT_ROOT  =", OUT_ROOT)

    verification_fp = OUT_ROOT / "verification_summary_all_cities.csv"
    analysis_fp = OUT_ROOT / "analysis_summary_per_city.csv"

    if USE_EXISTING_SUMMARY_IF_AVAILABLE and analysis_fp.exists():
        print(f"[info] Uso summary esistente senza ricalcolare le città: {analysis_fp}")
        plot_aggregate_results(analysis_fp)
        return

    cities = get_cities(DATA_ROOT)
    print("Found cities:", cities)

    summary = []

    for city in cities:
        try:
            summary.append(process_city(city))
        except Exception as e:
            warnings.warn(f"[{city}] errore inatteso: {e}")
            summary.append({"city": city, "status": "error", "error": str(e)})

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(verification_fp, index=False)

    ok_summary = summary_df[summary_df["status"] == "ok"].copy()
    ok_summary.to_csv(analysis_fp, index=False)

    print("Saved summaries:")
    print(" -", verification_fp)
    print(" -", analysis_fp)

    plot_aggregate_results(analysis_fp)


if __name__ == "__main__":
    main()