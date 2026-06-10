#!/usr/bin/env python3
"""Graph transformation utilities: translations, rotations, and scaling."""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _get_node_latlon(nodedata):
    """Ritorna (lat, lon) provando varie chiavi comuni."""

    if not isinstance(nodedata, dict):
        return None
    if "lat" in nodedata and "lon" in nodedata:
        return float(nodedata["lat"]), float(nodedata["lon"])
    if "y" in nodedata and "x" in nodedata:
        return float(nodedata["y"]), float(nodedata["x"])
    if "latitude" in nodedata and "longitude" in nodedata:
        return float(nodedata["latitude"]), float(nodedata["longitude"])
    return None


def graph_to_gdf_edges(G, prefer_nodes=False):
    """
    Converte un (Multi)DiGraph in GeoDataFrame di archi.
    Se prefer_nodes=True, IGNORA l'attributo edge['geometry'] e
    ricostruisce le LineString dalle coordinate dei nodi.
    CRS di output: EPSG:4326 (lon/lat).
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except Exception as e:
        raise RuntimeError("geopandas e shapely sono richiesti") from e

    records = []

    # iteratore archi (compatibile anche con dict-of-dict)
    if hasattr(G, "edges"):
        iterator = G.edges(data=True, keys=True) if getattr(G, "is_multigraph", lambda: False)() else G.edges(data=True)
    else:
        iterator = []
        for u, nbrs in G.items():
            for v, data in nbrs.items():
                iterator.append((u, v, data))

    for item in iterator:
        if len(item) == 4:
            u, v, _k, data = item
        else:
            u, v, data = item
        if not isinstance(data, dict):
            continue

        line = None
        geom = data.get("geometry", None)

        if (not prefer_nodes) and (geom is not None) and getattr(geom, "geom_type", None) == "LineString":
            # usa la geometria salvata
            line = geom
        else:
            # ricostruisci da nodi
            nu = G.nodes[u] if hasattr(G, "nodes") else G.get(u, {})
            nv = G.nodes[v] if hasattr(G, "nodes") else G.get(v, {})
            pu = _get_node_latlon(nu)
            pv = _get_node_latlon(nv)
            if pu is None or pv is None:
                # fallback: se manca qualcosa prova comunque a usare la geometry
                if (geom is not None) and getattr(geom, "geom_type", None) == "LineString":
                    line = geom
                else:
                    continue
            else:
                # shapely vuole (x=lon, y=lat)
                line = LineString([(pu[1], pu[0]), (pv[1], pv[0])])

        records.append({"geometry": line})

    if not records:
        raise RuntimeError("Non sono riuscito a costruire geometrie di archi.")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf


def _format_lonlat_axes_3857(ax):
    """Mostra etichette di assi in lon/lat quando i dati sono in EPSG:3857."""
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(3857, 4326, always_xy=True)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        cx = 0.5*(xlim[0]+xlim[1]); cy = 0.5*(ylim[0]+ylim[1])

        xt = ax.get_xticks(); yt = ax.get_yticks()
        xtlabels = [f"{tr.transform(x, cy)[0]:.3f}" for x in xt]
        ytlabels = [f"{tr.transform(cx, y)[1]:.3f}" for y in yt]
        ax.set_xticklabels(xtlabels, rotation=45)
        ax.set_yticklabels(ytlabels)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    except Exception:
        pass


def make_overlay_pdf(orig_pkl, trans_pkl, out_pdf,
                     pad_frac=0.03,
                     basemap="CartoDB.PositronNoLabels",
                     figsize=(8.8, 6.2),
                     zoom=None,
                     trans_from_nodes=True):
    import contextily as ctx
    from matplotlib.lines import Line2D

    # --- carica grafi ---
    with open(orig_pkl, "rb") as f:
        G_orig = pickle.load(f)
    with open(trans_pkl, "rb") as f:
        G_trans = pickle.load(f)

    # Originale: usa geometry se c'è; Trasformato: ricostruisci dalle coordinate dei nodi
    gdf_o_wgs = graph_to_gdf_edges(G_orig,  prefer_nodes=False)
    gdf_t_wgs = graph_to_gdf_edges(G_trans, prefer_nodes=trans_from_nodes)

    # Proietta in Web Mercator per basemap
    gdf_o = gdf_o_wgs.to_crs(3857)
    gdf_t = gdf_t_wgs.to_crs(3857)

    # Estensione unita
    bounds = np.array([gdf_o.total_bounds, gdf_t.total_bounds])
    minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
    maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()
    pad_x = (maxx - minx) * pad_frac
    pad_y = (maxy - miny) * pad_frac

    # --- figura ---
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    # Estensione PRIMA di aggiungere il basemap (così contextily sceglie bene le tile)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")

    # Basemap sotto a tutto
    try:
        provider = ctx.providers
        for part in basemap.split("."):
            provider = getattr(provider, part)
        kwargs = dict(ax=ax, crs=gdf_o.crs, source=provider, alpha=1.0, attribution=False, zorder=0)
        if isinstance(zoom, int):
            kwargs["zoom"] = zoom
        ctx.add_basemap(**kwargs)
    except Exception as e:
        print(f"[info] Basemap saltato: {e}")

    # Disegna: originale (grigio) e trasformato (rosso) sopra
    gdf_o.plot(ax=ax, color="#747a82", linewidth=0.7, alpha=0.95, zorder=2, rasterized=True)
    gdf_t.plot(ax=ax, color="#d51616", linewidth=0.9, alpha=0.45, zorder=3, rasterized=True)

    # Assi in lon/lat
    _format_lonlat_axes_3857(ax)

    # Pulizia frame
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Legenda coerente con i colori disegnati
    legend_elems = [
        Line2D([0], [0], color="#747a82", lw=2.2, label="Original network"),
        Line2D([0], [0], color="#d51616", lw=2.2, label="Transformed network"),
    ]
    ax.legend(handles=legend_elems, frameon=True, loc="lower right", fontsize=9)

    # Salvataggio
    try:
        fig.tight_layout(pad=0.0)
    except Exception:
        pass
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"[ok] Saved overlay to: {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description="Overlay original vs transformed city network and save PDF.")
    ap.add_argument("--orig", required=True, help="Path to graph_original.pkl")
    ap.add_argument("--trans", required=True, help="Path to transformed graph .pkl (e.g., graph_translated_10.pkl)")
    ap.add_argument("--out", required=True, help="Output PDF path")
    ap.add_argument("--basemap", default="CartoDB.PositronNoLabels", help="Contextily provider")
    ap.add_argument("--zoom", type=int, default=None, help="Basemap zoom (omit for auto)")
    ap.add_argument("--pad-frac", type=float, default=0.03, help="Fractional padding of extent")
    ap.add_argument("--trans-from-nodes", action="store_true",
                    help="Rebuild transformed edges from node coordinates (recommended).")
    args = ap.parse_args()

    make_overlay_pdf(
        orig_pkl=args.orig,
        trans_pkl=args.trans,
        out_pdf=args.out,
        pad_frac=args.pad_frac,
        basemap=args.basemap,
        zoom=args.zoom,
        trans_from_nodes=True if args.trans_from_nodes else True  # default True per evitare il bug
    )


if __name__ == "__main__":
    main()
