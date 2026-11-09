#!/usr/bin/env python3
"""Rail networks (metro, commuter rail, tram) from OpenStreetMap route relations.

Each stop-to-stop edge keeps its track polyline and, for every polyline segment,
whether the track lies on the terrain. The vertical profile is built as follows:

* track on the terrain: DEM sampled every DS = 10 m along the polyline, as for
  roads (fine_grid_gravitational_work.py) and buses (bus_network.py);
* tunnel, bridge or viaduct (OSM `tunnel`, `bridge`, `location=underground`,
  negative `layer`): a uniform grade between the nearest samples on the terrain
  (tunnel portals, bridge abutments) or the stops at the edge ends.

Stops are always anchored to the surface DEM, since station depths are not
available. Uphill and downhill gains on non-terrain segments are also stored
separately (`up_t`, `down_t`) for sensitivity tests on that assumption.

The stop graph is undirected; per-edge gains refer to the direction from the
edge attribute `start` and are swapped when the edge is traversed backwards.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
from terrain import MetricFrame, clean_elevation_profile  # noqa: E402

DS = 10.0
ROUTE_MODE = {"subway": "metro", "train": "train", "tram": "tram", "light_rail": "tram"}
RAIL_MODES = ["metro", "train", "tram"]
OVERPASS_MIRRORS = ["https://overpass-api.de/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter"]
USER_AGENT = "topolity-research/1.0 (academic study of urban mobility energy)"
# Route choice minimises a time-like weight: walking (~5 km/h) costs WALK_FACTOR times the
# same distance on board (~15 km/h); energies use only the on-board legs.
WALK_FACTOR = 3.0
TERMS = 5          # length, up, down, up_t, down_t


def _norm(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().strip().lower()


def grounded(tags: dict) -> bool:
    """True if an OSM railway way lies on the terrain surface."""
    if tags.get("tunnel", "no") != "no":
        return False
    if tags.get("bridge", "no") != "no":
        return False
    if tags.get("location") == "underground":
        return False
    try:
        if float(tags.get("layer", 0)) < 0:
            return False
    except ValueError:
        pass
    return True


def download_rail_routes(bbox, cache: Path, networks=None) -> dict:
    """Route relations with member ways (with tags) and nodes; cached at `cache`.

    networks: optional list of OSM `network` values to download (exact match). Large rail hubs
    otherwise pull in every long-distance line crossing the bbox with all its ways."""
    if cache.exists():
        return json.loads(cache.read_text())
    s, w, n, e = bbox
    net = ""
    if networks:
        alt = "|".join(re.escape(x) for x in networks)
        net = f'["network"~"^({alt})$"]'
    query = (f'[out:json][timeout:600];'
             f'relation["type"="route"]["route"~"^(subway|train|tram|light_rail)$"]{net}({s},{w},{n},{e});'
             f'out body;>;out body qt;')
    last = None
    for attempt in range(6):          # Overpass often returns 429/502/504 under load
        url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=900)
            if r.ok:
                break
            last = f"{url}: HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = f"{url}: {exc}"
        time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"Overpass download failed: {last}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    return r.json()


def build_rail_graph(osm: dict, include_networks, metric_crs, bounds=None):
    """Undirected stop graph; each rail edge keeps its polyline (lon/lat) and segment flags.

    bounds (minlon, minlat, maxlon, maxlat): edges with a stop outside are dropped.
    """
    include = {_norm(n) for n in include_networks}
    lonlat = {e["id"]: (e["lon"], e["lat"]) for e in osm["elements"] if e["type"] == "node"}
    ways = {e["id"]: e for e in osm["elements"] if e["type"] == "way"}
    to_m = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)

    def xy(nid):
        return np.array(to_m.transform(*lonlat[nid]))

    def inside(nid):
        if bounds is None:
            return True
        lon, lat = lonlat[nid]
        return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]

    R = nx.Graph()
    stats = {"routes_used": 0, "routes_skipped": 0, "stop_pairs": 0, "track_fallbacks": 0,
             "edges_outside_bounds": 0}
    for rel in (e for e in osm["elements"] if e["type"] == "relation"):
        tags = rel.get("tags", {})
        mode = ROUTE_MODE.get(tags.get("route"))
        if mode is None or _norm(tags.get("network")) not in include:
            continue
        T = nx.Graph()
        for m in rel["members"]:
            if m["type"] == "way" and m["ref"] in ways:
                way = ways[m["ref"]]
                g = grounded(way.get("tags", {}))
                nds = [nd for nd in way["nodes"] if nd in lonlat]
                for a, b in zip(nds[:-1], nds[1:]):
                    T.add_edge(a, b, w=float(np.linalg.norm(xy(a) - xy(b))), grounded=g)
        stops = [m["ref"] for m in rel["members"]
                 if m["type"] == "node" and m.get("role", "").startswith("stop") and m["ref"] in lonlat]
        if T.number_of_nodes() == 0 or len(stops) < 2:
            stats["routes_skipped"] += 1
            continue
        stats["routes_used"] += 1
        track = list(T.nodes())
        tree = KDTree(np.array([xy(t) for t in track]))
        on_track = [s if s in T else track[tree.query(xy(s))[1]] for s in stops]
        for (s1, t1), (s2, t2) in zip(zip(stops[:-1], on_track[:-1]), zip(stops[1:], on_track[1:])):
            if s1 == s2:
                continue
            if not (inside(s1) and inside(s2)):
                stats["edges_outside_bounds"] += 1
                continue
            try:
                path = nx.shortest_path(T, t1, t2, weight="w")
                flags = [T[a][b]["grounded"] for a, b in zip(path[:-1], path[1:])]
            except nx.NetworkXNoPath:
                path, flags = [], []
                stats["track_fallbacks"] += 1
            pts = [s1] + path + [s2]
            # stop-to-track connectors inherit the flag of the adjacent track segment
            seg = ([flags[0] if flags else True] + flags + [flags[-1] if flags else True])
            keep = [0] + [i for i in range(1, len(pts)) if pts[i] != pts[i - 1]]
            poly = np.array([lonlat[pts[i]] for i in keep])
            seg = np.array([seg[i - 1] for i in keep[1:]], dtype=bool)
            if len(poly) < 2:
                continue
            pm = np.array(to_m.transform(poly[:, 0], poly[:, 1])).T
            track_len = float(np.linalg.norm(np.diff(pm, axis=0), axis=1).sum())
            stats["stop_pairs"] += 1
            for s in (s1, s2):
                if s not in R:
                    R.add_node(s, lon=lonlat[s][0], lat=lonlat[s][1])
            if not R.has_edge(s1, s2) or R[s1][s2]["track_len"] > track_len:
                R.add_edge(s1, s2, poly=poly, seg_grounded=seg, start=s1, mode=mode, kind="rail",
                           track_len=track_len)
    stats["stops"] = R.number_of_nodes()
    stats["rail_edges"] = R.number_of_edges()
    stats["grounded_share"] = _grounded_share(R)
    return R, stats


def _grounded_share(R) -> float:
    """Track-length share on the terrain (2-D, all rail edges)."""
    tot = gnd = 0.0
    for _, _, d in R.edges(data=True):
        if d.get("kind") != "rail":
            continue
        p = d["poly"]
        seg = np.hypot(np.diff(p[:, 0]) * np.cos(np.radians(p[:-1, 1])), np.diff(p[:, 1]))
        tot += seg.sum()
        gnd += seg[d["seg_grounded"]].sum()
    return float(gnd / tot) if tot > 0 else float("nan")


def profile_polyline(x, y, seg_grounded, z_fn, ds: float = DS) -> dict:
    """Vertical profile of a metric polyline.

    Samples every <= ds metres; samples on terrain segments (and both ends) take
    z_fn(x, y), the others are linearly interpolated between them along the track.
    Returns 3-D length, uphill/downhill gain and the part of each gain on
    non-terrain segments (up_t, down_t).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    seg_grounded = np.asarray(seg_grounded, bool)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    # same sampling as terrain.resample_polyline, keeping the arc-length positions s
    s = np.linspace(0.0, cum[-1], max(int(np.ceil(cum[-1] / ds)) + 1, 2))
    xs, ys = np.interp(s, cum, x), np.interp(s, cum, y)
    seg_of = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_grounded) - 1)
    anchor = seg_grounded[seg_of].copy()
    # a sample exactly at the end of a terrain segment (portal) is on the terrain too
    prev = np.clip(np.searchsorted(cum, s, side="left") - 1, 0, len(seg_grounded) - 1)
    anchor |= seg_grounded[prev]
    anchor[0] = anchor[-1] = True
    z = np.full(len(xs), np.nan)
    z[anchor] = clean_elevation_profile(z_fn(xs[anchor], ys[anchor]))
    z[~anchor] = np.interp(s[~anchor], s[anchor], z[anchor])
    dz = np.diff(z)
    dxy = np.diff(s)
    mid_t = ~seg_grounded[np.clip(np.searchsorted(cum, 0.5 * (s[:-1] + s[1:]), side="right") - 1,
                                  0, len(seg_grounded) - 1)]
    up, down = np.clip(dz, 0, None), np.clip(-dz, 0, None)
    return {"length": float(np.sqrt(dxy ** 2 + dz ** 2).sum()), "up": float(up.sum()), "down": float(down.sum()),
            "up_t": float(up[mid_t].sum()), "down_t": float(down[mid_t].sum())}


def profile_rail_graph(R0, dem, metric_crs, transfer_max_m: float, frame: MetricFrame = None, fn=None,
                       ds: float = DS):
    """Copy of R0 moved by fn (metric transformation in `frame`), every rail edge profiled on `dem`
    (an object with .sample(lon, lat)). Walking transfers between stops are rebuilt.

    Returns (R, stop_nodes, stop_xy) with stop_xy in `metric_crs`.
    """
    move = (lambda lon, lat: frame.apply(lon, lat, fn)) if fn is not None else (lambda lon, lat: (lon, lat))
    nodes0 = list(R0.nodes())
    pf = MetricFrame(*np.mean([[R0.nodes[n]["lon"], R0.nodes[n]["lat"]] for n in nodes0], axis=0))
    R = nx.Graph()
    for n in nodes0:
        lon, lat = move(np.array([R0.nodes[n]["lon"]]), np.array([R0.nodes[n]["lat"]]))
        R.add_node(n, lon=float(lon[0]), lat=float(lat[0]),
                   z=float(clean_elevation_profile(dem.sample(lon, lat))[0]))
    for u, v, d in R0.edges(data=True):
        if d.get("kind") != "rail":
            continue
        lon, lat = move(d["poly"][:, 0], d["poly"][:, 1])
        x, y = pf.to_metric(lon, lat)
        prof = profile_polyline(x, y, d["seg_grounded"], lambda xs, ys: dem.sample(*pf.to_lonlat(xs, ys)), ds)
        R.add_edge(u, v, kind="rail", mode=d["mode"], start=d["start"], weight=prof["length"], **prof)
    nodes = list(R.nodes())
    to_m = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    X, Y = to_m.transform([R.nodes[n]["lon"] for n in nodes], [R.nodes[n]["lat"] for n in nodes])
    P = np.column_stack([X, Y])
    for i, j in KDTree(P).query_pairs(transfer_max_m):
        a, b = nodes[i], nodes[j]
        if not R.has_edge(a, b):
            dist = float(np.linalg.norm(P[i] - P[j]))
            R.add_edge(a, b, kind="walk", length=dist, weight=WALK_FACTOR * dist)
    return R, nodes, P


def edge_terms(R, a, b):
    """(length, up, down, up_t, down_t) of rail edge a -> b."""
    d = R[a][b]
    if d["start"] == a:
        return d["length"], d["up"], d["down"], d["up_t"], d["down_t"]
    return d["length"], d["down"], d["up"], d["down_t"], d["up_t"]


def rail_source_sums(R, source):
    """Per reachable stop: {mode: array(length, up, down, up_t, down_t)} on the shortest path."""
    _, paths = nx.single_source_dijkstra(R, source, weight="weight")
    out = {}
    for t, path in paths.items():
        acc = {m: np.zeros(TERMS) for m in RAIL_MODES}
        for a, b in zip(path[:-1], path[1:]):
            if R[a][b]["kind"] == "rail":
                acc[R[a][b]["mode"]] += edge_terms(R, a, b)
        out[t] = acc
    return out
