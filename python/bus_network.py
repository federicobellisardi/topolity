"""Bus network of Barcelona from OpenStreetMap route relations (PTv2).

Stops are the ordered stop members of each route; consecutive stops are joined
along the route's own ways. Buses run on the surface, so every edge carries the
terrain profile sampled from the DEM every DS metres along the track (3-D length,
uphill and downhill gain), unlike rail where tunnel depths are unknown. The graph
is directed because the two directions of a line often use different streets.
Stops of any lines within `transfer_max_m` are joined by walking transfers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import KDTree

from barcelona_multimodal_robustness import OVERPASS_URL, TRANSIT_DIR, USER_AGENT, WALK_FACTOR, _norm
from terrain import MetricFrame, clean_elevation_profile, resample_polyline

BUS_CACHE = TRANSIT_DIR / "osm_bus_routes.json"
DS = 10.0
# long-distance coach services: not part of daily urban mobility
MIRRORS = [OVERPASS_URL, "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]
MAX_TORTUOSITY = 3.0
EXCLUDED = {"flixbus", "alsa", "blablacar bus", "eurolines", "ouibus", "iryo", "monbus internacional"}


def download_bus_routes(bbox, cache: Path = BUS_CACHE) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    s, w, n, e = bbox
    query = (f'[out:json][timeout:600];'
             f'relation["type"="route"]["route"~"^(bus|trolleybus)$"]({s},{w},{n},{e});'
             f'out body;>;out skel qt;')
    last = None
    for attempt in range(6):          # Overpass often returns 429/502/504 under load
        url = MIRRORS[attempt % len(MIRRORS)]
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


# Networks of a public transport authority: kept whatever the operator (in Madrid ALSA runs
# Consorcio interurban lines, which are part of daily mobility; ALSA coaches are excluded).
PUBLIC_NETWORKS = {"consorcio regional de transportes de madrid"}


def is_urban(tags: dict) -> bool:
    if tags.get("bus") == "coach":
        return False
    if _norm(tags.get("network")) in PUBLIC_NETWORKS:
        return True
    return not ({_norm(tags.get("network")), _norm(tags.get("operator"))} & EXCLUDED)


def build_bus_graph(osm: dict, metric_crs, transfer_max_m: float, bounds=None, max_tortuosity=MAX_TORTUOSITY):
    """Directed stop graph; bus edges keep their track polyline (lon/lat) in 'poly'.

    bounds (minlon, minlat, maxlon, maxlat): stops outside are dropped (outside the study area).
    Edges whose track is more than `max_tortuosity` times the straight distance are dropped as
    gaps in the OSM track.
    """
    lonlat = {e["id"]: (e["lon"], e["lat"]) for e in osm["elements"] if e["type"] == "node"}
    ways = {e["id"]: e["nodes"] for e in osm["elements"] if e["type"] == "way"}
    to_m = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)

    def xy(nid):
        return np.array(to_m.transform(*lonlat[nid]))

    B = nx.DiGraph()
    stats = {"routes_used": 0, "routes_excluded": 0, "routes_skipped": 0, "track_fallbacks": 0,
             "edges_outside_bounds": 0, "edges_tortuous": 0}

    def inside(nid):
        if bounds is None:
            return True
        lon, lat = lonlat[nid]
        return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]
    for rel in (e for e in osm["elements"] if e["type"] == "relation"):
        tags = rel.get("tags", {})
        if not is_urban(tags):
            stats["routes_excluded"] += 1
            continue
        T = nx.Graph()
        for m in rel["members"]:
            if m["type"] == "way" and m["ref"] in ways:
                nds = [nd for nd in ways[m["ref"]] if nd in lonlat]
                for a, b in zip(nds[:-1], nds[1:]):
                    T.add_edge(a, b, w=float(np.linalg.norm(xy(a) - xy(b))))
        node_members = [m for m in rel["members"] if m["type"] == "node" and m["ref"] in lonlat]
        stops = [m["ref"] for m in node_members if m.get("role", "").startswith("stop")]
        if len(stops) < 2:
            stops = [m["ref"] for m in node_members if m.get("role", "").startswith("platform")]
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
                poly = np.array([lonlat[s1]] + [lonlat[p] for p in path] + [lonlat[s2]])
            except nx.NetworkXNoPath:
                poly = np.array([lonlat[s1], lonlat[s2]])
                stats["track_fallbacks"] += 1
            pm = np.array(to_m.transform(poly[:, 0], poly[:, 1])).T
            track_len = float(np.sum(np.linalg.norm(np.diff(pm, axis=0), axis=1)))
            if track_len > max_tortuosity * max(float(np.linalg.norm(pm[-1] - pm[0])), 1.0):
                stats["edges_tortuous"] += 1
                continue
            for s in (s1, s2):
                if s not in B:
                    B.add_node(s, lon=lonlat[s][0], lat=lonlat[s][1])
            if not B.has_edge(s1, s2) or len(B[s1][s2]["poly"]) > len(poly):
                B.add_edge(s1, s2, poly=poly, kind="bus")
    stats["stops"] = B.number_of_nodes()
    stats["bus_edges"] = B.number_of_edges()
    return B, stats


def profile_bus_graph(B0, dem, metric_crs, transfer_max_m: float, frame: MetricFrame = None, fn=None):
    """Copy of B0 moved by fn (metric transformation in `frame`), with every bus edge profiled
    on `dem`: 3-D length, uphill and downhill gain. Walking transfers are rebuilt."""
    B = nx.DiGraph()
    move = (lambda lon, lat: frame.apply(lon, lat, fn)) if fn is not None else (lambda lon, lat: (lon, lat))
    pf = MetricFrame(*np.mean([[d["lon"], d["lat"]] for _, d in B0.nodes(data=True)], axis=0))
    for n, d in B0.nodes(data=True):
        lon, lat = move(np.array([d["lon"]]), np.array([d["lat"]]))
        B.add_node(n, lon=float(lon[0]), lat=float(lat[0]))
    for u, v, d in B0.edges(data=True):
        if d.get("kind") != "bus":
            continue
        lon, lat = move(d["poly"][:, 0], d["poly"][:, 1])
        x, y = pf.to_metric(lon, lat)
        xs, ys = resample_polyline(x, y, DS)
        z = clean_elevation_profile(dem.sample(*pf.to_lonlat(xs, ys)))
        dxy, dz = np.hypot(np.diff(xs), np.diff(ys)), np.diff(z)
        L3 = float(np.sqrt(dxy ** 2 + dz ** 2).sum())
        B.add_edge(u, v, kind="bus", length=L3, weight=L3,
                   up=float(np.clip(dz, 0, None).sum()), down=float(np.clip(-dz, 0, None).sum()))
    nodes = list(B.nodes())
    to_m = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    X, Y = to_m.transform([B.nodes[n]["lon"] for n in nodes], [B.nodes[n]["lat"] for n in nodes])
    P = np.column_stack([X, Y])
    for i, j in KDTree(P).query_pairs(transfer_max_m):
        a, b = nodes[i], nodes[j]
        dist = float(np.linalg.norm(P[i] - P[j]))
        for s, t in ((a, b), (b, a)):
            if not B.has_edge(s, t):
                B.add_edge(s, t, kind="walk", length=dist, weight=WALK_FACTOR * dist, up=0.0, down=0.0)
    return B, nodes, P


def bus_source_sums(B, source):
    """Per reachable stop: [length, uphill, downhill] of the bus legs of the shortest path."""
    _, paths = nx.single_source_dijkstra(B, source, weight="weight")
    out = {}
    for t, path in paths.items():
        acc = [0.0, 0.0, 0.0]
        for a, b in zip(path[:-1], path[1:]):
            d = B[a][b]
            if d["kind"] == "bus":
                acc[0] += d["length"]
                acc[1] += d["up"]
                acc[2] += d["down"]
        out[t] = acc
    return out


class BusRouter:
    """Shortest bus paths (time-like weight, walking transfers included) with compact per-source results.

    sums(i) returns a float32 array of shape (3, n_stops) with the length, uphill and
    downhill of the bus legs from stop index i to every stop (NaN if unreachable).
    """

    def __init__(self, B):
        from scipy import sparse
        self.nodes = list(B.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        u, v, w, attr = [], [], [], {}
        for a, b, d in B.edges(data=True):
            i, j = self.index[a], self.index[b]
            u.append(i)
            v.append(j)
            w.append(max(d["weight"], 1e-3))          # explicit zeros would be dropped
            bus = d["kind"] == "bus"
            attr[i * n + j] = (d["length"] if bus else 0.0, d["up"] if bus else 0.0, d["down"] if bus else 0.0)
        self.n = n
        self.attr = attr
        self.graph = sparse.csr_matrix((w, (u, v)), shape=(n, n))

    def sums(self, i):
        from scipy.sparse.csgraph import dijkstra
        dist, pred = dijkstra(self.graph, directed=True, indices=i, return_predecessors=True)
        out = np.full((3, self.n), np.nan, dtype=np.float32)
        out[:, i] = 0.0
        n, attr = self.n, self.attr
        for j in np.argsort(dist):
            if not np.isfinite(dist[j]):
                break
            if j == i:
                continue
            p = pred[j]
            out[:, j] = out[:, p] + attr[p * n + j]
        return out
