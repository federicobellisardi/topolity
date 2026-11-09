"""Tests for the round-2 corrections: metric transformations, DEM profile, O/D sampling."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrain import GeoDEM, MetricFrame, Z_MIN_VALID_M, clean_elevation_profile  # noqa: E402
from dem_extractor_fine_grid import rotate_graph, scale_graph, translate_graph  # noqa: E402
from fine_grid_gravitational_work import compute_edge_profile, sample_od_nodes  # noqa: E402

GEOD = Geod(ellps="WGS84")


def _grid_graph(lon0, lat0, n=6, step_deg=0.02):
    """Directed grid of n x n nodes with straight edge geometries."""
    G = nx.MultiDiGraph()
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, x=lon0 + i * step_deg, y=lat0 + j * step_deg)
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < n and j + dj < n:
                    u, v = i * n + j, (i + di) * n + (j + dj)
                    for a, b in ((u, v), (v, u)):
                        geom = LineString([(G.nodes[a]["x"], G.nodes[a]["y"]),
                                           (G.nodes[b]["x"], G.nodes[b]["y"])])
                        G.add_edge(a, b, geometry=geom)
    return G


def _pairwise_geodesic(G):
    nodes = sorted(G.nodes())
    lon = np.array([G.nodes[n]["x"] for n in nodes])
    lat = np.array([G.nodes[n]["y"] for n in nodes])
    i, j = np.triu_indices(len(nodes), k=1)
    _, _, d = GEOD.inv(lon[i], lat[i], lon[j], lat[j])
    return np.asarray(d)


# --------------------------------------------------------------------------
# Transformations preserve internal distances (reviewer point on rotations)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lat0", [-34.6, 41.4, 59.9])
@pytest.mark.parametrize("angle", [-20.0, 5.0, 20.0])
def test_rotation_preserves_distances(lat0, angle):
    G = _grid_graph(2.0, lat0)
    d0 = _pairwise_geodesic(G)
    d1 = _pairwise_geodesic(rotate_graph(G.copy(), angle))
    assert np.max(np.abs(d1 / d0 - 1)) < 1e-4


def test_degree_space_rotation_would_fail():
    """Sanity check that the test above is sensitive to the old bug."""
    G = _grid_graph(2.0, 59.9)
    d0 = _pairwise_geodesic(G)
    th = np.radians(20.0)
    x0 = np.mean([d["x"] for _, d in G.nodes(data=True)])
    y0 = np.mean([d["y"] for _, d in G.nodes(data=True)])
    for _, d in G.nodes(data=True):
        dx, dy = d["x"] - x0, d["y"] - y0
        d["x"], d["y"] = x0 + np.cos(th) * dx - np.sin(th) * dy, y0 + np.sin(th) * dx + np.cos(th) * dy
    d1 = _pairwise_geodesic(G)
    assert np.max(np.abs(d1 / d0 - 1)) > 0.05


def test_rotation_moves_edge_geometries_with_nodes():
    G = rotate_graph(_grid_graph(2.0, 41.4), 15.0)
    for u, v, d in G.edges(data=True):
        (x0, y0), (x1, y1) = d["geometry"].coords[0], d["geometry"].coords[-1]
        assert np.hypot(x0 - G.nodes[u]["x"], y0 - G.nodes[u]["y"]) < 1e-9
        assert np.hypot(x1 - G.nodes[v]["x"], y1 - G.nodes[v]["y"]) < 1e-9


def test_translation_is_rigid_and_has_requested_length():
    G = _grid_graph(-58.4, -34.6)
    d0 = _pairwise_geodesic(G)
    G1 = translate_graph(G.copy(), (1500.0 * np.cos(np.radians(75)), 1500.0 * np.sin(np.radians(75))))
    assert np.max(np.abs(_pairwise_geodesic(G1) / d0 - 1)) < 1e-4
    frame = MetricFrame.from_graph(G)
    x0, y0 = frame.to_metric(G.nodes[0]["x"], G.nodes[0]["y"])
    x1, y1 = frame.to_metric(G1.nodes[0]["x"], G1.nodes[0]["y"])
    assert np.hypot(x1 - x0, y1 - y0) == pytest.approx(1500.0, rel=1e-4)


def test_ns_scaling_changes_only_ns_distances():
    G = _grid_graph(2.0, 59.9)
    G1 = scale_graph(G.copy(), 1.1, axis="y")
    n = 6
    # E-W neighbours (same j): distance unchanged; N-S neighbours: x1.1
    _, _, d_ew0 = GEOD.inv(G.nodes[0]["x"], G.nodes[0]["y"], G.nodes[n]["x"], G.nodes[n]["y"])
    _, _, d_ew1 = GEOD.inv(G1.nodes[0]["x"], G1.nodes[0]["y"], G1.nodes[n]["x"], G1.nodes[n]["y"])
    _, _, d_ns0 = GEOD.inv(G.nodes[0]["x"], G.nodes[0]["y"], G.nodes[1]["x"], G.nodes[1]["y"])
    _, _, d_ns1 = GEOD.inv(G1.nodes[0]["x"], G1.nodes[0]["y"], G1.nodes[1]["x"], G1.nodes[1]["y"])
    assert d_ew1 / d_ew0 == pytest.approx(1.0, abs=2e-3)  # AEQD off-centre scale error
    assert d_ns1 / d_ns0 == pytest.approx(1.1, rel=2e-3)


# --------------------------------------------------------------------------
# Elevation validity rule
# --------------------------------------------------------------------------

def test_clean_profile_keeps_below_sea_level_and_fills_voids():
    z = np.array([-9.0, -8.0, -220.0, -6.0, np.nan, -4.0])
    out = clean_elevation_profile(z)
    assert out[0] == -9.0 and out[1] == -8.0 and out[3] == -6.0
    assert out[2] == pytest.approx(-7.0)
    assert out[4] == pytest.approx(-5.0)
    assert np.all(out >= Z_MIN_VALID_M)


def test_clean_profile_all_invalid_is_flat():
    assert np.all(clean_elevation_profile([np.nan, -500.0, -1000.0]) == 0.0)


# --------------------------------------------------------------------------
# DEM sampled along the edge (reviewer point on node interpolation)
# --------------------------------------------------------------------------

def _write_dem(path, lon0, lat0, n, res, fn):
    lon = lon0 + (np.arange(n) + 0.5) * res
    lat = lat0 + n * res - (np.arange(n) + 0.5) * res
    LON, LAT = np.meshgrid(lon, lat)
    z = fn(LON, LAT).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lon0, lat0 + n * res, res, res),
                       nodata=0) as dst:
        dst.write(z, 1)


@pytest.fixture
def hill_dem(tmp_path):
    """Gaussian hill (50 m) at lon 2.05 on a -5 m plain, with one -220 m void spike."""
    path = tmp_path / "dem.tif"

    def fn(lon, lat):
        z = -5.0 + 50.0 * np.exp(-((lon - 2.05) / 0.01) ** 2)
        z[(np.abs(lon - 2.02) < 0.0005) & (np.abs(lat - 41.4) < 0.0005)] = -220.0
        return z

    _write_dem(path, 1.98, 41.33, 1400, 0.0001, fn)
    return path


def _edge_graph(coords, reverse_geometry=False):
    G = nx.MultiDiGraph()
    G.add_node(0, x=coords[0][0], y=coords[0][1], z=-5.0)
    G.add_node(1, x=coords[-1][0], y=coords[-1][1], z=-5.0)
    line = LineString(coords[::-1] if reverse_geometry else coords)
    G.add_edge(0, 1, geometry=line)
    return G


def test_dem_profile_sees_intermediate_hill(hill_dem):
    G = _edge_graph([(2.0, 41.4), (2.1, 41.4)])
    frame = MetricFrame.from_graph(G)
    dem = GeoDEM(hill_dem, (2.0, 41.4, 2.1, 41.4))
    l2d, l3d, gain, _ = compute_edge_profile(G, 0, 1, dem, frame, elevation_source="dem")
    _, _, gain_node, _ = compute_edge_profile(G, 0, 1, dem, frame, elevation_source="node")
    assert gain == pytest.approx(50.0, abs=1.0)    # climbs the hill; the -220 m spike is filled
    assert gain_node == 0.0                         # endpoint interpolation misses it
    assert l3d > l2d
    _, _, d = GEOD.inv(2.0, 41.4, 2.1, 41.4)
    assert l2d == pytest.approx(d, rel=1e-3)


def test_dem_profile_respects_edge_direction(hill_dem):
    """Gain is computed u -> v even when the stored geometry runs v -> u."""
    coords = [(2.0, 41.4), (2.05, 41.4)]          # u on the plain, v on the hilltop
    dem = GeoDEM(hill_dem, (2.0, 41.4, 2.05, 41.4))
    G_fwd = _edge_graph(coords)
    G_rev = _edge_graph(coords, reverse_geometry=True)
    frame = MetricFrame.from_graph(G_fwd)
    _, _, g_fwd, _ = compute_edge_profile(G_fwd, 0, 1, dem, frame)
    _, _, g_rev, _ = compute_edge_profile(G_rev, 0, 1, dem, frame)
    assert g_fwd == pytest.approx(50.0, abs=1.0)
    assert g_rev == pytest.approx(g_fwd)


def test_gain_and_3d_length_share_the_profile(hill_dem):
    """Below-sea-level plain: gain 0 and L3D == L2D (no clipping in one term only)."""
    G = _edge_graph([(1.99, 41.36), (2.01, 41.36)])
    dem = GeoDEM(hill_dem, (1.99, 41.36, 2.01, 41.36))
    l2d, l3d, gain, _ = compute_edge_profile(G, 0, 1, dem, MetricFrame.from_graph(G))
    assert gain == pytest.approx(0.0, abs=1e-3)   # float32 DEM noise only
    assert l3d == pytest.approx(l2d, rel=1e-9)


# --------------------------------------------------------------------------
# O/D points sampled uniformly within cells (reviewer point on centroids)
# --------------------------------------------------------------------------

def _cells():
    return pd.DataFrame({
        "cell": [0, 1],
        "lon_min": [2.0, 2.1], "lon_max": [2.1, 2.2],
        "lat_min": [41.4, 41.4], "lat_max": [41.5, 41.5],
        "cent_lon": [2.05, 2.15], "cent_lat": [41.45, 41.45],
    })


def test_od_sampling_is_reproducible_and_inside_cells():
    G = _grid_graph(2.005, 41.405, n=10, step_deg=0.01)   # nodes only in cell 0 and its edge
    a = sample_od_nodes(G, _cells(), 20, seed=1)
    b = sample_od_nodes(G, _cells(), 20, seed=1)
    assert a == b
    for o_map, d_map in a:
        for m in (o_map, d_map):
            n = m[0]
            assert 2.0 <= G.nodes[n]["x"] <= 2.1 and 41.4 <= G.nodes[n]["y"] <= 41.5


def test_od_sampling_covers_the_cell_uniformly():
    G = _grid_graph(2.005, 41.405, n=10, step_deg=0.01)   # 10 x 10 nodes, one per sub-square
    samples = sample_od_nodes(G, _cells().iloc[:1], 4000, seed=0)
    picked = [m[0] for pair in samples for m in pair]
    counts = np.bincount(picked, minlength=100)
    assert (counts > 0).all()
    # Interior nodes own equal Voronoi areas -> equal expected frequency
    interior = [i * 10 + j for i in range(1, 9) for j in range(1, 9)]
    expected = 8000 / 100
    assert np.abs(counts[interior] / expected - 1).max() < 0.5
    # Origins and destinations are independent draws
    assert np.mean([o[0] != d[0] for o, d in samples]) > 0.9
