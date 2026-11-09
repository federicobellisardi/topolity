"""Tests for python/rail_network.py (rail profiles sampled along the track)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline_production"))

import rail_network as rn  # noqa: E402
from terrain import scaling_fn  # noqa: E402


def bump(x, y, x0=500.0, h=40.0, w=100.0):
    """Gaussian hill of height h centred on x = x0."""
    return h * np.exp(-((np.asarray(x) - x0) / w) ** 2)


class CountingZ:
    def __init__(self, f):
        self.f, self.x = f, []

    def __call__(self, x, y):
        self.x.append(np.asarray(x))
        return self.f(x, y)


LINE_X, LINE_Y = np.array([0.0, 1000.0]), np.array([0.0, 0.0])


def test_flat_surface_track():
    p = rn.profile_polyline(LINE_X, LINE_Y, np.array([True]), lambda x, y: np.zeros_like(x))
    assert p["length"] == pytest.approx(1000.0)
    assert p["up"] == p["down"] == p["up_t"] == p["down_t"] == 0.0


def test_surface_track_follows_hill_with_10m_sampling():
    z = CountingZ(bump)
    p = rn.profile_polyline(LINE_X, LINE_Y, np.array([True]), z, ds=10.0)
    xs = np.concatenate(z.x)
    assert np.max(np.diff(np.sort(xs))) <= 10.0 + 1e-9
    assert p["up"] == pytest.approx(40.0, abs=0.1) and p["down"] == pytest.approx(40.0, abs=0.1)
    assert p["up_t"] == 0.0 and p["length"] > 1000.0


def test_tunnel_does_not_follow_terrain():
    p = rn.profile_polyline(LINE_X, LINE_Y, np.array([False]), bump)
    assert p["up"] == pytest.approx(0.0, abs=1e-6) and p["down"] == pytest.approx(0.0, abs=1e-6)
    assert p["length"] == pytest.approx(1000.0)


def test_tunnel_between_stations_at_different_heights_is_a_uniform_grade():
    ramp = lambda x, y: 0.05 * np.asarray(x)          # noqa: E731  station 1 at 0 m, station 2 at 50 m
    p = rn.profile_polyline(LINE_X, LINE_Y, np.array([False]), ramp)
    assert p["up"] == pytest.approx(50.0) and p["up_t"] == pytest.approx(50.0)
    assert p["down"] == 0.0 and p["down_t"] == 0.0


def test_mixed_track_uses_portal_elevation_inside_tunnel():
    # surface 0-300 m over flat ground, tunnel 300-1000 m under a hill; terrain at x=1000 is 20 m
    x, y = np.array([0.0, 300.0, 1000.0]), np.zeros(3)
    terrain = lambda x, y: bump(x, y, x0=650.0, h=60.0) + 0.02 * np.asarray(x)  # noqa: E731
    p = rn.profile_polyline(x, y, np.array([True, False]), terrain)
    z300 = float(terrain(np.array([300.0]), 0)[0])
    z1000 = float(terrain(np.array([1000.0]), 0)[0])
    assert p["up_t"] == pytest.approx(z1000 - z300, abs=0.05)       # straight grade portal -> station
    assert p["up"] == pytest.approx(z1000, abs=0.05)                 # no 60 m hill
    assert p["down_t"] == pytest.approx(0.0, abs=1e-9)


def test_grounded_tag_rules():
    assert rn.grounded({}) and rn.grounded({"tunnel": "no"}) and rn.grounded({"cutting": "yes"})
    assert not rn.grounded({"tunnel": "yes"})
    assert not rn.grounded({"bridge": "viaduct"})
    assert not rn.grounded({"location": "underground"})
    assert not rn.grounded({"layer": "-1"})
    assert rn.grounded({"layer": "1"})                     # elevated layer without bridge tag: kept


def _osm():
    # metro: stops 1 -> 2 -> 3 along lat 41.40; way 900 surface, way 901 tunnel
    nodes = {1: (2.100, 41.40), 100: (2.105, 41.40), 2: (2.110, 41.40), 101: (2.115, 41.40), 3: (2.120, 41.40),
             10: (2.120, 41.4015), 11: (2.120, 41.42)}
    el = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    el += [{"type": "way", "id": 900, "nodes": [1, 100, 2], "tags": {"railway": "subway"}},
           {"type": "way", "id": 901, "nodes": [3, 101, 2], "tags": {"railway": "subway", "tunnel": "yes"}},
           {"type": "way", "id": 902, "nodes": [10, 11], "tags": {"railway": "rail"}},
           {"type": "relation", "id": 1, "tags": {"route": "subway", "network": "Metro de Barcelona"},
            "members": [{"type": "node", "ref": 1, "role": "stop"}, {"type": "node", "ref": 2, "role": "stop"},
                        {"type": "node", "ref": 3, "role": "stop"}, {"type": "way", "ref": 900, "role": ""},
                        {"type": "way", "ref": 901, "role": ""}]},
           {"type": "relation", "id": 2, "tags": {"route": "train", "network": "Rodalies de Catalunya"},
            "members": [{"type": "node", "ref": 10, "role": "stop"}, {"type": "node", "ref": 11, "role": "stop"},
                        {"type": "way", "ref": 902, "role": ""}]},
           {"type": "relation", "id": 3, "tags": {"route": "train", "network": "Renfe AVE"},
            "members": [{"type": "node", "ref": 1, "role": "stop"}, {"type": "node", "ref": 11, "role": "stop"},
                        {"type": "way", "ref": 900, "role": ""}]}]
    return {"elements": el}


class EastHillDEM:
    """Terrain rising linearly east of lon 2.10: 0 m at 2.10, 100 m at 2.12 (lon/lat input)."""

    def sample(self, lon, lat):
        return np.clip((np.asarray(lon) - 2.10) / 0.02, 0, None) * 100.0


@pytest.fixture
def graph():
    R0, stats = rn.build_rail_graph(_osm(), ["Metro de Barcelona", "Rodalies de Catalunya"], "EPSG:25831")
    return R0, stats


def test_build_rail_graph_flags_and_exclusions(graph):
    R0, stats = graph
    assert stats["routes_used"] == 2                         # AVE not in the included networks
    assert set(R0.nodes()) == {1, 2, 3, 10, 11}
    assert R0[1][2]["mode"] == "metro" and R0[10][11]["mode"] == "train"
    assert R0[1][2]["seg_grounded"].all()
    assert not R0[2][3]["seg_grounded"].any()               # tunnel way, stored reversed in OSM
    assert not R0.has_edge(1, 11)


def test_profiled_graph_direction_and_transfers(graph):
    R0, _ = graph
    R, nodes, P = rn.profile_rail_graph(R0, EastHillDEM(), "EPSG:25831", transfer_max_m=300)
    assert R[3][10]["kind"] == "walk"
    L12, up12, dn12, upt12, dnt12 = rn.edge_terms(R, 1, 2)
    assert up12 == pytest.approx(50.0, abs=0.5) and dn12 == pytest.approx(0.0, abs=1e-6) and upt12 == 0.0
    L21, up21, dn21, _, _ = rn.edge_terms(R, 2, 1)
    assert (L21, up21, dn21) == pytest.approx((L12, dn12, up12))
    _, up23, _, upt23, _ = rn.edge_terms(R, 2, 3)
    assert up23 == pytest.approx(50.0, abs=0.5) and upt23 == pytest.approx(up23)   # tunnel: straight grade
    sums = rn.rail_source_sums(R, 1)
    assert sums[3]["metro"][1] == pytest.approx(100.0, abs=1.0)
    assert sums[3]["metro"][3] == pytest.approx(sums[3]["metro"][1] - up12)
    back = rn.rail_source_sums(R, 3)[1]["metro"]
    assert back[2] == pytest.approx(sums[3]["metro"][1]) and back[1] == pytest.approx(sums[3]["metro"][2])
    assert sums[11]["train"][0] > 0


def test_scaling_stretches_track_length(graph):
    from terrain import MetricFrame
    R0, _ = graph
    flat = type("Flat", (), {"sample": lambda self, lon, lat: np.zeros_like(np.asarray(lon, float))})()
    frame = MetricFrame(2.11, 41.40)
    R1, _, _ = rn.profile_rail_graph(R0, flat, "EPSG:25831", 300)
    R2, _, _ = rn.profile_rail_graph(R0, flat, "EPSG:25831", 300, frame=frame, fn=scaling_fn(1.2, "x"))
    assert R2[1][2]["length"] == pytest.approx(1.2 * R1[1][2]["length"], rel=1e-3)
