#!/usr/bin/env python3
"""
Render Folium HTML maps from existing graph_*.pkl files, then build an index page.

Use this when you ran dem_extractor.py with --no-maps and want to generate maps later.

Examples:
  python tools/render_maps_from_pickles.py --base_path /data/workspaces/fbellisardi/data_processed --city madrid
  python tools/render_maps_from_pickles.py --base_path /data/workspaces/fbellisardi/data_processed --overwrite

By default, only missing maps are generated (skip existing HTML files).
"""
import os
import re
import glob
import argparse
import pickle
from typing import List, Tuple

import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Sorting order for consistent colormap assignment
SECTION_ORDER = (
    'original',
    'translated_',
    'rot_',
    'scale_x_',
    'scale_y_',
)


def variant_sort_key(variant: str) -> Tuple[int, float, str]:
    """Sort variants deterministically: original, translated_N, rot_deg, scale_x_v, scale_y_v."""
    for i, prefix in enumerate(SECTION_ORDER):
        if variant == 'original':
            return (0, 0.0, variant)
        if variant.startswith(prefix):
            # Try to extract trailing numeric value (int/float, possibly signed)
            num = None
            m = re.search(r'([-+]?\d+(?:\.\d+)?)$', variant)
            if m:
                try:
                    num = float(m.group(1))
                except Exception:
                    num = None
            return (i, num if num is not None else float('inf'), variant)
    return (len(SECTION_ORDER), float('inf'), variant)


def list_city_pickles(graphs_dir: str) -> List[Tuple[str, str]]:
    """Return list of (variant, pickle_path), sorted for stable coloring."""
    out = []
    for p in glob.glob(os.path.join(graphs_dir, 'graph_*.pkl')):
        name = os.path.basename(p)
        if not name.startswith('graph_') or not name.endswith('.pkl'):
            continue
        variant = name[len('graph_'):-len('.pkl')]
        out.append((variant, p))
    return sorted(out, key=lambda t: variant_sort_key(t[0]))


def compute_center_from_graph(G) -> Tuple[float, float]:
    ys = []
    xs = []
    for _, data in G.nodes(data=True):
        y = data.get('y')
        x = data.get('x')
        if y is not None and x is not None:
            ys.append(float(y))
            xs.append(float(x))
    if ys and xs:
        return (sum(ys)/len(ys), sum(xs)/len(xs))
    return (40.0, -3.7)  # fallback near Madrid


def render_city(base_path: str, city: str, overwrite: bool, zoom: int = 13) -> None:
    city_dir = os.path.join(base_path, city)
    graphs_dir = os.path.join(city_dir, 'graphs')
    if not os.path.isdir(graphs_dir):
        print(f"[skip] {city}: graphs dir not found: {graphs_dir}")
        return

    pairs = list_city_pickles(graphs_dir)
    if not pairs:
        print(f"[skip] {city}: no graph_*.pkl found")
        return

    # Build a stable colormap for all variants present
    N = len(pairs)
    cmap = plt.get_cmap('Set1', N)

    for idx, (variant, pkl_path) in enumerate(pairs):
        html_path = os.path.join(graphs_dir, f'graph_{variant}_map.html')
        if os.path.exists(html_path) and not overwrite:
            continue
        with open(pkl_path, 'rb') as f:
            G = pickle.load(f)
        cy, cx = compute_center_from_graph(G)
        fmap = folium.Map(location=[cy, cx], zoom_start=zoom)
        color = mcolors.to_hex(cmap(idx))
        # Draw edges
        for u, v in G.edges():
            y1, x1 = G.nodes[u].get('y'), G.nodes[u].get('x')
            y2, x2 = G.nodes[v].get('y'), G.nodes[v].get('x')
            if None in (y1, x1, y2, x2):
                continue
            folium.PolyLine([(y1, x1), (y2, x2)], color=color, weight=2).add_to(fmap)
        fmap.save(html_path)
        print(f"[map] {city}: {os.path.basename(html_path)}")

    # Build/update index
    try:
        from tools.build_maps_index import process_city as _build_index
        _build_index(base_path, city, iframes=False)
        print(f"[index] {city}: maps_index.html updated")
    except Exception as e:
        print(f"[warn] {city}: could not update index: {e}")


def main():
    parser = argparse.ArgumentParser(description='Render maps from graph_*.pkl files')
    parser.add_argument('--base_path', type=str, default='/data/workspaces/fbellisardi/data_processed',
                        help='Base path containing city folders')
    parser.add_argument('--city', type=str, default=None,
                        help='Render only this city (folder name)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate maps even if they already exist')
    parser.add_argument('--zoom', type=int, default=13,
                        help='Leaflet zoom level')
    args = parser.parse_args()

    if args.city:
        render_city(args.base_path, args.city, overwrite=args.overwrite, zoom=args.zoom)
    else:
        for name in sorted(os.listdir(args.base_path)):
            if os.path.isdir(os.path.join(args.base_path, name)):
                render_city(args.base_path, name, overwrite=args.overwrite, zoom=args.zoom)


if __name__ == '__main__':
    main()
