#!/usr/bin/env python3
"""
author: Federico Bellisardi
"""

import os
import argparse
import json
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from pyproj import Transformer


def load_graphs(graphs_dir, stats_df):
    graphs = {}
    for _, row in stats_df.iterrows():
        variant = row['variant']
        filename = 'graph_original.pkl' if variant == 'original' else f'graph_{variant}.pkl'
        path = os.path.join(graphs_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: pickle for variant '{variant}' not found at {path}")
            continue
        with open(path, 'rb') as f:
            G = pickle.load(f)
        graphs[variant] = {'graph': G, 'stats': row.to_dict()}
    return graphs


def load_cell_coordinates(path, cells_crs):
    df = pd.read_csv(path)
    req = {'cell_id','x_min','y_min','x_max','y_max'}
    if not req.issubset(df.columns):
        raise ValueError(f"cell_coordinates.csv must have as columns: {req}")

    transformer = Transformer.from_crs(cells_crs, "EPSG:4326", always_xy=True)

    df['x_cent'] = (df.x_min + df.x_max)/2
    df['y_cent'] = (df.y_min + df.y_max)/2
    df[['x','y']] = df.apply(
        lambda r: transformer.transform(r.x_cent, r.y_cent),
        axis=1, result_type='expand'
    )

    df[['xmin_wgs','ymin_wgs']] = df.apply(
        lambda r: transformer.transform(r.x_min, r.y_min),
        axis=1, result_type='expand'
    )
    df[['xmax_wgs','ymax_wgs']] = df.apply(
        lambda r: transformer.transform(r.x_max, r.y_max),
        axis=1, result_type='expand'
    )

    return df.rename(columns={'cell_id':'cell'})[
        ['cell','x','y','xmin_wgs','ymin_wgs','xmax_wgs','ymax_wgs']
    ]


def load_od_list(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'cell_origin': 'origin',
        'cell_destination': 'dest',
        'count': 'flow'
    })
    df['origin'] = df['origin'].astype(str)
    df['dest']   = df['dest'].astype(str)
    df['flow']   = pd.to_numeric(df['flow'], errors='coerce').fillna(0.0)
    return df[df['flow'] > 0].reset_index(drop=True)


def map_cells_to_nodes(G, cell_df):
    node_items = list(G.nodes(data=True))
    node_ids   = np.array([n for n,_ in node_items])
    lons = np.array([data['x'] for _,data in node_items])
    lats = np.array([data['y'] for _,data in node_items])

    mapping = {}
    for _, row in cell_df.iterrows():
        cid = row['cell']
        mask = (
            (lons >= row['xmin_wgs']) & (lons <= row['xmax_wgs']) &
            (lats >= row['ymin_wgs']) & (lats <= row['ymax_wgs'])
        )
        nodes_in = node_ids[mask].tolist()
        mapping[str(cid)] = nodes_in
    return mapping


def compute_total_work(G, cell_map, od_list):
    total_work = 0.0
    for origin, dest, flow in od_list.itertuples(index=False, name=None):
        src_nodes = cell_map.get(origin, [])
        dst_nodes = cell_map.get(dest,   [])
        if not src_nodes or not dst_nodes:
            continue
        n_pairs = len(src_nodes) * len(dst_nodes)
        if n_pairs == 0:
            continue
        q = flow / n_pairs
        for u in src_nodes:
            for v in dst_nodes:
                try:
                    path = nx.shortest_path(G, source=u, target=v, weight='length')
                except nx.NetworkXNoPath:
                    continue
                dz_sum = 0.0
                for a,b in zip(path[:-1], path[1:]):
                    dz = G.nodes[b].get('z', 0) - G.nodes[a].get('z', 0)
                    if dz > 0:
                        dz_sum += dz
                total_work += q * dz_sum
    return total_work


def parse_config_or_args():
    parser = argparse.ArgumentParser(description='Compute gravitational work for city graph variants.')
    parser.add_argument("-c", "--conf", default="conf_wheight.json", help='JSON config file')
    args = parser.parse_args()

    with open(args.conf) as cf:
        conf = json.load(cf)

    missing = [k for k in ('graphs_dir','cells_file','od_work_file','od_holiday_file','cells_crs')
               if k not in conf]
    if missing:
        parser.error(f"Missing in config: {', '.join(missing)}")

    return (
        conf['graphs_dir'],
        conf['cells_file'],
        conf['od_work_file'],
        conf['od_holiday_file'],
        conf['cells_crs']
    )


def main():
    graphs_dir, cells_file, od_work_file, od_holiday_file, cells_crs = parse_config_or_args()

    stats_df = pd.read_csv(os.path.join(graphs_dir, 'graph_stats.csv'))
    print(f"Loaded statistics for {len(stats_df)} variants.")
    graphs = load_graphs(graphs_dir, stats_df)
    print(f"Loaded {len(graphs)} graph variants.\n")

    cell_df = load_cell_coordinates(cells_file, cells_crs)
    od_work = load_od_list(od_work_file)
    od_hol  = load_od_list(od_holiday_file)
    print(f"Loaded {len(cell_df)} cells, OD working (rows={len(od_work)}), holiday (rows={len(od_hol)}).\n")

    original_graph = graphs['original']['graph']
    cell_map = map_cells_to_nodes(original_graph, cell_df)

    results = []
    for variant, data in graphs.items():
        print(f"Processing '{variant}'…")
        work_wd  = compute_total_work(data['graph'], cell_map, od_work)
        work_hol = compute_total_work(data['graph'], cell_map, od_hol)
        results.append({
            'variant': variant,
            'work_working_day': work_wd,
            'work_holiday':     work_hol
        })
        print(f"  → WD:  {work_wd:.2f},  Holiday: {work_hol:.2f}\n")

    out_df = pd.DataFrame(results)
    out_path = os.path.join(graphs_dir, 'gravitational_work_by_variant.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
