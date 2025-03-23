"""
author: Federico Bellisardi

This module contains functions to compute models.
"""

import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import Point

from tqdm import tqdm

from utils import logger, haversine, gravitational_work, friction_work, integrator

def compute_gravitational_model(grid, G, dyn_conf, cell_mass_threshold=0, model_type="gravitational"):
    """
    Compute a gravitational model between grid cells based on population mass and network distances.
    
    Parameters:
        grid: GeoDataFrame containing grid cells with a 'mass' column and 'centroid'.
        G: networkx graph representing the road network.
        dyn_conf: dictionary with dynamics configuration (expects at least "epsilon").
        cell_mass_threshold: cells with mass <= threshold are ignored.
        model_type: string indicating which model to compute. Default is "gravitational".
                    (You can add alternative models later.)
                    
    Returns:
        grav_norm: a dictionary with normalized gravitational weights between grid cells.
        grid_nonzero: the subset of grid cells with mass above the threshold.
    """

    grid_nonzero = grid[grid['mass'] > cell_mass_threshold].copy()
    grid_nonzero = grid_nonzero.to_crs(epsg=4326)
    
    grid_nonzero['node'] = grid_nonzero['centroid'].apply(
        lambda geom: ox.distance.nearest_nodes(G, X=geom.x, Y=geom.y)
    )
    
    epsilon = dyn_conf.get("epsilon", 1e-6)
    grav = {}
    cell_ids_list = grid_nonzero['cell_id'].tolist()
    
    for i in tqdm(cell_ids_list, desc="Computing gravitational model", unit="cell"):
        grav[i] = {}
        mass_i = grid_nonzero.loc[grid_nonzero['cell_id'] == i, 'mass'].values[0]
        node_i = grid_nonzero.loc[grid_nonzero['cell_id'] == i, 'node'].values[0]
        for j in cell_ids_list:
            if i == j:
                grav[i][j] = 0
            else:
                mass_j = grid_nonzero.loc[grid_nonzero['cell_id'] == j, 'mass'].values[0]
                node_j = grid_nonzero.loc[grid_nonzero['cell_id'] == j, 'node'].values[0]
                try:
                    d = nx.shortest_path_length(G, source=node_i, target=node_j, weight='length')
                except nx.NetworkXNoPath:
                    d = float('inf')
                grav[i][j] = 0 if d == float('inf') else (mass_i * mass_j) / (d**2 + epsilon)

    grav_norm = {}
    for i in grav:
        total = sum(grav[i].values())
        if total == 0:
            n = len(grav[i])
            grav_norm[i] = {j: 1/n for j in grav[i]}
        else:
            grav_norm[i] = {j: grav[i][j] / total for j in grav[i]}
    
    logger.info("Gravitational model computed using model type '%s'.", model_type)
    return grav_norm, grid_nonzero


def grav_high(G, source, target, m=1, g=1, dh=0.1, mu=0.1, model_type="work_based"):
    """
    Compute the shortest path between source and target in graph G using a work-based edge weight.
    
    For each edge from node u to node v, the weight is computed as:
      - If the altitude at v is higher than at u: gravitational work is computed.
      - Otherwise: friction work is computed.
    
    The work-based weight is computed for every edge along the path.
    
    Parameters:
        G: networkx graph with nodes that have an 'alt' attribute (altitude), as well as 'x' and 'y' coordinates.
        source: source node id.
        target: target node id.
        m: mass (default: 1).
        g: gravitational acceleration (default: 1).
        dh: step size used in gravitational work integration (default: 0.1).
        mu: friction coefficient (default: 0.1).
        model_type: string to choose model type. Currently supports "work_based".
        
    Returns:
        path: list of nodes representing the shortest path (or None if no path exists).
        total_work: total work computed along the path.
    """
    
    def get_altitude_for_point(lat, lon, dem_gdf, radius=50):
        # Convert point to shapely Point
        pt = Point(lon, lat)
        delta = radius / 111000.0  # approximate conversion from meters to degrees
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        candidate_indices = list(dem_gdf.sindex.intersection(bbox))
        within_candidates = []
        for idx in candidate_indices:
            row = dem_gdf.iloc[idx]
            d = haversine(lat, lon, row['lat'], row['lon'])
            if d <= radius:
                within_candidates.append((d, row['alt']))
        if within_candidates:
            within_candidates.sort(key=lambda x: x[0])
            return float(within_candidates[0][1])
        else:
            nearest_idx = list(dem_gdf.sindex.nearest([pt.coords[0]], 1))[0]
            return float(dem_gdf.iloc[nearest_idx]['alt'])
    
    def work_weight(u, v, d):
        if 'alt' not in G.nodes[u]:
            G.nodes[u]['alt'] = get_altitude_for_point(G.nodes[u]['y'], G.nodes[u]['x'], dem_gdf, search_radius)
        if 'alt' not in G.nodes[v]:
            G.nodes[v]['alt'] = get_altitude_for_point(G.nodes[v]['y'], G.nodes[v]['x'], dem_gdf, search_radius)
        h1 = G.nodes[u]['alt']
        h2 = G.nodes[v]['alt']
        if h2 > h1:
            weight = gravitational_work(h1, h2, m, g, dh)
        else:
            weight = friction_work(G.nodes[u]['y'], G.nodes[u]['x'],
                                    G.nodes[v]['y'], G.nodes[v]['x'], m, g, mu)
        return weight
    
    try:
        path = nx.shortest_path(G, source=source, target=target, weight=work_weight)
    except nx.NetworkXNoPath:
        logger.warning("No path found between %s and %s using work-based model.", source, target)
        return None, None
    
    total_work = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_work = work_weight(u, v, None)
        total_work += edge_work
    
    logger.info("Work-based model computed path with total work: %.2f", total_work)
    return path, total_work