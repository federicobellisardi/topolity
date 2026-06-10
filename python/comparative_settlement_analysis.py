#!/usr/bin/env python3
"""Comparative settlement analysis of gravitational work across candidate locations."""


import os
import sys
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon, box
import networkx as nx
import rasterio
from rasterio.mask import mask
from pyproj import CRS, Transformer
from tqdm.auto import tqdm
from scipy.spatial import KDTree
import contextily as ctx

# Constants
DS = 10.0

M_PHYS_KG = 1200.0
G_PHYS = 9.81

def compute_lambda_from_fuel_params(
    consumption_l_per_100km=(5.0, 8.0),
    energy_mj_per_l=36.0,
    efficiency=0.25,
):
    """Compute lambda (horizontal cost weight) in J/m from fuel parameters."""
    c_min, c_max = consumption_l_per_100km
    c_mean = 0.5 * (c_min + c_max)
    lambda_mean_mj_per_100km = efficiency * energy_mj_per_l * c_mean
    return lambda_mean_mj_per_100km * 10.0  # MJ/100km -> J/m


# Horizontal travel cost weight (J/m), aligned with pipeline_production logic.
HORIZONTAL_COST_WEIGHT = compute_lambda_from_fuel_params()

plt.rcParams['figure.figsize'] = (18, 16)
plt.rcParams['font.size'] = 14
sns.set_style('whitegrid')

print("=" * 80)
print("COMPARATIVE SETTLEMENT ANALYSIS - GRAVITATIONAL WORK")
print("=" * 80)



CITY = "barcelone"
NEW_RESIDENTS = 25000  # Number of new residents to add to each test cell

# Gravity model parameters
D_0 = 25000.0  # Distance decay parameter (meters)
ALPHA = 1.0   # Population exponent

# Display settings for final chart labels: "GJ" or "SCI"
WORK_DISPLAY_MODE = "GJ"

# Paths
BASE_DIR = Path(f"/home/fbellisardi/code/data/data_processed/{CITY}")
CELLS_FILE = BASE_DIR / f"{CITY}_basic_model" / "1000_cells" / "cell_coordinates.csv"
GRAPH_FILE = BASE_DIR / "graphs_fine_grid" / "graph_original.pkl"
DEM_FILE = BASE_DIR / "dem" / f"{CITY}_dem.tif"
OUTPUT_DIR = Path(f"/home/fbellisardi/code/topolity/output/comparative_settlement_{CITY}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# WorldPop data
WORLDPOP_DIR = Path("/home/fbellisardi/code/topolity/data/worldpop/raw/2020")
WORLDPOP_FILE = WORLDPOP_DIR / "esp_ppp_2020.tif"

# City bounding box from metropolis.json (lat, lon in EPSG:4326)
CITY_BBOX_FILE = Path("/home/fbellisardi/code/data/metropolis.json")

# Hardcoded test cells (cell_id) - strategically distributed
# These will be selected based on position: center, north, south, periphery
TEST_CELLS_IDS = []  # Will be determined automatically based on spatial distribution

print(f"\nConfiguration:")
print(f"  City: {CITY}")
print(f"  New residents per test cell: {NEW_RESIDENTS:,}")
print(f"  Gravity model d_0: {D_0:,.0f} m")
print(f"  Horizontal cost weight λ: {HORIZONTAL_COST_WEIGHT:.1f} J/m")
print(f"  Work display mode: {WORK_DISPLAY_MODE}")
print(f"  Cells file: {CELLS_FILE}")
print(f"  Graph file: {GRAPH_FILE}")
print(f"  DEM file: {DEM_FILE}")
print(f"  WorldPop file: {WORLDPOP_FILE}")
print(f"  City bbox file: {CITY_BBOX_FILE}")

# Load city bounding box
import json
with open(CITY_BBOX_FILE) as f:
    metropolis_data = json.load(f)
    city_bbox_coords = metropolis_data[CITY]  # List of [lat, lon] points
    
# Extract bbox (min/max lat/lon)
lats = [coord[0] for coord in city_bbox_coords]
lons = [coord[1] for coord in city_bbox_coords]
BBOX_WGS84 = {
    'min_lat': min(lats),
    'max_lat': max(lats),
    'min_lon': min(lons),
    'max_lon': max(lons)
}

print(f"  City bbox (WGS84): lat[{BBOX_WGS84['min_lat']:.5f}, {BBOX_WGS84['max_lat']:.5f}], "
      f"lon[{BBOX_WGS84['min_lon']:.5f}, {BBOX_WGS84['max_lon']:.5f}]")



print("\n" + "-" * 80)
print("Loading grid cells...")

cells_df = pd.read_csv(CELLS_FILE)
print(f"✓ Loaded {len(cells_df):,} cells from grid")

# Create geometries (cell centroids and polygons)
cells_df['centroid_x'] = (cells_df['x_min'] + cells_df['x_max']) / 2
cells_df['centroid_y'] = (cells_df['y_min'] + cells_df['y_max']) / 2

cells_gdf = gpd.GeoDataFrame(
    cells_df,
    geometry=[box(row.x_min, row.y_min, row.x_max, row.y_max) 
              for _, row in cells_df.iterrows()],
    crs="EPSG:3857"  # Web Mercator
)

cells_gdf['centroid'] = cells_gdf.geometry.centroid

print(f"  Grid bounds (Web Mercator EPSG:3857):")
print(f"    X: {cells_df['x_min'].min():.0f} - {cells_df['x_max'].max():.0f} m")
print(f"    Y: {cells_df['y_min'].min():.0f} - {cells_df['y_max'].max():.0f} m")

# Check a sample cell transformation
sample_cell_wm = cells_gdf.iloc[0]
sample_cell_wgs = gpd.GeoDataFrame([sample_cell_wm], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
sample_bounds_wm = sample_cell_wm.geometry.bounds
sample_bounds_wgs = sample_cell_wgs.geometry.bounds
print(f"  Sample cell (first cell):")
print(f"    Web Mercator bounds: {sample_bounds_wm}")
print(f"    WGS84 bounds: {sample_bounds_wgs}")



print("\nLoading road network graph...")

with open(GRAPH_FILE, 'rb') as f:
    G = pickle.load(f)

print(f"✓ Graph loaded:")
print(f"    Nodes: {G.number_of_nodes():,}")
print(f"    Edges: {G.number_of_edges():,}")

# Extract node positions and elevations
nodes_data = []
for node_id in G.nodes():
    node_data = G.nodes[node_id]
    nodes_data.append({
        'node_id': node_id,
        'x': node_data.get('x', 0),
        'y': node_data.get('y', 0),
        'elevation': node_data.get('elevation', 0)
    })

nodes_df = pd.DataFrame(nodes_data)
nodes_gdf = gpd.GeoDataFrame(
    nodes_df,
    geometry=gpd.points_from_xy(nodes_df['x'], nodes_df['y']),
    crs="EPSG:4326"
)

# Convert to Web Mercator to match cells CRS for distance calculations
nodes_gdf_wm = nodes_gdf.to_crs("EPSG:3857")

# Assign elevations from DEM if nodes don't have them
if nodes_df['elevation'].max() == 0:
    print("  Nodes have no elevation, sampling from DEM...")
    with rasterio.open(DEM_FILE) as dem_src:
        # Nodes are in WGS84 (lon, lat)
        node_coords_ll = [(row.x, row.y) for _, row in nodes_df.iterrows()]
        elevations = [val[0] for val in dem_src.sample(node_coords_ll)]
        
        # Update nodes in graph and dataframe
        for i, node_id in enumerate(nodes_df['node_id']):
            G.nodes[node_id]['elevation'] = elevations[i]
        nodes_df['elevation'] = elevations
    
    print(f"  ✓ Elevations assigned from DEM")

print(f"  Elevation range: {nodes_df['elevation'].min():.1f} - {nodes_df['elevation'].max():.1f} m")



print("\nPrecomputing gravitational work for all edges...")

def compute_edge_work(graph, u, v, dem_src, ds=10.0, mass=1.0, grav=1.0):
    """
    Sample along edge and compute uphill work (matching wheight.py).
    Returns total work for traversing this edge.
    """
    from shapely.geometry import LineString
    
    data = graph.get_edge_data(u, v)
    geom = None
    
    # Handle MultiDiGraph with multiple edges
    if isinstance(data, dict):
        geom = data.get('geometry')
    
    # If no geometry, create straight line
    if geom is None:
        x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
        x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
        geom = LineString([(x1, y1), (x2, y2)])
    
    # Sample points along edge
    length = geom.length
    n_pts = max(int(length / ds) + 1, 2)
    dists = np.linspace(0, length, n_pts)
    pts = [geom.interpolate(d) for d in dists]
    coords = [(pt.x, pt.y) for pt in pts]
    
    # Get elevations
    elevs = [val[0] for val in dem_src.sample(coords)]
    
    # Compute uphill work
    work = 0.0
    for h1, h2 in zip(elevs[:-1], elevs[1:]):
        if h2 > h1:  # Only uphill
            work += mass * grav * (h2 - h1)
    
    return work

# Open DEM for sampling
dem_src = rasterio.open(DEM_FILE)

# Precompute work for all edges
edge_work = {}
for u, v, key in tqdm(G.edges(keys=True), desc="Computing edge work", total=G.number_of_edges()):
    work = compute_edge_work(G, u, v, dem_src, ds=DS, mass=M_PHYS_KG, grav=G_PHYS)
    edge_work[(u, v, key)] = work

print(f"✓ Edge work precomputed for {len(edge_work):,} edges")
print(f"  Mean edge work: {np.mean(list(edge_work.values())):.2f}")
print(f"  Max edge work: {np.max(list(edge_work.values())):.2f}")



print("\nExtracting population from WorldPop...")

if not WORLDPOP_FILE.exists():
    raise FileNotFoundError(f"WorldPop file not found: {WORLDPOP_FILE}")

populations = []
with rasterio.open(WORLDPOP_FILE) as src:
    pop_crs = src.crs
    pop_bounds = src.bounds  # (minx, miny, maxx, maxy) in WorldPop CRS
    
    print(f"  WorldPop CRS: {pop_crs}")
    print(f"  WorldPop full bounds: {pop_bounds}")
    print(f"  Cells CRS: {cells_gdf.crs}")
    
    # Create bbox geometry in WGS84 for filtering
    from shapely.geometry import box as shapely_box
    bbox_geom_wgs84 = shapely_box(
        BBOX_WGS84['min_lon'], 
        BBOX_WGS84['min_lat'],
        BBOX_WGS84['max_lon'], 
        BBOX_WGS84['max_lat']
    )
    
    # Transform cells to WorldPop CRS
    cells_pop = cells_gdf.to_crs(pop_crs)
    
    # Filter cells within city bbox (use WGS84 bbox)
    bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_geom_wgs84}], crs="EPSG:4326").to_crs(pop_crs)
    bbox_geom_pop_crs = bbox_gdf.iloc[0].geometry
    
    cells_pop['in_bbox'] = cells_pop.geometry.intersects(bbox_geom_pop_crs)
    n_in_bbox = cells_pop['in_bbox'].sum()
    
    print(f"  Cells within city bbox: {n_in_bbox:,} / {len(cells_pop):,}")
    
    if n_in_bbox == 0:
        raise ValueError(
            f"No cells within city bounding box!\n"
            f"  City bbox: {BBOX_WGS84}\n"
            f"  Check bbox definition in {CITY_BBOX_FILE}"
        )
    
    # Process only cells in bbox
    cells_to_process = cells_pop[cells_pop['in_bbox']].copy()
    print(f"  Processing {len(cells_to_process):,} cells within bbox...")
    
    error_count = 0
    pop_extracted_count = 0
    
    for idx, row in tqdm(cells_to_process.iterrows(), total=len(cells_to_process), 
                        desc="Extracting population"):
        try:
            out_image, out_transform = mask(src, [row.geometry], crop=True, nodata=0)
            population = float(out_image.sum())
            if population > 0:
                pop_extracted_count += 1
            populations.append((row.name, max(0, population)))  # Store (original_index, population)
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"    Warning: Error extracting population for cell {idx}: {e}")
            populations.append((row.name, 0))
    
    if error_count > 0:
        print(f"  Total extraction errors: {error_count:,}")
    
    print(f"  Cells with population > 0: {pop_extracted_count:,}")

# Assign population back to original cells_gdf
cells_gdf['population'] = 0  # Initialize all to 0
for orig_idx, pop in populations:
    cells_gdf.loc[orig_idx, 'population'] = pop

print(f"\n  Population extraction summary:")
print(f"    Total population extracted: {cells_gdf['population'].sum():,.0f}")
print(f"    Cells with population > 0: {(cells_gdf['population'] > 0).sum():,}")
print(f"    Cells with population > 10: {(cells_gdf['population'] > 10).sum():,}")
print(f"    Max population in a cell: {cells_gdf['population'].max():,.0f}")

# Filter cells with very low population (keep threshold low to retain cells)
cells_gdf = cells_gdf[cells_gdf['population'] > 10].copy()
cells_gdf.reset_index(drop=True, inplace=True)

if len(cells_gdf) == 0:
    raise ValueError(
        f"No cells with population found!\n"
        f"  Cells CRS: {cells_gdf.crs}\n"
        f"  WorldPop CRS: {pop_crs}\n"
        f"  Cells in bbox: {n_in_bbox:,}\n"
        f"  Cells with extracted pop > 0: {pop_extracted_count:,}\n"
        f"  Total cells after filtering (pop > 10): {len(cells_gdf):,}\n"
        f"  Check WorldPop file coverage and city bbox."
    )

total_pop = cells_gdf['population'].sum()
print(f"\n✓ Population extracted:")
print(f"    Total population: {total_pop:,.0f}")
print(f"    Cells with population: {len(cells_gdf):,}")
print(f"    Mean per cell: {cells_gdf['population'].mean():,.0f}")
print(f"    Median per cell: {cells_gdf['population'].median():,.0f}")



print("\nAssigning nearest network node to each cell...")

# Build KD-tree for network nodes
node_coords = np.array([[geom.x, geom.y] for geom in nodes_gdf_wm.geometry])
node_tree = KDTree(node_coords)

# Find nearest node for each cell
cell_coords = np.array([[row.centroid_x, row.centroid_y] for _, row in cells_gdf.iterrows()])
distances, indices = node_tree.query(cell_coords, k=1)

cells_gdf['nearest_node'] = nodes_df.iloc[indices]['node_id'].values
cells_gdf['node_distance'] = distances

print(f"✓ Nearest nodes assigned:")
print(f"    Mean distance to node: {distances.mean():.0f} m")
print(f"    Max distance to node: {distances.max():.0f} m")



print("\nSelecting strategic test cells...")

# Calculate center of mass
center_x = cells_gdf['centroid_x'].mean()
center_y = cells_gdf['centroid_y'].mean()

cells_gdf['dist_to_center'] = np.sqrt(
    (cells_gdf['centroid_x'] - center_x)**2 + 
    (cells_gdf['centroid_y'] - center_y)**2
)

# Select 4 strategic cells
# 1. Center: closest to center of mass, high population
center_candidates = cells_gdf.nsmallest(50, 'dist_to_center')
foothills_cell = center_candidates.nlargest(1, 'population').iloc[0]

# 2. North: high y, medium distance from center
north_candidates = cells_gdf[
    (cells_gdf['centroid_y'] > center_y + 2000) &
    (cells_gdf['dist_to_center'] < cells_gdf['dist_to_center'].quantile(0.7))
]
if len(north_candidates) > 0:
    mountain_cell = north_candidates.nlargest(1, 'population').iloc[0]
else:
    mountain_cell = cells_gdf.nlargest(1, 'centroid_y').iloc[0]

# 3. South: low y, medium distance from center
south_candidates = cells_gdf[
    (cells_gdf['centroid_y'] < center_y - 2000) &
    (cells_gdf['dist_to_center'] < cells_gdf['dist_to_center'].quantile(0.7))
]
if len(south_candidates) > 0:
    urban_center_cell = south_candidates.nlargest(1, 'population').iloc[0]
else:
    urban_center_cell = cells_gdf.nsmallest(1, 'centroid_y').iloc[0]

# 4. Periphery: far from center, but still populated
periphery_candidates = cells_gdf[
    cells_gdf['dist_to_center'] > cells_gdf['dist_to_center'].quantile(0.75)
]
if len(periphery_candidates) > 0:
    coastal_periphery_cell = periphery_candidates.nlargest(1, 'population').iloc[0]
else:
    coastal_periphery_cell = cells_gdf.nlargest(1, 'dist_to_center').iloc[0]

# 5. Castelldefels direction: south-west coastal area
castelldefels_candidates = cells_gdf[
    (cells_gdf['centroid_y'] < center_y - 1000) &  # South
    (cells_gdf['centroid_x'] < center_x - 1000) &  # West
    (cells_gdf['dist_to_center'] > cells_gdf['dist_to_center'].quantile(0.75))  # Medium-far from center
]
if len(castelldefels_candidates) > 0:
    castelldefels_cell = castelldefels_candidates.nlargest(1, 'population').iloc[0]
else:
    # Fallback: just southwest
    sw_candidates = cells_gdf[
        (cells_gdf['centroid_y'] < center_y) &
        (cells_gdf['centroid_x'] < center_x)
    ]
    if len(sw_candidates) > 0:
        castelldefels_cell = sw_candidates.nlargest(1, 'dist_to_center').iloc[0]
    else:
        castelldefels_cell = cells_gdf.nsmallest(1, 'centroid_x').iloc[0]

test_cells = [
    ('urban_center', urban_center_cell),
    ('foothills', foothills_cell),
    ('mountain', mountain_cell),
    ('coastal_periphery', coastal_periphery_cell),
    ('castelldefels', castelldefels_cell)
]

print(f"\n✓ Selected {len(test_cells)} test cells:")
for name, cell in test_cells:
    print(f"\n  {name.upper()}:")
    print(f"    Cell ID: {cell['cell_id']}")
    print(f"    Position: ({cell['centroid_x']:.0f}, {cell['centroid_y']:.0f})")
    print(f"    Current population: {cell['population']:,.0f}")
    print(f"    Distance to center: {cell['dist_to_center']:,.0f} m")



print("\n" + "-" * 80)
print("Computing pairwise cell distances...")

n_cells = len(cells_gdf)
cell_centroids = np.array([[row.centroid_x, row.centroid_y] 
                           for _, row in cells_gdf.iterrows()])

# Compute distance matrix (Euclidean for gravity model)
from scipy.spatial.distance import cdist
distance_matrix = cdist(cell_centroids, cell_centroids, metric='euclidean')

print(f"✓ Distance matrix computed: {n_cells} × {n_cells}")
print(f"  Mean inter-cell distance: {distance_matrix[distance_matrix > 0].mean():,.0f} m")



print("\n" + "=" * 80)
print("COMPUTING GRAVITATIONAL WORK FOR EACH TEST CELL")
print("=" * 80)

results = []

for test_idx, (test_name, test_cell) in enumerate(test_cells):
    print(f"\n{'-' * 80}")
    print(f"TEST CELL {test_idx + 1}/{len(test_cells)}: {test_name.upper()}")
    print(f"{'-' * 80}")
    print(f"  Cell ID: {test_cell['cell_id']}")
    print(f"  Position: ({test_cell['centroid_x']:.0f}, {test_cell['centroid_y']:.0f})")
    print(f"  Baseline population: {test_cell['population']:,.0f}")
    print(f"  Adding: {NEW_RESIDENTS:,} new residents")
    
    # Create modified population array
    populations = cells_gdf['population'].values.copy()
    test_cell_idx = cells_gdf[cells_gdf['cell_id'] == test_cell['cell_id']].index[0]
    populations[test_cell_idx] += NEW_RESIDENTS
    
    print(f"  New population in test cell: {populations[test_cell_idx]:,.0f}")
    
    # Get distances from test cell to all other cells
    distances_from_test = distance_matrix[test_cell_idx, :]
    
    # Compute gravity model probabilities
    # P(destination j | origin i) = P_j * exp(-d_ij / d_0) / Σ_k(P_k * exp(-d_ik / d_0))
    attractions = populations ** ALPHA * np.exp(-distances_from_test / D_0)
    attractions[test_cell_idx] = 0  # No self-trips
    
    # Normalize to get probabilities
    total_attraction = attractions.sum()
    if total_attraction > 0:
        trip_probabilities = attractions / total_attraction
    else:
        trip_probabilities = np.zeros(n_cells)
    
    # Total trips generated from test cell (proportional to new population)
    # Assume each new resident makes 2 trips per day
    trips_per_person_per_day = 2.0
    total_trips = NEW_RESIDENTS * trips_per_person_per_day
    
    # Distribute trips according to gravity model
    trips_to_destinations = total_trips * trip_probabilities
    
    print(f"\n  Gravity model:")
    print(f"    Total trips from test cell: {total_trips:,.0f} trips/day")
    print(f"    Destinations with >1% of trips: {(trip_probabilities > 0.01).sum()}")
    
    # Top 5 destinations
    top_dest_idx = np.argsort(trips_to_destinations)[::-1][:5]
    print(f"\n  Top 5 destinations:")
    for rank, dest_idx in enumerate(top_dest_idx, 1):
        if trips_to_destinations[dest_idx] > 0:
            dest_cell = cells_gdf.iloc[dest_idx]
            print(f"    {rank}. Cell {dest_cell['cell_id']}: "
                  f"{trips_to_destinations[dest_idx]:,.0f} trips/day "
                  f"({trip_probabilities[dest_idx]*100:.1f}%), "
                  f"distance: {distances_from_test[dest_idx]:,.0f} m")
    
    # ========================================================================
    # COMPUTE ROUTING WORK USING DIJKSTRA (matching wheight.py)
    # ========================================================================
    
    print(f"\n  Computing routing-based gravitational work...")
    print(f"  (Using Dijkstra + precomputed edge work, matching wheight.py)")
    print(f"  (Including both outbound AND return trips)")
    print(f"  (Total cost = vertical_work + λ * horizontal_distance)")
    
    # Get nearest node to test cell
    test_node = test_cell['nearest_node']
    
    # Compute work for trips to significant destinations (>0.1% of trips)
    significant_mask = trip_probabilities > 0.001
    significant_destinations = np.where(significant_mask)[0]
    
    print(f"    Processing {len(significant_destinations)} significant destinations...")
    
    total_work = 0.0
    total_vertical_work = 0.0
    total_horizontal_distance = 0.0
    total_horizontal_cost = 0.0
    successful_routes = 0
    failed_routes = 0
    total_route_distance = 0.0
    weighted_distance = 0.0  # Distance weighted by trips
    
    for dest_idx in tqdm(significant_destinations, desc=f"  Routing for {test_name}"):
        dest_cell = cells_gdf.iloc[dest_idx]
        dest_node = dest_cell['nearest_node']
        n_trips = trips_to_destinations[dest_idx]
        
        if n_trips < 1:  # Skip very small trip counts
            continue
        
        try:
            # ================================================================
            # OUTBOUND TRIP: test_cell → destination
            # ================================================================
            path_outbound = nx.shortest_path(G, source=test_node, target=dest_node, weight='length')
            
            # Calculate outbound path components
            path_length_outbound = 0.0
            path_vertical_outbound = 0.0
            for i in range(len(path_outbound) - 1):
                u, v = path_outbound[i], path_outbound[i+1]
                edge_keys = G[u][v].keys() if hasattr(G[u][v], 'keys') else [0]
                key = list(edge_keys)[0]
                edge_length = G[u][v][key].get('length', 0)
                path_length_outbound += edge_length
                path_vertical_outbound += edge_work.get((u, v, key), 0.0)
            
            total_route_distance += path_length_outbound
            weighted_distance += path_length_outbound * n_trips
            
            # ================================================================
            # RETURN TRIP: destination → test_cell
            # ================================================================
            path_return = nx.shortest_path(G, source=dest_node, target=test_node, weight='length')
            
            # Calculate return path components
            path_length_return = 0.0
            path_vertical_return = 0.0
            for i in range(len(path_return) - 1):
                u, v = path_return[i], path_return[i+1]
                edge_keys = G[u][v].keys() if hasattr(G[u][v], 'keys') else [0]
                key = list(edge_keys)[0]
                edge_length = G[u][v][key].get('length', 0)
                path_length_return += edge_length
                path_vertical_return += edge_work.get((u, v, key), 0.0)
            
            # ================================================================
            # TOTAL WORK: outbound + return
            # ================================================================
            # Each trip includes both outbound and return journey.
            vertical_per_trip = path_vertical_outbound + path_vertical_return
            horizontal_distance_per_trip = path_length_outbound + path_length_return
            horizontal_cost_per_trip = HORIZONTAL_COST_WEIGHT * horizontal_distance_per_trip
            work_per_trip = vertical_per_trip + horizontal_cost_per_trip

            total_vertical_work += n_trips * vertical_per_trip
            total_horizontal_distance += n_trips * horizontal_distance_per_trip
            total_horizontal_cost += n_trips * horizontal_cost_per_trip

            work_ij = n_trips * work_per_trip
            total_work += work_ij
            
            successful_routes += 1
            
        except (nx.NetworkXNoPath, KeyError):
            # No path found (disconnected nodes)
            failed_routes += 1
            continue
    
    avg_route_distance = total_route_distance / successful_routes if successful_routes > 0 else 0
    avg_weighted_distance = weighted_distance / total_trips if total_trips > 0 else 0
    
    print(f"    ✓ Routing complete:")
    print(f"      Successful routes: {successful_routes}")
    print(f"      Failed routes: {failed_routes}")
    print(f"      Average route distance: {avg_route_distance:,.0f} m")
    print(f"      Weighted average distance: {avg_weighted_distance:,.0f} m (by trips)")
    print(f"      Vertical work: {total_vertical_work:,.0f} J")
    print(f"      Horizontal distance (weighted): {total_horizontal_distance:,.0f} m")
    print(f"      Horizontal cost λ·d: {total_horizontal_cost:,.0f} J")
    print(f"      Total gravitational work: {total_work:,.0f} J")
    print(f"      Work per new resident: {total_work / NEW_RESIDENTS:,.0f} J/person")
    print(f"      Work per trip: {total_work / total_trips:,.0f} J/trip")
    
    # Store results
    results.append({
        'test_cell_name': test_name,
        'cell_id': test_cell['cell_id'],
        'centroid_x': test_cell['centroid_x'],
        'centroid_y': test_cell['centroid_y'],
        'baseline_population': test_cell['population'],
        'new_population': populations[test_cell_idx],
        'new_residents_added': NEW_RESIDENTS,
        'dist_to_center': test_cell['dist_to_center'],
        'total_trips': total_trips,
        'successful_routes': successful_routes,
        'failed_routes': failed_routes,
        'avg_route_distance_m': avg_route_distance,
        'avg_weighted_distance_m': avg_weighted_distance,
        'vertical_work_joules': total_vertical_work,
        'horizontal_distance_weighted_m': total_horizontal_distance,
        'horizontal_cost_joules': total_horizontal_cost,
        'total_work_joules': total_work,
        'work_per_resident': total_work / NEW_RESIDENTS,
        'work_per_trip': total_work / total_trips if total_trips > 0 else 0
    })



print("\n" + "=" * 80)
print("COMPARATIVE RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_work_joules')

print(f"\nRanking by total gravitational work (lower is better):\n")
for rank, (idx, row) in enumerate(results_df.iterrows(), 1):
    print(f"{rank}. {row['test_cell_name'].upper()}")
    print(f"   Cell ID: {row['cell_id']}")
    print(f"   Distance to center: {row['dist_to_center']:,.0f} m")
    print(f"   Avg route distance: {row['avg_route_distance_m']:,.0f} m")
    print(f"   Weighted avg distance: {row['avg_weighted_distance_m']:,.0f} m")
    print(f"   Total work: {row['total_work_joules']:,.0f} J")
    print(f"   Work per resident: {row['work_per_resident']:,.0f} J/person")
    print(f"   Work per trip: {row['work_per_trip']:,.0f} J/trip")
    print()

# Compute relative differences
best_work = results_df['total_work_joules'].min()
results_df['work_increase_vs_best'] = results_df['total_work_joules'] - best_work
results_df['work_increase_pct'] = (results_df['total_work_joules'] / best_work - 1) * 100

print("Energy efficiency comparison:")
print(f"  Best option: {results_df.iloc[0]['test_cell_name'].upper()}")
print(f"  Work: {results_df.iloc[0]['total_work_joules']:,.0f} J\n")

for idx, row in results_df.iloc[1:].iterrows():
    print(f"  {row['test_cell_name'].upper()}: "
          f"+{row['work_increase_pct']:.1f}% more work "
          f"(+{row['work_increase_vs_best']:,.0f} J)")



print("\n" + "-" * 80)
print("Saving results...")

# Save detailed results to CSV
results_csv = OUTPUT_DIR / f"{CITY}_comparative_work_results.csv"
results_df.to_csv(results_csv, index=False, sep=';')
print(f"✓ Results saved to: {results_csv}")

# Save test cells as GeoPackage
test_cells_gdf = cells_gdf[cells_gdf['cell_id'].isin(results_df['cell_id'])].copy()
test_cells_gdf = test_cells_gdf.merge(
    results_df[['cell_id', 'test_cell_name', 'total_work_joules', 
                'work_per_resident', 'work_increase_pct']],
    on='cell_id',
    how='left'
)

# Drop centroid column (it's a geometry column, keep only main geometry)
if 'centroid' in test_cells_gdf.columns:
    test_cells_gdf = test_cells_gdf.drop(columns=['centroid'])

test_cells_gpkg = OUTPUT_DIR / f"{CITY}_test_cells.gpkg"
test_cells_gdf.to_file(test_cells_gpkg, driver='GPKG')
print(f"✓ Test cells saved to: {test_cells_gpkg}")



print("\nGenerating visualizations...")

# Convert to WGS84 for mapping
cells_ll = cells_gdf.to_crs("EPSG:4326")
test_cells_ll = test_cells_gdf.to_crs("EPSG:4326")
nodes_ll = nodes_gdf.to_crs("EPSG:4326")

# Define colors for each test cell (ranked by work)
color_map = {
    results_df.iloc[0]['test_cell_name']: "#d7191c",  # Best: Red
    results_df.iloc[1]['test_cell_name']: '#fdae61',  # 2nd: Green
    results_df.iloc[2]['test_cell_name']: '#ffffbf',  # 3rd: Yellow
    results_df.iloc[3]['test_cell_name']: '#abdda4',  # 4th: Orange
    results_df.iloc[4]['test_cell_name']: '#2b83ba',  # 5th/Worst: Red
}

# Figure 1: Main map with highlighted test cells
fig, ax = plt.subplots(1, 1, figsize=(18, 16))

# Background: all cells in light gray
cells_ll.plot(ax=ax, facecolor='#e0e0e0', edgecolor='none', alpha=0.3)

# Network nodes
# nodes_ll.plot(ax=ax, color='#333333', markersize=0.8, alpha=0.4, label='Road network', zorder=10)

# Highlight test cells with colors
for idx, row in test_cells_ll.iterrows():
    color = color_map[row['test_cell_name']]
    centroid = row.geometry.centroid  # Calculate centroid on the fly
    
    # Plot cell
    gpd.GeoSeries([row['geometry']], crs=test_cells_ll.crs).plot(
        ax=ax, facecolor=color, edgecolor='black', alpha=0.6, linewidth=2, zorder=100
    )
    
    # Add marker at centroid
    ax.scatter(
        centroid.x, centroid.y,
        c=color, s=4000, alpha=0.8,
        edgecolors='black', linewidth=3,
        zorder=110
    )
    
    # Add label
    # label_text = f"{row['test_cell_name'].upper()}\n{row['work_per_resident']:,.0f} J/res"
    # ax.text(
    #     centroid.x, centroid.y,
    #     label_text,
    #     ha='center', va='center',
    #     fontsize=11, fontweight='bold',
    #     color='black',
    #     zorder=111
    # )

# Add basemap
try:
    ctx.add_basemap(ax, crs=cells_ll.crs, source=ctx.providers.CartoDB.Positron, 
                    zoom=12, alpha=1, attribution=False)
except:
    print("⚠ Could not add basemap")

# Add legend with work values
from matplotlib.patches import Patch

def work_value_for_chart(work_joules: float) -> float:
    if WORK_DISPLAY_MODE == "SCI":
        return work_joules
    return work_joules / 1e9  # GJ

def work_label_for_text(work_joules: float) -> str:
    if WORK_DISPLAY_MODE == "SCI":
        return f"{work_joules:.2e} J"
    return f"{work_joules / 1e9:.2f} GJ"

def work_per_resident_label(work_per_resident_j: float) -> str:
    if WORK_DISPLAY_MODE == "SCI":
        return f"{work_per_resident_j:.2e} J/res"
    return f"{work_per_resident_j / 1e6:.1f} MJ/res"

legend_elements = []
for idx, row in results_df.iterrows():
    color = color_map[row['test_cell_name']]
    label = f"{row['test_cell_name'].upper()}: {work_label_for_text(row['total_work_joules'])} ({work_per_resident_label(row['work_per_resident'])})"
    legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label, linewidth=2))

# ax.legend(handles=legend_elements, loc='lower right', fontsize=16, 
#           frameon=True, fancybox=True, shadow=True, framealpha=0.9)

# ax.set_title(
#     f'Comparative Settlement Analysis - {CITY.upper()}\n'
#     f'Gravitational Work for {NEW_RESIDENTS:,} New Residents\n'
#     f'(Green = Most Efficient, Red = Least Efficient)',
#     fontsize=18, fontweight='bold', pad=20
# )
# ax.set_xlabel('Longitude', fontsize=20, fontweight='normal')
# ax.set_ylabel('Latitude', fontsize=20, fontweight='normal')
ax.tick_params(axis='both', which='major', labelsize=26)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{CITY}_comparative_map.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / f'{CITY}_comparative_map.pdf', dpi=300, bbox_inches='tight')
print(f"✓ Map saved: {CITY}_comparative_map.png/pdf")

# Figure 2: Bar chart comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

colors_ranked = [color_map[name] for name in results_df['test_cell_name']]

bars = ax.bar(
    range(len(results_df)),
    results_df['total_work_joules'].apply(work_value_for_chart),
    color=colors_ranked,
    edgecolor='black',
    linewidth=2,
    alpha=0.8
)

# Add value labels on bars
for i, (idx, row) in enumerate(results_df.iterrows()):
    height = work_value_for_chart(row['total_work_joules'])
    ax.text(
        i, height + height * 0.02,
        f"{work_label_for_text(row['total_work_joules'])}\n({work_per_resident_label(row['work_per_resident'])})",
        ha='center', va='bottom',
        fontsize=24, fontweight='normal'
    )

# Custom labels for xticks (edit as needed)
xtick_label_map = {
    'urban_center': 'Urban Core',
    'foothills': 'Foothills',
    'mountain': 'Terrassa',
    'coastal_periphery': 'Matarò',
    'castelldefels': 'Castelldefels'
}

ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(
    [xtick_label_map.get(name, name.upper()) for name in results_df['test_cell_name']],
    fontsize=24, fontweight='normal'
)
if WORK_DISPLAY_MODE == "SCI":
    ax.set_ylabel(r'$W_{TOT}$' " (J)", fontsize=24, fontweight='normal')
else:
    ax.set_ylabel(r'$W_{TOT}$' " (GJ)", fontsize=24, fontweight='normal')
ax.tick_params(axis='y', which='major', labelsize=20)
# ax.set_title(
#     f'Energy Efficiency Comparison - {CITY.upper()}\n'
#     f'Total Work for {NEW_RESIDENTS:,} New Residents',
#     fontsize=24, fontweight='normal', pad=20
# )
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{CITY}_work_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / f'{CITY}_work_comparison.pdf', dpi=300, bbox_inches='tight')
print(f"✓ Chart saved: {CITY}_work_comparison.png/pdf")



print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nCity: {CITY.upper()}")
print(f"New residents per test: {NEW_RESIDENTS:,}")
print(f"Test cells analyzed: {len(results_df)}")

print(f"\n🏆 MOST ENERGY-EFFICIENT LOCATION:")
best_result = results_df.iloc[0]
print(f"   {best_result['test_cell_name'].upper()}")
print(f"   Cell ID: {best_result['cell_id']}")
print(f"   Total work: {best_result['total_work_joules']:,.0f} J ({best_result['total_work_joules']/1e6:.2f} MJ)")
print(f"   Work per resident: {best_result['work_per_resident']:,.0f} J")
print(f"   Distance to center: {best_result['dist_to_center']:,.0f} m")

print(f"\n📊 OUTPUTS:")
print(f"   • {results_csv.name}")
print(f"   • {test_cells_gpkg.name}")
print(f"   • {CITY}_comparative_map.png/pdf")
print(f"   • {CITY}_work_comparison.png/pdf")

print("\n" + "=" * 80)
print("All files saved to:")
print(f"{OUTPUT_DIR}")
print("=" * 80)
