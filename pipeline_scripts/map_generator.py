#!/usr/bin/env python3
"""HTML map generator for DEM extractor results."""


import os
import argparse
import pickle
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium import FeatureGroup
from utils import logger
import time

def get_polygon_coordinates(geom):
    """Extract coordinates from Polygon or MultiPolygon geometries"""
    coords = []
    if geom.geom_type == 'Polygon':
        coords.append([(lat, lon) for lon, lat in geom.exterior.coords])
    elif geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:
            coords.append([(lat, lon) for lon, lat in polygon.exterior.coords])
    return coords

def load_metropolis_bbox(json_path, city_key):
    """Load city bounding box from JSON file"""
    with open(json_path) as f:
        data = json.load(f)
    if city_key not in data:
        raise KeyError(f"City '{city_key}' not found in {json_path}")
    
    raw_coords = data[city_key]           
    swapped_coords = [(lon, lat) for lat, lon in raw_coords]
    
    from shapely.geometry import Polygon
    return Polygon(swapped_coords)

def generate_city_bbox_maps(city_folder, base_path):
    """Generate city bounding box maps (original and land-clipped)"""
    logger.info(f"[{city_folder}] Generating city bbox maps")
    
    folder_path = os.path.join(base_path, city_folder)
    graphs_dir = os.path.join(folder_path, 'graphs')
    
    # Load city data
    csv_file = os.path.join(folder_path, 'data_useful.csv')
    if not os.path.exists(csv_file):
        logger.error(f"[{city_folder}] No data_useful.csv found")
        return
    
    df = pd.read_csv(csv_file, sep=';').dropna(subset=['geometry'])
    df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    minx, miny, maxx, maxy = gdf.total_bounds
    center_y = (miny + maxy) / 2
    center_x = (minx + maxx) / 2
    
    # Load city polygon
    json_bbox = "/data/workspaces/fbellisardi/metropolis.json"
    try:
        city_poly = load_metropolis_bbox(json_bbox, city_folder)
    except KeyError:
        logger.error(f"[{city_folder}] City not found in metropolis.json")
        return
    
    # Generate original city bbox map
    folium_map = folium.Map(location=[center_y, center_x], zoom_start=13)
    
    for coord_list in get_polygon_coordinates(city_poly):
        folium.Polygon(
            locations=coord_list,
            color='blue', fill=True, fill_opacity=0.1
        ).add_to(folium_map)
    
    city_map_file = os.path.join(graphs_dir, f'{city_folder}_city_bbox.html')
    folium_map.save(city_map_file)
    logger.info(f"[{city_folder}] Saved city bbox map to {city_map_file}")
    
    # Generate land-clipped bbox map if land shapefile exists
    land_shp = os.path.join(folder_path, 'land', f'{city_folder}_clipped_land.shp')
    if os.path.exists(land_shp):
        land_gdf = gpd.read_file(land_shp).to_crs('EPSG:4326')
        polygon = land_gdf.geometry.union_all()
        
        folium_map2 = folium.Map(location=[center_y, center_x], zoom_start=13)
        
        for coord_list in get_polygon_coordinates(polygon):
            folium.Polygon(
                locations=coord_list,
                color='blue', fill=True, fill_opacity=0.1
            ).add_to(folium_map2)
        
        city_map_file = os.path.join(graphs_dir, f'{city_folder}_city_bbox_landed.html')
        folium_map2.save(city_map_file)
        logger.info(f"[{city_folder}] Saved land-clipped city bbox map to {city_map_file}")
    else:
        logger.warning(f"[{city_folder}] No land shapefile found, skipping land-clipped map")

def generate_variant_maps(city_folder, base_path):
    """Generate individual maps for each graph variant"""
    logger.info(f"[{city_folder}] Generating variant maps")
    
    folder_path = os.path.join(base_path, city_folder)
    graphs_dir = os.path.join(folder_path, 'graphs')
    stats_file = os.path.join(graphs_dir, 'graph_stats.csv')
    
    if not os.path.exists(stats_file):
        logger.error(f"[{city_folder}] No graph_stats.csv found")
        return
    
    # Load variant statistics
    stats_df = pd.read_csv(stats_file)
    
    # Load city data for center coordinates
    csv_file = os.path.join(folder_path, 'data_useful.csv')
    df = pd.read_csv(csv_file, sep=';').dropna(subset=['geometry'])
    df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    minx, miny, maxx, maxy = gdf.total_bounds
    center_y = (miny + maxy) / 2
    center_x = (minx + maxx) / 2
    
    # Set up colormap
    cmap = plt.get_cmap('Set1', len(stats_df))
    
    # Generate map for each variant
    for idx, row in stats_df.iterrows():
        variant = row['variant']
        pkl_path = os.path.join(graphs_dir, f'graph_{variant}.pkl')
        
        if not os.path.exists(pkl_path):
            logger.warning(f"[{city_folder}] Pickle file not found for variant {variant}")
            continue
        
        # Load graph
        try:
            with open(pkl_path, 'rb') as f:
                G_var = pickle.load(f)
        except Exception as e:
            logger.error(f"[{city_folder}] Could not load graph for variant {variant}: {e}")
            continue
        
        # Create map
        variant_map = folium.Map(location=[center_y, center_x], zoom_start=13)
        color = mcolors.to_hex(cmap(idx))
        
        # Add edges to map
        edge_count = 0
        for u, v in G_var.edges():
            try:
                y1, x1 = G_var.nodes[u]['y'], G_var.nodes[u]['x']
                y2, x2 = G_var.nodes[v]['y'], G_var.nodes[v]['x']
                folium.PolyLine([(y1, x1), (y2, x2)], color=color, weight=2).add_to(variant_map)
                edge_count += 1
            except KeyError as e:
                logger.warning(f"[{city_folder}] Missing node coordinates in variant {variant}: {e}")
                continue
        
        # Save map
        variant_map_file = os.path.join(graphs_dir, f'graph_{variant}_map.html')
        variant_map.save(variant_map_file)
        logger.info(f"[{city_folder}] Saved variant map for {variant} ({edge_count} edges)")

def generate_overview_map(city_folder, base_path):
    """Generate overview map with all variants"""
    logger.info(f"[{city_folder}] Generating overview map with all variants")
    
    folder_path = os.path.join(base_path, city_folder)
    graphs_dir = os.path.join(folder_path, 'graphs')
    stats_file = os.path.join(graphs_dir, 'graph_stats.csv')
    
    if not os.path.exists(stats_file):
        logger.error(f"[{city_folder}] No graph_stats.csv found")
        return
    
    # Load variant statistics
    stats_df = pd.read_csv(stats_file)
    
    # Load city data for center coordinates
    csv_file = os.path.join(folder_path, 'data_useful.csv')
    df = pd.read_csv(csv_file, sep=';').dropna(subset=['geometry'])
    df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    minx, miny, maxx, maxy = gdf.total_bounds
    center_y = (miny + maxy) / 2
    center_x = (minx + maxx) / 2
    
    # Create overview map
    overview_map = folium.Map(location=[center_y, center_x], zoom_start=12)
    
    # Set up colormap
    cmap = plt.get_cmap('Set1', len(stats_df))
    
    # Add each variant as a separate layer
    for idx, row in stats_df.iterrows():
        variant = row['variant']
        pkl_path = os.path.join(graphs_dir, f'graph_{variant}.pkl')
        
        if not os.path.exists(pkl_path):
            continue
        
        try:
            with open(pkl_path, 'rb') as f:
                G_var = pickle.load(f)
        except Exception as e:
            logger.warning(f"[{city_folder}] Could not load graph for variant {variant}: {e}")
            continue
        
        # Create feature group for this variant
        fg = FeatureGroup(name=f'{variant} ({G_var.number_of_nodes()} nodes)')
        color = mcolors.to_hex(cmap(idx))
        
        # Add edges to feature group (sample for performance)
        edges = list(G_var.edges())
        sample_size = min(len(edges), 1000)  # Limit edges for performance
        import random
        sampled_edges = random.sample(edges, sample_size) if len(edges) > sample_size else edges
        
        for u, v in sampled_edges:
            try:
                y1, x1 = G_var.nodes[u]['y'], G_var.nodes[u]['x']
                y2, x2 = G_var.nodes[v]['y'], G_var.nodes[v]['x']
                folium.PolyLine([(y1, x1), (y2, x2)], color=color, weight=1, opacity=0.7).add_to(fg)
            except KeyError:
                continue
        
        fg.add_to(overview_map)
    
    # Add layer control
    folium.LayerControl().add_to(overview_map)
    
    # Save overview map
    overview_map_file = os.path.join(graphs_dir, f'{city_folder}_variants_overview.html')
    overview_map.save(overview_map_file)
    logger.info(f"[{city_folder}] Saved variants overview map to {overview_map_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate HTML maps for DEM extractor results')
    parser.add_argument('--city', type=str, required=True,
                        help='City folder to process')
    parser.add_argument('--base-path', type=str, 
                        default='/data/workspaces/fbellisardi/data_processed',
                        help='Base path containing city folders')
    parser.add_argument('--bbox-only', action='store_true',
                        help='Generate only city bounding box maps')
    parser.add_argument('--variants-only', action='store_true',
                        help='Generate only variant maps')
    parser.add_argument('--overview-only', action='store_true',
                        help='Generate only overview map')
    parser.add_argument('--all', action='store_true',
                        help='Generate all types of maps (default)')
    
    args = parser.parse_args()
    
    # Default to all maps if no specific type is requested
    if not (args.bbox_only or args.variants_only or args.overview_only):
        args.all = True
    
    city_folder = args.city
    base_path = args.base_path
    
    # Check if city folder exists
    folder_path = os.path.join(base_path, city_folder)
    if not os.path.isdir(folder_path):
        logger.error(f"City folder not found: {folder_path}")
        return
    
    # Create graphs directory if it doesn't exist
    graphs_dir = os.path.join(folder_path, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    start_time = time.time()
    
    try:
        if args.bbox_only or args.all:
            generate_city_bbox_maps(city_folder, base_path)
        
        if args.variants_only or args.all:
            generate_variant_maps(city_folder, base_path)
        
        if args.overview_only or args.all:
            generate_overview_map(city_folder, base_path)
        
        # Update maps index
        try:
            from tools.build_maps_index import process_city as _build_maps_index
            _build_maps_index(base_path, city_folder, iframes=False)
            logger.info(f"[{city_folder}] Updated maps index in graphs directory")
        except Exception as e:
            logger.warning(f"[{city_folder}] Could not update maps index: {e}")
        
        end_time = time.time()
        logger.info(f"[{city_folder}] Map generation completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"[{city_folder}] Map generation failed: {e}")
        raise

if __name__ == '__main__':
    main()