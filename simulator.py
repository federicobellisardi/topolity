#!/usr/bin/env python
"""
author: Federico Bellisardi
"""

import os
import argparse
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely import wkt
from shapely.geometry import Point, box
from tqdm import tqdm
from collections import defaultdict
import rasterio
import requests
from rasterio.mask import mask

from utils import logger, read_conf, haversine
from data_processing import process_dataset, DEMReader
from od_generator import main as od_main
from work import Work, WorkPlot



class MovementSimulator:
    def __init__(self, OD_matrix, dem_gdf, G, radius=50):
        self.OD_matrix = OD_matrix
        self.dem_gdf = dem_gdf
        self.G = G
        self.radius = radius    
        self.OD_matrix.index = self.OD_matrix.index.astype(int)
        self.OD_matrix.columns = self.OD_matrix.columns.astype(int)


    def compute_edge_costs_from_path_cost_matrix(self):
        edge_weights = defaultdict(float)

        for origin in self.cost_matrix.index:
            for dest in self.cost_matrix.columns:
                if origin == dest:
                    continue
                try:
                    path = nx.shortest_path(self.G, source=origin, target=dest, weight='length')
                    total_cost = self.cost_matrix.loc[origin, dest]
                    if len(path) < 2 or total_cost == 0:
                        continue
                    per_edge_cost = total_cost / (len(path) - 1)
                    for u, v in zip(path[:-1], path[1:]):
                        edge_weights[(u, v)] += per_edge_cost
                except nx.NetworkXNoPath:
                    continue

        return edge_weights

    def simulate_movements(self, save_path):
        w1 = 1
        w2 = 0
        # edge_weights = self.compute_edge_costs_from_path_cost_matrix()

        # def weighted_cost(u, v, d):
        #     return edge_weights.get((u, v), 0)
            
        movements = []
        for origin_node, row in tqdm(self.OD_matrix.iterrows(), total=len(self.OD_matrix), desc="Simulating movements", unit="node"):
            for dest_node, _ in row.items():
                on = int(origin_node)
                dn = int(dest_node)
                if on == dn: 
                    continue
                try:
                    weight = lambda u, v, d: w1 * d.get('length', 0) #+ w2 * weighted_cost(u, v, d)
                    path_nodes = nx.shortest_path(self.G, source=on, target=dn, 
                                                    weight=weight, method='dijkstra'
                                                  )
                except nx.NetworkXNoPath:
                    path_nodes = [origin_node, dest_node]
                
                polyline = []
                for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                    u_data = self.G.nodes[u]
                    v_data = self.G.nodes[v]
                    polyline.extend([(u_data['y'], u_data['x']), (v_data['y'], v_data['x'])])
                
                movements.append({
                    'origin_node': origin_node,
                    'destination_node': dest_node,
                    'path': path_nodes,
                    'polyline': polyline
                })

        movements_df = pd.DataFrame(movements)
        logger.info(f"Simulated {len(movements_df)} movements.")

        def get_altitude_for_point(lat, lon, dem_gdf):
            pt = Point(lon, lat)
            delta = self.radius / 111000.0 
            bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
            candidate_indices = list(dem_gdf.sindex.intersection(bbox))
            within_candidates = []
            for idx in candidate_indices:
                row = dem_gdf.iloc[idx]
                d = haversine(lat, lon, row['lat'], row['lon'])
                if d <= self.radius:
                    within_candidates.append((d, row['alt']))
            if within_candidates:
                within_candidates.sort(key=lambda x: x[0])
                return float(within_candidates[0][1])
            else:
                nearest_idx = list(dem_gdf.sindex.nearest(pt, 1))[0]

                return float(dem_gdf.iloc[nearest_idx]['alt'].iloc[0])

        enriched_polylines = []
        for _, row in tqdm(movements_df.iterrows(), total=len(movements_df), desc="Enriching polylines with altitude"):
            poly_entry = row['polyline']
            enriched = []
            for lat, lon in poly_entry:
                alt = get_altitude_for_point(lat, lon, self.dem_gdf)
                enriched.append((lat, lon, alt))
            enriched_polylines.append(enriched)

        movements_df['polyline_with_alt'] = enriched_polylines

        return movements_df

def main():
    parser = argparse.ArgumentParser(
        description="Simulate movement using different models."
    )
    parser.add_argument("-c", "--conf", required=True, help="Path to configuration JSON file")
    args = parser.parse_args()


    conf = read_conf(args.conf)
    dp_conf = conf.get("data_processing", {})
    data_src = conf.get("data_processing", {}).get("twitter", {})
    if data_src:
        ds_conf = conf.get("data_processing", {}).get("data_source", {}).get("twitter", {})
    else:
        ds_conf = conf.get("data_processing", {}).get("data_source", {}).get("worldpop", {})  # population file settings now reside here

    dyn_conf = conf.get("dynamics", {})
    tag = dp_conf.get("tag", "default")
    radius = dyn_conf.get("search_radius", 50)

    translation = dyn_conf.get("translation_dem", False)
    api_key = dp_conf.get("altitude", {}).get("api_key",{})

    dem_foder = os.path.join(os.environ.get("WORKSPACE", "."), "topolity", "data", "dem")
    demfile = os.path.join(dem_foder, f"{tag}_dem.tif")

    save_folder = os.path.join(os.environ.get("WORKSPACE", "."), "topolity", 'output', f"tag_{tag}")    
    _, G = od_main()

    if not translation:
        logger.info("Original demographic used.")
        bbox_dic = dp_conf.get("bbox", {}).get(tag, {})

        if not bbox_dic:
            raise ValueError("Bounding box info missing in config.")
        bbox = (
            bbox_dic["min_lon"],
            bbox_dic["min_lat"],
            bbox_dic["max_lon"],
            bbox_dic["max_lat"]
        )
        bbox_geom = [box(*bbox)]
        
        with rasterio.open(demfile) as dem:
            if dem.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform_bounds
                bbox = transform_bounds("EPSG:4326", dem.crs, *bbox)
                bbox_geom = [box(*bbox)]
            dem_data, dem_transform = mask(dem, bbox_geom, crop=True)
            dem_data = dem_data[0]
        dem_file = demfile
    else:
        logger.info("Translated demographic used.")
        translation_direction = conf.get("translation_direction", "north")  # e.g., "north", "south", "east", "west"
        translation_offset = conf.get("translation_offset", 0.1)  # offset in degrees (adjust as needed)
        
        bbox_dic = dp_conf.get("bbox", {}).get(tag, {})
        if not bbox_dic:
            raise ValueError("Bounding box info missing in config.")
        min_lon = bbox_dic["min_lon"]
        min_lat = bbox_dic["min_lat"]
        max_lon = bbox_dic["max_lon"]
        max_lat = bbox_dic["max_lat"]
        
        if translation_direction.lower() == "north":
            new_bbox = (min_lon, min_lat + translation_offset, max_lon, max_lat + translation_offset)
        elif translation_direction.lower() == "south":
            new_bbox = (min_lon, min_lat - translation_offset, max_lon, max_lat - translation_offset)
        elif translation_direction.lower() == "east":
            new_bbox = (min_lon + translation_offset, min_lat, max_lon + translation_offset, max_lat)
        elif translation_direction.lower() == "west":
            new_bbox = (min_lon - translation_offset, min_lat, max_lon - translation_offset, max_lat)
        else:
            raise ValueError("Invalid translation direction specified in config.")
        
        new_min_lon, new_min_lat, new_max_lon, new_max_lat = new_bbox
        width = new_max_lon - new_min_lon
        height = new_max_lat - new_min_lat
        expanded_min_lon = new_min_lon - width
        expanded_min_lat = new_min_lat - height
        expanded_max_lon = new_max_lon + width
        expanded_max_lat = new_max_lat + height
        
        if not api_key:
            raise ValueError("API key for DEM download missing in config.")
        
        url = (
            f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&"
            f"south={expanded_min_lat}&north={expanded_max_lat}&west={expanded_min_lon}&east={expanded_max_lon}&"
            f"outputFormat=GTiff&API_Key={api_key}"
        )
        logger.info("Downloading translated DEM from: " + url)
        response = requests.get(url)
        if response.status_code == 200:
            translated_dem_file = f"{dem_foder}/{tag}_translated_dem.tif"
            with open(translated_dem_file, "wb") as f:
                f.write(response.content)
            logger.info("Translated DEM downloaded and saved as " + translated_dem_file)
        else:
            raise Exception("Failed to download translated DEM: " + str(response.json()))
        
        bbox_geom = [box(*new_bbox)]
        with rasterio.open(translated_dem_file) as dem:
            if dem.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform_bounds
                new_bbox_transformed = transform_bounds("EPSG:4326", dem.crs, *new_bbox)
                bbox_geom = [box(*new_bbox_transformed)]
            dem_data, dem_transform = mask(dem, bbox_geom, crop=True)
            dem_data = dem_data[0]
        
        dem_file = translated_dem_file
    
    dem_reader = DEMReader(dem_file, search_radius=50)
    dem_gdf = dem_reader.get_pixel_centroids()
    OD_folder = os.path.join(os.environ.get("WORKSPACE", "."), "topolity", 'output', f"tag_{tag}", 'OD_matrices')

    OD_files = [f for f in os.listdir(OD_folder) if f.endswith('.csv')]
    OD_matrices = {}
    for file in OD_files:
        file_path = os.path.join(OD_folder, file)
        df = pd.read_csv(file_path, index_col=[0], header=[0], sep=';')
        OD_matrices[file] = df

    for filename, df in OD_matrices.items():
        model_type = filename.split('_')[1]
        simulator = MovementSimulator(df, dem_gdf, G, radius=radius)
        save_path = os.path.join(save_folder, "movements")
        if not translation:
            save_path = os.path.join(save_path, f"original")
        else:
            save_path = os.path.join(save_path, f"translated")
        os.makedirs(save_path, exist_ok=True)

        save_path = os.path.join(save_path, f"{model_type}")
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        file_save = os.path.join(save_path, f"{tag}_{model_type}_movements.csv")

        movements_df = simulator.simulate_movements(file_save)

        movements_df.to_csv(file_save, index=False, sep=';')
        logger.info(f"Simulated movements saved to {file_save}")

        od_matrix = df
        work_obj = Work(
            movements=movements_df,
            od_matrix=od_matrix,
            G=G,
            conf=dp_conf,
            dem=dem_file             
        )
        edge_work_df = work_obj.compute_edge_work(od_matrix, movements_df)
        plot_obj = WorkPlot(
            df=edge_work_df,
            G=G,
            conf=dp_conf,
            dem=dem_file, 
            output=f"{save_path}/{tag}_{model_type}_work_heatmap.png"
        )
        plot_obj.heatmap(edge_work_df, tag)


if __name__ == "__main__":
    main()