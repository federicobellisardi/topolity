#!/usr/bin/env python
"""
author: Federico Bellisardi
"""

import os
import random
import argparse
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely import wkt
from shapely.geometry import Point

from utils import logger, read_conf
from data_processing import process_dataset
from tools.python.od_matrix_generator import Model as odmg
from tools.python.od_matrix_generator import CostModel as codmg

class GraphBuilder:
    def __init__(self, data_folder, pop_file, dyn_conf):
        self.data_folder = data_folder
        self.pop_file = pop_file
        self.dyn_conf = dyn_conf
        self.population_gdf = None
        self.agent_points = None
        self.graph = None
        self.grid = None
        self.agents_with_cell = None

    @staticmethod
    def compute_population_metrics(gdf):
        gdf_proj = gdf.to_crs(epsg=3395)
        gdf_proj['centroid'] = gdf_proj['geometry'].centroid
        gdf['centroid'] = gdf_proj['centroid'].to_crs(epsg=4326)
        gdf_proj['area'] = gdf_proj['geometry'].area
        gdf_proj['weight'] = gdf_proj['population'] / gdf_proj['area']
        min_weight = gdf_proj['weight'].min()
        max_weight = gdf_proj['weight'].max()
        gdf_proj['norm_weight'] = 100 * (gdf_proj['weight'] - min_weight) / (max_weight - min_weight)
        gdf_proj['n_points'] = gdf_proj['norm_weight'].round().astype(int)
        logger.info("Population metrics computed.")
        return gdf_proj

    @staticmethod
    def generate_agent_points(gdf_proj):
        def random_point_in_polygon(polygon):
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(p):
                    return p
        points = []
        for idx, row in gdf_proj.iterrows():
            cod = row['cod']
            polygon = row['geometry']
            n_points = int(row['n_points'])
            for _ in range(n_points):
                pt = random_point_in_polygon(polygon)
                points.append({'cod': cod, 'geometry': pt})
        points_gdf = gpd.GeoDataFrame(points, geometry='geometry', crs=gdf_proj.crs)
        points_gdf = points_gdf.to_crs(epsg=4326)
        logger.info(f"{len(points_gdf)} Agent points generated.")
        return points_gdf

    def build_graph(self):
        if self.population_gdf is None:
            logger.error("Population data is not loaded.")
            return None
        minx, miny, maxx, maxy = self.population_gdf.total_bounds
        padding = self.dyn_conf.get("osm_padding", 0.01)
        bbox = (minx + padding, miny - padding, maxx + padding, maxy - padding)
        network_type = self.dyn_conf.get("network_type", "drive")
        simplify = self.dyn_conf.get("simplify", True)
        logger.info("Downloading OSMnx graph...")
        self.graph = ox.graph_from_bbox(bbox, network_type=network_type, simplify=simplify)
        logger.info(f"Graph downloaded with {len(self.graph.nodes)} nodes.")
        return self.graph

    @staticmethod
    def snap_agents_to_graph(points_gdf, G):
        def get_nearest_node(row):
            return ox.distance.nearest_nodes(G, X=row.geometry.x, Y=row.geometry.y)
        points_gdf['nearest_node'] = points_gdf.apply(get_nearest_node, axis=1)
        points_gdf['nearest_node_coords'] = points_gdf['nearest_node'].apply(
            lambda node: (G.nodes[node]['y'], G.nodes[node]['x'])
        )
        points_gdf['node_pnt'] = points_gdf['nearest_node_coords'].apply(
        lambda coords: Point(coords[1], coords[0])
        )

        logger.info("Agent points snapped to graph nodes.")
        return points_gdf

    @staticmethod
    def node_mass(points_gdf):
        node_masses = points_gdf.groupby('nearest_node').agg(
            node_coordinate=('node_pnt', 'first'),
            m=('nearest_node', 'size')
        ).reset_index().rename(columns={'nearest_node': 'node_id'})
        return node_masses

def main():
    parser = argparse.ArgumentParser(
        description="Generate OD from population using different models."
    )
    parser.add_argument("-c", "--conf", required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    test = True

    conf = read_conf(args.conf)
    data_src = conf.get("data_processing", {}).get("twitter", {})
    if data_src:
        logger.info("Using Twitter data source.")
        dp_conf = conf.get("data_processing", {}).get("data_source", {}).get("twitter", {})
    else:
        logger.info("Using Worldpop data source.")
        dp_conf = conf.get("data_processing", {}).get("data_source", {}).get("worldpop", {})  # population file settings now reside here

    dyn_conf = conf.get("dynamics", {})
    tag = dp_conf.get("tag", "default")
    radius = dyn_conf.get("search_radius", 50)

    data_folder = os.path.join(os.environ.get("WORKSPACE", "."), "topolity", "data")
    pop_file = dp_conf.get("input_file", {})
    save_folder = os.path.join(os.environ.get("WORKSPACE", "."), "topolity", 'output', f"tag_{tag}")
    os.makedirs(save_folder, exist_ok=True)
    pop_df, dem_data  = process_dataset(conf)

    pop_df['geometry'] = pop_df['geometry'].apply(wkt.loads)
    pop_gdf = gpd.GeoDataFrame(pop_df, geometry='geometry', crs="EPSG:4326")

    if test:
        pop_gdf = pop_gdf.sample(150)

    gb = GraphBuilder(data_folder, pop_gdf, dyn_conf)
    gb.population_gdf = pop_gdf
    pop_metrics = gb.compute_population_metrics(pop_gdf)
    points_gdf = gb.generate_agent_points(pop_metrics)

    # G = gb.build_graph()
    graph_folder = f"{save_folder}/graph"
    os.makedirs(graph_folder, exist_ok=True)
    graph_file = os.path.join(graph_folder, f"{tag}_graph.gpickle")
    if not os.path.exists(graph_file):
        G = gb.build_graph()
        ox.save_graphml(G, graph_file)
        logger.info(f"Graph saved in {graph_file}.")
    else:
        G = ox.load_graphml(graph_file)
        logger.info(f"Graph loaded from {graph_file}.")

    points_gdf = GraphBuilder.snap_agents_to_graph(points_gdf, G)
    node_masses = GraphBuilder.node_mass(points_gdf)
    
    OD_folder = os.path.join(save_folder, "OD_matrices")
    os.makedirs(OD_folder, exist_ok=True)

    model = odmg(node_masses, G, pop_col='m')

    gravity_od = model.compute_gravity_od(epsilon=1e-6)
    g_path = os.path.join(OD_folder, f"{tag}_gravity_od_matrix.csv")
    gravity_od.to_csv(g_path, sep=";")
    logger.info(f"Gravity OD matrix computxed and saved in {g_path}.")

    radiation_od = model.compute_radiation_od()
    r_path = os.path.join(OD_folder, f"{tag}_radiation_od_matrix.csv")
    radiation_od.to_csv(r_path, sep=";")
    logger.info(f"Radiation OD matrix computed and saved in {r_path}.")

    logit_od = model.compute_logit_od(beta=-0.01)
    l_path = os.path.join(OD_folder, f"{tag}_logit_od_matrix.csv")
    logit_od.to_csv(l_path, sep=";")
    logger.info(f"Logit OD matrix computed and saved in {l_path}.")

    cost_folder = os.path.join(save_folder, "cost_matrices")
    os.makedirs(cost_folder, exist_ok=True)

    cost_model = codmg(node_masses, G, pop_col='m')

    gravity_cost = cost_model.compute_gravity_cost(epsilon=1e-6)
    g_path_cost = os.path.join(cost_folder, f"{tag}_gravity_cost_matrix.csv")
    gravity_cost.to_csv(g_path_cost, sep=";")
    logger.info(f"Gravity cost matrix computxed and saved in {g_path_cost}.")

    radiation_cost = cost_model.compute_radiation_cost()
    r_path_cost = os.path.join(cost_folder, f"{tag}_radiation_cost_matrix.csv")
    radiation_cost.to_csv(r_path_cost, sep=";")
    logger.info(f"Radiation cost matrix computed and saved in {r_path_cost}.")

    logit_cost = cost_model.compute_logit_cost(beta=-0.01)
    l_path_cost = os.path.join(cost_folder, f"{tag}_logit_cost_matrix.csv")
    logit_cost.to_csv(l_path_cost, sep=";")
    logger.info(f"Logit cost matrix computed and saved in {l_path_cost}.")
    
    return dem_data, G

    # movements_file = os.path.join(save_folder, f"{tag}_movements.csv")
    # simulator = MovementSimulator(points_gdf, agents_with_cell, grid_nonzero, grav_norm, G)
    # movements = simulator.simulate_movements(movements_file, dem_data)
    # movements.to_csv(movements_file, index=False, sep=';')
    # logger.info("Simulation completed.")

if __name__ == "__main__":
    main()
