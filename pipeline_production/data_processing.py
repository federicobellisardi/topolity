#!/usr/bin/env python3
"""Data processing pipeline: graph generation and transformation."""


import argparse
import os
import csv
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask
from tqdm import tqdm

from shapely.geometry import Point, box
from utils import logger, read_conf, polygon_to_wkt

class PopulationReader:
    def __init__(self, input_file, bbox_config, tag, output_folder, search_radius=50):
        self.input_file = input_file
        self.output_folder = output_folder
        self.bbox = bbox_config
        self.tag = tag
        self.search_radius = search_radius
        self.data = None

    def get_band_and_transform(self, dataset):
        min_lon = self.bbox.get("min_lon")
        min_lat = self.bbox.get("min_lat")
        max_lon = self.bbox.get("max_lon")
        max_lat = self.bbox.get("max_lat")
        logger.info(f"Using bounding box for region '{self.tag}': {min_lon}, {min_lat}, {max_lon}, {max_lat}")
        window = rasterio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=dataset.transform)
        band = dataset.read(1, window=window, masked=True)
        transform = dataset.window_transform(window)
        return band, transform,

    def process_features(self, band, transform, dataset, csv_writer):
        features = list(rasterio.features.shapes(band.data, mask=~band.mask, transform=transform))
        logger.info(f"Total features found: {len(features)}")
        idx = 0
        for geom, val in tqdm(features, desc="Processing population", unit="feature"):
            geom_transformed = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)
            if val != 0:
                try:
                    wkt_polygon = polygon_to_wkt(geom_transformed)
                except ValueError as e:
                    logger.warning(f"Skipping feature at index {idx}: {e}")
                    continue
                csv_writer.writerow([idx, idx, round(val), wkt_polygon])
                idx += 1

    def read_population(self):
        with rasterio.open(self.input_file) as dataset:
            band, transform = PopulationReader.get_band_and_transform(self, dataset)
            output_file = os.path.join(self.output_folder, f"{self.tag}_population.csv")
            with open(output_file, mode='w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';')
                csv_writer.writerow(["index", "cod", "population", "geometry"])
                self.process_features(band, transform, dataset, csv_writer)
                csvfile.flush()
        return pd.read_csv(output_file, sep=";")

# @todo: Implement the TwitterOD class to fetch OD matrix data from Twitter.
# class TwitterOD:

class DEMReader:
    def __init__(self, dem_file, search_radius=50):
        self.dem_file = dem_file
        self.search_radius = search_radius
        self.dem_data = None

    def download_dem(self, api_key, bbox_dict, dem_file="dem.tif"):
        min_lon = bbox_dict.get("min_lon", 11.2)
        min_lat = bbox_dict.get("min_lat", 44.4)
        max_lon = bbox_dict.get("max_lon", 11.7)
        max_lat = bbox_dict.get("max_lat", 44.7)
        width = max_lon - min_lon
        height = max_lat - min_lat
        expanded_min_lon = min_lon - width
        expanded_min_lat = min_lat - height
        expanded_max_lon = max_lon + width
        expanded_max_lat = max_lat + height
        url = (
            f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&"
            f"south={expanded_min_lat}&north={expanded_max_lat}&west={expanded_min_lon}&east={expanded_max_lon}&"
            f"outputFormat=GTiff&API_Key={api_key}"
        )
        
        logger.info(f"Requesting DEM from URL: {url.replace(api_key, 'HIDDEN')}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(dem_file, 'wb') as f:
                f.write(response.content)
            logger.info(f"DEM downloaded and saved as {dem_file}")
            return dem_file
        else:
            try:
                error_details = response.json()
                logger.error(f"Error fetching DEM (status {response.status_code}): {error_details}")
            except (ValueError, requests.exceptions.JSONDecodeError):
                logger.error(f"Error fetching DEM (status {response.status_code}): {response.text[:500]}")
            raise Exception(f"Failed to download DEM - HTTP {response.status_code}")

    def get_pixel_centroids(self, bbox=None):
        logger.info(f"Reading DEM from {self.dem_file}")
        centroids = []
        lats = []
        lons = []
        alts = []
        with rasterio.open(self.dem_file) as src:
            if bbox:
                bbox_geom = [box(*bbox)]
                image, transform = mask(src, bbox_geom, crop=True)
                elevation = image[0]
            else:
                elevation = src.read(1, masked=True)
                transform = src.transform

            rows, cols = elevation.shape
            for row in range(rows):
                for col in range(cols):
                    if np.ma.is_masked(elevation[row, col]):
                        continue
                    lon, lat = rasterio.transform.xy(transform, row, col, offset='center')
                    alt = float(elevation[row, col])
                    lats.append(lat)
                    lons.append(lon)
                    alts.append(alt)
                    centroids.append(Point(lon, lat))
        gdf = gpd.GeoDataFrame({'lat': lats, 'lon': lons, 'alt': alts, 'geometry': centroids})
        gdf.crs = src.crs
        logger.info("DEM pixel centroids extracted successfully.")
        return gdf

def process_dataset(conf):
    dp_conf = conf.get("data_processing", {})
    wp_conf = conf.get("data_processing", {}).get("data_source", {}).get("worldpop", {})
    al_conf = conf.get("data_processing", {}).get("altitude", {})
    tag = dp_conf.get("tag", {})
    input_file = wp_conf.get("input_file")
    input_file = os.path.join(os.environ['WORKSPACE'], 'data', f'{input_file}')
    dem_path = os.path.join(os.environ['WORKSPACE'], 'data', 'dem')
    if not os.path.exists(dem_path):
        os.makedirs(dem_path)
    dem_file = os.path.join(os.environ['WORKSPACE'], 'data', 'dem', f'{tag}_dem.tif')

    bbox_config = dp_conf.get("bbox", {})
    selected_bbox = bbox_config.get(tag, {})
    search_radius = conf.get("data_processing", {}).get("search_radius", 50)
    output_folder = conf.get("data_processing", {}).get("output_folder", {})
    output_folder = os.path.join(os.environ['WORKSPACE'], 'topolity', f'{output_folder}', f'tag_{tag}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder: {output_folder}")

    if not input_file:
        logger.error("'Input_file' must be provided in the configuration.")
        return
    
    pop_reader = PopulationReader(input_file, selected_bbox, tag, output_folder, search_radius=search_radius)
    pop_data = pop_reader.read_population()

    dem_reader = DEMReader(dem_file, search_radius=search_radius)
    if not os.path.exists(dem_file):
        dem_download = dem_reader.download_dem(al_conf.get("api_key"), selected_bbox, dem_file)
    else:
        logger.info(f"Using existing DEM file: {dem_file}")

    dem_gdf = dem_reader.get_pixel_centroids()

    return pop_data, dem_gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process population and DEM files to produce enriched data."
    )
    parser.add_argument("-c", "--conf", required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    conf = read_conf(args.conf)

    process_dataset(conf)
