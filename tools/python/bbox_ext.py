"""
author: Federico Bellisardi

execution: python bbox_ext.py -c ../conf/conf_extractor.json
"""

import os
import io
import zipfile
import json
import argparse

import requests
import geopandas as gpd
import folium
from shapely.geometry import Polygon


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def download_and_extract(url, extract_to):
    resp = requests.get(url)
    resp.raise_for_status()
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(extract_to)


def build_bbox_polygon(coords):
    return Polygon([(lon, lat) for lat, lon in coords])


def compute_center(coords):
    lats, lons = zip(*coords)
    return sum(lats) / len(lats), sum(lons) / len(lons)


def main(config_path):
    cfg = load_config(config_path)
    city, coords = next(iter(cfg.items()))

    raw_dir = "/data/workspaces/fbellisardi/land/"
    download_and_extract(
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
        raw_dir
    )

    shp_file = os.path.join(raw_dir, "ne_10m_land.shp")
    land = gpd.read_file(shp_file)

    bbox_poly = build_bbox_polygon(coords)
    land_clip = gpd.clip(land, bbox_poly)

    out_dir = f"/home/fbellisardi/code/geo_flow/data/data_processed/{city}/land"
    os.makedirs(out_dir, exist_ok=True)
    shp_out = os.path.join(out_dir, f"{city}_clipped_land.shp")
    land_clip.to_file(shp_out)
    print(f"Shapefile exported in: {shp_out}")

    center = compute_center(coords)
    m = folium.Map(location=center, zoom_start=10)

    folium.GeoJson(
        land_clip,
        name="Land (clipped)",
        style_function=lambda _:{
            "fillColor":"green","color":"green","weight":1,"fillOpacity":0.5
        }
    ).add_to(m)

    folium.GeoJson(
        {"type":"Feature","geometry":{
            "type":"Polygon",
            "coordinates":[[ [lon,lat] for lat,lon in coords ]]
        }},
        name="Bounding Box",
        style_function=lambda _:{
            "fill":False,"color":"blue","weight":2
        }
    ).add_to(m)

    titles = ["SW","SE","NE","NW","SW"]
    for (lat, lon), t in zip(coords, titles):
        folium.Marker([lat, lon], popup=t).add_to(m)

    folium.LayerControl().add_to(m)

    map_html = f"{out_dir}/{city}_bbox_map.html"
    m.save(map_html)
    print(f"Map saved in: {map_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip a Natural Earth land shapefile to a bbox defined in a JSON config")
    parser.add_argument("-c", "--config",default="config.json",help="Path to JSON config file (default: config.json)")
    args = parser.parse_args()
    
    main(args.config)
