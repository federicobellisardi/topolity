#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: Federico Bellisardi
This script generates maps of Functional Urban Areas (FUA) from a GeoPackage file.
It allows users to search for cities, convert coordinates to WGS84 if necessary,
and generate PNG maps with bounding boxes.
"""

import os
import logging
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt/X11 errors
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box
import numpy as np
from pyproj import Transformer
from typing import Optional, Tuple

log_dir = '/home/fbellisardi/code/topolity/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'fua_map_generator.log'))
    ]
)
logger = logging.getLogger(__name__)


class FUAMapGenerator:
    
    def __init__(self, gpkg_file: str, output_dir: str = "./output_maps"):
        plt.rcParams.update({'font.size': 18})
        self.gpkg_file = gpkg_file
        self.output_dir = output_dir
        self.gdf = None
        logger.info(f"Initialized FUAMapGenerator with gpkg_file: {gpkg_file}, output_dir: {output_dir}")
        
    def load_data(self) -> gpd.GeoDataFrame:
        logger.info(f"Loading GeoPackage: {self.gpkg_file}")
        logger.info(f"File exists: {os.path.exists(self.gpkg_file)}")
        
        try:
            self.gdf = gpd.read_file(self.gpkg_file)
            logger.info(f"GeoDataFrame loaded with {len(self.gdf)} rows")
            logger.info(f"CRS: {self.gdf.crs}")
            logger.debug(f"Columns: {self.gdf.columns.tolist()}")
            return self.gdf
        except Exception as e:
            logger.error(f"Error loading GeoPackage: {e}")
            raise
    
    def find_city_row(self, city_name: str) -> gpd.GeoSeries:
        if self.gdf is None:
            raise ValueError("Data not loaded. Run load_data() first")
            
        logger.info(f"Searching for city: {city_name}")
        
        mask = self.gdf['eFUA_name'].str.lower() == city_name.lower()
        result = self.gdf[mask]
        
        if not result.empty:
            if len(result) > 1:
                logger.warning(f"Multiple rows found ({len(result)}) for eFUA_name == '{city_name}'")
                raise ValueError(f"Multiple rows found ({len(result)}) for eFUA_name == '{city_name}'")
            logger.info(f"Exact match found for '{city_name}'")
            return result.iloc[0]
        
        return self._search_partial_match(city_name)
    
    def _search_partial_match(self, city_name: str) -> gpd.GeoSeries:
        logger.info(f"No exact match found for '{city_name}'")
        logger.info("Searching for partial matches...")
        
        partial_mask = self.gdf['eFUA_name'].str.lower().str.contains(city_name.lower(), na=False)
        partial_results = self.gdf[partial_mask]
        
        if partial_results.empty:
            reverse_mask = self.gdf['eFUA_name'].str.lower().apply(
                lambda x: city_name.lower() in x if pd.notna(x) else False
            )
            partial_results = self.gdf[reverse_mask]
        
        if partial_results.empty:
            logger.warning(f"No matches found for '{city_name}'")
            print(f"No matches found for '{city_name}'")
            print("\nHere are some available cities in the dataset:")
            print(self.gdf['eFUA_name'].head(20).tolist())
            raise ValueError(f"No matches found for '{city_name}'")
        
        logger.info(f"Found {len(partial_results)} partial matches")
        return self._interactive_city_selection(partial_results)
    
    def _interactive_city_selection(self, partial_results: gpd.GeoDataFrame) -> gpd.GeoSeries:
        print(f"\nFound {len(partial_results)} partial matches:")
        index_map = {}  # Map displayed indices to rows
        for i, row in partial_results.iterrows():
            print(f"{i}: {row['eFUA_name']}")
            index_map[i] = row
        
        while True:
            try:
                choice = input(f"\nEnter the desired city index or 'q' to quit: ")
                if choice.lower() == 'q':
                    logger.info("Operation cancelled by user")
                    raise ValueError("Operation cancelled by user")
                
                index = int(choice)
                if index in index_map:
                    selected_row = index_map[index]
                    logger.info(f"Selected: {selected_row['eFUA_name']}")
                    print(f"Selected: {selected_row['eFUA_name']}")
                    return selected_row
                else:
                    available_indices = list(index_map.keys())
                    print(f"Enter a valid index from: {available_indices}")
            except ValueError as e:
                if "Operation cancelled" in str(e):
                    raise e
                print("Enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                logger.info("Operation cancelled by user (KeyboardInterrupt)")
                raise ValueError("Operation cancelled by user")
    
    def convert_to_wgs84(self, row: gpd.GeoSeries) -> Tuple[float, float, float, float]:
        minx, miny, maxx, maxy = row.geometry.bounds
        logger.info(f"Original bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")
        
        if abs(minx) > 180 or abs(maxx) > 180 or abs(miny) > 90 or abs(maxy) > 90:
            logger.warning("Coordinates don't seem to be in EPSG:4326!")
            logger.info(f"GeoDataFrame CRS: {self.gdf.crs}")
            
            if self.gdf.crs != 'EPSG:4326':
                logger.info("Converting coordinates to EPSG:4326...")
                gdf_single = gpd.GeoDataFrame([row], crs=self.gdf.crs)
                gdf_single_4326 = gdf_single.to_crs('EPSG:4326')
                row_4326 = gdf_single_4326.iloc[0]
                minx, miny, maxx, maxy = row_4326.geometry.bounds
                logger.info(f"Converted bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")
        
        return minx, miny, maxx, maxy
    
    def generate_map(self, city_name: str, south: float, north: float, 
                    west: float, east: float) -> str:
        logger.info(f"Generating PNG for {city_name}")
        logger.info(f"Bounds: South={south}, North={north}, West={west}, East={east}")
        
        bbox_poly = box(west, south, east, north)
        gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs='EPSG:4326')
        
        logger.debug(f"Bbox bounds in EPSG:4326: {gdf_bbox.total_bounds}")
        
        gdf_bbox_3857 = gdf_bbox.to_crs(epsg=3857)
        logger.debug(f"Bbox bounds in EPSG:3857: {gdf_bbox_3857.total_bounds}")

        fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
        gdf_bbox_3857.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=3, alpha=0.8)
        self._add_basemap(ax, gdf_bbox_3857)
        self._configure_axes(ax, gdf_bbox_3857, city_name)
        png_file = self._save_map(fig, city_name)
        
        return png_file
    
    def _add_basemap(self, ax, gdf_bbox_3857: gpd.GeoDataFrame):
        try:
            import contextily as ctx
            logger.info("Adding basemap...")
            ctx.add_basemap(ax, crs=gdf_bbox_3857.crs, source=ctx.providers.OpenStreetMap.Mapnik)
            logger.info("Basemap added successfully")
        except ImportError:
            logger.warning("Contextily not available, using gray background")
            ax.set_facecolor('lightgray')
        except Exception as e:
            logger.error(f"Error adding basemap: {e}")
            ax.set_facecolor('lightgray')
    
    def _configure_axes(self, ax, gdf_bbox_3857: gpd.GeoDataFrame, city_name: str):
        minx, miny, maxx, maxy = gdf_bbox_3857.total_bounds
        
        x_margin = (maxx - minx) * 0.1
        y_margin = (maxy - miny) * 0.1
        ax.set_xlim(minx - x_margin, maxx + x_margin)
        ax.set_ylim(miny - y_margin, maxy + y_margin)
        
        transformer = Transformer.from_crs(3857, 4326, always_xy=True)
        
        xticks = np.linspace(minx, maxx, 5)
        yticks = np.linspace(miny, maxy, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{transformer.transform(x, miny)[0]:.3f}°" for x in xticks], rotation=45)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{transformer.transform(minx, y)[1]:.3f}°" for y in yticks])
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # ax.set_title(f'Bounding Box of {city_name}', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _save_map(self, fig, city_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        png_file = os.path.join(self.output_dir, f'{city_name}_bbox.png')
        plt.savefig(png_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"PNG saved to: {png_file}")
        return png_file
    
    def process_city(self, city_name: str) -> str:
        if self.gdf is None:
            self.load_data()
        
        print("\nFirst 10 available cities:")
        print(self.gdf['eFUA_name'].head(10).tolist())
        
        logger.info(f"Searching for city: {city_name}")
        row = self.find_city_row(city_name)
        selected_city_name = row['eFUA_name']
        logger.info(f"Row found for {selected_city_name}")
        logger.debug(f"Geometry type: {type(row.geometry)}")
       
        minx, miny, maxx, maxy = self.convert_to_wgs84(row)
        
        png_file = self.generate_map(selected_city_name, south=miny, north=maxy, 
                                   west=minx, east=maxx)
        
        self.export_bbox_coordinates(selected_city_name, minx, miny, maxx, maxy)
        
        return png_file
    
    def export_bbox_coordinates(self, city_name: str, minx: float, miny: float, 
                              maxx: float, maxy: float) -> bool:
        try:
            choice = input(f"\nDo you want to export the bounding box coordinates for {city_name}? (y/n): ")
            if choice.lower() in ['y', 'yes']:
                logger.info(f"Exporting bounding box coordinates for {city_name}")
                
                bbox_coords = [
                    [miny, minx],  # Bottom-left
                    [miny, maxx],  # Bottom-right  
                    [maxy, maxx],  # Top-right
                    [maxy, minx],  # Top-left
                    [miny, minx]   # Close the polygon
                ]
                
                city_key = city_name.lower().replace(' ', '_')
                print(f'\n  "{city_key}": [')
                for i, coord in enumerate(bbox_coords):
                    if i < len(bbox_coords) - 1:
                        print(f'    [{coord[0]:.4f}, {coord[1]:.4f}],')
                    else:
                        print(f'    [{coord[0]:.4f}, {coord[1]:.4f}]')
                print('  ],')
                
                logger.info("Bounding box coordinates exported successfully")
                return True
            else:
                logger.info("User chose not to export coordinates")
                return False
                
        except KeyboardInterrupt:
            logger.info("Export operation cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Error during coordinates export: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate maps of Functional Urban Areas (FUA) from a GeoPackage file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Examples:
                    %(prog)s Barcelona
                    %(prog)s Madrid --output ./maps
                    %(prog)s "New York" --gpkg /path/to/file.gpkg
                    %(prog)s Rome --output ./output --gpkg /path/to/data.gpkg
                            """
    )
    
    parser.add_argument('city_name',help='Name of the city to process (e.g., "Barcelona", "Madrid")')
    
    parser.add_argument('--gpkg', '--gpkg-file', default="/home/fbellisardi/code/topolity/vars/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg", help='Path to the GeoPackage file (default: %(default)s)')
    
    parser.add_argument('--output', '--output-dir', default="/home/fbellisardi/code/topolity/output/bbox",help='Output directory for generated maps (default: %(default)s)')
    
    parser.add_argument('--log-level',choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],default='INFO',help='Set the logging level (default: %(default)s)')
    
    parser.add_argument('--version',action='version',version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    config = {
        'gpkg_file': args.gpkg,
        'city_name': args.city_name,
        'output_dir': args.output
    }
    
    logger.info(f"Starting FUA Map Generator v1.0.0")
    logger.info(f"Configuration: {config}")
    
    try:
        generator = FUAMapGenerator(config['gpkg_file'], config['output_dir'])
        png_file = generator.process_city(config['city_name'])
        logger.info("Process completed successfully!")
        logger.info(f"File saved: {png_file}")
        print(f"\nProcess completed successfully!")
        print(f"File saved: {png_file}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"\nError during execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
