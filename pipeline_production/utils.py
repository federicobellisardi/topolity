#!/usr/bin/env python3
"""Shared utilities for the pipeline."""
import logging
import json
import math

def setup_logger(name=__name__, level=logging.INFO, fmt="%(asctime)s - %(levelname)s - %(message)s"):
    """
    Set up and return a logger with the given name and log level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(fmt)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger("utils")

def read_conf(conf_path):
    """
    Read and return the configuration dictionary from a JSON file.
    """
    with open(conf_path, 'r') as f:
        return json.load(f)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points using the haversine formula.
    Returns the distance in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def polygon_to_wkt(geom):
    if geom['type'] != 'Polygon':
        raise ValueError("Geometry type is not Polygon")
    coords = geom['coordinates'][0]
    coord_strings = ", ".join(f"{pt[0]} {pt[1]}" for pt in coords)
    return f"POLYGON (({coord_strings}))"

def friction_work(lat1, lon1, lat2, lon2, m, g, mu):
    """
    Calculate friction work along the horizontal distance between two geographic points.
    """
    distance = haversine(lat1, lon1, lat2, lon2)
    return mu * m * g * distance

