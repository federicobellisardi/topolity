#!/usr/bin/env python3
"""
Plotting module for gravitational work analysis.
Contains all plotting functions separated from the main analysis code.

Author: Federico Bellisardi
Usage: python plot_gravitational_work.py --city santiago
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import networkx as nx
import rasterio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pickle
try:
    import contextily as ctx
    import pyproj
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Warning: contextily/pyproj not available. Install with: pip install contextily pyproj")
import glob
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def plot_work_by_variant(results_df, out_png):
    """
    Explanatory bar plot showing W_WD and W_HOL for each variant side by side.
    """
    variants = results_df['variant']
    wd = results_df['work_wd']
    hol = results_df['work_hol']

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(variants) * 0.8), 6))
    
    # Enhanced color scheme
    colors = ['#2E86AB', '#A23B72']
    bars1 = ax.bar(x - width/2, wd, width, label='Working Day', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, hol, width, label='Holiday', color=colors[1], alpha=0.8)
    
    # Value labels removed for better readability

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Total uphill work (m·units)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved work-by-variant plot → {out_png}")


def plot_variant_differences(results_df, out_png):
    """
    Plot percent difference in W_WD and W_HOL relative to the original network.
    """
    orig = results_df[results_df.variant == 'original'].iloc[0]
    pert = results_df[results_df.variant != 'original'].copy()
    pert['pct_diff_wd']  = (pert.work_wd  - orig.work_wd)  / orig.work_wd  * 100
    pert['pct_diff_hol'] = (pert.work_hol - orig.work_hol) / orig.work_hol * 100

    variants = pert['variant']
    wd_diff  = pert['pct_diff_wd']
    hol_diff = pert['pct_diff_hol']

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(variants) * 0.8), 6))
    
    # Enhanced colors with positive/negative distinction
    colors = ['#E63946', '#F77F00']  # Red and orange tones
    bars1 = ax.bar(x - width/2, wd_diff, width, label='Working Day Δ%', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, hol_diff, width, label='Holiday Δ%', color=colors[1], alpha=0.8)
    
    # Color bars based on positive/negative values
    for bar, val in zip(bars1, wd_diff):
        bar.set_color('#2A9D8F' if val < 0 else '#E63946')
    for bar, val in zip(bars2, hol_diff):
        bar.set_color('#2A9D8F' if val < 0 else '#F77F00')
    
    # Percentage labels removed for better readability

    ax.axhline(0, color='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Percent difference vs original (%)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved percent-difference plot → {out_png}")


def plot_arc_work_distribution(arc_work_csv, out_png):
    """
    Plot histogram of arc_work values from CSV file.
    Expected CSV columns: u, v, start_x, start_y, end_x, end_y, segment_work
    """
    df = pd.read_csv(arc_work_csv)
    vals = df['segment_work'].values
    
    plt.figure(figsize=(10,6))
    n, bins, patches = plt.hist(vals, bins=50, color='#457B9D', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Color gradient for histogram
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.viridis(i / len(patches)))
    
    plt.xlabel('Uphill work per edge (J)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    plt.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nCount: {len(vals)}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved arc-work distribution plot → {out_png}")


def plot_dem_elevation_distribution(dem_file, out_png):
    """
    Plot histogram of DEM elevation values (masked nodata).
    """
    with rasterio.open(dem_file) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
    plt.figure(figsize=(10,6))
    arr_flat = arr.flatten()
    n, bins, patches = plt.hist(arr_flat, bins=100, color='#2D5016', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Color gradient based on elevation
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.terrain(i / len(patches)))
    
    plt.xlabel('Elevation (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add elevation statistics
    min_elev, max_elev = np.min(arr_flat), np.max(arr_flat)
    mean_elev, std_elev = np.mean(arr_flat), np.std(arr_flat)
    plt.text(0.02, 0.98, f'Min: {min_elev:.0f}m\nMax: {max_elev:.0f}m\nMean: {mean_elev:.0f}m\nStd: {std_elev:.0f}m', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved DEM elevation distribution plot → {out_png}")


def plot_arc_work_boxplot(arc_work_csv, out_png):
    """
    Boxplot of uphill work per edge from CSV file.
    """
    df = pd.read_csv(arc_work_csv)
    vals = df['segment_work'].values
    
    plt.figure(figsize=(8,10))
    box_plot = plt.boxplot(vals, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#A8DADC', color='#1D3557', linewidth=2),
                          medianprops=dict(color='#E63946', linewidth=2),
                          whiskerprops=dict(color='#1D3557', linewidth=2),
                          capprops=dict(color='#1D3557', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='#F77F00', alpha=0.5, markersize=4))
    
    plt.ylabel('Uphill work per edge (J)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add statistics annotation
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    plt.text(1.1, median, f'Q1: {q1:.2f}\nMedian: {median:.2f}\nQ3: {q3:.2f}', 
             verticalalignment='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().set_xticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved arc-work boxplot → {out_png}")


def plot_cell_node_mapping(cells_csv, graph_pkl, out_png):
    """
    Scatter plot of cell centroids and their mapped node.
    Requires loading cell coordinates and graph data.
    """
    import pickle
    
    # Load cell data
    cells_df = pd.read_csv(cells_csv)
    
    # Check if we have the transformed columns or need to compute centroids
    if 'cent_lon' in cells_df.columns and 'cent_lat' in cells_df.columns:
        centroids = cells_df[['cent_lon','cent_lat']].values
    else:
        # Compute centroids from x_min, y_min, x_max, y_max
        cells_df['cent_x'] = 0.5 * (cells_df['x_min'] + cells_df['x_max'])
        cells_df['cent_y'] = 0.5 * (cells_df['y_min'] + cells_df['y_max'])
        centroids = cells_df[['cent_x','cent_y']].values
    
    # Load original graph
    with open(graph_pkl, 'rb') as f:
        G = pickle.load(f)
    
    # For simplicity, we'll just plot the centroids and graph nodes
    # The exact mapping logic would require the cell_map which isn't easily recreated
    node_coords = []
    for node, data in G.nodes(data=True):
        lon = data.get('x', data.get('lon'))
        lat = data.get('y', data.get('lat'))
        if lon is not None and lat is not None:
            node_coords.append((lon, lat))
    
    node_coords = np.array(node_coords)
    
    plt.figure(figsize=(12,10))
    
    # Enhanced scatter plot with better colors and styling
    if len(node_coords) > 0:
        plt.scatter(node_coords[:,0], node_coords[:,1], s=3, c='#457B9D', 
                   label='Graph nodes', alpha=0.6, marker='.')
    plt.scatter(centroids[:,0], centroids[:,1], s=25, c='#E63946', 
               label='Cell centroids', alpha=0.8, marker='s', edgecolors='white', linewidth=0.5)
    
    plt.xlabel('X coordinate', fontsize=12, fontweight='bold')
    plt.ylabel('Y coordinate', fontsize=12, fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect ratio for geographic data
    plt.axis('equal')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cell-node mapping plot → {out_png}")


def remove_duplicate_variants(results_df):
    """
    Remove variants that have identical work values (both work_wd and work_hol).
    Keep the first occurrence and prioritize 'original' variant.
    """
    # Create a composite key for work values (rounded to avoid floating point issues)
    results_df['work_key'] = results_df.apply(
        lambda row: (round(row['work_wd'], 6), round(row['work_hol'], 6)), axis=1
    )
    
    # Group by work values
    duplicates = results_df.groupby('work_key')
    
    filtered_rows = []
    removed_variants = []
    
    for work_key, group in duplicates:
        if len(group) > 1:
            # Multiple variants with same work values
            variants_list = group['variant'].tolist()
            
            # Prioritize 'original' variant if present
            if 'original' in variants_list:
                kept_variant = 'original'
                kept_row = group[group['variant'] == 'original'].iloc[0]
            else:
                # Keep the first one alphabetically for consistency
                kept_variant = sorted(variants_list)[0]
                kept_row = group[group['variant'] == kept_variant].iloc[0]
            
            filtered_rows.append(kept_row)
            
            # Track removed variants
            for variant in variants_list:
                if variant != kept_variant:
                    removed_variants.append(variant)
                    
            logger.info(f"Duplicate work values found: {variants_list}")
            logger.info(f"Keeping '{kept_variant}', removing: {[v for v in variants_list if v != kept_variant]}")
        else:
            # Unique work values
            filtered_rows.append(group.iloc[0])
    
    if removed_variants:
        logger.info(f"Removed {len(removed_variants)} duplicate variants: {removed_variants}")
    else:
        logger.info("No duplicate variants found")
    
    # Create new dataframe without the work_key column
    filtered_df = pd.DataFrame(filtered_rows).drop('work_key', axis=1).reset_index(drop=True)
    return filtered_df


def extract_scale_factor(variant_name):
    """
    Extract scale factor from variant name.
    """
    import re
    
    # Try different patterns for scale factors
    patterns = [
        r'scale.*?(\d+\.?\d*)',  # scale_x_1.2, scale_y_0.8, etc.
        r'scal.*?(\d+\.?\d*)',   # scaling_1.5, etc.
        r'(\d+\.?\d*).*scale',   # 1.2_scale, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, variant_name.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Default to 1.0 if no scale factor found
    return 1.0


def group_variants_by_transformation_type(results_df, min_scale_factor=1.0):
    """
    Group variants by transformation type and filter scaling variants.
    
    Parameters:
    - results_df: DataFrame with variant results
    - min_scale_factor: Minimum scale factor to include (default: 1.0)
    
    Returns:
    - Dictionary with grouped variants by type
    """
    variants = results_df.copy()
    
    # Initialize groups
    groups = {
        'original': variants[variants['variant'] == 'original'],
        'translations': pd.DataFrame(),
        'rotations': pd.DataFrame(), 
        'scaling': pd.DataFrame()
    }
    
    # Group by transformation type
    for _, row in variants.iterrows():
        variant = row['variant'].lower()
        
        if variant == 'original':
            continue
        elif 'translat' in variant:
            groups['translations'] = pd.concat([groups['translations'], row.to_frame().T])
        elif 'rotat' in variant or 'rot_' in variant:
            groups['rotations'] = pd.concat([groups['rotations'], row.to_frame().T])
        elif 'scal' in variant or 'scale' in variant:
            # Extract scale factor and filter
            scale_factor = extract_scale_factor(row['variant'])
            if scale_factor >= min_scale_factor:
                groups['scaling'] = pd.concat([groups['scaling'], row.to_frame().T])
    
    # Remove empty groups and reset indices
    for key in list(groups.keys()):
        if len(groups[key]) == 0:
            del groups[key]
        else:
            groups[key] = groups[key].reset_index(drop=True)
    
    # Log grouping results
    total_variants = sum(len(group) for group in groups.values())
    logger.info(f"Grouped {len(variants)} variants into {len(groups)} types:")
    for group_name, group_df in groups.items():
        logger.info(f"  {group_name}: {len(group_df)} variants")
    
    return groups


def create_grouped_summary_data(groups):
    """
    Create summary statistics for each transformation group.
    """
    summary_data = []
    
    for group_name, group_df in groups.items():
        if len(group_df) == 0:
            continue
            
        if group_name == 'original':
            # Original is just one variant
            summary_data.append({
                'group': 'Original',
                'variant_count': 1,
                'work_wd_mean': group_df.iloc[0]['work_wd'],
                'work_hol_mean': group_df.iloc[0]['work_hol'],
                'work_wd_std': 0,
                'work_hol_std': 0,
                'work_wd_min': group_df.iloc[0]['work_wd'],
                'work_wd_max': group_df.iloc[0]['work_wd'],
                'work_hol_min': group_df.iloc[0]['work_hol'],
                'work_hol_max': group_df.iloc[0]['work_hol']
            })
        else:
            # Calculate statistics for the group
            summary_data.append({
                'group': group_name.title(),
                'variant_count': len(group_df),
                'work_wd_mean': group_df['work_wd'].mean(),
                'work_hol_mean': group_df['work_hol'].mean(),
                'work_wd_std': group_df['work_wd'].std() if len(group_df) > 1 else 0,
                'work_hol_std': group_df['work_hol'].std() if len(group_df) > 1 else 0,
                'work_wd_min': group_df['work_wd'].min(),
                'work_wd_max': group_df['work_wd'].max(),
                'work_hol_min': group_df['work_hol'].min(),
                'work_hol_max': group_df['work_hol'].max()
            })
    
    return pd.DataFrame(summary_data)


def plot_grouped_transformation_comparison(groups, out_png):
    """
    Plot comparison of transformation groups with error bars showing variation.
    """
    summary_df = create_grouped_summary_data(groups)
    
    if len(summary_df) == 0:
        logger.warning("No groups to plot")
        return
    
    fig, ax = plt.subplots(figsize=(max(10, len(summary_df) * 1.5), 8))
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    # Plot bars with error bars
    bars1 = ax.bar(x - width/2, summary_df['work_wd_mean'], width, 
                   yerr=summary_df['work_wd_std'], 
                   label='Working Day', color='#2E86AB', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, summary_df['work_hol_mean'], width, 
                   yerr=summary_df['work_hol_std'],
                   label='Holiday', color='#A23B72', alpha=0.8, capsize=5)
    
    # Highlight original in gold
    for i, group in enumerate(summary_df['group']):
        if group == 'Original':
            bars1[i].set_color('#FFD700')
            bars2[i].set_color('#FFD700')
            bars1[i].set_edgecolor('#2E86AB')
            bars2[i].set_edgecolor('#A23B72')
            bars1[i].set_linewidth(2)
            bars2[i].set_linewidth(2)
    
    # Add variant count labels
    for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, summary_df['variant_count'])):
        max_height = max(bar1.get_height() + summary_df.iloc[i]['work_wd_std'], 
                        bar2.get_height() + summary_df.iloc[i]['work_hol_std'])
        ax.text(i, max_height + max(summary_df['work_wd_mean']) * 0.02, 
                f'n={count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['group'], fontsize=12)
    ax.set_ylabel('Total uphill work (m·units)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved grouped transformation comparison plot → {out_png}")


def check_available_variant_types(results_df):
    """
    Check which types of variants are available in the results.
    Returns a dictionary with boolean flags for each type.
    """
    variants = results_df['variant'].tolist()
    
    has_translations = any('translat' in v.lower() for v in variants)
    has_rotations = any('rotat' in v.lower() for v in variants)
    has_scaling = any('scal' in v.lower() for v in variants)
    has_original = 'original' in variants
    
    logger.info(f"Available variant types: translations={has_translations}, "
                f"rotations={has_rotations}, scaling={has_scaling}, original={has_original}")
    
    return {
        'translations': has_translations,
        'rotations': has_rotations, 
        'scaling': has_scaling,
        'original': has_original
    }


def plot_work_parabolic_translations(results_df, stats_df, out_png):
    orig_row = results_df[results_df.variant == 'original']
    translation_variants = results_df[results_df.variant.str.contains('translat', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for translation plot")
        return
    
    if len(translation_variants) == 0:
        logger.warning("No translation variants found")
        return
    
    merged = translation_variants.merge(stats_df, on='variant', how='left')
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    orig_offset_x = 0.0
    orig_offset_y = 0.0
    
    translated_variants = []
    for _, row in merged.iterrows():
        offset_x = row['offset_x']
        offset_y = row['offset_y']
        sort_key = (row['work_wd'] + row['work_hol']) / 2
        translated_variants.append((sort_key, offset_x, offset_y, row['work_wd'], row['work_hol'], row['variant']))
    
    translated_variants.sort(key=lambda x: x[0])
    
    n_variants = len(translated_variants)
    center_idx = n_variants // 2
    
    all_sort_keys = [None] * (n_variants + 1)
    all_wd = [None] * (n_variants + 1)
    all_hol = [None] * (n_variants + 1)
    all_offset_x = [None] * (n_variants + 1)
    all_offset_y = [None] * (n_variants + 1)
    all_names = [None] * (n_variants + 1)
    
    orig_idx = center_idx
    orig_sort_key = (orig_wd + orig_hol) / 2
    all_sort_keys[orig_idx] = orig_sort_key
    all_wd[orig_idx] = orig_wd
    all_hol[orig_idx] = orig_hol
    all_offset_x[orig_idx] = orig_offset_x
    all_offset_y[orig_idx] = orig_offset_y
    all_names[orig_idx] = 'Original'
    
    for i, (sort_key, offset_x, offset_y, wd, hol, name) in enumerate(translated_variants):
        if i < center_idx:
            pos = center_idx - 1 - i
        else:
            pos = center_idx + 1 + (i - center_idx)
        
        all_sort_keys[pos] = sort_key
        all_wd[pos] = wd
        all_hol[pos] = hol
        all_offset_x[pos] = offset_x
        all_offset_y[pos] = offset_y
        all_names[pos] = name
    
    x_positions = list(range(len(all_sort_keys)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append(f'Original\n({orig_offset_x:.2f},{orig_offset_y:.2f})')
        else:
            offset_x = all_offset_x[i]
            offset_y = all_offset_y[i]
            x_labels.append(f'Δlat:{offset_x:.2f}\nΔlon:{offset_y:.2f}')
    
    x_positions = np.array(x_positions)
    wd_values = np.array(all_wd)
    hol_values = np.array(all_hol)
    
    orig_x = x_positions[orig_idx]
    orig_wd = wd_values[orig_idx]
    orig_hol = hol_values[orig_idx]
    
    def fit_constrained_parabola(x_data, y_data, x0, y0):
        u_data = x_data - x0
        v_data = y_data - y0
        a = np.sum(v_data * u_data**2) / np.sum(u_data**4)
        return a, x0, y0
    
    a_wd, x0_wd, y0_wd = fit_constrained_parabola(x_positions, wd_values, orig_x, orig_wd)
    a_hol, x0_hol, y0_hol = fit_constrained_parabola(x_positions, hol_values, orig_x, orig_hol)
    
    x_smooth = np.linspace(x_positions.min(), x_positions.max(), 100)
    y_smooth_wd = a_wd * (x_smooth - x0_wd)**2 + y0_wd
    y_smooth_hol = a_hol * (x_smooth - x0_hol)**2 + y0_hol
    
    y_pred_wd = a_wd * (x_positions - x0_wd)**2 + y0_wd
    y_pred_hol = a_hol * (x_positions - x0_hol)**2 + y0_hol
    
    r_squared_wd = 1 - (np.sum((wd_values - y_pred_wd) ** 2) / np.sum((wd_values - np.mean(wd_values)) ** 2))
    r_squared_hol = 1 - (np.sum((hol_values - y_pred_hol) ** 2) / np.sum((hol_values - np.mean(hol_values)) ** 2))
    
    fig, ax = plt.subplots(figsize=(max(16, len(x_positions) * 1.2), 10))
    
    width = 0.35
    x_offset = np.arange(len(x_positions))
    
    bars_wd = ax.bar(x_offset - width/2, wd_values, width, 
                     label='Working Day', color='blue', alpha=0.8)
    bars_hol = ax.bar(x_offset + width/2, hol_values, width, 
                      label='Holiday', color='red', alpha=0.8)
    
    bars_wd[orig_idx].set_facecolor('gold')
    bars_wd[orig_idx].set_edgecolor('blue')
    bars_wd[orig_idx].set_linewidth(3)
    
    bars_hol[orig_idx].set_facecolor('gold')
    bars_hol[orig_idx].set_edgecolor('red')
    bars_hol[orig_idx].set_linewidth(3)
    
    ax.axvline(x=orig_idx, color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = max(wd_values[orig_idx], hol_values[orig_idx])
    y_range = max(max(wd_values), max(hol_values)) - min(min(wd_values), min(hol_values))
    ax.annotate('Original Network', 
               xy=(orig_idx, max_y), 
               xytext=(orig_idx, max_y + y_range * 0.08),
               ha='center', fontsize=16, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(x_offset)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Translation Variants', fontsize=18, fontweight='bold')
    ax.set_ylabel('Total Uphill Work (m·units)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    y_min = min(min(wd_values), min(hol_values))
    y_max = max(max(wd_values), max(hol_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved translation work plot → {out_png}")


def plot_work_parabolic_translations_average(results_df, stats_df, out_png):
    """
    Plot translation variants with average of working day and holiday work values.
    """
    orig_row = results_df[results_df.variant == 'original']
    translation_variants = results_df[results_df.variant.str.contains('translat', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for translation average plot")
        return
    
    if len(translation_variants) == 0:
        logger.warning("No translation variants found for average plot")
        return
    
    merged = translation_variants.merge(stats_df, on='variant', how='left')
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    orig_avg = (orig_wd + orig_hol) / 2
    orig_offset_x = 0.0
    orig_offset_y = 0.0
    
    translated_variants = []
    for _, row in merged.iterrows():
        offset_x = row['offset_x']
        offset_y = row['offset_y']
        avg_work = (row['work_wd'] + row['work_hol']) / 2
        translated_variants.append((avg_work, offset_x, offset_y, avg_work, row['variant']))
    
    translated_variants.sort(key=lambda x: x[0])
    
    n_variants = len(translated_variants)
    center_idx = n_variants // 2
    
    all_avg_work = [None] * (n_variants + 1)
    all_offset_x = [None] * (n_variants + 1)
    all_offset_y = [None] * (n_variants + 1)
    all_names = [None] * (n_variants + 1)
    
    orig_idx = center_idx
    all_avg_work[orig_idx] = orig_avg
    all_offset_x[orig_idx] = orig_offset_x
    all_offset_y[orig_idx] = orig_offset_y
    all_names[orig_idx] = 'Original'
    
    for i, (_, offset_x, offset_y, avg_work, name) in enumerate(translated_variants):
        if i < center_idx:
            pos = center_idx - 1 - i
        else:
            pos = center_idx + 1 + (i - center_idx)
        
        all_avg_work[pos] = avg_work
        all_offset_x[pos] = offset_x
        all_offset_y[pos] = offset_y
        all_names[pos] = name
    
    x_positions = list(range(len(all_avg_work)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append(f'Original\n({orig_offset_x:.2f},{orig_offset_y:.2f})')
        else:
            offset_x = all_offset_x[i]
            offset_y = all_offset_y[i]
            x_labels.append(f'Δlat:{offset_x:.2f}\nΔlon:{offset_y:.2f}')
    
    x_positions = np.array(x_positions)
    avg_values = np.array(all_avg_work)
    
    orig_x = x_positions[orig_idx]
    orig_avg = avg_values[orig_idx]
    
    def fit_constrained_parabola(x_data, y_data, x0, y0):
        u_data = x_data - x0
        v_data = y_data - y0
        a = np.sum(v_data * u_data**2) / np.sum(u_data**4)
        return a, x0, y0
    
    a, x0, y0 = fit_constrained_parabola(x_positions, avg_values, orig_x, orig_avg)
    
    y_pred = a * (x_positions - x0)**2 + y0
    ss_res = np.sum((avg_values - y_pred) ** 2)
    ss_tot = np.sum((avg_values - np.mean(avg_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    coeffs_free = np.polyfit(x_positions, avg_values, 2)
    a_free, b_free, c_free = coeffs_free
    y_pred_free = a_free * x_positions**2 + b_free * x_positions + c_free
    r_squared_free = 1 - (np.sum((avg_values - y_pred_free) ** 2) / np.sum((avg_values - np.mean(avg_values)) ** 2))
    
    fig, ax = plt.subplots(figsize=(max(16, len(x_positions) * 1.2), 10))
    
    ax.bar(range(len(avg_values)), avg_values, color='purple', alpha=0.8, label='Average Work')
    ax.bar(orig_idx, avg_values[orig_idx], color='gold', edgecolor='purple', linewidth=3)
    
    ax.axvline(x=orig_idx, color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = avg_values[orig_idx]
    y_range = max(avg_values) - min(avg_values)
    ax.annotate('Original Network', 
               xy=(orig_idx, max_y), 
               xytext=(orig_idx, max_y + y_range * 0.08),
               ha='center', fontsize=16, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Translation Variants', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Total Uphill Work (m·units)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    y_min = min(avg_values)
    y_max = max(avg_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved translation average work plot → {out_png}")


def plot_work_parabolic_rotations(results_df, stats_df, out_png):
    orig_row = results_df[results_df.variant == 'original']
    rotation_variants = results_df[results_df.variant.str.contains('rotat', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for rotation plot")
        return
    
    if len(rotation_variants) == 0:
        logger.warning("No rotation variants found")
        return
    
    merged = rotation_variants.merge(stats_df, on='variant', how='left')
    
    rotated_variants = []
    for _, row in merged.iterrows():
        angle = row['angle_deg']
        if pd.isna(angle):
            continue
        rotated_variants.append((angle, row['work_wd'], row['work_hol'], row['variant']))
    
    rotated_variants.sort(key=lambda x: x[0])
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    
    all_angles = [angle for angle, _, _, _ in rotated_variants]
    all_wd = [wd for _, wd, _, _ in rotated_variants]
    all_hol = [hol for _, _, hol, _ in rotated_variants]
    all_names = [name for _, _, _, name in rotated_variants]
    
    n_total = len(rotated_variants) + 1
    center_idx = len(rotated_variants) // 2
    
    orig_angle = 0.0
    
    all_angles.insert(center_idx, orig_angle)
    all_wd.insert(center_idx, orig_wd)
    all_hol.insert(center_idx, orig_hol)
    all_names.insert(center_idx, 'Original')
    
    x_positions = list(range(len(all_angles)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append('Original\n(0°)')
        else:
            angle = all_angles[i]
            x_labels.append(f'{angle:.1f}°')
    
    x_positions = np.array(x_positions)
    wd_values = np.array(all_wd)
    hol_values = np.array(all_hol)
    
    orig_x = x_positions[center_idx]
    orig_wd = wd_values[center_idx]
    orig_hol = hol_values[center_idx]
    
    def fit_constrained_parabola(x_data, y_data, x0, y0):
        u_data = x_data - x0
        v_data = y_data - y0
        a = np.sum(v_data * u_data**2) / np.sum(u_data**4)
        return a, x0, y0
    
    a_wd, x0_wd, y0_wd = fit_constrained_parabola(x_positions, wd_values, orig_x, orig_wd)
    a_hol, x0_hol, y0_hol = fit_constrained_parabola(x_positions, hol_values, orig_x, orig_hol)
    
    x_smooth = np.linspace(x_positions.min(), x_positions.max(), 100)
    y_smooth_wd = a_wd * (x_smooth - x0_wd)**2 + y0_wd
    y_smooth_hol = a_hol * (x_smooth - x0_hol)**2 + y0_hol
    
    y_pred_wd = a_wd * (x_positions - x0_wd)**2 + y0_wd
    y_pred_hol = a_hol * (x_positions - x0_hol)**2 + y0_hol
    
    r_squared_wd = 1 - (np.sum((wd_values - y_pred_wd) ** 2) / np.sum((wd_values - np.mean(wd_values)) ** 2))
    r_squared_hol = 1 - (np.sum((hol_values - y_pred_hol) ** 2) / np.sum((hol_values - np.mean(hol_values)) ** 2))
    
    fig, ax = plt.subplots(figsize=(max(16, len(x_positions) * 1.2), 10))
    
    width = 0.35
    x_offset = np.arange(len(x_positions))
    
    bars_wd = ax.bar(x_offset - width/2, wd_values, width, 
                     label='Working Day', color='green', alpha=0.8)
    bars_hol = ax.bar(x_offset + width/2, hol_values, width, 
                      label='Holiday', color='orange', alpha=0.8)
    
    orig_idx = center_idx
    bars_wd[orig_idx].set_facecolor('gold')
    bars_wd[orig_idx].set_edgecolor('green')
    bars_wd[orig_idx].set_linewidth(3)
    
    bars_hol[orig_idx].set_facecolor('gold')
    bars_hol[orig_idx].set_edgecolor('orange')
    bars_hol[orig_idx].set_linewidth(3)
    
    ax.axvline(x=orig_idx, color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = max(wd_values[orig_idx], hol_values[orig_idx])
    y_range = max(max(wd_values), max(hol_values)) - min(min(wd_values), min(hol_values))
    ax.annotate('Original Network', 
               xy=(orig_idx, max_y), 
               xytext=(orig_idx, max_y + y_range * 0.08),
               ha='center', fontsize=16, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(x_offset)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Rotation Variants (degrees)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Total Uphill Work (m·units)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    y_min = min(min(wd_values), min(hol_values))
    y_max = max(max(wd_values), max(hol_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved rotation work plot → {out_png}")


def plot_work_parabolic_scaling(results_df, stats_df, out_png):
    orig_row = results_df[results_df.variant == 'original']
    scaling_variants = results_df[results_df.variant.str.contains('scal', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for scaling plot")
        return
    
    if len(scaling_variants) == 0:
        logger.warning("No scaling variants found")
        return
    
    merged = scaling_variants.merge(stats_df, on='variant', how='left')
    
    scaled_variants = []
    for _, row in merged.iterrows():
        scale = row['scale_factor']
        if pd.isna(scale):
            continue
        scaled_variants.append((scale, row['work_wd'], row['work_hol'], row['variant']))
    
    scaled_variants.sort(key=lambda x: x[0])
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    
    all_scales = [scale for scale, _, _, _ in scaled_variants]
    all_wd = [wd for _, wd, _, _ in scaled_variants]
    all_hol = [hol for _, _, hol, _ in scaled_variants]
    all_names = [name for _, _, _, name in scaled_variants]
    
    n_total = len(scaled_variants) + 1
    center_idx = len(scaled_variants) // 2
    
    orig_scale = 1.0
    
    all_scales.insert(center_idx, orig_scale)
    all_wd.insert(center_idx, orig_wd)
    all_hol.insert(center_idx, orig_hol)
    all_names.insert(center_idx, 'Original')
    
    x_positions = list(range(len(all_scales)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append('Original\n(×1.0)')
        else:
            scale = all_scales[i]
            x_labels.append(f'×{scale:.2f}')
    
    x_positions = np.array(x_positions)
    wd_values = np.array(all_wd)
    hol_values = np.array(all_hol)
    
    orig_x = x_positions[center_idx]
    orig_wd = wd_values[center_idx]
    orig_hol = hol_values[center_idx]
    
    def fit_constrained_parabola(x_data, y_data, x0, y0):
        u_data = x_data - x0
        v_data = y_data - y0
        a = np.sum(v_data * u_data**2) / np.sum(u_data**4)
        return a, x0, y0
    
    a_wd, x0_wd, y0_wd = fit_constrained_parabola(x_positions, wd_values, orig_x, orig_wd)
    a_hol, x0_hol, y0_hol = fit_constrained_parabola(x_positions, hol_values, orig_x, orig_hol)
    
    x_smooth = np.linspace(x_positions.min(), x_positions.max(), 100)
    y_smooth_wd = a_wd * (x_smooth - x0_wd)**2 + y0_wd
    y_smooth_hol = a_hol * (x_smooth - x0_hol)**2 + y0_hol
    
    y_pred_wd = a_wd * (x_positions - x0_wd)**2 + y0_wd
    y_pred_hol = a_hol * (x_positions - x0_hol)**2 + y0_hol
    
    r_squared_wd = 1 - (np.sum((wd_values - y_pred_wd) ** 2) / np.sum((wd_values - np.mean(wd_values)) ** 2))
    r_squared_hol = 1 - (np.sum((hol_values - y_pred_hol) ** 2) / np.sum((hol_values - np.mean(hol_values)) ** 2))
    
    fig, ax = plt.subplots(figsize=(max(16, len(x_positions) * 1.2), 10))
    
    width = 0.35
    x_offset = np.arange(len(x_positions))
    
    bars_wd = ax.bar(x_offset - width/2, wd_values, width, 
                     label='Working Day', color='purple', alpha=0.8)
    bars_hol = ax.bar(x_offset + width/2, hol_values, width, 
                      label='Holiday', color='brown', alpha=0.8)
    
    orig_idx = center_idx
    bars_wd[orig_idx].set_facecolor('gold')
    bars_wd[orig_idx].set_edgecolor('purple')
    bars_wd[orig_idx].set_linewidth(3)
    
    bars_hol[orig_idx].set_facecolor('gold')
    bars_hol[orig_idx].set_edgecolor('brown')
    bars_hol[orig_idx].set_linewidth(3)
    
    ax.axvline(x=orig_idx, color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = max(wd_values[orig_idx], hol_values[orig_idx])
    y_range = max(max(wd_values), max(hol_values)) - min(min(wd_values), min(hol_values))
    ax.annotate('Original Network', 
               xy=(orig_idx, max_y), 
               xytext=(orig_idx, max_y + y_range * 0.08),
               ha='center', fontsize=16, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(x_offset)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Scaling Variants (scale factor)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Total Uphill Work (m·units)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    y_min = min(min(wd_values), min(hol_values))
    y_max = max(max(wd_values), max(hol_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scaling work plot → {out_png}")


def plot_absolute_work_differences(results_df, out_png):
    """
    Plot absolute differences in work values compared to original.
    """
    if 'original' not in results_df['variant'].values:
        logger.warning("No 'original' variant found for absolute differences plot")
        return
    
    orig = results_df[results_df.variant == 'original'].iloc[0]
    variants = results_df[results_df.variant != 'original'].copy()
    
    variants['abs_diff_wd'] = variants.work_wd - orig.work_wd
    variants['abs_diff_hol'] = variants.work_hol - orig.work_hol
    
    fig, ax = plt.subplots(figsize=(max(12, len(variants) * 0.8), 6))
    
    x = np.arange(len(variants))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, variants['abs_diff_wd'], width, 
                   label='Working Day Δ', alpha=0.8)
    bars2 = ax.bar(x + width/2, variants['abs_diff_hol'], width, 
                   label='Holiday Δ', alpha=0.8)
    
    # Color bars based on positive/negative
    for bar, val in zip(bars1, variants['abs_diff_wd']):
        bar.set_color('#2A9D8F' if val < 0 else '#E63946')
    for bar, val in zip(bars2, variants['abs_diff_hol']):
        bar.set_color('#2A9D8F' if val < 0 else '#F77F00')
    
    ax.axhline(0, color='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants['variant'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Absolute work difference (m·units)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved absolute differences plot → {out_png}")


def plot_combined_work_comparison(results_df, out_png):
    """
    Plot combined work (WD + Holiday) for easier comparison.
    """
    results_df = results_df.copy()
    results_df['total_work'] = results_df['work_wd'] + results_df['work_hol']
    results_df = results_df.sort_values('total_work')
    
    fig, ax = plt.subplots(figsize=(max(10, len(results_df) * 0.6), 6))
    
    # Create color map - highlight original in gold
    colors = ['#FFD700' if v == 'original' else '#457B9D' for v in results_df['variant']]
    
    bars = ax.bar(range(len(results_df)), results_df['total_work'], 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels
    # for i, (bar, val) in enumerate(zip(bars, results_df['total_work'])):
    #     ax.text(bar.get_x() + bar.get_width()/2., val + max(results_df['total_work']) * 0.01,
    #             f'{val:.0f}', ha='center', va='bottom', fontsize=9, alpha=0.8)
    
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['variant'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Total work (WD + Holiday)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined work comparison plot → {out_png}")


def plot_work_efficiency_comparison(results_df, out_png):
    """
    Plot work efficiency ratio (Holiday/Working Day) for each variant.
    """
    results_df = results_df.copy()
    # Avoid division by zero
    results_df['efficiency_ratio'] = np.where(results_df['work_wd'] > 0, 
                                            results_df['work_hol'] / results_df['work_wd'], 
                                            0)
    
    fig, ax = plt.subplots(figsize=(max(10, len(results_df) * 0.6), 6))
    
    # Sort by efficiency ratio
    results_df = results_df.sort_values('efficiency_ratio')
    
    # Color based on ratio - closer to 1.0 is more balanced
    colors = []
    for ratio in results_df['efficiency_ratio']:
        if abs(ratio - 1.0) < 0.1:
            colors.append('#2A9D8F')  # Green for balanced
        elif ratio > 1.0:
            colors.append('#E63946')  # Red for holiday > working day
        else:
            colors.append('#F77F00')  # Orange for working day > holiday
    
    bars = ax.bar(range(len(results_df)), results_df['efficiency_ratio'], 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add ratio values on bars
    for i, (bar, ratio) in enumerate(zip(bars, results_df['efficiency_ratio'])):
        ax.text(bar.get_x() + bar.get_width()/2., ratio + max(results_df['efficiency_ratio']) * 0.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=9, alpha=0.8)
    
    # Add reference line at 1.0 (perfect balance)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(len(results_df) * 0.02, 1.02, 'Perfect balance', fontsize=10, alpha=0.7)
    
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['variant'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Work ratio (Holiday / Working Day)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved work efficiency comparison plot → {out_png}")


def plot_grouped_transformation_comparison(groups, output_path):
    """
    Plot comparison of different transformation groups.
    
    Args:
        groups (dict): Dictionary with group names as keys and DataFrames as values
        output_path (str): Path to save the plot
    """
    logger.info(f"Creating grouped transformation comparison plot: {output_path}")
    
    # Prepare data for plotting
    group_data = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    for i, (group_name, group_df) in enumerate(groups.items()):
        if len(group_df) > 0:
            mean_work = group_df['total_work'].mean()
            std_work = group_df['total_work'].std() if len(group_df) > 1 else 0
            count = len(group_df)
            group_data.append({
                'group': group_name,
                'mean_work': mean_work,
                'std_work': std_work,
                'count': count,
                'color': colors[i]
            })
    
    if not group_data:
        logger.warning("No data to plot in grouped comparison")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Mean work by group with error bars
    groups_names = [d['group'] for d in group_data]
    means = [d['mean_work'] for d in group_data]
    stds = [d['std_work'] for d in group_data]
    colors_list = [d['color'] for d in group_data]
    
    bars1 = ax1.bar(groups_names, means, yerr=stds, capsize=5, 
                   color=colors_list, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Mean Total Work', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Transformation Group', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Count labels removed for better readability
    
    # Right plot: Box plot for each group
    all_data = []
    positions = []
    group_labels = []
    
    for i, (group_name, group_df) in enumerate(groups.items()):
        if len(group_df) > 0:
            all_data.append(group_df['total_work'].values)
            positions.append(i + 1)
            group_labels.append(group_name)
    
    if all_data:
        bp = ax2.boxplot(all_data, positions=positions, patch_artist=True,
                        tick_labels=group_labels, notch=True, whis=1.5)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            if element in bp:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(0.8)
        
        ax2.set_ylabel('Total Work Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Transformation Group', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Grouped transformation comparison plot saved to: {output_path}")


def plot_transformation_boxplots(groups, output_path):
    """
    Plot boxplots to characterize the distribution of work across transformation groups.
    
    Args:
        groups (dict): Dictionary with group names as keys and DataFrames as values
        output_path (str): Path to save the plot
    """
    logger.info(f"Creating transformation boxplot characterization: {output_path}")
    
    # Prepare data for plotting - separate original from transformations
    all_data = []
    group_labels = []
    is_original = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    for group_name, group_df in groups.items():
        if len(group_df) > 0:
            all_data.append(group_df['total_work'].values)
            group_labels.append(group_name)
            is_original.append(group_name == 'original')
    
    if not all_data:
        logger.warning("No data to plot in transformation boxplots")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create boxplot only for non-original groups
    positions = []
    boxplot_data = []
    boxplot_labels = []
    point_positions = []
    point_values = []
    
    pos_counter = 1
    for i, (data, label, is_orig) in enumerate(zip(all_data, group_labels, is_original)):
        if is_orig:
            # For original, just plot as a point
            point_positions.append(pos_counter)
            point_values.append(data[0] if len(data) > 0 else 0)
            positions.append(pos_counter)
            boxplot_labels.append(label)
        else:
            # For transformations, add to boxplot data
            boxplot_data.append(data)
            positions.append(pos_counter)
            boxplot_labels.append(label)
        pos_counter += 1
    
    # Create boxplot for transformation groups only
    if boxplot_data:
        bp_positions = [p for p, is_orig in zip(positions, is_original) if not is_orig]
        bp = ax.boxplot(boxplot_data, positions=bp_positions, patch_artist=True,
                        notch=False, whis=1.5,
                        showmeans=True, meanline=False,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                      markeredgecolor='darkred', markersize=8, alpha=0.8))
    
    # Color the boxes
    color_idx = 0
    for i, (patch, is_orig) in enumerate(zip(bp['boxes'], [is_original[j] for j in range(len(is_original)) if not is_original[j]])):
        patch.set_facecolor(colors[color_idx + 1])  # Skip first color for original
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        color_idx += 1
    
    # Plot original as a point
    if point_positions:
        ax.scatter(point_positions, point_values, s=200, c='#FFD700', 
                  marker='*', edgecolors='black', linewidths=2, 
                  zorder=5, label='Original', alpha=0.9)
    
    # Style medians
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2.5)
    
    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_color('gray')
        whisker.set_linewidth(1.2)
        whisker.set_linestyle('--')
    
    for cap in bp['caps']:
        cap.set_color('gray')
        cap.set_linewidth(1.2)
    
    # Style fliers (outliers)
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('lightcoral')
        flier.set_markeredgecolor('darkred')
        flier.set_markersize(5)
        flier.set_alpha(0.5)
    
    # Add statistical annotations
    for i, (group_name, data, pos, is_orig) in enumerate(zip(group_labels, all_data, positions, is_original), 1):
        if is_orig:
            # For original, just show the value
            stats_text = f'n=1\nW={data[0]:.2e}'
            ax.text(pos, data[0] * 1.15, stats_text, 
                   ha='center', va='bottom', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7, edgecolor='black'))
        else:
            # Calculate statistics for transformations
            median_val = np.median(data)
            mean_val = np.mean(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            # Add text annotation with key statistics
            stats_text = f'n={len(data)}\nμ={mean_val:.2e}\nM={median_val:.2e}\nIQR={iqr:.2e}'
            ax.text(pos, ax.get_ylim()[1] * 0.95, stats_text, 
                   ha='center', va='top', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(boxplot_labels, rotation=45, ha='right')
    
    # Labels and formatting
    ax.set_ylabel('Total Gravitational Work', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transformation Type', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Gravitational Work Across Transformation Groups', 
                fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                  markeredgecolor='black', markersize=15, label='Original', linewidth=0),
        plt.Line2D([0], [0], color='darkblue', linewidth=2.5, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
                  markeredgecolor='darkred', markersize=8, label='Mean'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                  markeredgecolor='darkred', markersize=5, label='Outliers')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Transformation boxplot characterization saved to: {output_path}")


def plot_transformation_violinplots(groups, output_path):
    """
    Plot violin plots to characterize the distribution of work across transformation groups.
    
    Args:
        groups (dict): Dictionary with group names as keys and DataFrames as values
        output_path (str): Path to save the plot
    """
    logger.info(f"Creating transformation violin plot characterization: {output_path}")
    
    # Prepare data for plotting - separate original from transformations
    all_data = []
    group_labels = []
    is_original = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    for group_name, group_df in groups.items():
        if len(group_df) > 0:
            # Convert to numpy array and ensure it's a 1D array of floats
            data = np.asarray(group_df['total_work'].values, dtype=float).flatten()
            all_data.append(data)
            group_labels.append(group_name)
            is_original.append(group_name == 'original')
    
    if not all_data:
        logger.warning("No data to plot in transformation violin plots")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create violin plot only for non-original groups
    positions = []
    violin_data = []
    violin_labels = []
    violin_positions = []
    point_positions = []
    point_values = []
    point_labels = []
    
    pos_counter = 1
    for i, (data, label, is_orig) in enumerate(zip(all_data, group_labels, is_original)):
        # Check if data has enough variance for violin plot (need at least 3 points and meaningful variance)
        unique_values = len(np.unique(data))
        std_dev = np.std(data)
        mean_val = np.mean(data)
        # Coefficient of variation to check if variance is meaningful relative to mean
        cv = std_dev / mean_val if mean_val > 0 else 0
        
        has_variance = len(data) >= 3 and unique_values >= 3 and cv > 1e-6
        
        if is_orig or not has_variance:
            # For original, single-point groups, or low-variance groups, plot as a point
            point_positions.append(pos_counter)
            point_values.append(data[0] if len(data) == 1 else np.mean(data))
            point_labels.append(label)
            positions.append(pos_counter)
            violin_labels.append(label)
            if not is_orig:
                logger.info(f"Group '{label}': n={len(data)}, unique={unique_values}, std={std_dev:.2e}, cv={cv:.2e} - plotting as point")
        else:
            # For transformations with multiple points and variance, add to violin plot data
            violin_data.append(data)
            violin_positions.append(pos_counter)
            positions.append(pos_counter)
            violin_labels.append(label)
            logger.info(f"Group '{label}': n={len(data)}, unique={unique_values}, std={std_dev:.2e}, cv={cv:.2e} - plotting as violin")
        pos_counter += 1
    
    
    # Create violin plot for transformation groups only (with multiple data points and variance)
    if violin_data:
        try:
            # Simply try to plot - if it fails, we'll fall back to all points
            parts = ax.violinplot(violin_data, positions=violin_positions, 
                                 showmeans=True, showmedians=True,
                                 widths=0.7)
        
            # Color the violin bodies
            color_idx = 0
            for pc in parts['bodies']:
                pc.set_facecolor(colors[color_idx + 1])  # Skip first color for original
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.2)
                color_idx += 1
            
            # Style medians
            if 'cmedians' in parts:
                parts['cmedians'].set_color('darkblue')
                parts['cmedians'].set_linewidth(2.5)
            
            # Style means
            if 'cmeans' in parts:
                parts['cmeans'].set_color('red')
                parts['cmeans'].set_linewidth(2.5)
                parts['cmeans'].set_linestyle('--')
            
            # Style other elements
            for key in ['cbars', 'cmaxes', 'cmins']:
                if key in parts:
                    parts[key].set_color('gray')
                    parts[key].set_linewidth(1.2)
        except Exception as e:
            logger.warning(f"Failed to create violin plot: {e}. Falling back to points only.")
            # If violin plot fails, convert all violin data to points
            for vpos, vdata in enumerate(violin_data):
                actual_pos = violin_positions[vpos] if vpos < len(violin_positions) else vpos + 1
                if actual_pos not in point_positions:  # Avoid duplicates
                    point_positions.append(actual_pos)
                    point_values.append(np.mean(vdata))
                    point_labels.append(f'group_{vpos}')    # Plot points for original and single-value groups
    if point_positions:
        # Use different colors/markers for original vs single-point transformations
        for pos, val, label in zip(point_positions, point_values, point_labels):
            if label == 'original':
                ax.scatter([pos], [val], s=200, c='#FFD700', 
                          marker='*', edgecolors='black', linewidths=2, 
                          zorder=5, label='Original', alpha=0.9)
            else:
                # Single-point transformation group
                ax.scatter([pos], [val], s=150, c='lightgray', 
                          marker='o', edgecolors='black', linewidths=1.5, 
                          zorder=5, alpha=0.7)
    
    # Add statistical annotations
    annotation_counter = 0
    for i, (group_name, data, pos, is_orig) in enumerate(zip(group_labels, all_data, positions, is_original), 1):
        # Check if this group has variance
        has_variance = len(data) > 1 and len(np.unique(data)) > 1 and np.std(data) > 0
        
        if is_orig or not has_variance:
            # For original, single-point groups, or zero-variance groups, just show the value
            value = data[0] if len(data) == 1 else np.mean(data)
            if is_orig:
                stats_text = f'n=1\nW={value:.2e}'
                bg_color = 'gold'
            else:
                stats_text = f'n={len(data)}\nW={value:.2e}'
                bg_color = 'lightgray'
            ax.text(pos , value * 1.5, stats_text, 
                   ha='center', va='bottom', fontsize=14, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.7, edgecolor='black'))
        else:
            # Calculate statistics for transformations with distribution
            median_val = np.median(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Add text annotation with key statistics
            stats_text = f'n={len(data)}\nμ={mean_val:.2e}\nM={median_val:.2e}\nσ={std_val:.2e}'
            ax.text(pos, ax.get_ylim()[1] * 0.95, stats_text, 
                   ha='center', va='top', fontsize=14, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'))
            annotation_counter += 1
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(violin_labels, rotation=45, ha='right')
    
    # Labels and formatting
    ax.set_ylabel('Total Gravitational Work', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transformation Type', fontsize=14, fontweight='bold')
    # ax.set_title('Distribution of Gravitational Work Across Transformation Groups (Violin Plot)', 
    #             fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                  markeredgecolor='black', markersize=15, label='Original', linewidth=0),
        plt.Line2D([0], [0], color='darkblue', linewidth=2.5, label='Median'),
        plt.Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
             fancybox=True, shadow=True, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Transformation violin plot characterization saved to: {output_path}")


def load_graph_variant(graphs_dir, variant_name):
    """
    Load a graph variant pickle file.
    """
    possible_files = [
        os.path.join(graphs_dir, f'graph_{variant_name}.pkl'),
        os.path.join(graphs_dir, f'{variant_name}.pkl')
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            logger.info(f"Loading graph variant: {filepath}")
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    logger.warning(f"Graph file not found for variant: {variant_name}")
    return None


def get_graph_bounds(graph):
    """
    Get the bounding box of all nodes in the graph.
    """
    if len(graph.nodes()) == 0:
        return None
    
    lons = []
    lats = []
    
    for node, data in graph.nodes(data=True):
        lon = data.get('x', data.get('lon'))
        lat = data.get('y', data.get('lat'))
        if lon is not None and lat is not None:
            lons.append(lon)
            lats.append(lat)
    
    if not lons or not lats:
        return None
    
    return {
        'min_lon': min(lons),
        'max_lon': max(lons),
        'min_lat': min(lats),
        'max_lat': max(lats)
    }


def plot_graph_frame(graph, ax, variant_name, bounds=None, node_size=0.5, edge_width=0.2):
    """
    Plot a single frame of the graph for animation.
    """
    ax.clear()
    
    if graph is None:
        ax.text(0.5, 0.5, f'Graph not found:\n{variant_name}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        return
    
    # Extract node coordinates
    pos = {}
    for node, data in graph.nodes(data=True):
        lon = data.get('x', data.get('lon'))
        lat = data.get('y', data.get('lat'))
        if lon is not None and lat is not None:
            pos[node] = (lon, lat)
    
    if not pos:
        ax.text(0.5, 0.5, f'No node coordinates:\n{variant_name}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        return
    
    # Draw the network
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#457B9D', 
                          width=edge_width, alpha=0.6)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='#E63946', 
                          node_size=node_size, alpha=0.8)
    
    # Add white map background if contextily is available
    if HAS_CONTEXTILY and bounds and len(pos) > 0:
        try:
            # Set bounds first for basemap
            ax.set_xlim(bounds['min_lon'], bounds['max_lon'])
            ax.set_ylim(bounds['min_lat'], bounds['max_lat'])
            
            # Add white/light basemap (CartoDB Positron is a clean white style)
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=0.7)
            logger.debug(f"Added white basemap for {variant_name}")
        except Exception as e:
            logger.warning(f"Could not add basemap for {variant_name}: {e}")
    
    # Set consistent bounds if provided
    if bounds:
        ax.set_xlim(bounds['min_lon'], bounds['max_lon'])
        ax.set_ylim(bounds['min_lat'], bounds['max_lat'])
    
    # Set title and styling
    ax.set_title(f'Graph Transformation: {variant_name}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_graph_transformation_gif(graphs_dir, results_df, out_gif, duration=1.5):
    """
    Create an animated GIF showing graph transformations.
    
    Parameters:
    - graphs_dir: Directory containing graph pickle files
    - results_df: DataFrame with variant information
    - out_gif: Output GIF file path
    - duration: Duration in seconds for each frame
    """
    logger.info(f"Creating graph transformation GIF: {out_gif}")
    
    # Get list of variants in logical transformation order
    variants = []
    
    # Start with original if available
    if 'original' in results_df['variant'].values:
        variants.append('original')
    
    # Define transformation order: translations → rotations → x scaling → y scaling
    transformation_order = []
    
    # 1. Add translations first
    translations = results_df[results_df['variant'].str.contains('translated', na=False)]['variant'].tolist()
    transformation_order.extend(sorted(translations))
    
    # 2. Add rotations second
    rotations = results_df[results_df['variant'].str.contains('rot_', na=False)]['variant'].tolist()
    transformation_order.extend(sorted(rotations))
    
    # 3. Add x scaling third
    x_scaling = results_df[results_df['variant'].str.contains('scale_x', na=False)]['variant'].tolist()
    transformation_order.extend(sorted(x_scaling))
    
    # 4. Add y scaling last
    y_scaling = results_df[results_df['variant'].str.contains('scale_y', na=False)]['variant'].tolist()
    transformation_order.extend(sorted(y_scaling))
    
    # Add any remaining variants not caught by the patterns above
    remaining = [v for v in results_df['variant'].tolist() 
                if v not in variants and v not in transformation_order]
    transformation_order.extend(sorted(remaining))
    
    variants.extend(transformation_order)
    
    logger.info(f"Animation sequence: {variants}")
    
    # Load all graphs and determine global bounds
    loaded_graphs = {}
    all_bounds = []
    
    for variant in variants:
        graph = load_graph_variant(graphs_dir, variant)
        loaded_graphs[variant] = graph
        
        if graph is not None:
            bounds = get_graph_bounds(graph)
            if bounds:
                all_bounds.append(bounds)
    
    # Calculate global bounds for consistent scaling
    if all_bounds:
        global_bounds = {
            'min_lon': min(b['min_lon'] for b in all_bounds),
            'max_lon': max(b['max_lon'] for b in all_bounds),
            'min_lat': min(b['min_lat'] for b in all_bounds),
            'max_lat': max(b['max_lat'] for b in all_bounds)
        }
        
        # Add some padding
        lon_range = global_bounds['max_lon'] - global_bounds['min_lon']
        lat_range = global_bounds['max_lat'] - global_bounds['min_lat']
        padding = max(lon_range, lat_range) * 0.05
        
        global_bounds['min_lon'] -= padding
        global_bounds['max_lon'] += padding
        global_bounds['min_lat'] -= padding
        global_bounds['max_lat'] += padding
    else:
        global_bounds = None
        logger.warning("Could not determine global bounds for animation")
    
    # Create the animation
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    def animate(frame):
        variant = variants[frame]
        graph = loaded_graphs[variant]
        
        # Determine node and edge sizes based on graph size
        if graph is not None:
            n_nodes = len(graph.nodes())
            n_edges = len(graph.edges())
            
            # Scale sizes inversely with graph complexity
            base_node_size = max(1.0, min(50.0, 5000.0 / max(n_nodes, 1)))
            base_edge_width = max(0.1, min(2.0, 500.0 / max(n_edges, 1)))
        else:
            base_node_size = 1.0
            base_edge_width = 0.2
        
        plot_graph_frame(graph, ax, variant, global_bounds, 
                        node_size=base_node_size, edge_width=base_edge_width)
        
        # Add progress indicator
        progress_text = f"Frame {frame + 1}/{len(variants)}"
        ax.text(0.02, 0.98, progress_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax.artists + ax.collections + ax.patches + ax.texts
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(variants), 
                                 interval=int(duration * 1000), repeat=True, blit=False)
    
    # Save as GIF
    try:
        logger.info("Saving animation (this may take a while)...")
        anim.save(out_gif, writer='pillow', fps=1/duration, dpi=100)
        plt.close()
        logger.info(f"Successfully saved graph transformation GIF → {out_gif}")
        
        # Log file size
        file_size = os.path.getsize(out_gif) / (1024 * 1024)  # MB
        logger.info(f"GIF file size: {file_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save GIF: {e}")
        plt.close()
        
        # Try to save individual frames instead
        frames_dir = out_gif.replace('.gif', '_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        logger.info(f"Saving individual frames to: {frames_dir}")
        for i, variant in enumerate(variants):
            fig, ax = plt.subplots(figsize=(12, 10))
            graph = loaded_graphs[variant]
            
            if graph is not None:
                n_nodes = len(graph.nodes())
                n_edges = len(graph.edges())
                base_node_size = max(1.0, min(50.0, 5000.0 / max(n_nodes, 1)))
                base_edge_width = max(0.1, min(2.0, 500.0 / max(n_edges, 1)))
            else:
                base_node_size = 1.0
                base_edge_width = 0.2
            
            plot_graph_frame(graph, ax, variant, global_bounds,
                           node_size=base_node_size, edge_width=base_edge_width)
            
            frame_file = os.path.join(frames_dir, f'frame_{i:03d}_{variant}.png')
            plt.savefig(frame_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved {len(variants)} individual frames")


def create_transformation_comparison_gif(graphs_dir, results_df, out_gif, duration=2.0):
    """
    Create a comparison GIF showing original vs transformed graphs side by side.
    """
    logger.info(f"Creating transformation comparison GIF: {out_gif}")
    
    # Load original graph
    original_graph = load_graph_variant(graphs_dir, 'original')
    if original_graph is None:
        logger.error("Original graph not found - cannot create comparison GIF")
        return
    
    # Get other variants
    other_variants = results_df[results_df['variant'] != 'original']['variant'].tolist()
    if not other_variants:
        logger.warning("No transformation variants found for comparison")
        return
    
    # Load all variants
    loaded_graphs = {'original': original_graph}
    for variant in other_variants:
        graph = load_graph_variant(graphs_dir, variant)
        if graph is not None:
            loaded_graphs[variant] = graph
    
    # Determine bounds for both original and transformed graphs
    original_bounds = get_graph_bounds(original_graph)
    
    # Create the animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    def animate_comparison(frame):
        if frame >= len(other_variants):
            return []
        
        variant = other_variants[frame]
        transformed_graph = loaded_graphs.get(variant)
        
        # Plot original on left
        if original_graph is not None:
            n_nodes = len(original_graph.nodes())
            n_edges = len(original_graph.edges())
            node_size = max(1.0, min(50.0, 5000.0 / max(n_nodes, 1)))
            edge_width = max(0.1, min(2.0, 500.0 / max(n_edges, 1)))
            
            plot_graph_frame(original_graph, ax1, 'Original Network', 
                           original_bounds, node_size, edge_width)
        
        # Plot transformed on right
        if transformed_graph is not None:
            n_nodes = len(transformed_graph.nodes())
            n_edges = len(transformed_graph.edges())
            node_size = max(1.0, min(50.0, 5000.0 / max(n_nodes, 1)))
            edge_width = max(0.1, min(2.0, 500.0 / max(n_edges, 1)))
            
            transformed_bounds = get_graph_bounds(transformed_graph)
            plot_graph_frame(transformed_graph, ax2, f'Transformed: {variant}', 
                           transformed_bounds, node_size, edge_width)
        else:
            ax2.clear()
            ax2.text(0.5, 0.5, f'Graph not found:\n{variant}', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        
        # Add overall title
        fig.suptitle(f'Graph Transformation Comparison - Step {frame + 1}/{len(other_variants)}', 
                    fontsize=18, fontweight='bold')
        
        return ax1.artists + ax1.collections + ax1.patches + ax1.texts + \
               ax2.artists + ax2.collections + ax2.patches + ax2.texts
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate_comparison, frames=len(other_variants), 
                                 interval=int(duration * 1000), repeat=True, blit=False)
    
    # Save as GIF
    try:
        logger.info("Saving comparison animation (this may take a while)...")
        anim.save(out_gif, writer='pillow', fps=1/duration, dpi=100)
        plt.close()
        logger.info(f"Successfully saved transformation comparison GIF → {out_gif}")
        
        file_size = os.path.getsize(out_gif) / (1024 * 1024)  # MB
        logger.info(f"GIF file size: {file_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save comparison GIF: {e}")
        plt.close()


def main():
    """
    Main function to generate all plots from saved CSV data.
    """
    p = argparse.ArgumentParser(description="Generate gravitational work plots from CSV results")
    p.add_argument('-c', '--conf', default='tools/conf/conf_wheight.json', 
                   help='Configuration file path')
    p.add_argument('-city', '--city', help='City name to override conf file')
    p.add_argument('--results-csv', help='Path to results CSV file (optional - will auto-detect)')
    p.add_argument('--stats-csv', help='Path to stats CSV file (optional - will auto-detect)')
    p.add_argument('--create-gif', action='store_true', 
                   help='Create animated GIF showing graph transformations')
    p.add_argument('--gif-duration', type=float, default=1.5,
                   help='Duration in seconds for each frame in the GIF (default: 1.5)')
    p.add_argument('--create-comparison-gif', action='store_true',
                   help='Create side-by-side comparison GIF of original vs transformed graphs')
    args = p.parse_args()
    
    # Load configuration
    conf = json.load(open(args.conf))
    
    city = args.city if args.city else conf.get("city")
    if not city:
        logger.error("City not specified in command line or config file")
        sys.exit(1)
    else:
        logger.info(f"Using city: {city}")
    
    # Set up directory paths
    base_dir = os.path.join(os.environ['WORKSPACE'], 'topolity', 'data', 'data_processed')
    graphs_dir = os.path.join(base_dir, city, "graphs")
    cells_dir = os.path.join(base_dir, city, f"{city}_basic_model/1000_cells")
    
    # Create images directory for plots
    images_dir = os.path.join(graphs_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    logger.info(f"Created images directory: {images_dir}")
    
    # Auto-detect CSV files if not provided
    results_csv = args.results_csv or os.path.join(graphs_dir, 'gravitational_work_by_variant.csv')
    stats_csv = args.stats_csv or os.path.join(graphs_dir, 'variant_stats.csv')
    
    # Check if required files exist
    if not os.path.exists(results_csv):
        logger.error(f"Results CSV file not found: {results_csv}")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading results from: {results_csv}")
    df = pd.read_csv(results_csv)
    
    # Remove duplicate variants with identical work values
    df = remove_duplicate_variants(df)
    
    stats = None
    if os.path.exists(stats_csv):
        logger.info(f"Loading stats from: {stats_csv}")
        stats = pd.read_csv(stats_csv)
    else:
        logger.warning(f"Stats CSV file not found: {stats_csv}")
        logger.warning("Parabolic plots may not work without stats data")
    
    # Check if we have too many variants (threshold: more than 15 variants)
    use_grouping = len(df) > 10
    
    if use_grouping:
        logger.info(f"Large number of variants ({len(df)}), using grouped visualization")
        
        # Add total_work column for grouping (sum of work_wd and work_hol)
        df_with_total = df.copy()
        df_with_total['total_work'] = df_with_total['work_wd'] + df_with_total['work_hol']
        
        # Group variants by transformation type
        groups = group_variants_by_transformation_type(df_with_total, min_scale_factor=1.0)
        
        # Generate grouped plots
        grouped_comparison_png = os.path.join(images_dir, 'grouped_transformation_comparison.png')
        plot_grouped_transformation_comparison(groups, grouped_comparison_png)
        
        # Generate boxplot characterization
        boxplot_png = os.path.join(images_dir, 'transformation_boxplots.png')
        plot_transformation_boxplots(groups, boxplot_png)
        
        # Generate violin plot characterization
        violinplot_png = os.path.join(images_dir, 'transformation_violinplots.png')
        plot_transformation_violinplots(groups, violinplot_png)
        
        # Create individual plots for each transformation group (excluding 'original')
        orig_df = groups.get('original', pd.DataFrame())
        
        for group_name, group_df in groups.items():
            if group_name != 'original' and len(group_df) > 0:  # Skip original group, process all others
                logger.info(f"Creating individual plots for {group_name} group ({len(group_df)} variants)")
                
                # Remove total_work column before passing to existing plot functions
                plot_df = group_df.drop(columns=['total_work'])
                
                # Create work by variant plot for this group (including original)
                if len(orig_df) > 0:
                    combined_plot_df = pd.concat([orig_df.drop(columns=['total_work']), plot_df]).reset_index(drop=True)
                    group_work_png = os.path.join(images_dir, f'work_by_variant_{group_name}.png')
                    plot_work_by_variant(combined_plot_df, group_work_png)
                    
                    # Create differences plot for this group vs original
                    group_diff_png = os.path.join(images_dir, f'variant_differences_{group_name}.png')
                    plot_variant_differences(combined_plot_df, group_diff_png)
                else:
                    # If no original, just plot the group alone
                    group_work_png = os.path.join(images_dir, f'work_by_variant_{group_name}.png')
                    plot_work_by_variant(plot_df, group_work_png)
                    logger.warning(f"No original variant found for comparison with {group_name}")
    else:
        logger.info(f"Moderate number of variants ({len(df)}), using standard visualization")
        
        # Generate basic plots
        work_png = os.path.join(images_dir, 'work_by_variant.png')
        plot_work_by_variant(df, work_png)

        diff_png = os.path.join(images_dir, 'variant_differences.png')
        plot_variant_differences(df, diff_png)
    
    # Generate additional comparison plots
    if len(df) > 1:  # Only if we have multiple variants
        absolute_diff_png = os.path.join(images_dir, 'absolute_work_differences.png')
        plot_absolute_work_differences(df, absolute_diff_png)
        
        combined_work_png = os.path.join(images_dir, 'combined_work_comparison.png')
        plot_combined_work_comparison(df, combined_work_png)
        
        efficiency_png = os.path.join(images_dir, 'work_efficiency_comparison.png')
        plot_work_efficiency_comparison(df, efficiency_png)

    # Check available variant types
    available_types = check_available_variant_types(df)
    
    if not available_types['original']:
        logger.warning("No 'original' variant found - parabolic plots may not work correctly")
    
    # Generate parabolic plots if stats are available
    if stats is not None:
        if available_types['translations']:
            translations_png = os.path.join(images_dir, 'work_parabolic_translations.png')
            plot_work_parabolic_translations(df, stats, translations_png)
            
            translations_avg_png = os.path.join(images_dir, 'work_parabolic_translations_average.png')
            plot_work_parabolic_translations_average(df, stats, translations_avg_png)
        else:
            logger.info("No translation variants found - skipping translation plot")
        
        if available_types['rotations']:
            rotations_png = os.path.join(images_dir, 'work_parabolic_rotations.png')
            plot_work_parabolic_rotations(df, stats, rotations_png)
        else:
            logger.info("No rotation variants found - skipping rotation plot")
        
        if available_types['scaling']:
            scaling_png = os.path.join(images_dir, 'work_parabolic_scaling.png')
            plot_work_parabolic_scaling(df, stats, scaling_png)
        else:
            logger.info("No scaling variants found - skipping scaling plot")
    
    # Generate additional plots (these require additional data files)
    original_arc_work_csv = os.path.join(graphs_dir, 'arc_work_segments_original.csv')
    if os.path.exists(original_arc_work_csv):
        aw_png = os.path.join(images_dir, 'arc_work_distribution.png')
        plot_arc_work_distribution(original_arc_work_csv, aw_png)

        aw_box_png = os.path.join(images_dir, 'arc_work_boxplot.png')
        plot_arc_work_boxplot(original_arc_work_csv, aw_box_png)
    else:
        logger.warning(f"Arc work segments CSV not found: {original_arc_work_csv}")
        logger.warning("Skipping arc work distribution plots")
    
    # DEM elevation distribution
    dem_file = os.path.join(base_dir, city, "dem", f"{city}_dem.tif")
    if os.path.exists(dem_file):
        dem_hist_png = os.path.join(images_dir, 'dem_elevation_distribution.png')
        plot_dem_elevation_distribution(dem_file, dem_hist_png)
    else:
        logger.warning(f"DEM file not found: {dem_file}")
        logger.warning("Skipping DEM elevation distribution plot")
    
    # Cell-node mapping plot
    cells_file = os.path.join(cells_dir, "cell_coordinates.csv")
    # Try both naming conventions for the original graph file
    original_graph_pkl = os.path.join(graphs_dir, 'graph_original.pkl')
    if not os.path.exists(original_graph_pkl):
        original_graph_pkl = os.path.join(graphs_dir, 'original.pkl')
    
    if os.path.exists(cells_file) and os.path.exists(original_graph_pkl):
        cnm_png = os.path.join(images_dir, 'cell_node_mapping.png')
        plot_cell_node_mapping(cells_file, original_graph_pkl, cnm_png)
    else:
        logger.warning("Missing files for cell-node mapping plot")
        logger.warning(f"Cells file: {cells_file} (exists: {os.path.exists(cells_file)})")
        logger.warning(f"Graph file: {original_graph_pkl} (exists: {os.path.exists(original_graph_pkl)})")
    
    # Generate animated GIFs if requested
    if args.create_gif and len(df) > 1:
        gif_path = os.path.join(images_dir, 'graph_transformation_animation.gif')
        create_graph_transformation_gif(graphs_dir, df, gif_path, args.gif_duration)
    elif args.create_gif:
        logger.warning("Cannot create GIF - need at least 2 variants")
    
    if args.create_comparison_gif and len(df) > 1 and 'original' in df['variant'].values:
        comparison_gif_path = os.path.join(images_dir, 'transformation_comparison_animation.gif')
        create_transformation_comparison_gif(graphs_dir, df, comparison_gif_path, args.gif_duration)
    elif args.create_comparison_gif:
        logger.warning("Cannot create comparison GIF - need original variant and at least one transformation")
    
    logger.info("Plot generation complete!")


if __name__ == '__main__':
    main()