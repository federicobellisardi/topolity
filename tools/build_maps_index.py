#!/usr/bin/env python3
"""
Build per-city HTML index pages that link to (or embed) all generated Folium maps.

Usage:
  python build_maps_index.py --base_path /data/workspaces/fbellisardi/data_processed --city madrid
  python build_maps_index.py --base_path /data/workspaces/fbellisardi/data_processed --iframes

By default, it creates a lightweight links index at graphs/maps_index.html for each city.
With --iframes it embeds previews (heavier to load) in graphs/maps_index_iframes.html.
"""
import os
import argparse
import glob
import html
from datetime import datetime

SECTION_ORDER = (
    'city_bbox_landed',
    'city_bbox',
    'graph_original_map',
    'graph_translated_',
    'graph_rot_',
    'graph_scale_x_',
    'graph_scale_y_',
)


def sort_key(fn: str):
    name = os.path.basename(fn)
    # Prioritize by known prefixes, then natural name
    for i, prefix in enumerate(SECTION_ORDER):
        if name.startswith(prefix):
            return (i, name)
    return (len(SECTION_ORDER), name)


def build_links_index(city: str, graphs_dir: str, files: list[str]) -> str:
    items = '\n'.join(
        f'<li><a href="{html.escape(os.path.basename(f))}" target="_blank">{html.escape(os.path.basename(f))}</a></li>'
        for f in files
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(city)} — Maps Index</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; }}
    h1 {{ margin: 0 0 12px; font-size: 22px; }}
    .meta {{ color: #666; margin-bottom: 16px; }}
    ul {{ line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>{html.escape(city)} — Maps Index</h1>
  <div class="meta">Generated {html.escape(datetime.now().isoformat(timespec='seconds'))} • {len(files)} files</div>
  <ul>
    {items}
  </ul>
</body>
</html>
"""


essential_css = """
  body { font-family: system-ui, sans-serif; margin: 16px; }
  h1 { margin: 0 0 12px; font-size: 22px; }
  .meta { color: #666; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
  .card { border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
  .card h3 { margin: 0; font-size: 14px; font-weight: 600; padding: 8px 10px; border-bottom: 1px solid #eee; background: #fafafa; }
  iframe { width: 100%; height: 340px; border: 0; }
"""


def build_iframes_index(city: str, graphs_dir: str, files: list[str]) -> str:
    cards = '\n'.join(
        f'<div class="card"><h3>{html.escape(os.path.basename(f))}</h3>'
        f'<iframe loading="lazy" src="{html.escape(os.path.basename(f))}"></iframe></div>'
        for f in files
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(city)} — Maps Gallery</title>
  <style>{essential_css}</style>
</head>
<body>
  <h1>{html.escape(city)} — Maps Gallery</h1>
  <div class="meta">Generated {html.escape(datetime.now().isoformat(timespec='seconds'))} • {len(files)} files</div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def process_city(base_path: str, city: str, iframes: bool) -> None:
    city_dir = os.path.join(base_path, city)
    graphs_dir = os.path.join(city_dir, 'graphs')
    if not os.path.isdir(graphs_dir):
        return
    # collect html map files
    htmls = sorted(
        glob.glob(os.path.join(graphs_dir, 'graph_*_map.html'))
        + glob.glob(os.path.join(graphs_dir, f'{city}_city_bbox*.html')),
        key=sort_key
    )
    if not htmls:
        return
    if iframes:
        out = build_iframes_index(city, graphs_dir, htmls)
        out_path = os.path.join(graphs_dir, 'maps_index_iframes.html')
    else:
        out = build_links_index(city, graphs_dir, htmls)
        out_path = os.path.join(graphs_dir, 'maps_index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f"Written {out_path} ({len(htmls)} items)")


def main():
    parser = argparse.ArgumentParser(description='Build per-city maps index HTML')
    parser.add_argument('--base_path', type=str, default='/data/workspaces/fbellisardi/data_processed',
                        help='Base path containing city folders')
    parser.add_argument('--city', type=str, default=None,
                        help='Process only this city (folder name)')
    parser.add_argument('--iframes', action='store_true',
                        help='Embed HTML maps as iframes (heavier to load)')
    args = parser.parse_args()

    if args.city:
        process_city(args.base_path, args.city, args.iframes)
    else:
        for name in sorted(os.listdir(args.base_path)):
            if os.path.isdir(os.path.join(args.base_path, name)):
                process_city(args.base_path, name, args.iframes)


if __name__ == '__main__':
    main()
