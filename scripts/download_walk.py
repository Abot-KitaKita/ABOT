"""Download the OSM walking network for Mandaluyong (owner B).

Run:  .venv/bin/python scripts/download_walk.py
Output: data/raw/walk.graphml  (not committed; large file, shared drive)

Uses the Nominatim geocoder for the city boundary, then downloads the walkable
street graph and saves it. Test it afterwards with a route:
    .venv/bin/python -c "import osmnx as ox; G=ox.load_graphml('data/raw/walk.graphml'); ..."
"""
from __future__ import annotations

import sys
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abot import paths  # noqa: E402

PLACE = "Mandaluyong, Metro Manila, Philippines"
OUT = paths.RAW / "walk.graphml"


def main() -> None:
    paths.RAW.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    print(f"Downloading walking network for {PLACE} ...")
    G = ox.graph_from_place(PLACE, network_type="walk", which_result=1)
    nodes, edges = len(G.nodes), len(G.edges)
    print(f"Graph: {nodes} nodes, {edges} edges; CRS {G.graph.get('crs')}")
    ox.save_graphml(G, OUT)
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
