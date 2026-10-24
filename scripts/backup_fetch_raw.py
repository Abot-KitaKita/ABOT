"""BACKUP raw fetch (owner B acting as A's backup, per E's Fri 3:30 PM status block).

Fetches into data/raw/: Mandaluyong boundary, barangay boundaries (with PSGC where OSM
tags it), health facilities and candidate public sites from OpenStreetMap, and probes
the CCHAIN Zenodo record. Everything records its OSM id for the QA log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abot import paths  # noqa: E402

UA = {"User-Agent": "abot-hackathon/1.0"}
REL = 2284209  # OSM relation: Mandaluyong


def overpass(query: str) -> dict:
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": query}, headers=UA, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    raw = paths.RAW
    raw.mkdir(parents=True, exist_ok=True)

    # 1. boundary polygon (Nominatim details, geojson)
    d = requests.get("https://nominatim.openstreetmap.org/details",
                     params={"osmtype": "R", "osmid": REL, "polygon_geojson": 1, "format": "json"},
                     headers=UA, timeout=60).json()
    geom = d["geometry"]
    (raw / "boundary.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": "Mandaluyong", "osm": f"R{REL}"}, "geometry": geom}]}))
    print("boundary saved")

    area = f'rel({REL});map_to_area->.a;'

    # 2. barangay boundaries (admin_level 10) with geometry for spatial join
    brgy = overpass(f"[out:json][timeout:120];{area}"
                    '(relation["admin_level"="10"]["boundary"="administrative"](area.a););out geom;')
    (raw / "barangays.json").write_text(json.dumps(brgy))
    tagged = sum(1 for e in brgy["elements"] if e.get("tags", {}).get("ref:psgc"))
    print(f"barangays: {len(brgy['elements'])} relations, {tagged} with ref:psgc")

    # 3. health facilities
    fac = overpass(f"[out:json][timeout:120];{area}"
                   '(node["amenity"~"^(hospital|clinic|doctors)$"](area.a);'
                   'way["amenity"~"^(hospital|clinic|doctors)$"](area.a);'
                   'node["healthcare"~"^(hospital|clinic|doctor)$"](area.a);'
                   'way["healthcare"~"^(hospital|clinic|doctor)$"](area.a););out center tags;')
    (raw / "facilities_osm.json").write_text(json.dumps(fac))
    print(f"facility candidates from OSM: {len(fac['elements'])}")

    # 4. candidate public sites (barangay halls, schools, community centres, courts)
    cand = overpass(f"[out:json][timeout:180];{area}"
                    '(node["amenity"="barangay_hall"](area.a);way["amenity"="barangay_hall"](area.a);'
                    'node["amenity"="school"](area.a);way["amenity"="school"](area.a);'
                    'node["amenity"="community_centre"](area.a);way["amenity"="community_centre"](area.a);'
                    'way["leisure"="pitch"]["sport"="basketball"](area.a);'
                    'node["leisure"="pitch"]["sport"="basketball"](area.a););out center tags;')
    (raw / "candidates_osm.json").write_text(json.dumps(cand))
    print(f"candidate sites from OSM: {len(cand['elements'])}")

    # 5. CCHAIN Zenodo probe (file list only; join if small enough)
    try:
        z = requests.get("https://zenodo.org/api/records/15251382", headers=UA, timeout=60).json()
        files = [(f["key"], f["size"] // 1_000_000) for f in z.get("files", [])]
        print("CCHAIN Zenodo files (key, MB):", files[:20])
    except Exception as exc:  # noqa: BLE001 - probe must not kill the fetch
        print(f"CCHAIN probe failed: {exc}")


if __name__ == "__main__":
    main()
