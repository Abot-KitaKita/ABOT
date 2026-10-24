"""ABOT planner app (owner C): Streamlit + Folium.

Run:  streamlit run app.py
Synthetic development data:  ABOT_DATA=fixtures streamlit run app.py
Teammates' functions that are not implemented yet show a notice instead of a number.
Layout: a header with the Plan settings pop-up and the
View menu, then one view. The Map view is one component (map, floating cards and results panel), so the site chips,
the list and the markers can work together. Screen text is plain language; the downloaded brief keeps the formal wording.
Styles: app.css for the page, MAP_CSS for the inside of the map component.
"""
from __future__ import annotations

import html
import json
import math
import re

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import box

from abot import config, explain, io, need, paths, place, service_type
from abot.brief import (LIMITATIONS, distinct_names, fmt_num, fmt_range, kind_text, make_brief, placement_note,
                        site_details)
from abot.contracts import CONDITIONS, SCENARIOS, SERVICE_LABELS

SERVICE_PENDING = "Service type pending (rule not merged yet)"
SCENARIO_LABEL = {"low": "Low", "mid": "Middle", "high": "High"}
COND_LABEL = {"hypertension": "High blood pressure", "diabetes": "Diabetes"}
# Formal names, for the downloaded brief.
METHOD_LABEL = {
    "abot": "ABOT (equity-weighted covering)",
    "most_populous_barangay": "Most populous barangays",
    "nearest_city_hall": "Nearest City Hall",
    "random": "Random sites (mean of draws)",
}
SCREEN_METHOD = {"abot": "ABOT", "most_populous_barangay": "Biggest barangays first",
                 "nearest_city_hall": "Closest to City Hall", "random": "Random places"}
VIEWS = {"map": ("Map", "Who lives far from care, and where to add spots", ":material/map:"),
         "compare": ("Compare", "ABOT next to simpler ways of picking spots", ":material/bar_chart:"),
         "brief": ("Placement brief", "A short write-up you can download", ":material/description:"),
         "assume": ("Assumptions & sources", "What we assumed and where the data came from", ":material/info:")}
# Wealth groups in plain words. They rank areas inside Mandaluyong only, and the city's range is narrow
# (docs/qa_log.md), so group 1 is "lowest in the city", not "poor".
PLAIN_WEALTH = {1: "In the city's lowest wealth group (1 of 5).", 2: "In the city's second-lowest wealth group (2 of 5).",
                3: "In the city's middle wealth group (3 of 5).", 4: "In the city's second-highest wealth group (4 of 5).",
                5: "In the city's highest wealth group (5 of 5)."}
SITE_NOTE = "Suggested public places only: not yet checked for permission, space or readiness."

st.set_page_config(page_title="ABOT | Chronic-care access planner", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(paths.ROOT / 'app.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def pending(fn, *args):
    """Call a teammate's function; None while its owner has not implemented it yet."""
    try:
        return fn(*args)
    except NotImplementedError:
        return None


@st.cache_data
def load():
    cfg = config.load()
    d = {k: io.read(k) for k in ("grid", "grid_cells", "facilities", "candidates",
                                 "reach_facilities", "reach_candidates")}
    d["grid"]["cell_id"] = d["grid"]["cell_id"].astype(str)
    return cfg, d


@st.cache_data(show_spinner="Updating plan…")
def run_model(condition: str, speed: float, lam: float, counted: tuple[str, ...], k: int) -> dict:
    """Unmet need and greedy picks for every prevalence scenario under the plan settings."""
    cfg, d = load()
    limit = config.walk_limit_m(cfg, speed_kmh=speed)
    covered = io.covered_cells(d["reach_facilities"], counted, limit)
    g = d["grid"].set_index("cell_id")
    weight = need.equity_weight(g["wealth_quintile"], lam)
    poorest2 = (pd.to_numeric(g["wealth_quintile"], errors="coerce") <= 2).fillna(False)
    reach = io.reach_sets(d["reach_candidates"], "cand_id", limit)
    share = {s: config.prevalence(cfg, condition, s) for s in SCENARIOS}
    out = {"covered": covered, "limit_m": limit, "scale": {s: share[s] / share["mid"] for s in SCENARIOS}}
    for s in SCENARIOS:
        n = need.need(d["grid"], share[s])
        out[s] = {"need": n, "unmet": need.unmet(n, covered)}
    # One prevalence rate scales every cell equally, so picks are the same in every scenario and
    # gains scale by the rate. Running greedy once avoids float-tie flips between scenario runs.
    out["picks"] = place.greedy(out["mid"]["unmet"], weight, reach, k, poorest2)
    # Why fewer than k sites: does any unchosen candidate still reach a cell with unmet need?
    picked = {p["cand_id"] for p in out["picks"]}
    taken = covered.union(*(reach[c] for c in picked))
    u = out["mid"]["unmet"]
    left = set(u[(u > 0) & ~u.index.isin(taken)].index)
    out["more_possible"] = any(cells & left for c, cells in reach.items() if c not in picked)
    out["reach"] = reach
    return out


@st.cache_data
def city_outline() -> tuple[str, str, str]:
    """The city's outline, the area around it (both GeoJSON, EPSG:4326) and a layer label.

    Uses data/processed/city_boundary.geojson (OpenStreetMap's administrative boundary) when present. Otherwise,
    e.g. in fixtures mode, it traces the union of the 100 m grid cells, which can stray by a few hundred metres.
    """
    cfg, d = load()
    path = paths.PROCESSED / "city_boundary.geojson"
    if path.exists():
        city, label = gpd.read_file(path).to_crs("EPSG:4326").geometry, "City boundary (OpenStreetMap)"
    else:
        cells = d["grid_cells"].to_crs("EPSG:32651")
        shape = cells.union_all().buffer(1).simplify(60).buffer(-1)
        city = gpd.GeoSeries([shape], crs="EPSG:32651").to_crs("EPSG:4326")
        label = "City outline (traced from the 100 m grid)"
    x0, y0, x1, y1 = city.total_bounds
    outside = gpd.GeoSeries([box(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3).difference(city.union_all())], crs="EPSG:4326")
    return city.to_json(), outside.to_json(), label


@st.cache_data
def baseline_picks(k: int) -> dict:
    """Sites chosen by the three simple rules (abot.place). They depend only on k, not on the plan settings."""
    cfg, d = load()
    grid, cand, rc = d["grid"], d["candidates"], d["reach_candidates"]
    out = {"most_populous_barangay": place.baseline_most_populous_barangay(grid, cand, k),
           "random": place.baseline_random(rc["cand_id"].astype(str).unique().tolist(), k,
                                           int(cfg["random_draws"]), int(cfg["seed"]))}
    hall = cfg.get("city_hall") or {}
    if hall.get("lon") is not None and hall.get("lat") is not None:
        # B's own distance source (cached OSM walk, else B's fallback), so this rule always matches baselines.csv.
        walk, _ = need._city_hall_walk(cand, cfg)
        out["nearest_city_hall"] = place.baseline_nearest_city_hall(walk, k)
    return out


@st.cache_data
def data_sources() -> list[dict]:
    """docs/data_sources.csv as plain rows; a missing value stays empty, never "nan".

    Rows whose use in ABOT is "Not used" stay in the CSV as a record but are not shown or listed in the brief.
    """
    src = pd.read_csv(paths.DOCS / "data_sources.csv", dtype=str).fillna("")
    src = src[~src["use_in_abot"].str.strip().str.lower().str.startswith("not used")]
    return src.to_dict("records")


def service_reason(label: str, tercile: int | None, referral_min) -> str:
    """Why A's rule (abot/service_type.py) gave this label, using the same barangay network tercile."""
    mobile, teleconsult, needs_network = SERVICE_LABELS[:3]
    if label == needs_network:
        return "We have no phone-signal data for this barangay."
    if label == mobile and tercile == 1:
        return "Phone signal here is among the weakest third in the city."
    if label == mobile and tercile is not None:
        return f"No clinic we count is within a {referral_min}-minute walk to send patients to."
    if label == teleconsult:
        return f"Phone signal is good enough, and a clinic we count is within a {referral_min}-minute walk."
    return ""


def plain_sentence(walk_min: float | None, quintile: int | None) -> str:
    """The screen's version of E's site sentence (abot.explain.sentence), from the same walk and wealth values."""
    parts = []
    if walk_min is not None:
        m_ = max(1, int(round(walk_min)))
        parts.append(f"The nearest clinic to send patients to is {explain.article(m_)} {m_}-minute walk away.")
    if quintile in PLAIN_WEALTH:
        parts.append(PLAIN_WEALTH[quintile])
    return " ".join(parts)


def plain_note(n_sites: int, k: int, beyond_high: float, more_possible: bool, still: tuple[float, float],
               walk_min: int) -> str | None:
    """The screen's version of brief.placement_note, under the same conditions."""
    if fmt_num(beyond_high) == "0":
        return "Everyone is already close to a clinic we count under these settings, so no new spot is needed."
    if n_sites >= k:
        return None
    if more_possible:
        return f"{n_sites} of {k} new care spots found. The search stopped before finding them all."
    # No candidate reaches anyone still far from care, so more spots would add no one.
    return (f"Only {n_sites} of the {k} spots help. The {fmt_range(*still)} people still far from care live where no "
            f"possible spot is within a {walk_min}-minute walk, so more spots would add no one.")


def nice_ticks(top: float) -> list[float]:
    """Zero-based axis ticks (about four steps of 1, 2 or 5 x 10^n) that cover `top`."""
    if top <= 0:
        return [0.0, 1.0]
    raw = top / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    return [i * step for i in range(int(math.ceil(top / step)) + 1)]


URL = re.compile(r"https?://[^\s)]+")


def source_link(text: str) -> tuple[str, str]:
    """(short label, url) from a config source string such as 'Resolve to Save Lives (2025). How to … https://…'."""
    urls = URL.findall(text or "")
    url = urls[0].rstrip(".,") if urls else ""
    cut = [i for i in (text.find(". "), text.find(", ")) if i > 0]
    return (text[:min(cut)] if cut else text).strip(), url


def evidence_type(source: str) -> str:
    s = (source or "").lower()
    if "sensitivity assumption" in s:
        return "Sensitivity assumption (±20% of the middle value)"
    if "uncertainty interval" in s:
        return "Source uncertainty bound (95% interval)"
    return "Published estimate"


def esc(x) -> str:
    return html.escape(str(x))


# ---- Inside the map component: its own document, so its own styles and script ----------------------------
ICON = {  # 24px line icons from the canvas
    "reset": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>', "minus": '<path d="M5 12h14"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 12.5l9 5 9-5"/><path d="M3 16.5l9 5 9-5"/>',
    "left": '<path d="M15 18l-6-6 6-6"/>', "right": '<path d="M9 18l6-6-6-6"/>',
    "pin": '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5h.01"/>',
    "range": '<path d="M4 12h16"/><path d="M4 8v8"/><path d="M20 8v8"/>',
    "alert": '<circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/>',
}


def svg(name: str, size: int = 20) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON[name]}</svg>')


PLUS_SVG = ('<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1v8M1 5h8" '
            'stroke="currentColor" stroke-width="2"/></svg>')

MAP_CSS = """
:root{--page:#DCE2E0;--panel:#EDF1EF;--card:#FFFFFF;--well:#F3F6F5;--text:#16201F;--text-2:#34433F;--muted:#5E6E6A;
--accent:#0F6E56;--accent-text:#0A5442;--rule:#E4EAE7;--track:#E4EAE7;--selected:#E3F1EC;--still:#7A8986;--fac:#2A64A8;
--map-bg:#E3E7E5;--glass:rgba(252,253,253,0.84);--glass-text:#16201F;--glass-muted:#4E5D59;--glass-line:rgba(22,32,31,0.12);
--seg-track:rgba(228,234,231,0.88);--seg-on:#FFFFFF;--seg-on-text:#16201F;
--seg-shadow:0 1px 2px rgba(22,32,31,0.10),0 4px 12px rgba(22,32,31,0.14);
--glass-shadow:inset 0 1px 0 #FFFFFF,inset 0 0 0 1px rgba(255,255,255,0.7),0 2px 4px rgba(22,32,31,0.08),0 16px 36px -10px rgba(22,32,31,0.32);
--map-shadow:0 0 0 1px rgba(255,255,255,0.7),0 1px 2px rgba(22,32,31,0.10),0 10px 20px -6px rgba(22,32,31,0.14),0 36px 72px -20px rgba(22,32,31,0.45);
--panel-shadow:inset 0 1px 0 #FFFFFF,inset 0 0 0 1px rgba(255,255,255,0.6),0 1px 2px rgba(22,32,31,0.08),0 8px 16px -4px rgba(22,32,31,0.10),0 32px 64px -20px rgba(22,32,31,0.38);
--hero-shadow:inset 0 1px 0 #FFFFFF,0 0 0 1px rgba(15,110,86,0.10),0 2px 4px rgba(22,32,31,0.06),0 22px 44px -14px rgba(15,110,86,0.38);
--card-shadow:inset 0 1px 0 #FFFFFF,0 0 0 1px rgba(22,32,31,0.05),0 2px 4px rgba(22,32,31,0.06),0 18px 36px -14px rgba(22,32,31,0.30);
--ring:0 0 0 4px rgba(15,110,86,0.35),0 0 0 9px rgba(15,110,86,0.15),0 4px 10px rgba(0,0,0,0.4);
--tiles:grayscale(1) contrast(0.95);--pw:420px}
html{background:var(--page)}
html,body.ws{height:100%;margin:0}
body.ws{box-sizing:border-box;display:grid;grid-template-columns:minmax(0,1fr) var(--pw);grid-template-rows:minmax(0,1fr);
grid-template-areas:"map panel";gap:16px;padding:16px;background:var(--page);color:var(--text);
font-family:"Funnel Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;-webkit-font-smoothing:antialiased}
@media (max-width:1600px){body.ws{--pw:380px}}
body.ws.rail{--pw:64px}
.ws *{box-sizing:border-box}
.ws button{font:inherit;color:inherit;cursor:pointer}
.ws :focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.num,.ws b{font-variant-numeric:tabular-nums}
.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.d{font-family:"Funnel Display","Funnel Sans",system-ui,sans-serif}
.folium-map{grid-area:map;border-radius:28px;box-shadow:var(--map-shadow);background:var(--map-bg)!important;isolation:isolate}
.leaflet-tile-pane{filter:var(--tiles)}
.leaflet-container{font-family:inherit}
.leaflet-container a:focus-visible,.leaflet-marker-icon:focus-visible,.leaflet-container input:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.leaflet-control-attribution{background:transparent!important;color:var(--glass-muted)!important;font-size:11px;margin:0 24px 6px 0!important}
.leaflet-control-attribution a{color:inherit!important}
.leaflet-control-layers{display:none;margin:0 76px 32px 0!important;border:0!important;border-radius:20px!important;padding:6px 10px!important;
background:var(--glass)!important;color:var(--glass-text);backdrop-filter:blur(14px);box-shadow:var(--glass-shadow)!important;font-size:13px}
.layers-open .leaflet-control-layers{display:block}
.leaflet-control-layers label{margin:4px 0}
.leaflet-popup-content-wrapper{border-radius:24px;box-shadow:0 2px 4px rgba(0,0,0,0.15),0 20px 48px rgba(0,0,0,0.35);background:#FFFFFF;color:#16201F}
.leaflet-popup-content{margin:16px 18px 14px;font-size:13px;line-height:1.4}
.leaflet-popup-tip{background:#FFFFFF}
.leaflet-container a.leaflet-popup-close-button{top:8px;right:10px}
.abot-pin{width:28px;height:28px;border-radius:8px;background:#0F6E56;color:#FFFFFF;border:2px solid #FFFFFF;display:flex;
align-items:center;justify-content:center;font:600 16px/1 "Funnel Display","Funnel Sans",sans-serif;box-shadow:0 3px 8px rgba(0,0,0,0.45)}
.is-on .abot-pin{box-shadow:var(--ring)}
.abot-fac{width:22px;height:22px;border-radius:50%;background:var(--fac);color:#FFFFFF;border:2px solid #FFFFFF;display:flex;
align-items:center;justify-content:center;box-shadow:0 2px 6px rgba(0,0,0,0.45)}
.abot-fac.off{background:#FFFFFF;color:var(--fac);border-color:var(--fac)}
.pop-head{display:flex;gap:10px;align-items:flex-start;padding-right:14px}
.pop-rank{flex:none;width:26px;height:26px;border-radius:8px;background:#0F6E56;color:#FFFFFF;display:flex;align-items:center;justify-content:center;font:600 15px/1 "Funnel Display","Funnel Sans",sans-serif}
.pop-head b{font-size:17px;line-height:1.25;font-weight:600}
.pop-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.pop-chips span{font-size:12px;padding:3px 10px;border-radius:999px;background:#F3F6F5;color:#34433F}
.pop-gain{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-top:12px}
.pop-gain b{font-size:22px;font-weight:700;color:#0F6E56}
.pop-gain span,.pop-sub{font-size:13px;color:#5E6E6A}
.pop-sub{display:block}
.pop-note{margin:8px 0 0;font-size:12px;color:#5E6E6A}
.ov{grid-area:map;position:relative;z-index:1;pointer-events:none;min-width:0;min-height:0}
.ov>*{pointer-events:auto;position:absolute}
.glass{background:var(--glass);color:var(--glass-text);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:var(--glass-shadow)}
.plan{left:20px;top:20px;min-width:288px;max-width:360px;padding:16px 18px;border-radius:22px}
.plan p{margin:0}
.plan .t{font-size:22px;line-height:1.1;font-weight:600;letter-spacing:-0.01em}
.plan .s{margin-top:4px;font-size:14px;color:var(--glass-muted)}
.plan dl{margin:12px 0 0;padding-top:12px;border-top:1px solid var(--glass-line);display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:13px;line-height:1.3}
.plan dt{color:var(--glass-muted)}.plan dd{margin:0;font-weight:600}
.chips{right:20px;top:20px;max-width:calc(100% - 360px);display:flex;flex-wrap:wrap;justify-content:flex-end;gap:2px;padding:4px;border-radius:22px;background:var(--seg-track);backdrop-filter:blur(14px);box-shadow:var(--glass-shadow)}
.chips button{height:34px;min-width:40px;padding:0 12px;border:0;border-radius:999px;background:transparent;color:var(--glass-text);font-size:14px;font-weight:600}
.chips button[aria-pressed="true"]{background:var(--seg-on);color:var(--seg-on-text);box-shadow:var(--seg-shadow)}
.tools{right:20px;bottom:32px;display:flex;flex-direction:column;gap:10px}
.tools button{width:44px;height:44px;border-radius:50%;border:0;padding:0;display:flex;align-items:center;justify-content:center}
.tools button[aria-expanded="true"]{background:var(--seg-on-text);color:var(--page)}
.legend{left:20px;bottom:20px;padding:12px 16px;border-radius:20px;display:flex;flex-direction:column;gap:8px;font-size:13px;line-height:1.25;max-width:calc(100% - 100px)}
.legend .h{font-weight:600}.legend .h span{font-weight:400;color:var(--glass-muted)}
.ramp{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.ramp i{display:flex;gap:1px;width:120px;height:8px;border-radius:999px;overflow:hidden}.ramp i span{flex:1}
.keys{display:flex;gap:16px;flex-wrap:wrap}.keys>span{display:flex;align-items:center;gap:6px}
.k-site{width:14px;height:14px;border-radius:4px;background:#0F6E56;box-shadow:0 0 0 1.5px #FFFFFF}
.k-fac{width:14px;height:14px;border-radius:50%;background:var(--fac);box-shadow:0 0 0 1.5px #FFFFFF}
.panel{grid-area:panel;min-height:0;overflow-y:auto;border-radius:28px;background:var(--panel);box-shadow:var(--panel-shadow);padding:20px}
.full{display:flex;flex-direction:column;gap:16px}
.sec{display:flex;flex-direction:column;gap:12px}
.head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 0 4px}
.head h2{margin:0;font-size:24px;line-height:1.15;font-weight:600;letter-spacing:-0.015em}
.round{flex:none;width:36px;height:36px;border-radius:50%;border:0;padding:0;background:var(--card);box-shadow:var(--card-shadow);display:flex;align-items:center;justify-content:center}
.hero{border-radius:24px;background:var(--card);box-shadow:var(--hero-shadow);padding:22px 22px 16px;display:flex;flex-direction:column;gap:20px}
.hero p{margin:0}
.big{font-size:40px;line-height:1;font-weight:700;letter-spacing:-0.02em;color:var(--accent)}
.big-sub{margin-top:6px!important;font-size:17px;color:var(--text-2)}
.today{display:flex;flex-direction:column;gap:8px}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.row .l{font-size:14px;color:var(--text-2)}.row b{font-size:16px;font-weight:600}
.split{display:flex;gap:3px;height:10px}
.split span:first-child{border-radius:999px 0 0 999px;background:var(--accent)}
.split span:last-child{border-radius:0 999px 999px 0;background:var(--still)}
.split span:only-child{border-radius:999px}
.split-key{display:flex;justify-content:space-between;gap:12px;font-size:13px;color:var(--text-2)}
.split-key span{display:flex;align-items:center;gap:6px}.split-key b{font-weight:600;color:var(--text)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}.dot.still{background:var(--still)}
.where{display:flex;gap:8px;font-size:14px;line-height:1.4;color:var(--text-2)}.where svg{flex:none;margin-top:2px}
.how summary{list-style:none;display:flex;align-items:center;justify-content:space-between;gap:8px;padding-top:14px;border-top:1px solid var(--well);font-size:12px;color:var(--muted);cursor:pointer}
.how summary::-webkit-details-marker{display:none}
.pill{display:inline-flex;align-items:center;gap:6px;height:32px;padding:0 12px;border-radius:999px;background:var(--well);font-size:13px;font-weight:600;color:var(--accent-text)}
.how summary:focus-visible{outline:0}.how summary:focus-visible .pill{outline:2px solid var(--accent);outline-offset:2px}
.defs{margin:12px 0 0;display:flex;flex-direction:column;gap:8px;font-size:13px;line-height:1.45;color:var(--text-2)}
.defs dt{font-weight:700;color:var(--text)}.defs dd{margin:0}
.found{font-size:13px;font-weight:600;color:var(--text-2);padding:4px 12px;border-radius:999px;background:var(--card);box-shadow:var(--card-shadow);white-space:nowrap}
.list{border-radius:24px;background:var(--card);box-shadow:var(--card-shadow);padding:6px 14px}
.list ol{list-style:none;margin:0;padding:0}
.site{margin:0 -8px;padding:0 8px;border-bottom:1px solid var(--rule)}
.site:last-child{border-bottom:0}
.site.is-on{background:var(--selected);border-radius:16px;border-bottom-color:transparent}
.site summary{list-style:none;display:grid;grid-template-columns:24px minmax(0,1fr) auto;column-gap:14px;align-items:start;padding:10px 0;cursor:pointer;border-radius:12px}
.site summary::-webkit-details-marker{display:none}
.rank{margin-top:-4px;font-size:30px;line-height:1;font-weight:600;color:var(--accent);text-align:center}
.who{min-width:0;display:flex;flex-direction:column;gap:2px}
.name{font-size:15px;line-height:1.3;font-weight:600;color:var(--text)}
.type{font-size:13px;color:var(--muted)}
.gain{display:flex;flex-direction:column;align-items:flex-end;gap:6px}
.gain b{font-size:15px;font-weight:700;color:var(--accent-text);white-space:nowrap}
.share{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
.share i{width:44px;height:4px;border-radius:999px;background:var(--track);overflow:hidden}
.share i span{display:block;height:4px;border-radius:999px;background:var(--accent)}
.more{padding:0 0 12px 38px;font-size:13px;line-height:1.45;color:var(--text-2);display:flex;flex-direction:column;gap:4px}
.note{margin:0 4px;font-size:13px;line-height:1.4;color:var(--text-2)}
.good{margin:0 4px;padding:0;list-style:none;display:flex;flex-direction:column;gap:6px;font-size:13px;line-height:1.4;color:var(--text-2)}
.good li{display:flex;gap:8px}.good svg{flex:none;margin-top:2px}
.railv{display:none;height:100%;flex-direction:column;align-items:center;gap:18px}
.rail .panel{padding:16px 10px}.rail .full{display:none}.rail .railv{display:flex}
.railv .round{width:44px;height:44px}
.railv p{margin:0;writing-mode:vertical-rl;transform:rotate(180deg);display:flex;align-items:center;gap:10px;font-size:15px;color:var(--text-2);white-space:nowrap}
.railv p b{font-size:22px;font-weight:700;color:var(--accent)}
.railv hr{width:32px;height:1px;margin:0;border:0;background:var(--rule)}
.railv ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.railv ol button{width:44px;height:44px;border-radius:14px;border:0;padding:0;background:var(--card);color:var(--accent);font-size:22px;font-weight:600}
.railv ol button[aria-pressed="true"]{background:var(--selected)}
@media (max-width:900px){
 html,body.ws{height:auto}
 body.ws,body.ws.rail{grid-template-columns:minmax(0,1fr);grid-template-rows:520px auto;grid-template-areas:"map" "panel"}
 .panel{overflow:visible}.to-rail{display:none}
 .rail .panel{padding:20px}.rail .full{display:flex}.rail .railv{display:none}
}
@media (max-width:700px){
 .plan{display:none}
 .chips{left:12px;right:12px;top:12px;max-width:none;justify-content:flex-start}
 .legend{left:12px;bottom:28px;padding:10px 12px;max-width:calc(100% - 84px)}
 .tools{right:12px;bottom:28px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""
MAP_DARK = """
:root{--page:#0B1110;--panel:#141D1C;--card:#1F2B29;--well:#16201F;--text:#F3F6F5;--text-2:#C9D3CF;--muted:#A7B5B1;
--accent:#8FD3BB;--accent-text:#8FD3BB;--rule:#2E3B38;--track:#16201F;--selected:#1F3A33;--still:#7A8986;--fac:#3B7DD8;
--map-bg:#1C2523;--glass:rgba(22,32,31,0.74);--glass-text:#FFFFFF;--glass-muted:#B9C5C1;--glass-line:rgba(255,255,255,0.12);
--seg-track:rgba(22,32,31,0.74);--seg-on:#F3F6F5;--seg-on-text:#16201F;--seg-shadow:0 2px 8px rgba(0,0,0,0.35);
--glass-shadow:inset 0 1px 0 rgba(255,255,255,0.14),inset 0 0 0 1px rgba(255,255,255,0.08),0 4px 10px rgba(0,0,0,0.3),0 18px 40px -10px rgba(0,0,0,0.6);
--map-shadow:0 0 0 1px rgba(255,255,255,0.06),0 2px 6px rgba(0,0,0,0.4),0 32px 70px -18px rgba(0,0,0,0.8);
--panel-shadow:inset 0 1px 0 rgba(255,255,255,0.07),inset 0 0 0 1px rgba(255,255,255,0.04),0 2px 6px rgba(0,0,0,0.35),0 28px 60px -16px rgba(0,0,0,0.7);
--hero-shadow:inset 0 1px 0 rgba(255,255,255,0.09),0 2px 4px rgba(0,0,0,0.3),0 20px 40px -12px rgba(0,0,0,0.6),0 12px 32px -14px rgba(143,211,187,0.25);
--card-shadow:inset 0 1px 0 rgba(255,255,255,0.07),0 2px 4px rgba(0,0,0,0.3),0 16px 32px -12px rgba(0,0,0,0.6);
--ring:0 0 0 4px rgba(143,211,187,0.55),0 0 0 9px rgba(143,211,187,0.22),0 4px 10px rgba(0,0,0,0.5);
--tiles:grayscale(1) invert(1) brightness(0.9) contrast(0.85)}
"""
# Links the chips, the list rows, the rail and the markers. ABOT = {map, pins, home} is written before this.
MAP_JS = """
document.addEventListener("DOMContentLoaded", function () {
  var map = window[ABOT.map], pins = ABOT.pins.map(function (n) { return window[n]; });
  var calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var home = L.latLngBounds(ABOT.home), panel = document.querySelector(".panel");
  var rows = Array.prototype.slice.call(document.querySelectorAll(".site")), current = 0;
  function q(s) { return Array.prototype.slice.call(document.querySelectorAll(s)); }
  // Fit inside the part of the map the floating cards leave clear. The plan card is cleared from above or from
  // the left, whichever leaves the larger view (tall maps: above; wide maps: left).
  function clear(b) {
    var mr = map.getContainer().getBoundingClientRect();
    function box(s) { var e = document.querySelector(s), r = e && e.getBoundingClientRect(); return r && r.width ? r : null; }
    var plan = box(".plan"), chips = box(".chips"), legend = box(".legend"), tools = box(".tools"), g = 16;
    var top = (chips ? chips.bottom - mr.top : 0) + g, right = tools ? mr.right - tools.left + g : 24;
    var low = legend ? mr.bottom - legend.top + g : 24;
    var opts = [{paddingTopLeft: [24, Math.max(top, plan ? plan.bottom - mr.top + g : 0)], paddingBottomRight: [right, low]}];
    if (plan || legend) {
      var left = Math.max(plan ? plan.right - mr.left : 0, legend ? legend.right - mr.left : 0) + g;
      opts.push({paddingTopLeft: [left, top], paddingBottomRight: [right, 24]});
    }
    // An option that leaves less than 160 px either way (the legend on a phone) is not a view; the fallback is plain.
    opts = opts.filter(function (o) {
      return mr.width - o.paddingTopLeft[0] - o.paddingBottomRight[0] >= 160 && mr.height - o.paddingTopLeft[1] - o.paddingBottomRight[1] >= 160;
    });
    if (!opts.length) opts = [{paddingTopLeft: [24, top], paddingBottomRight: [24, 24]}];
    function zoom(o) { return map.getBoundsZoom(b, false, L.point(o.paddingTopLeft).add(o.paddingBottomRight)); }
    var best = opts.reduce(function (a, o) { return zoom(o) > zoom(a) ? o : a; });
    best.maxZoom = 16;
    return best;
  }
  function fit(b) { var o = clear(b); if (calm) { map.fitBounds(b, o); } else { o.duration = 0.6; map.flyToBounds(b, o); } }
  function focus(n, bounds) {
    current = n;
    q("[data-focus]").forEach(function (b) { b.setAttribute("aria-pressed", String(+b.dataset.focus === n)); });
    pins.forEach(function (p, i) { var el = p.getElement(); if (el) el.classList.toggle("is-on", i + 1 === n); });
    rows.forEach(function (r) {
      var on = +r.dataset.rank === n;
      r.classList.toggle("is-on", on);
      r.querySelector("details").open = on;
      if (on && panel.scrollHeight > panel.clientHeight) {
        var pr = panel.getBoundingClientRect(), rr = r.getBoundingClientRect();
        if (rr.top < pr.top || rr.bottom > pr.bottom) panel.scrollTop += rr.top - pr.top - 96;
      }
    });
    if (!n) { map.closePopup(); fit(bounds || (pins.length ? L.featureGroup(pins).getBounds() : home)); return; }
    var p = pins[n - 1], ll = p.getLatLng(), z = Math.max(map.getZoom(), 15);
    if (calm || (map.getCenter().distanceTo(ll) < 5 && map.getZoom() === z)) { map.setView(ll, z, {animate: false}); p.openPopup(); }
    else { map.once("moveend", function () { p.openPopup(); }); map.flyTo(ll, z, {duration: 0.6}); }
  }
  pins.forEach(function (p, i) { p.on("click", function () { if (current !== i + 1) focus(i + 1); }); });
  q("[data-focus]").forEach(function (b) { b.addEventListener("click", function () { focus(+b.dataset.focus); }); });
  rows.forEach(function (r) {
    var d = r.querySelector("details");
    d.addEventListener("toggle", function () { if (d.open && current !== +r.dataset.rank) focus(+r.dataset.rank); });
  });
  document.getElementById("zin").onclick = function () { map.zoomIn(); };
  document.getElementById("zout").onclick = function () { map.zoomOut(); };
  document.getElementById("home").onclick = function () { focus(0, home); };
  var layers = document.getElementById("layers");
  layers.onclick = function () {
    var open = document.body.classList.toggle("layers-open");
    layers.setAttribute("aria-expanded", String(open));
  };
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && document.body.classList.contains("layers-open")) { layers.click(); layers.focus(); }
  });
  function rail(on) {
    document.body.classList.toggle("rail", on);
    q(".to-rail,.expand").forEach(function (b) { b.setAttribute("aria-expanded", String(!on)); });
    try { window.sessionStorage.setItem("abot-rail", on ? "1" : ""); } catch (e) {}
  }
  q(".to-rail").forEach(function (b) { b.onclick = function () { rail(true); document.querySelector(".expand").focus(); }; });
  q(".expand").forEach(function (b) { b.onclick = function () { rail(false); document.querySelector(".to-rail").focus(); }; });
  try { if (window.sessionStorage.getItem("abot-rail")) rail(true); } catch (e) {}
  // Shaded squares are clickable for their tooltip, so Chrome made each of them a tab stop (hundreds before the
  // markers). The results panel gives the same numbers, so they leave the tab order; squares redrawn on zoom too.
  var shapes = map.getPanes().overlayPane;
  function untab() { q(".leaflet-overlay-pane .leaflet-interactive").forEach(function (e) { e.setAttribute("tabindex", "-1"); }); }
  untab();
  // Markers are focusable but had no name ("DIV" to a screen reader): their tooltip text becomes the label.
  map.eachLayer(function (l) {
    var t = l instanceof L.Marker && l.getTooltip(), el = t && l.getElement();
    if (el) { var d = document.createElement("div"); d.innerHTML = t.getContent(); el.setAttribute("aria-label", d.textContent); el.setAttribute("role", "button"); }
  });
  new MutationObserver(untab).observe(shapes, {childList: true, subtree: true});
  // The map's box changes with the rail and the window; Leaflet alone only listens for window resizes.
  new ResizeObserver(function () { map.invalidateSize(); }).observe(map.getContainer());
  map.fitBounds(home, clear(home));
});
"""

# ---- Data and widget state ---------------------------------------------------------------------------------
try:
    cfg, d = load()
except FileNotFoundError as e:
    st.error(f"Data file missing: {e.filename}. Until owner A delivers it, run with ABOT_DATA=fixtures "
             "(synthetic data).")
    st.stop()
grid, fac, cand = d["grid"], d["facilities"], d["candidates"]
# Six candidate names are shared by several sites; tell them apart everywhere a site name is shown.
cand = cand.assign(name=distinct_names(cand["name"].astype(str).tolist(), cand.geometry.x.tolist(),
                                       cand.geometry.y.tolist()))
conditions = config.available_conditions(cfg)
default_counted = sorted(fac.loc[fac["counted"].astype(bool), "fac_id"].astype(str))
fac_names = {str(r.fac_id): f"{r.name} ({r.type})" for r in fac.itertuples()}
wealth_known = grid["wealth_quintile"].notna()
speeds = [float(cfg["walk_speed_kmh"]), float(cfg["alt_walk_speed_kmh"])]
walk_min, referral_min = cfg["walk_minutes"], cfg.get("referral_walk_minutes", 30)
# One checkbox per facility (key fac_<id>): readable full names and "untick one" in the demo.
PRESET = {"condition": "hypertension" if "hypertension" in conditions else (conditions or [None])[0],
          "scenario": "mid", "k": 5, "equity": True, "speed": speeds[0], "fac_search": "",
          **{f"fac_{f}": f in default_counted for f in fac["fac_id"].astype(str)}}
for key, value in PRESET.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("view", "map")
# The colour-mode pills can be clicked off; keep the last mode instead of none.
if st.session_state.get("mode") is None:
    st.session_state["mode"] = st.session_state.get("_mode", "light")
st.session_state["_mode"] = st.session_state["mode"]
# Streamlit drops the state of widgets that are not drawn; re-assigning keeps facilities hidden by the search.
for f in fac_names:
    st.session_state[f"fac_{f}"] = st.session_state[f"fac_{f}"]
dark = st.session_state["mode"] == "dark"
view = st.session_state["view"]

# ---- Header: Plan settings pop-up (left), ABOT (centre), status, View menu and colour mode (right) ----------
cond_now = st.session_state.condition
with st.container(key="appbar"):
    left, middle, right = st.columns([5, 2, 5], vertical_alignment="center")
with left:
    chip = f"{COND_LABEL.get(cond_now, 'No condition')} · {st.session_state.k} spot{'' if st.session_state.k == 1 else 's'}"
    with st.popover(f"Plan settings **{chip}**", icon=":material/tune:"):
        with st.container(key="planpanel"):
            st.markdown('<h2>Plan settings</h2><p class="abot-sub">The map and numbers update each time you change '
                        "something.</p>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                if conditions:
                    st.radio("Health condition", conditions, key="condition", horizontal=True,
                             format_func=lambda c: COND_LABEL.get(c, c.capitalize()))
                    st.caption("We use one rate for the whole city, so this changes the numbers, not which spots "
                               "are picked.")
                missing = [c for c in CONDITIONS if c not in conditions]
                if missing:
                    st.caption(f"Not available yet: {', '.join(COND_LABEL.get(c, c) for c in missing)} "
                               "(no cited rate yet).")
                with st.container(key="spots"):
                    st.markdown(f'<div class="abot-kbig"><span>New care spots</span><b>{st.session_state.k}</b></div>',
                                unsafe_allow_html=True)
                    st.slider("New care spots", 1, int(cfg["k_max"]), key="k", label_visibility="collapsed")
                    spots_note = st.empty()  # filled below when fewer spots help than were asked for
                st.toggle("Poorer areas first", key="equity", disabled=not wealth_known.any(),
                          help=f"Weight = 1 + λ × (5 − wealth group) / 4, with λ = {cfg['equity_lambda']:g}. Off: λ = 0.")
                # One block for the help line and, once the plan is known, whether the switch changes anything
                # (two separate blocks overlapped: Streamlit pulls each text block up under the one before).
                equity_help = ('<p class="abot-help">' + (
                    "Not available: the map has no wealth data." if not wealth_known.any() else
                    "Picks spots that help poorer areas more. Everyone is still counted the same.") + "</p>")
                equity_note = st.empty()
                equity_note.markdown(equity_help, unsafe_allow_html=True)
            with col_b:
                with st.container(key="assume"):
                    st.markdown('<p class="abot-group">What we assume</p>', unsafe_allow_html=True)
                    st.radio("How common the condition is", SCENARIOS, key="scenario", format_func=SCENARIO_LABEL.get,
                             horizontal=True)
                    st.caption("Changes the numbers and map colours, not which spots are picked.")
                    st.radio("Walking speed", speeds, key="speed", horizontal=True,
                             format_func=lambda v: f"{'Normal' if v == speeds[0] else 'Slower'} · {v:g} km/h")
                n_on = sum(st.session_state[f"fac_{f}"] for f in fac_names)
                with st.expander(f"Clinics we count ({n_on} of {len(fac)} listed)"):
                    st.caption("Tick the ones that already give care. Nobody has checked this list yet.")
                    q = st.text_input("Search clinics", key="fac_search", placeholder="Type part of a name").strip().lower()
                    shown = [(f, lab) for f, lab in sorted(fac_names.items(), key=lambda kv: kv[1].lower())
                             if q in lab.lower()]
                    if q:
                        st.caption(f"Showing {len(shown)} of {len(fac)}.")
                    with st.container(height=220, border=False):
                        if not shown:
                            st.caption("No clinic matches that search.")
                        for f, label in shown:
                            st.checkbox(label, key=f"fac_{f}")
                st.button("Back to the starting plan", on_click=lambda: st.session_state.update(PRESET),
                          use_container_width=True)
middle.markdown('<p class="abot-mark">ABOT</p>' + ('<span class="abot-dark" hidden></span>' if dark else ""),
                unsafe_allow_html=True)
with right:
    with st.container(key="appright"):
        slots = st.columns(3 if paths.USING_FIXTURES else 2, vertical_alignment="center")
    if paths.USING_FIXTURES:
        slots[0].markdown('<p class="abot-synth" role="status" title="Synthetic development data: not for reporting or '
                          'submission."><b>Synthetic data</b><span>&nbsp;· not for reporting</span></p>',
                          unsafe_allow_html=True)
    with slots[-2]:
        name_, _, icon_ = VIEWS[view]
        with st.popover(name_, icon=icon_):
            with st.container(key="viewmenu"):
                st.radio("View", list(VIEWS), key="view", format_func=lambda v: VIEWS[v][0],
                         captions=[VIEWS[v][1] for v in VIEWS], label_visibility="collapsed")
    with slots[-1], st.container(key="modepick"):
        st.segmented_control("Colour mode", ["light", "dark"], key="mode", label_visibility="collapsed",
                             format_func={"light": ":material/light_mode:", "dark": ":material/dark_mode:"}.get)

condition, scenario, k = st.session_state.condition, st.session_state.scenario, st.session_state.k
speed = float(st.session_state.speed)
lam = float(cfg["equity_lambda"]) if st.session_state.equity and wealth_known.any() else 0.0
counted = tuple(sorted(f for f in fac_names if st.session_state[f"fac_{f}"]))
cond_label = COND_LABEL.get(condition, str(condition).capitalize())

# ---- The plan: computed once; every view below reads it ---------------------------------------------------
model, unavailable = None, None
if not conditions:
    unavailable = "No condition has a cited rate for all three estimates yet (config.yaml, owner B)."
else:
    model = pending(run_model, condition, speed, lam, counted, k)
    if model is None:
        unavailable = "Numbers and spots appear once owner B's need.py and place.py are merged."

sites = pd.DataFrame(columns=["rank", "cand_id", "name", "kind", "service_type", "added_low", "added_mid",
                              "added_high", "sentence", "plain", "why", "lon", "lat"])
plan = None
if model:
    lo_x, hi_x = model["scale"]["low"], model["scale"]["high"]
    cand_i = cand.assign(cand_id=cand["cand_id"].astype(str)).set_index("cand_id")
    nearest_m = io.nearest_walk_m(d["reach_facilities"], counted)
    fac_now = fac.assign(counted=fac["fac_id"].astype(str).isin(counted))
    quintile = grid.set_index("cell_id")["wealth_quintile"]
    rows = []
    for p in model["picks"]:
        cid = p["cand_id"]
        lo, hi = p["added_people"] * lo_x, p["added_people"] * hi_x
        pt = cand_i.at[cid, "geometry"]
        cell = service_type.site_cell(grid, pt.x, pt.y)  # A's rule uses this cell too, so label and sentence agree
        walk = nearest_m.get(cell)
        walk_minutes = None if walk is None else walk / (speed * 1000 / 60)
        q_ = quintile.get(cell)
        q_ = None if pd.isna(q_) else int(q_)
        label = pending(service_type.assign, cid, grid, fac_now, d["reach_facilities"], cfg) or SERVICE_PENDING
        formal = pending(explain.sentence, lo, hi, walk_minutes, q_)
        rows.append({
            "rank": p["rank"], "cand_id": cid, "name": cand_i.at[cid, "name"], "kind": cand_i.at[cid, "kind"],
            "service_type": label,
            "why": service_reason(label, service_type.barangay_tercile(grid, cand_i.at[cid, "psgc"]), referral_min),
            "added_low": lo, "added_mid": p["added_people"], "added_high": hi,
            "sentence": site_details(formal or ""), "plain": plain_sentence(walk_minutes, q_) if formal else "",
            "lon": pt.x, "lat": pt.y,
        })
    sites = pd.DataFrame(rows, columns=sites.columns)
    before = tuple(model[s]["unmet"].sum() for s in ("low", "high"))
    before_mid = float(model["mid"]["unmet"].sum())
    reached_mid = sum(p["added_people"] for p in model["picks"])
    reached = (reached_mid * lo_x, reached_mid * hi_x)
    poorest = tuple(sum(p["added_poorest2"] for p in model["picks"]) * model["scale"][s] for s in ("low", "high"))
    still_after = (before[0] - reached[0], before[1] - reached[1])
    stop_note = placement_note(len(sites), k, before[1], model["more_possible"], still_after, walk_min)
    screen_note = plain_note(len(sites), k, before[1], model["more_possible"], still_after, walk_min)
    if len(sites) < k and not model["more_possible"] and fmt_num(before[1]) != "0":
        spots_note.markdown(f'<p class="abot-help abot-live">Only {len(sites)} spot{"" if len(sites) == 1 else "s"} '
                            "help with these settings; more would add no one.</p>", unsafe_allow_html=True)
    # Does "Poorer areas first" change anything under these settings? Run the other position and say so, so a
    # switch that moves nothing does not look broken.
    if wealth_known.any():
        other = pending(run_model, condition, speed, 0.0 if lam else float(cfg["equity_lambda"]), counted, k)
        if other is not None:
            now_ids = [p_["cand_id"] for p_ in model["picks"]]
            n_diff = len(set(now_ids) ^ {p_["cand_id"] for p_ in other["picks"]})
            equity_note.markdown(equity_help + '<p class="abot-help abot-live">' + (
                "With these settings, the same spots are picked with this on or off." if not n_diff else
                f"With these settings, turning this {'off' if lam else 'on'} changes which spots are picked.")
                + "</p>", unsafe_allow_html=True)
    # Comparison: the three simple rules under the same unmet need, walking reach and candidate sites.
    picks_by_rule = baseline_picks(k)
    u_mid, reach_now = model["mid"]["unmet"], model["reach"]
    compare_mid = {"abot": reached_mid}
    for rule in ("most_populous_barangay", "nearest_city_hall"):
        if rule in picks_by_rule:
            compare_mid[rule] = place.people_reached(u_mid, reach_now, picks_by_rule[rule])
    draws = [place.people_reached(u_mid, reach_now, d_) for d_ in picks_by_rule["random"]]
    compare_mid["random"] = float(np.mean(draws))
    spread = (float(np.percentile(draws, 5)), float(np.percentile(draws, 95)))
    plan = {"before": before, "reached": reached, "poorest": poorest, "stop_note": stop_note,
            "compare_mid": compare_mid, "spread": spread, "n_draws": len(draws)}
    # A plain-data snapshot of the plan, for regression checks (no UI reads it).
    st.session_state["_plan"] = {
        "settings": {"condition": condition, "k": k, "lam": lam, "speed": speed, "counted": list(counted)},
        "unmet": {s: float(model[s]["unmet"].sum()) for s in SCENARIOS},
        "sites": [{"rank": int(r.rank), "cand_id": r.cand_id, "service_type": r.service_type,
                   "added": {"low": r.added_low, "mid": r.added_mid, "high": r.added_high}} for r in sites.itertuples()],
        "compare_mid": {**compare_mid, "random_p5": spread[0], "random_p95": spread[1]},
    }

if unavailable:
    st.info(unavailable)

# ---- Map: one component holding the map, its floating cards and the results panel ----------------------------
if view == "map":
    m = folium.Map(location=[(grid["y"].min() + grid["y"].max()) / 2, (grid["x"].min() + grid["x"].max()) / 2],
                   zoom_start=14, tiles=None, zoom_control=False)
    # A grey, faded base map (dark mode inverts the same tiles): the need shading and markers carry the colour.
    folium.TileLayer("OpenStreetMap", name="Base map (OpenStreetMap)", control=False, opacity=0.6).add_to(m)
    root = m.get_root()
    root.header.add_child(folium.Element(
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Funnel+Display:wght@400..800&'
        'family=Funnel+Sans:wght@300..700&display=swap">'
        f"<style>{MAP_CSS}{MAP_DARK if dark else ''}</style>"))
    cells = d["grid_cells"].assign(cell_id=d["grid_cells"]["cell_id"].astype(str))
    outline, outside, outline_label = city_outline()
    folium.GeoJson(outside, name="Outside the city (dimmed)", control=False, interactive=False,
                   style_function=lambda f: {"fillColor": "#000000" if dark else "#FFFFFF",
                                             "fillOpacity": 0.38 if dark else 0.5, "stroke": False}).add_to(m)

    def cell_layer(values: pd.Series, name: str, show: bool):
        """Shaded 100 m cells; returns the colour scale for the key (None if nothing to draw)."""
        gj = cells.merge(values.round(1).rename("v"), left_on="cell_id", right_index=True)
        gj = gj[gj["v"] > 0][["cell_id", "v", "geometry"]]  # only cells with need: keeps the map fast
        if gj.empty:
            return None
        # Heat ramp, yellow (few) to deep red (most); lightness falls steadily, so the order also reads in greyscale
        # and to colour-blind readers. The two palest yellows vanished on the faded base map, so they are skipped.
        cmap = cm.LinearColormap(cm.linear.YlOrRd_09.colors[2:], vmin=0, vmax=float(gj["v"].max()))
        folium.GeoJson(
            gj, name=name, show=show,
            style_function=lambda f: {"fillColor": cmap(f["properties"]["v"]), "color": "none", "fillOpacity": 0.82},
            tooltip=folium.GeoJsonTooltip(["v"], aliases=["People in this 100 m square (estimate)"]),
        ).add_to(m)
        return cmap

    unmet_scale = None
    scen = SCENARIO_LABEL[scenario].lower()
    if model:
        unmet_scale = cell_layer(model[scenario]["unmet"], "People far from care", True)
        cell_layer(model[scenario]["need"], f"Everyone likely to have {cond_label.lower()}", False)
        walk_cells = cells[cells["cell_id"].isin(model["covered"])]
        if not walk_cells.empty:
            folium.GeoJson(walk_cells[["cell_id", "geometry"]], name="Close to a clinic we count", show=False,
                           style_function=lambda f: {"fillColor": "#378ADD", "color": "none",
                                                     "fillOpacity": 0.25}).add_to(m)

    included = folium.FeatureGroup(name="Clinics we count")
    excluded = folium.FeatureGroup(name="Listed clinics we do not count", show=False)  # 28 markers buried the 14
    for r in fac.itertuples():
        on = str(r.fac_id) in counted
        folium.Marker(
            [r.geometry.y, r.geometry.x], tooltip=esc(r.name),
            icon=folium.DivIcon(html=f'<div class="abot-fac{"" if on else " off"}">{PLUS_SVG}</div>',
                                icon_size=(22, 22), icon_anchor=(11, 11), class_name="abot-fac-wrap"),
            popup=folium.Popup(f"<b>{esc(r.name)}</b><br>{esc(r.type)}<br>{esc(r.status)}<br>"
                               f"{'Counted as a clinic that gives care now' if on else 'Not counted'}", max_width=260),
        ).add_to(included if on else excluded)
    included.add_to(m)
    excluded.add_to(m)

    layer = folium.FeatureGroup(name="Places that could host a spot", show=False)
    for r in cand.itertuples():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3, color="#5F5E5A", fill=True,
                            tooltip=esc(f"{r.name} ({r.kind.replace('_', ' ')})")).add_to(layer)
    layer.add_to(m)

    total_mid = float(sites["added_mid"].sum()) if not sites.empty else 0.0
    share = {r.rank: (100 * r.added_mid / total_mid if total_mid > 0 else 0.0) for r in sites.itertuples()}
    layer = folium.FeatureGroup(name="New care spots")
    pins = []
    for r in sites.itertuples():
        kind = kind_text(str(r.name), r.kind)
        chips = "".join(f"<span>{esc(x)}</span>" for x in (r.service_type, kind) if x)
        popup = (f'<div class="pop-head"><span class="pop-rank">{r.rank}</span><b>{esc(r.name)}</b></div>'
                 f'<div class="pop-chips">{chips}</div><div class="pop-gain"><b>+{esc(fmt_range(r.added_low, r.added_high))}'
                 f'</b><span>{share[r.rank]:.0f}% of the total</span></div><span class="pop-sub">more people close to '
                 f'care</span><p class="pop-note">{esc(SITE_NOTE)}</p>')
        pin = folium.Marker([r.lat, r.lon], popup=folium.Popup(popup, max_width=292),
                            tooltip=esc(f"{r.rank}. {r.name}"),
                            icon=folium.DivIcon(html=f'<div class="abot-pin">{r.rank}</div>', icon_size=(28, 28),
                                                icon_anchor=(14, 14), class_name="abot-pin-wrap"),
                            z_index_offset=1000)  # never under a facility badge
        pin.add_to(layer)
        pins.append(pin.get_name())
    layer.add_to(m)
    folium.GeoJson(outline, name=outline_label, interactive=False,
                   style_function=lambda f: {"color": "#E9EFEC" if dark else "#1E2423", "weight": 2,
                                             "dashArray": "6 4", "fill": False}).add_to(m)
    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)

    # Floating cards on the map
    found = ("No plan yet." if not plan else
             f"{len(sites)} of {k} new care spot{'' if k == 1 else 's'} found." if len(sites) >= k or model["more_possible"]
             else f"Only {len(sites)} of {k} spots help; more would add no one.")
    walk_text = f"A {walk_min}-minute walk" + ("" if speed == speeds[0] else f" at {speed:g} km/h")
    plan_card = (
        f'<section class="plan glass" aria-label="Current plan"><p class="t d">{esc(cond_label)} plan</p>'
        f'<p class="s">{esc(found)}</p><dl><dt>“Close” means</dt><dd>{esc(walk_text)}</dd>'
        f'<dt>Poorer areas first</dt><dd>{"Yes" if lam else "No" if wealth_known.any() else "Not available"}</dd>'
        f'<dt>Clinics we count</dt><dd class="num">{len(counted)} of {len(fac)}</dd>'
        f'<dt>Map colours</dt><dd>{esc(SCENARIO_LABEL[scenario])} estimate</dd></dl></section>')
    chip_html = ('<div class="chips" role="group" aria-label="Show on map"><button data-focus="0" aria-pressed="true">'
                 'All spots</button>' + "".join(
                     f'<button data-focus="{r.rank}" aria-pressed="false" aria-label="Show care spot {r.rank}">{r.rank}'
                     "</button>" for r in sites.itertuples()) + "</div>") if len(sites) else ""
    tools = ('<div class="tools">'
             f'<button id="home" class="glass" aria-label="Reset view" title="Reset view">{svg("reset")}</button>'
             f'<button id="zin" class="glass" aria-label="Zoom in" title="Zoom in">{svg("plus")}</button>'
             f'<button id="zout" class="glass" aria-label="Zoom out" title="Zoom out">{svg("minus")}</button>'
             f'<button id="layers" class="glass" aria-label="Map layers" title="Map layers" aria-expanded="false">'
             f'{svg("layers")}</button></div>')
    ramp = ""
    if unmet_scale is not None:
        stops = "".join(f'<span style="background:{unmet_scale(unmet_scale.vmax * (i + 0.5) / 6)}"></span>'
                        for i in range(6))
        ramp = (f'<span class="h">Yellow to red = more people living far from care <span>· {esc(scen)} estimate</span></span>'
                f'<span class="ramp num">0 <i aria-hidden="true">{stops}</i> {esc(fmt_num(unmet_scale.vmax))} people '
                "per 100 m square</span>")
    legend = (f'<div class="legend glass">{ramp}<span class="keys"><span><span class="k-site" aria-hidden="true">'
              '</span>New care spot</span><span><span class="k-fac" aria-hidden="true"></span>Clinic we count</span>'
              "</span></div>")
    summary = (f'<p class="vh">Map summary: people far from care are shaded from yellow (few) to red (most) per 100 m square; numbered squares mark '
               f"the {len(sites)} new care spots listed beside the map; round plus markers are the {len(counted)} "
               "clinics we count.</p>")

    # Results panel
    if not plan:
        panel = f'<div class="full"><p class="note">{esc(unavailable or "No plan yet.")}</p></div>'
        rail = ""
    else:
        by_brgy = (model["mid"]["unmet"].rename("u").to_frame().join(grid.set_index("cell_id")["barangay"])
                   .groupby("barangay")["u"].sum().sort_values(ascending=False))
        by_brgy = by_brgy[by_brgy > 0]
        where = ""
        if not by_brgy.empty:
            top = by_brgy.index[:2].tolist()
            where = (f'<p class="where">{svg("pin", 16)}<span>{by_brgy.iloc[:2].sum() / by_brgy.sum():.0%} of them live in '
                     f"{esc(' and '.join(top))}.</span></p>")
        still = (before[0] - reached[0], before[1] - reached[1])
        close_pct = 100 * reached_mid / before_mid if before_mid > 0 else 0.0
        bar = ""
        if before_mid > 0:
            bar = ('<div class="split" aria-hidden="true">'
                   + (f'<span style="flex:{close_pct:.2f} 1 0"></span>' if close_pct > 0 else "")
                   + (f'<span style="flex:{100 - close_pct:.2f} 1 0"></span>' if close_pct < 100 else "") + "</div>")
        if not wealth_known.any():
            wealth_line = "We have no wealth data, so we cannot say how many live in poorer areas."
        else:
            wealth_line = f"{fmt_range(*poorest)} of them live in the city's two lowest wealth groups."
            if not wealth_known.all():
                wealth_line += f" We do not know the wealth of {1 - wealth_known.mean():.0%} of the map; it gets no extra weight."
        n_s = len(sites)
        defs = [
            ("Far from care today",
             (f"Adults aged 30–79 who likely have {cond_label.lower()} and live more than a {walk_min}-minute walk "
              f"({model['limit_m']:,.0f} m) from all {len(counted)} clinics we count." if counted else
              "We count no clinics, so everyone is far from care.")
             + " We give a low–high range because no one knows the exact number."),
            ("Now close", f"People who would be a {walk_min}-minute walk from one of the {n_s} new spot"
                          f"{'' if n_s == 1 else 's'}. Each person is counted once. {wealth_line} “Poorer areas first” "
                          "changes which spots we pick, not how we count people."),
            ("Still far", "Far from care today, minus the people now close. Living close to a clinic does not mean a "
                          "person gets care."),
        ]
        rows_html = []
        for r in sites.itertuples():
            kind = kind_text(str(r.name), r.kind)
            brings = f"brings {share[r.rank]:.0f}% of the people reached"
            more = [f"{kind} · {brings}" if kind else brings[0].upper() + brings[1:], r.plain, r.why]
            rows_html.append(
                f'<li class="site" data-rank="{r.rank}"><details><summary><span class="rank d">{r.rank}</span>'
                f'<span class="who"><span class="name">{esc(r.name)}</span><span class="type">{esc(r.service_type)}'
                f'</span></span><span class="gain"><b>+{esc(fmt_range(r.added_low, r.added_high))}</b><span class="share">'
                f'<i role="img" aria-label="{share[r.rank]:.0f}% of the people reached"><span style="width:'
                f'{share[r.rank]:.1f}%"></span></i>{share[r.rank]:.0f}%</span></span></summary><div class="more">'
                + "".join(f"<span>{esc(x)}</span>" for x in more if x) + "</div></details></li>")
        site_list = (f'<div class="list"><ol aria-label="New care spots, in the order picked">{"".join(rows_html)}</ol></div>'
                     if rows_html else "")
        note = f'<p class="note">{esc(screen_note)}</p>' if screen_note else ""
        gain = esc(fmt_range(*reached))
        panel = (
            f'<div class="full"><section class="sec" aria-labelledby="h-adds"><div class="head"><h2 id="h-adds" class="d">'
            f"What {n_s} new care spot{'' if n_s == 1 else 's'} would do</h2><button class=\"round to-rail\" "
            f'aria-label="Collapse results panel" aria-expanded="true">{svg("right", 16)}</button></div>'
            f'<div class="hero"><div><p class="big d num" data-v="reached">+{gain}</p><p class="big-sub">more people close '
            f'to care</p></div><div class="today num"><div class="row"><span class="l">Far from care today</span>'
            f'<b data-v="before">{esc(fmt_range(*before))}</b></div>{bar}<div class="split-key"><span>'
            f'<span class="dot" aria-hidden="true"></span>Now close</span><span><span class="dot still" aria-hidden="true">'
            f'</span>Still far <b data-v="still">{esc(fmt_range(*still))}</b></span></div></div>{where}'
            f'<details class="how"><summary><span>Bar: middle estimate</span><span class="pill">{svg("info", 14)}How we '
            f'counted</span></summary><dl class="defs">'
            + "".join(f"<div><dt>{esc(a)}</dt><dd>{esc(b)}</dd></div>" for a, b in defs)
            + "</dl></details></div></section>"
            f'<section class="sec" aria-labelledby="h-sites"><div class="head"><h2 id="h-sites" class="d">New care spots'
            f'</h2><span class="found num">{n_s} of {k} found</span></div>{note}{site_list}</section>'
            f'<ul class="good" aria-label="Good to know"><li>{svg("range", 14)}All numbers are estimates, shown as a '
            f'low–high range.</li><li>{svg("alert", 14)}Living close to a clinic does not mean people get care. We have '
            "not checked the clinics on our list, and the new spots are not approved yet.</li></ul></div>")
        rail = (f'<aside class="railv" aria-label="Results (collapsed)"><button class="round expand" '
                f'aria-label="Open results panel" aria-expanded="true">{svg("left", 18)}</button><p><b class="d">+{gain}'
                '</b>more people close to care</p><hr>' + (
                    '<ol aria-label="New care spots">' + "".join(
                        f'<li><button class="d" data-focus="{r.rank}" aria-pressed="false" aria-label="Show care spot '
                        f'{r.rank}">{r.rank}</button></li>' for r in sites.itertuples()) + "</ol>" if len(sites) else "")
                + "</aside>")
    root.html.add_child(folium.Element(
        f'<div class="ov">{plan_card}{chip_html}{tools}{legend}{summary}</div>'
        f'<aside class="panel" aria-label="Results">{panel}{rail}</aside>'))
    home = [[float(grid["y"].min()), float(grid["x"].min())], [float(grid["y"].max()), float(grid["x"].max())]]
    root.script.add_child(folium.Element(
        f"var ABOT = {json.dumps({'map': m.get_name(), 'pins': pins, 'home': home})};{MAP_JS}"))
    # body.ws scopes the component styles; Folium's own template writes a bare <body>.
    doc = root.render().replace("<body>", '<body class="ws">', 1)
    with st.container(key="workspace"):
        components.html(doc, height=900, scrolling=True)

# ---- Compare ------------------------------------------------------------------------------------------------
baseline_rows, compare_note = None, None
if plan:
    baseline_rows = [(METHOD_LABEL[m_], fmt_num(compare_mid[m_]), fmt_range(compare_mid[m_] * lo_x,
                                                                             compare_mid[m_] * hi_x))
                     for m_ in compare_mid]
else:
    compare_note = "The comparison with simple placement rules needs need and placement first."
if view == "compare":
    with st.container(border=True, key="sheet"):
        if not plan:
            st.markdown('<h2 class="abot-h2">ABOT next to simpler ways of picking spots</h2>', unsafe_allow_html=True)
            st.info("The comparison appears once the numbers and spots are available.")
        else:
            # One HTML block: headline, chart, one card per way of picking, then the note on uncertainty.
            n_cand = int(d["reach_candidates"]["cand_id"].nunique())
            order = sorted(compare_mid, key=lambda m_: -compare_mid[m_])
            best = max((m_ for m_ in compare_mid if m_ != "abot"), key=compare_mid.get)
            gap = compare_mid["abot"] - compare_mid[best]
            settings = [cond_label, f"Up to {k} new spot{'' if k == 1 else 's'}",
                        f"{walk_min}-minute walk at {speed:g} km/h", f"{len(counted)} clinics we count",
                        f"{n_cand} places that could host a spot"]
            if gap > 1e-6:
                big, line = f"+{fmt_num(gap)}", "more people close to care with ABOT than with the best simple way"
            elif gap < -1e-6:
                big, line = f"−{fmt_num(-gap)}", "fewer people close to care with ABOT than with the best simple way"
            else:
                big, line = "Same", "number of people close to care as the best simple way"
            times = (f'<span class="cmp-times">about {compare_mid["abot"] / compare_mid[best]:.1f} times as many</span>'
                     if gap > 1e-6 and compare_mid[best] >= 1 else "")
            hero = (f'<section class="cmp-hero" aria-label="Main result"><p class="cmp-big d num">{esc(big)}</p>'
                    f'<div><p class="cmp-line">{esc(line)} ({esc(SCREEN_METHOD[best].lower())}).</p>'
                    f'<p class="cmp-vs num"><span><i class="sw is-abot"></i>ABOT <b>{esc(fmt_num(compare_mid["abot"]))}'
                    f'</b></span><span><i class="sw"></i>{esc(SCREEN_METHOD[best])} <b>{esc(fmt_num(compare_mid[best]))}</b>'
                    f"</span>{times}</p></div></section>")

            ticks = nice_ticks(max(max(compare_mid.values()), spread[1]))
            pct = (lambda v: 100 * v / ticks[-1])
            grid_html = "".join(f'<i style="left:{pct(t):.2f}%"></i>' for t in ticks[1:])
            rows = []
            for m_ in order:
                v, rng = compare_mid[m_], fmt_range(compare_mid[m_] * lo_x, compare_mid[m_] * hi_x)
                tip = f"Middle {fmt_num(v)} · low–high {rng}"
                whisker = ""
                if m_ == "random":
                    tip += f" · middle 90% of tries {fmt_range(*spread)}"
                    whisker = (f'<span class="cmp-whisk" style="left:{pct(spread[0]):.2f}%;width:'
                               f'{max(pct(spread[1]) - pct(spread[0]), 0.3):.2f}%"></span>')
                rows.append(
                    f'<div class="cmp-row" tabindex="0" aria-label="{esc(SCREEN_METHOD[m_])}: {esc(tip)}">'
                    f'<span class="cmp-name">{esc(SCREEN_METHOD[m_])}</span><span class="cmp-track" aria-hidden="true">'
                    f'<span class="cmp-grid">{grid_html}</span>'
                    f'<span class="cmp-bar{" is-abot" if m_ == "abot" else ""}" style="width:{pct(v):.2f}%"></span>{whisker}'
                    f'<b class="cmp-val num" style="left:{pct(v):.2f}%">{esc(fmt_num(v))}</b>'
                    f'<span class="cmp-tip num{" is-start" if pct(v) < 25 else " is-end" if pct(v) > 75 else ""}" '
                    f'style="left:{pct(v):.2f}%">{esc(tip)}</span></span></div>')
            axis = "".join(f'<span style="left:{pct(t):.2f}%">{esc(fmt_num(t)) if t else "0"}</span>' for t in ticks)
            chart = ('<figure class="cmp-card cmp-chart" aria-label="More people close to care by way of picking, middle '
                     'estimate"><figcaption class="cmp-cap"><b>More people close to care</b><span>middle estimate</span>'
                     f'</figcaption><div class="cmp-plot">{"".join(rows)}<div class="cmp-axis" aria-hidden="true">'
                     f'<span class="cmp-name"></span><span class="cmp-ticks num">{axis}</span></div></div>'
                     '<p class="cmp-foot">Bars start at zero. The bracket under random places spans the middle 90% of '
                     f"{plan['n_draws']} random tries. Hover or tab to a row for its low–high range.</p></figure>")

            how = {"abot": ("Adds spots one at a time, each time picking the place that brings the most people still far "
                            "from care close to it" + (", with extra weight for poorer areas" if lam else "")
                            + ". This step-by-step way is not guaranteed to find the best possible set."),
                   "most_populous_barangay": "One spot near the middle of each of the most populated barangays.",
                   "nearest_city_hall": "The places with the shortest walk from City Hall.",
                   "random": (f"The average of {plan['n_draws']} random sets of {k} places (seed {cfg['seed']}). The "
                              f"middle 90% of tries bring {fmt_range(*spread)} people close to care.")}
            cards = "".join(
                f'<article class="cmp-card cmp-way{" is-abot" if m_ == "abot" else ""}"><h4><i class="sw'
                f'{" is-abot" if m_ == "abot" else ""}"></i>{esc(SCREEN_METHOD[m_])}</h4><p class="cmp-num num"><b>'
                f'{esc(fmt_num(compare_mid[m_]))}</b><span>low–high {esc(fmt_range(compare_mid[m_] * lo_x, compare_mid[m_] * hi_x))}'
                f'</span></p><p class="cmp-how">{esc(how[m_])}</p></article>' for m_ in order)
            missing_hall = ('<p class="abot-note">Closest to City Hall is not shown: City Hall\'s location is missing '
                            "from config.yaml.</p>" if "nearest_city_hall" not in picks_by_rule else "")
            st.markdown(
                '<div class="cmp"><h2 class="abot-h2">ABOT next to simpler ways of picking spots</h2>'
                '<p class="cmp-sub">How many more people each way brings close to care, all with the same settings:</p><ul class="cmp-chips" aria-label="Settings">'
                + "".join(f"<li>{esc(x)}</li>" for x in settings) + f"</ul>{hero}{chart}"
                '<h3 class="abot-h3">How each way picks its spots</h3><p class="cmp-sub">More people close to care, '
                f'middle estimate and low–high range.</p><div class="cmp-ways">{cards}</div>{missing_hall}'
                '<aside class="cmp-aside"><b>Two kinds of uncertainty</b><span>The low–high range comes from how common '
                "the condition might be. The random-places spread comes from which places get drawn. They are "
                "different, so we show them apart.</span></aside></div>", unsafe_allow_html=True)

# ---- Placement brief (formal wording: it goes to a city health office) ----------------------------------------
if view == "brief":
    # A document view: no outer panel, one centred column (title and download, then the brief as a page).
    with st.container(key="briefview"):
        if not plan:
            st.markdown('<div class="abot-toolbar"><h2>Placement brief</h2></div>', unsafe_allow_html=True)
            st.info("The brief appears once the numbers and spots are available.")
        else:
            status = []
            if paths.USING_FIXTURES:
                status.append("Synthetic development data: not real places or numbers.")
            if (sites["service_type"] == SERVICE_PENDING).any():
                status.append("Service-type rule not merged yet: service types are pending.")
            if pending(explain.sentence, 1.0, 2.0, None, None) is None:
                status.append("Plain-language site sentences not available yet.")
            if not wealth_known.any():
                status.append("No wealth data: lower-wealth priority could not be applied.")
            elif not wealth_known.all():
                status.append(f"Wealth unknown for {1 - wealth_known.mean():.0%} of cells (weighted as 1.0 there).")
            text = make_brief(sites, {
                "condition": condition, "k": k, "walk_speed_kmh": speed, "walk_minutes": walk_min,
                "walk_m": model["limit_m"], "equity_lambda": lam, "n_facilities_counted": len(counted),
                "n_facilities_total": len(fac), "unmet_before": before, "reached": reached,
                "poorest2": poorest if wealth_known.any() else None, "placement_note": stop_note,
                "baselines": baseline_rows, "compare_note": compare_note, "data_status": status,
                "synthetic": paths.USING_FIXTURES,
                "sources": [{"dataset": s_["dataset"], "source": s_["source"], "url": s_["url"],
                             "period": s_["unit_and_period"], "accessed": s_["date_accessed"], "licence": s_["licence"]}
                            for s_ in data_sources()],
            })
            bar_title, bar_button = st.columns([3, 1], vertical_alignment="bottom")
            bar_title.markdown('<div class="abot-toolbar"><h2>Placement brief</h2><span>The plan on screen as a short '
                               "write-up, in the formal wording a health office expects.</span></div>",
                               unsafe_allow_html=True)
            bar_button.download_button("Download brief (.md)", text, type="primary", mime="text/markdown",
                                       icon=":material/download:", use_container_width=True,
                                       file_name=f"abot_placement_brief{'_SYNTHETIC' if paths.USING_FIXTURES else ''}.md")
            with st.container(border=True, key="brief_doc"):
                st.markdown(text)

# ---- Assumptions & sources ----------------------------------------------------------------------------------
if view == "assume":
    with st.container(border=True, key="sheet"):
        # A reference page: text at a reading width, the roadmap wide, tables for the numbers.
        steps = [("People", "Adults aged 30–79 on a grid of 100 m squares (WorldPop 2020, a modelled population)."),
                 ("Who likely has the condition", "Those adults times a published rate for the chosen condition. A "
                  "low, middle and high rate give the range. The two conditions are never added together."),
                 ("Close to care", f"A square is close when a clinic we count is within a {walk_min}-minute walk along "
                  "OpenStreetMap paths."),
                 ("Far from care", "People likely to have the condition in squares that are not close to any clinic "
                  "we count."),
                 ("New care spots", "Public places are added one at a time, each where it brings the most people still "
                  "far from care close (poorer areas weigh more while that option is on). Each person counts once."),
                 ("Kind of service", "A simple rule using phone signal and the walk to the nearest clinic we count."),
                 ("Compare", "Three simpler ways of picking spots, with the same settings.")]
        # Wide: a road across the page, pins at 1/8 … 7/8 of its length, labels alternating above and below, each
        # two-eighths wide, so they tile without overlapping. Narrow: the road runs down the left. Positions come
        # from the class s1…s7 in app.css (Streamlit strips CSS variables from inline styles).
        road = ['<div class="road-strip" aria-hidden="true"></div>']
        for i, (t, x) in enumerate(steps, 1):
            side = "up" if i % 2 else "down"
            road.append(f'<div class="road-pin {side} s{i}" aria-hidden="true"><i></i><b class="d">{i}</b></div>')
            road.append(f'<div class="road-stop {side} s{i}" role="listitem"><b>{esc(t)}</b><span>{esc(x)}</span></div>')
        roadmap = f'<div class="road" role="list" aria-label="How ABOT works, in seven steps">{"".join(road)}</div>'

        settings = [("Walking speed", "Planning assumption", f"{speed:g} km/h",
                     (f"{model['limit_m']:,.0f} m in {walk_min} minutes. " if model else "")
                     + f"How far “close” reaches, for clinics and new spots. {speeds[1]:g} km/h checks slower walkers."),
                    ("Poorer-areas weight λ", "Planning assumption", f"{lam:g} ({'on' if lam else 'off'})",
                     "Weight = 1 + λ × (5 − wealth group) / 4. Changes which spots are picked, not how people are "
                     "counted. Wealth groups rank areas inside Mandaluyong only (2022 relative wealth), and the city's "
                     "range is narrow, so the lowest group is not necessarily poor."),
                    ("Walk to send patients on", "Planning assumption", f"{referral_min} min",
                     "Used by the kind-of-service rule: is a clinic we count within this walk of the spot?"),
                    ("Random tries", "Method setting", f"{cfg['random_draws']}",
                     f"For the random-places comparison (seed {cfg['seed']}).")]
        settings_table = ('<div class="ref-scroll"><table class="ref"><thead><tr><th>Assumption</th><th>Now</th><th>What it means</th></tr>'
                          "</thead><tbody>" + "".join(
                              f'<tr><td><b>{esc(n)}</b><small>{esc(kind)}</small></td><td class="val num">{esc(v)}</td>'
                              f"<td>{esc(m)}</td></tr>" for n, kind, v, m in settings) + "</tbody></table></div>")

        def link(label, url):
            return (f'<a class="abot-link" href="{esc(url)}" target="_blank" rel="noopener">{esc(label)}</a>'
                    if url else esc(label))

        prev_rows = []
        for c in conditions:
            mid_label, mid_url = source_link(cfg["prevalence"][c]["mid_source"])
            ends = sorted({evidence_type(cfg["prevalence"][c][f"{s_}_source"]) for s_ in ("low", "high")})
            note = "Low and high: " + "; ".join(e[0].lower() + e[1:] for e in ends) + "."
            if c == "diabetes":
                note += " The source covers adults 30 and over (including 80+); ABOT applies it to adults 30–79."
            prev_rows.append(
                f"<tr><td><b>{esc(COND_LABEL.get(c, c))}</b></td>" + "".join(
                    f'<td class="val num r{" mid" if s_ == "mid" else ""}">{config.prevalence(cfg, c, s_):.1%}</td>'
                    for s_ in SCENARIOS) + f"<td>{link(mid_label, mid_url)}<small>{esc(note)}</small></td></tr>")
        prev_table = ('<div class="ref-scroll"><table class="ref"><thead><tr><th>Condition</th><th class="r">Low</th><th class="r">Middle'
                      '</th><th class="r">High</th><th>Source</th></tr></thead><tbody>' + "".join(prev_rows)
                      + "</tbody></table></div>")

        def src_row(s_):
            meta = " · ".join(x for x in (s_["unit_and_period"], s_["observed_or_modelled"],
                                          s_["licence"] or "licence not recorded",
                                          f"accessed {s_['date_accessed']}" if s_["date_accessed"] else "") if x)
            return (f'<li><div class="ds-who"><b>{link(s_["dataset"], s_["url"])}</b><span>{esc(s_["source"])}</span></div>'
                    f'<div class="ds-what">' + (f'<p><span>Used for</span> {esc(s_["use_in_abot"])}</p>'
                                                if s_["use_in_abot"] else "")
                    + f'<small>{esc(meta)}</small></div></li>')
        sources = f'<ul class="ds">{"".join(src_row(s_) for s_ in data_sources())}</ul>'
        limits = '<ul class="lim">' + "".join(f"<li>{esc(x)}</li>" for x in LIMITATIONS) + "</ul>"

        tech = [("Formulas", "need = adults 30–79 × prevalence; unmet = need in cells outside every counted facility's "
                             "walking reach; weight = 1 + λ × (5 − wealth quintile) / 4 (missing wealth = 1.0); each "
                             "greedy step adds the site with the largest sum of uncovered unmet × weight, and reports "
                             "unweighted people."),
                ("Configuration file", "config.yaml (every assumption above, with its source)"),
                ("Data in use", f"{'Synthetic fixtures (tests/fixtures)' if paths.USING_FIXTURES else 'Processed data (data/processed)'}; "
                                f"{len(fac)} listed facilities, {len(counted)} counted; {len(cand)} candidate sites; "
                                f"{fmt_num(len(grid))} grid cells.")]
        tech += [(f"{c.capitalize()} prevalence source, {SCENARIO_LABEL[s_].lower()} (as recorded)",
                  cfg["prevalence"][c][f"{s_}_source"]) for c in conditions for s_ in SCENARIOS]
        technical = ('<details class="tech"><summary>Formulas, configuration and the full source records</summary><dl>'
                     + "".join(f"<div><dt>{esc(a_)}</dt><dd>{esc(b_)}</dd></div>" for a_, b_ in tech) + "</dl></details>")

        toc = [("asm-how", "How ABOT works"), ("asm-now", "What the plan assumes"), ("asm-data", "Data sources"),
               ("asm-lim", "Limitations"), ("asm-tec", "Technical details")]
        st.markdown(
            '<div class="ref-page"><h2 class="abot-h2">What we assumed and where the data came from</h2>'
            '<p class="ref-intro">How the numbers are made, what the plan on screen assumes, where each dataset comes '
            "from, and what the results cannot tell you.</p><p class=\"ref-toc\">"
            + "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in toc) + "</p>"
            f'<section id="asm-how"><h3 class="abot-h3">How ABOT works</h3>{roadmap}<p class="ref-note">These are planning '
            "estimates of how close people live to care. They do not diagnose anyone or show that a clinic gives "
            "chronic care.</p></section>"
            f'<section id="asm-now"><h3 class="abot-h3">What the plan assumes</h3><p class="ref-sub">These follow Plan '
            f'settings.</p>{settings_table}<h4 class="ref-h4">How common each condition is</h4><p class="ref-sub">Share of '
            f"adults 30–79. Low and high give the range; the map shows the estimate chosen in Plan settings.</p>{prev_table}</section>"
            f'<section id="asm-data"><h3 class="abot-h3">Where the data came from</h3>{sources}</section>'
            f'<section id="asm-lim"><h3 class="abot-h3">Limitations</h3>{limits}</section>'
            f'<section id="asm-tec"><h3 class="abot-h3">Technical details</h3>{technical}</section></div>',
            unsafe_allow_html=True)
