"""One-page placement brief (Markdown) for a city health office.

OWNER: C. Must include: scenario and assumptions used, unmet need before/after (as ranges),
each chosen site with its service type and plain-language sentence, the comparison with
baselines, and the limitations list from CONTEXT.md. Label modelled numbers as modelled.
"""
from __future__ import annotations

import math

import pandas as pd

# CONTEXT.md section 3, "Always shown as limitations". The app shows the same list.
LIMITATIONS = (
    "Need is modelled from published prevalence and 2020 population estimates, not observed patient counts.",
    "Diabetes prevalence is a WHO estimate for adults 30 and over (including 80+), applied to adults 30–79, so diabetes need is slightly overstated.",
    "Facilities are listed, not verified for chronic-care services, staff, hours or capacity.",
    "Results are proximity-only: no facility capacity data.",
    "Walking times come from the OpenStreetMap network at an assumed speed and are not route-validated.",
    "Network quality uses Ookla speed tests (2019–2022) as a proxy.",
    "Candidate sites depend on OpenStreetMap completeness.",
    "Residents near the city edge may use facilities in neighbouring cities.",
    "These are planning estimates, not predictions of individual need or health outcomes.",
)

PLACEMENT_COLUMNS = ("rank", "cand_id", "name", "kind", "service_type", "added_low", "added_high", "sentence")
SUMMARY_KEYS = ("condition", "k", "walk_speed_kmh", "walk_minutes", "walk_m", "equity_lambda",
                "n_facilities_counted", "n_facilities_total", "unmet_before", "reached", "poorest2",
                "placement_note", "baselines", "compare_note", "data_status", "synthetic")

# Shared with the app so the screen and the brief use the same words.
METRIC_LABELS = ("Beyond reach now", "Additional reach", "Remaining beyond reach")
SCOPE_NOTE = "Modelled scenario ranges; walking proximity does not confirm access to care."
REACH_NOTE = "Being within walking reach is not the same as receiving care."
SITES_NOTE = "Suggested public sites only: not checked for permission, space or readiness."


def fmt_num(x: float) -> str:
    """Modelled counts are estimates: round to the nearest 10 from 100 up.

    A real value under 0.5 shows as "<1", never as 0; only float dust (|x| < 1e-6, e.g. before - reached
    leaving -1e-13) shows as 0.
    """
    if abs(x) < 1e-6:
        return "0"
    if 0 < x < 0.5:
        return "<1"
    return f"{round(x, -1):,.0f}" if x >= 100 else f"{x:,.0f}"


def fmt_range(low: float, high: float) -> str:
    """'1,900–2,600'; a single number when low and high round the same."""
    lo, hi = fmt_num(low), fmt_num(high)
    return lo if lo == hi else f"{lo}–{hi}"


def site_details(sentence: str) -> str:
    """A's site sentence without its opening "About X–Y likely patients" clause.

    Screens and the brief already show "+X–Y additional reach", so the clause only repeated it.
    shortcut: relies on explain.sentence's documented "About …; …" format; any other format is kept whole.
    """
    if not sentence.startswith("About "):
        return sentence
    rest = sentence.partition("; ")[2]
    return rest[:1].upper() + rest[1:]


def kind_text(name: str, kind: str) -> str:
    """'School', 'Barangay hall', …, or '' when the site name already starts with it (e.g. 'School (Hulo)')."""
    label = kind.replace("_", " ")
    return "" if name.lower().startswith(label) else label.capitalize()


_COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def distinct_names(names: list[str], lons: list[float], lats: list[float]) -> list[str]:
    """Names that tell same-named sites apart, e.g. 'Court (Mauway), north-east' and 'Court (Mauway), south-west'.

    Unique names are kept. A shared name gets its compass direction from the group's centre; if two in a group
    would still share a direction, the group is numbered west to east ('…, 1 of 3') instead.
    """
    out = list(names)
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(n, []).append(i)
    for n, idx in groups.items():
        if len(idx) < 2:
            continue
        cx, cy = sum(lons[i] for i in idx) / len(idx), sum(lats[i] for i in idx) / len(idx)
        dirs = {}
        for i in idx:
            dx, dy = (lons[i] - cx) * math.cos(math.radians(cy)), lats[i] - cy
            dirs[i] = _COMPASS[round(math.degrees(math.atan2(dy, dx)) / 45) % 8]
        if len(set(dirs.values())) == len(idx):
            for i in idx:
                out[i] = f"{n}, {dirs[i]}"
        else:
            for rank, i in enumerate(sorted(idx, key=lambda j: (lons[j], lats[j])), start=1):
                out[i] = f"{n}, {rank} of {len(idx)}"
    return out


def placement_note(n_sites: int, k: int, beyond_high: float, more_possible: bool,
                   remaining: tuple[float, float] | None = None, walk_minutes: int | None = None) -> str | None:
    """Why fewer sites than requested were recommended, or None when all k were.

    remaining: (low, high) likely patients still beyond reach after the chosen sites; when given with walk_minutes,
    the note says they are out of every candidate's walking reach (true whenever more_possible is False).
    """
    if fmt_num(beyond_high) == "0":
        return ("No modelled need lies beyond walking reach under these settings, so no new site is "
                f"recommended. {REACH_NOTE}")
    if n_sites >= k:
        return None
    reason = ("The placement stopped before reaching the requested number." if more_possible
              else "No remaining candidate adds further modelled reach under these settings.")
    if not more_possible and remaining is not None and walk_minutes and fmt_num(remaining[1]) != "0":
        reason += (f" The remaining {fmt_range(*remaining)} likely patients beyond reach are not within a "
                   f"{walk_minutes}-minute walk of any candidate site.")
    return f"{n_sites} of {k} requested sites recommended. {reason}"


def _cell(text) -> str:
    """Text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def make_brief(placements: pd.DataFrame, summary: dict) -> str:
    """Return the brief as Markdown text: the plan first, then results, comparison, assumptions and evidence.

    placements: one row per chosen site with PLACEMENT_COLUMNS (added_low/high = additional people within
    reach under the low/high prevalence scenario; sentence may be empty).
    summary: SUMMARY_KEYS. unmet_before and reached are (low, high); poorest2 is (low, high) or None when
    wealth data are unavailable; placement_note is placement_note() or None; baselines is a list of
    (rule label, middle-scenario text, low–high text) or None, with compare_note saying why when None;
    data_status is a list of short lines about inputs that are synthetic, pending or missing; synthetic is a
    bool. Optional: sources, a list of dicts with dataset, source, url, period, accessed, licence.
    """
    missing = [c for c in PLACEMENT_COLUMNS if c not in placements.columns]
    missing += [k for k in SUMMARY_KEYS if k not in summary]
    if missing:
        raise KeyError(f"make_brief is missing {missing}")
    s = summary
    before, reached = s["unmet_before"], s["reached"]
    after = (before[0] - reached[0], before[1] - reached[1])
    cond = s["condition"].capitalize()
    n_sites = len(placements)
    lines = []
    if s["synthetic"]:
        lines += ["> **SYNTHETIC DEVELOPMENT DATA: not real places or numbers. Do not use.**", ""]
    lines += [
        f"# Placement brief: {cond} care, Mandaluyong City (prototype)",
        "",
        f"{SCOPE_NOTE} Hypertension and diabetes are planned separately and never added together.",
        "",
        "## Decision summary",
        "",
        f"- New service points requested: {s['k']}; recommended: {n_sites}.",
        f"- {METRIC_LABELS[1]}: about {fmt_range(*reached)} modelled likely patients newly within a "
        f"{s['walk_minutes']:g}-minute walk, each counted once.",
        f"- {METRIC_LABELS[0]}: {fmt_range(*before)}; {METRIC_LABELS[2].lower()} after these sites: {fmt_range(*after)}.",
    ]
    if s["placement_note"]:
        lines.append(f"- {s['placement_note']}")
    lines += ["", REACH_NOTE, "", "## Recommended sites", ""]
    if placements.empty:
        lines.append("No site recommended under these settings.")
    else:
        lines += ["| # | Site | Suggested service | Additional reach | Notes |", "| --- | --- | --- | --- | --- |"]
        for r in placements.sort_values("rank").itertuples():
            kind = kind_text(r.name, r.kind)
            site = f"**{_cell(r.name)}**" + (f" ({kind.lower()})" if kind else "")
            lines.append(f"| {r.rank} | {site} | {_cell(r.service_type)} | +{fmt_range(r.added_low, r.added_high)} "
                         f"| {_cell(r.sentence)} |")
        lines += ["", SITES_NOTE]
    lines += [
        "", "## Modelled results", "",
        "| | Likely patients (low–high scenario range) |",
        "| --- | --- |",
        f"| {METRIC_LABELS[0]} | {fmt_range(*before)} |",
        f"| {METRIC_LABELS[1]} (from {n_sites} new sites, each person counted once) | {fmt_range(*reached)} |",
        f"| {METRIC_LABELS[2]} | {fmt_range(*after)} |",
        f"| Of the additional people, in the city's two lowest wealth quintiles | "
        f"{fmt_range(*s['poorest2']) if s['poorest2'] is not None else 'wealth data unavailable'} |",
        "",
        "## Compared with simple placement rules",
        "",
    ]
    if s["baselines"]:
        lines += ["Same condition, walking reach, counted facilities, candidate sites and number of sites for every "
                  "rule.", "", "| Rule | Additional reach, middle scenario | Low–high scenario range |",
                  "| --- | --- | --- |"]
        for row in s["baselines"]:
            rule, *vals = row
            mid_text, range_text = (vals[0], vals[1]) if len(vals) > 1 else ("", vals[0])
            lines.append(f"| {_cell(rule)} | {mid_text} | {range_text} |")
    else:
        lines.append(s["compare_note"] or "Comparison not available for these settings.")
    lines += [
        "", "## Planning assumptions", "",
        f"- Condition: {cond}",
        f"- Prioritize lower-wealth areas: {'on' if s['equity_lambda'] else 'off'} (λ = {s['equity_lambda']:g}); "
        "it steers which sites are chosen, and counts stay unweighted people",
        f"- Walking reach: {s['walk_minutes']:g} minutes at {s['walk_speed_kmh']:g} km/h "
        f"= {s['walk_m']:,.0f} m network distance",
        f"- Counted as existing care: {s['n_facilities_counted']} of {s['n_facilities_total']} listed facilities "
        "(listed, not verified)",
        "- Prevalence: low–high scenario range, not a confidence interval; scenarios change counts, "
        "not the site ranking",
    ]
    if s["data_status"]:
        lines += ["", "## Data status", ""] + [f"- {x}" for x in s["data_status"]]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    if s.get("sources"):
        lines += ["", "## Sources", ""]
        for src in s["sources"]:
            bits = [b for b in (src.get("period"), f"accessed {src['accessed']}" if src.get("accessed") else "",
                                src.get("licence")) if b]
            lines.append(f"- {src['dataset']}: {src['source']}" + (f" ({'; '.join(bits)})" if bits else "")
                         + (f". {src['url']}" if src.get("url") else ""))
    return "\n".join(lines) + "\n"
