"""Placement brief (owner C): required sections, ranges and limitations."""
from __future__ import annotations

import pandas as pd
import pytest

from abot.brief import (LIMITATIONS, METRIC_LABELS, distinct_names, fmt_range, kind_text, make_brief, placement_note,
                        site_details)


def _summary(**over):
    s = {"condition": "hypertension", "k": 2, "walk_speed_kmh": 4.5, "walk_minutes": 15, "walk_m": 1125.0,
         "equity_lambda": 0.5, "n_facilities_counted": 3, "n_facilities_total": 4,
         "unmet_before": (800.0, 1200.0), "reached": (300.0, 450.0), "poorest2": (100.0, 150.0),
         "placement_note": None, "baselines": [("Nearest City Hall", "120", "100–150")], "compare_note": None,
         "data_status": ["Synthetic development data: not real places or numbers."], "synthetic": True}
    s.update(over)
    return s


def _sites():
    return pd.DataFrame([
        {"rank": 1, "cand_id": "C1", "name": "Hall A", "kind": "barangay_hall", "service_type": "Mobile clinic stop",
         "added_low": 200.0, "added_high": 300.0, "sentence": ""},
        {"rank": 2, "cand_id": "C2", "name": "School B", "kind": "school", "service_type": "Mobile clinic stop",
         "added_low": 100.0, "added_high": 150.0, "sentence": "Lowest wealth quintile."},
    ])


def test_fmt_range():
    assert fmt_range(1894, 2612) == "1,890–2,610"
    assert fmt_range(42, 42.3) == "42"
    assert fmt_range(0, -1e-13) == "0"
    assert fmt_range(0.2, 0.3) == "<1" and fmt_range(0.3, 0.7) == "<1–1"  # a real fraction is never shown as 0


def test_brief_has_ranges_sites_and_limitations():
    text = make_brief(_sites(), _summary())
    assert "SYNTHETIC" in text and "modelled" in text
    assert "800–1,200" in text and "300–450" in text and "500–750" in text  # before, reached, after
    assert all(label in text for label in METRIC_LABELS) and "3 of 4" in text
    assert "## Data status" in text and "not checked for permission" in text
    assert text.index("Hall A") < text.index("School B")
    assert "| Nearest City Hall | 120 | 100–150 |" in text  # middle scenario and range in separate columns
    assert all(x in text for x in LIMITATIONS)
    assert text.index("## Recommended sites") < text.index("## Planning assumptions")  # the plan comes first
    assert "requested: 2; recommended: 2" in text
    assert "nan" not in text.lower()


def test_brief_handles_no_sites_no_baselines_no_wealth():
    note = placement_note(0, 5, 0.0, False)
    text = make_brief(_sites().iloc[0:0], _summary(baselines=None, compare_note="Not computed yet.",
                                                   poorest2=None, placement_note=note, data_status=[],
                                                   synthetic=False))
    assert "SYNTHETIC" not in text and "No site recommended" in text and "Not computed yet." in text
    assert "wealth data unavailable" in text and note in text and "## Data status" not in text


def test_brief_lists_sources_and_escapes_table_cells():
    src = [{"dataset": "Population grid", "source": "WorldPop", "url": "https://example.org/x", "period": "2020",
            "accessed": "2026-10-23", "licence": "CC BY 4.0"}]
    sites = _sites().assign(name=["Hall | A", "School B"])
    text = make_brief(sites, _summary(sources=src))
    assert "## Sources" in text and "- Population grid: WorldPop (2020; accessed 2026-10-23; CC BY 4.0). https://example.org/x" in text
    assert "Hall \\| A" in text  # a pipe in a site name cannot break the table


def test_site_details_and_kind_text():
    assert site_details("About 260–390 likely patients; lowest wealth quintile.") == "Lowest wealth quintile."
    assert site_details("About 11–16 likely patients; nearest listed facility for referral is an 8-minute walk "
                        "away; lowest wealth quintile.").startswith("Nearest listed facility for referral")
    assert site_details("About 40 likely patients.") == ""
    assert site_details("Some other format.") == "Some other format."
    assert kind_text("School (Hulo)", "school") == ""
    assert kind_text("Bulwagang Pambarangay", "barangay_hall") == "Barangay hall"


def test_distinct_names():
    assert distinct_names(["Hall", "Court (Hulo)", "Court (Hulo)"], [0.0, 1.0, 1.0], [0.0, 14.0, 14.01]) == \
        ["Hall", "Court (Hulo), south", "Court (Hulo), north"]
    # four sites where two share a direction from the centre: numbered west to east instead
    out = distinct_names(["C"] * 4, [0.0, 0.1, 0.2, 0.3], [14.0, 14.0, 14.0, 14.0])
    assert out == ["C, 1 of 4", "C, 2 of 4", "C, 3 of 4", "C, 4 of 4"]


def test_placement_note():
    assert placement_note(5, 5, 900.0, False) is None
    assert placement_note(2, 5, 900.0, False).startswith("2 of 5 requested sites recommended. No remaining")
    assert "stopped before" in placement_note(2, 5, 900.0, True)
    assert placement_note(2, 5, 900.0, False, (150.0, 220.0), 15).endswith(
        "The remaining 150–220 likely patients beyond reach are not within a 15-minute walk of any candidate site.")
    assert placement_note(0, 5, 0.0, False).startswith("No modelled need lies beyond walking reach")
    # 0.2 likely patients is a real (if tiny) remainder, so it is not reported as "no need"
    assert placement_note(0, 5, 0.2, False).startswith("0 of 5 requested sites recommended")


def test_brief_rejects_missing_fields():
    with pytest.raises(KeyError):
        make_brief(_sites(), {"condition": "hypertension"})
