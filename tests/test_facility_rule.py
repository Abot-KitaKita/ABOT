"""The 'existing chronic care' rule: public health centres and barangay health stations only."""
from __future__ import annotations

import pytest

from abot.build_grid import public_primary_care
from abot.contracts import SPECS


@pytest.mark.parametrize("name,kind", [
    ("Makabayan Health Center", "health_post"),
    ("Unnamed health_post", "health_post"),
    ("Namayan Barangay Health Center", "clinic"),
    ("Burol Puericulture & Health Center", "clinic"),
    ("Some Barangay Health Station", "clinic"),
    ("Riverside Health Centre", "doctors"),
])
def test_counted(name, kind):
    assert public_primary_care(name, kind)


@pytest.mark.parametrize("name,kind", [
    ("The Medical City", "clinic"),
    ("TMS dialysis center", "clinic"),
    ("Mandaluyong City Dialysis Center", "doctors"),
    ("TB DOTS Clinic", "clinic"),
    ("Mandaluyong Social Hygiene Clinic", "doctors"),
    ("School clinic", "clinic"),
    ("Xanderm", "clinic"),
    ("Asian Hyperbaric Healthcare", "doctors"),
    ("Victor R. Potenciano Medical Center", "hospital"),
    ("Some Health Center Hospital", "hospital"),
])
def test_not_counted(name, kind):
    assert not public_primary_care(name, kind)


def test_delivered_facilities_follow_the_rule():
    spec = SPECS["facilities"]
    if not spec.path.exists():
        pytest.skip("facilities.geojson not delivered")
    from abot.contracts import read_file

    fac = read_file("facilities")
    expected = [public_primary_care(n, t) for n, t in zip(fac["name"], fac["type"])]
    assert list(fac["counted"].astype(bool)) == expected
