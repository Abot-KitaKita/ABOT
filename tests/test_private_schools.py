"""Private schools are dropped by the name list. The two public colleges stay."""
from __future__ import annotations

import pytest

from abot.build_grid import is_private_school
from abot.contracts import SPECS


@pytest.mark.parametrize("name", [
    "St. Augustine School of Nursing",
    "La Salle Green Hills",
    "St. Therese Private School",
    "Don Bosco Technical College",
    "International Baptist College",
    "International Baptists College",
])
def test_private_school_is_dropped(name):
    assert is_private_school(name)


@pytest.mark.parametrize("name", [
    "Mandaluyong City College",
    "Rizal Technological University",
    "Amado T. Reyes Elementary School",
    "School (Wack-wack Greenhills)",
])
def test_public_or_unnamed_school_stays(name):
    assert not is_private_school(name)


def test_delivered_candidates_have_no_private_names():
    spec = SPECS["candidates"]
    if not spec.path.exists():
        pytest.skip("candidates.geojson not delivered")
    from abot.contracts import read_file

    cand = read_file("candidates")
    assert not cand["name"].map(is_private_school).any()
