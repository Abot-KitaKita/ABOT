"""Plain-language sentences for areas and chosen sites.

Format (numbers illustrative only):
  "About 1,900–2,600 likely patients; nearest listed facility for referral is a 24-minute walk away; lowest wealth quintile."
The walk is measured from the site itself, so the clause says "for referral" (the people the site
reaches are mostly beyond a 15-minute walk; the site may sit close to a facility).
A clause whose value is missing is left out; the text never contains 'nan'.
"""
from __future__ import annotations

import math

QUINTILE_WORDS = {
    1: "lowest wealth quintile",
    2: "second-lowest wealth quintile",
    3: "middle wealth quintile",
    4: "second-highest wealth quintile",
    5: "highest wealth quintile",
}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def round_people(n: float) -> int:
    """Round a modelled count so it does not look more precise than it is."""
    if n < 100:
        return int(round(n))
    if n < 1000:
        return int(round(n, -1))
    return int(round(n, -2))


def people_phrase(low: float, high: float) -> str:
    lo, hi = sorted((round_people(low), round_people(high)))
    if hi < 1:
        return "Fewer than 1 likely patient"
    if lo == hi:
        return f"About {lo:,} likely patient{'s' if lo != 1 else ''}"
    return f"About {lo:,}–{hi:,} likely patients"


def article(n: int) -> str:
    """'an' before numbers read with a vowel sound (8, 11, 18, 80-89, 800-899), else 'a'."""
    return "an" if n in (8, 11, 18) or str(n).startswith("8") else "a"


def sentence(low: float, high: float, nearest_walk_min: float | None, wealth_quintile: int | None) -> str:
    """Build one sentence from a low–high need range, nearest walk time (minutes) and wealth quintile."""
    if _missing(low) or _missing(high):
        raise ValueError("low and high are required")
    parts = [people_phrase(float(low), float(high))]
    if not _missing(nearest_walk_min):
        minutes = max(1, int(round(float(nearest_walk_min))))
        parts.append(f"nearest listed facility for referral is {article(minutes)} {minutes}-minute walk away")
    if not _missing(wealth_quintile) and int(wealth_quintile) in QUINTILE_WORDS:
        parts.append(QUINTILE_WORDS[int(wealth_quintile)])
    return "; ".join(parts) + "."
