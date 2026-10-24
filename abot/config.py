"""Load config.yaml and look up assumptions safely.

prevalence() refuses to return a value that has no cited source, so an uncited
number can never flow into the results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from abot import paths
from abot.contracts import CONDITIONS, SCENARIOS


class ConfigError(ValueError):
    """Raised when a needed assumption is missing or has no source."""


def load(path: Path | str | None = None) -> dict[str, Any]:
    """Read config.yaml (or another path) and return it as a dict."""
    path = Path(path) if path else paths.CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("walk_speed_kmh", "walk_minutes", "equity_lambda", "seed", "prevalence"):
        if key not in cfg:
            raise ConfigError(f"config.yaml is missing '{key}'")
    return cfg


def walk_limit_m(cfg: dict[str, Any], minutes: float | None = None, speed_kmh: float | None = None) -> float:
    """Network distance in metres walkable in `minutes` at `speed_kmh` (defaults from config)."""
    minutes = cfg["walk_minutes"] if minutes is None else minutes
    speed_kmh = cfg["walk_speed_kmh"] if speed_kmh is None else speed_kmh
    if minutes <= 0 or speed_kmh <= 0:
        raise ConfigError("minutes and speed_kmh must be positive")
    return speed_kmh * 1000.0 / 60.0 * minutes


def prevalence(cfg: dict[str, Any], condition: str, scenario: str) -> float:
    """Return the cited prevalence share for a condition and scenario.

    Raises ConfigError if the value or its source is missing.
    """
    if condition not in CONDITIONS:
        raise ConfigError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if scenario not in SCENARIOS:
        raise ConfigError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    block = (cfg.get("prevalence") or {}).get(condition) or {}
    value = block.get(scenario)
    source = block.get(f"{scenario}_source")
    if value is None or not source:
        raise ConfigError(
            f"No cited prevalence for {condition}/{scenario}. "
            f"Fill prevalence.{condition}.{scenario} and {scenario}_source in config.yaml "
            "(owner B) and record it in docs/prevalence_sources.md."
        )
    value = float(value)
    if not 0 < value < 1:
        raise ConfigError(f"prevalence {condition}/{scenario} must be a share between 0 and 1, got {value}")
    return value


def available_conditions(cfg: dict[str, Any]) -> list[str]:
    """Conditions with a cited value for all three scenarios (the app shows only these)."""
    ok = []
    for condition in CONDITIONS:
        try:
            for scenario in SCENARIOS:
                prevalence(cfg, condition, scenario)
        except ConfigError:
            continue
        ok.append(condition)
    return ok
