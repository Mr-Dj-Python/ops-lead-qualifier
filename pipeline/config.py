"""Load config.yaml and parse dates used by the CLI."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when the config file or a config value is unusable."""


def parse_iso_date(text: str) -> date:
    """Parse a strict YYYY-MM-DD string."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()):
        raise ValueError(f"expected YYYY-MM-DD, got {text!r}")
    return date.fromisoformat(text.strip())


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError(f"config {path} must contain a mapping at the top level")

    missing_values = config.get("missing_values")
    if not isinstance(missing_values, list) or not all(isinstance(v, str) for v in missing_values):
        raise ConfigError("config 'missing_values' must be a list of strings")

    weights = config.get("weights")
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(isinstance(w, (int, float)) and not isinstance(w, bool) and w >= 0 for w in weights.values())
    ):
        raise ConfigError("config 'weights' must map each factor to a number of 0 or more")
    if abs(sum(weights.values()) - 1) > 1e-6:
        raise ConfigError(f"config 'weights' must add up to 1, got {sum(weights.values()):g}")

    if not isinstance(config.get("rules"), dict) or not config["rules"]:
        raise ConfigError("config 'rules' must be a non-empty mapping")
    return config
