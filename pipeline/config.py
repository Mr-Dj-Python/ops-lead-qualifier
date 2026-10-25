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
    return config
