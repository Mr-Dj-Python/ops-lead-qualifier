"""Load config.yaml, apply WEIGHT_* environment overrides, and parse dates used by the CLI."""
from __future__ import annotations

import math
import os
import re
from datetime import date
from pathlib import Path
from typing import Mapping

import yaml
from dotenv import dotenv_values

WEIGHT_PREFIX = "WEIGHT_"


class ConfigError(Exception):
    """Raised when the config file or a config value is unusable."""


def parse_iso_date(text: str) -> date:
    """Parse a strict YYYY-MM-DD string."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()):
        raise ValueError(f"expected YYYY-MM-DD, got {text!r}")
    return date.fromisoformat(text.strip())


def load_env(env_file: str | Path | None = None, required: bool = False) -> dict[str, str]:
    """Return the settings visible to the program: values from the .env file, then the real
    environment on top of them (a real environment variable wins over the file).

    The file is read without touching os.environ. A missing default .env is fine; a missing
    file that was asked for explicitly (required=True) is an error.
    """
    path = Path(env_file) if env_file else Path(".env")
    from_file: dict[str, str] = {}
    if path.is_file():
        try:
            from_file = {key: value for key, value in dotenv_values(path).items() if value is not None}
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"cannot read env file {path}: {exc}") from exc
    elif required:
        raise ConfigError(f"env file not found: {path}")
    return {**from_file, **os.environ}


def _apply_weight_overrides(weights: dict, env: Mapping[str, str]) -> list[str]:
    """Replace weights with WEIGHT_<FACTOR> values from env; return the factors overridden."""
    factors = {name.upper(): name for name in weights}
    overridden = []
    for key, raw in env.items():
        if not key.startswith(WEIGHT_PREFIX):
            continue
        factor = factors.get(key[len(WEIGHT_PREFIX):].upper())
        if factor is None:
            expected = ", ".join(f"{WEIGHT_PREFIX}{name}" for name in factors)
            raise ConfigError(f"unknown weight variable {key}; expected one of: {expected}")
        try:
            value = float(raw.strip())
        except ValueError:
            raise ConfigError(f"{key} must be a number, got {raw!r}") from None
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"{key} must be a number of 0 or more, got {raw!r}")
        weights[factor] = value
        overridden.append(factor)
    return overridden


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> dict:
    """Read and validate config.yaml.

    Weights in the file are defaults; WEIGHT_INDUSTRY, WEIGHT_SIZE, WEIGHT_SOURCE and
    WEIGHT_RECENCY in `env` (default: os.environ) override them. The result's
    `weights_overridden` lists the factors taken from the environment.
    """
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
    weights = dict(weights)
    config["weights_overridden"] = _apply_weight_overrides(weights, os.environ if env is None else env)
    if abs(sum(weights.values()) - 1) > 1e-6:
        raise ConfigError(
            f"weights must add up to 1, got {sum(weights.values()):g} "
            f"({', '.join(f'{name} {value:g}' for name, value in weights.items())})"
        )
    config["weights"] = weights

    if not isinstance(config.get("rules"), dict) or not config["rules"]:
        raise ConfigError("config 'rules' must be a non-empty mapping")
    return config
