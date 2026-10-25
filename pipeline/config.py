"""Load config.yaml (filling ${NAME} placeholders from .env) and parse dates used by the CLI."""
from __future__ import annotations

import math
import os
import re
from datetime import date
from pathlib import Path
from typing import Mapping

import yaml
from dotenv import dotenv_values

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


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


def _expand(value, env: Mapping[str, str], where: str):
    """Replace ${NAME} in every string of the loaded YAML with the value of NAME from env.

    ${NAME:-fallback} uses `fallback` when NAME is unset or empty; plain ${NAME} must be set.
    """
    if isinstance(value, str):

        def replace(match: re.Match) -> str:
            name, fallback = match.group(1), match.group(2)
            if fallback is not None:
                return env.get(name) or fallback
            if name not in env:
                raise ConfigError(f"config {where} refers to ${{{name}}}, which is not set; add {name} to .env")
            return env[name]

        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand(item, env, f"{where}.{key}" if where else str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, env, f"{where}[{index}]") for index, item in enumerate(value)]
    return value


def _parse_weights(weights) -> dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        raise ConfigError("config 'weights' must map each factor to a number of 0 or more")
    parsed = {}
    for name, raw in weights.items():
        try:
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw.strip()) if isinstance(raw, str) else float(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"config weights.{name} must be a number, got {raw!r}") from None
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"config weights.{name} must be a number of 0 or more, got {raw!r}")
        parsed[name] = value
    return parsed


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> dict:
    """Read and validate config.yaml.

    Any `${NAME}` in the file is replaced by NAME from `env` (default: os.environ). The
    weights are written that way, so their values live in .env.
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

    config = _expand(config, os.environ if env is None else env, "")

    missing_values = config.get("missing_values")
    if not isinstance(missing_values, list) or not all(isinstance(v, str) for v in missing_values):
        raise ConfigError("config 'missing_values' must be a list of strings")

    weights = _parse_weights(config.get("weights"))
    if abs(sum(weights.values()) - 1) > 1e-6:
        raise ConfigError(
            f"weights must add up to 1, got {sum(weights.values()):g} "
            f"({', '.join(f'{name} {value:g}' for name, value in weights.items())})"
        )
    config["weights"] = weights

    if not isinstance(config.get("rules"), dict) or not config["rules"]:
        raise ConfigError("config 'rules' must be a non-empty mapping")
    return config
