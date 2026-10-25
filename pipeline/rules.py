"""Stage 2: deterministic rule scoring for company size, lead source and recency.

Each attribute is scored by a RuleScorer. RuleFactory builds the scorers named in
config.yaml from a registry, so adding an attribute rule is one new class (decorated with
@register_rule) plus one entry under `rules:` in the config. Bands and scores live in the
config, never in code.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from pipeline.config import ConfigError
from pipeline.ingest import Lead


class RuleConfigError(ConfigError):
    """Raised when a rule's settings are missing or invalid."""


@dataclass(frozen=True)
class RuleResult:
    rule: str
    score: int  # 1-10
    detail: str  # short reason, reused in the per-lead explanation


class RuleScorer(ABC):
    """Scores one attribute of a lead on a 1-10 scale."""

    name: str  # set by @register_rule

    @abstractmethod
    def __init__(self, settings: dict) -> None:
        """Validate and store this rule's settings from config.yaml."""

    @abstractmethod
    def score(self, lead: Lead, reference_date: date) -> RuleResult:
        """Return the score for this lead."""


_REGISTRY: dict[str, type[RuleScorer]] = {}


def register_rule(name: str):
    """Class decorator that makes a RuleScorer available to RuleFactory under `name`."""

    def decorator(cls: type[RuleScorer]) -> type[RuleScorer]:
        if name in _REGISTRY:
            raise ValueError(f"rule {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


class RuleFactory:
    """Builds RuleScorer objects from the `rules` section of config.yaml."""

    @staticmethod
    def create(name: str, settings: dict) -> RuleScorer:
        try:
            cls = _REGISTRY[name]
        except KeyError:
            known = ", ".join(sorted(_REGISTRY))
            raise RuleConfigError(f"unknown rule {name!r}; available rules: {known}") from None
        if not isinstance(settings, dict):
            raise RuleConfigError(f"rule {name!r} settings must be a mapping")
        return cls(settings)

    @classmethod
    def create_all(cls, rules_config: dict) -> list[RuleScorer]:
        if not isinstance(rules_config, dict) or not rules_config:
            raise RuleConfigError("config 'rules' must be a non-empty mapping of rule name to settings")
        return [cls.create(name, settings) for name, settings in rules_config.items()]


# --- config helpers ------------------------------------------------------------------------

def _require(settings: dict, key: str, rule: str):
    if key not in settings:
        raise RuleConfigError(f"rule {rule!r} is missing setting {key!r}")
    return settings[key]


def _score_value(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise RuleConfigError(f"{where} must be a whole number from 1 to 10, got {value!r}")
    return value


def _whole_number(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuleConfigError(f"{where} must be a whole number of 0 or more, got {value!r}")
    return value


# --- rules ---------------------------------------------------------------------------------

@register_rule("size")
class SizeRule(RuleScorer):
    """Employee count mapped to a score by inclusive bands; the first matching band wins."""

    def __init__(self, settings: dict) -> None:
        bands = _require(settings, "bands", "size")
        if not isinstance(bands, list) or not bands:
            raise RuleConfigError("rule 'size' setting 'bands' must be a non-empty list")
        self.bands = [self._parse_band(band, index) for index, band in enumerate(bands, start=1)]
        self.missing_score = _score_value(_require(settings, "missing_score", "size"), "size missing_score")

    @staticmethod
    def _parse_band(band, index: int) -> tuple[int | None, int | None, int, str]:
        where = f"size band {index}"
        if not isinstance(band, dict):
            raise RuleConfigError(f"{where} must be a mapping")
        low = band.get("min")
        high = band.get("max")
        if low is None and high is None:
            raise RuleConfigError(f"{where} needs 'min', 'max' or both")
        low = None if low is None else _whole_number(low, f"{where} min")
        high = None if high is None else _whole_number(high, f"{where} max")
        if low is not None and high is not None and low > high:
            raise RuleConfigError(f"{where} has min greater than max")
        score = _score_value(band.get("score"), f"{where} score")
        return low, high, score, str(band.get("label", ""))

    def score(self, lead: Lead, reference_date: date) -> RuleResult:
        size = lead.company_size
        if size is None:
            return RuleResult(self.name, self.missing_score, "company size not provided (scored neutral)")
        for low, high, score, label in self.bands:
            if (low is None or size >= low) and (high is None or size <= high):
                detail = f"{size} employees" + (f", {label}" if label else "")
                return RuleResult(self.name, score, detail)
        raise RuleConfigError(f"rule 'size': no band covers {size} employees")


@register_rule("source")
class SourceRule(RuleScorer):
    """How the lead reached us, looked up case-insensitively in a score table."""

    def __init__(self, settings: dict) -> None:
        scores = _require(settings, "scores", "source")
        if not isinstance(scores, dict) or not scores:
            raise RuleConfigError("rule 'source' setting 'scores' must be a non-empty mapping")
        self.scores = {
            self._normalize(str(source)): _score_value(score, f"source score for {source!r}")
            for source, score in scores.items()
        }
        self.unknown_score = _score_value(_require(settings, "unknown_score", "source"), "source unknown_score")

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def score(self, lead: Lead, reference_date: date) -> RuleResult:
        if lead.source is None:
            return RuleResult(self.name, self.unknown_score, "source not provided (default score)")
        score = self.scores.get(self._normalize(lead.source))
        if score is None:
            return RuleResult(self.name, self.unknown_score, f"source {lead.source!r} not recognized (default score)")
        return RuleResult(self.name, score, lead.source)


@register_rule("recency")
class RecencyRule(RuleScorer):
    """Days between the last interaction and the reference date, mapped by bands."""

    def __init__(self, settings: dict) -> None:
        bands = _require(settings, "bands", "recency")
        if not isinstance(bands, list) or not bands:
            raise RuleConfigError("rule 'recency' setting 'bands' must be a non-empty list")
        parsed = []
        for index, band in enumerate(bands, start=1):
            where = f"recency band {index}"
            if not isinstance(band, dict):
                raise RuleConfigError(f"{where} must be a mapping")
            parsed.append(
                (
                    _whole_number(band.get("max_days"), f"{where} max_days"),
                    _score_value(band.get("score"), f"{where} score"),
                )
            )
        if [days for days, _ in parsed] != sorted({days for days, _ in parsed}):
            raise RuleConfigError("recency bands must have strictly increasing max_days")
        self.bands = parsed
        self.older_score = _score_value(_require(settings, "older_score", "recency"), "recency older_score")
        self.missing_score = _score_value(_require(settings, "missing_score", "recency"), "recency missing_score")

    def score(self, lead: Lead, reference_date: date) -> RuleResult:
        if lead.last_interaction_date is None:
            return RuleResult(self.name, self.missing_score, "last interaction date not provided")
        days = max((reference_date - lead.last_interaction_date).days, 0)
        for max_days, score in self.bands:
            if days <= max_days:
                return RuleResult(self.name, score, f"{days} days since contact")
        return RuleResult(self.name, self.older_score, f"{days} days since contact")


# --- running the rules ---------------------------------------------------------------------

def score_lead(lead: Lead, scorers: list[RuleScorer], reference_date: date) -> dict[str, RuleResult]:
    """Score one lead with every rule; results are keyed by rule name."""
    return {scorer.name: scorer.score(lead, reference_date) for scorer in scorers}


def score_leads(
    leads: list[Lead], scorers: list[RuleScorer], reference_date: date
) -> dict[str, dict[str, RuleResult]]:
    """Score every lead; results are keyed by lead_id, then rule name."""
    return {lead.lead_id: score_lead(lead, scorers, reference_date) for lead in leads}
