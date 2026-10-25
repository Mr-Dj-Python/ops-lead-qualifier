"""Stage 3: score industry fit with an LLM, in batches.

Flow per batch (docs/WORKFLOW.md Stage 3, docs/LOW_LEVEL_DESIGN.md section 3):
ask -> validate every item -> re-ask only the invalid or missing ones (at most
`validation_retries` times) -> clamp an out-of-range score if the tier is valid, otherwise
fall back to tier "unknown" / low confidence / llm_failed, which Stage 4 caps at review.
Transport problems (rate limits, timeouts) are retried inside LLMAPIClient; a bad API key
stops the run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline.config import ConfigError
from pipeline.ingest import Lead
from pipeline.llm_client import LLMAPIClient, LLMAuthError, LLMError
from pipeline.prompts import TIER_ORDER, build_industry_system_prompt, build_industry_user_prompt

log = logging.getLogger(__name__)

CONFIDENCES = ("high", "medium", "low")
MAX_COMPANY_CHARS = 100
MAX_REASON_CHARS = 200
FAILED_REASON = "LLM scoring failed"


@dataclass(frozen=True)
class IndustryResult:
    lead_id: str
    tier: str  # best | hard | none | unknown
    industry_fit: int  # 1-10, inside the tier's score range
    confidence: str  # high | medium | low
    reason: str
    llm_failed: bool = False

    @property
    def caps_at_review(self) -> bool:
        """Stage 4 may not qualify a lead whose industry is unknown, low confidence or failed."""
        return self.tier == "unknown" or self.confidence == "low" or self.llm_failed


@dataclass
class IndustryScoring:
    results: dict[str, IndustryResult]  # keyed by lead_id, one per lead
    api_calls: int = 0
    cache_hits: int = 0  # leads answered from the cache
    failed_ids: list[str] = field(default_factory=list)  # leads that fell back after retries


@dataclass(frozen=True)
class IndustrySettings:
    system_prompt: str
    tier_ranges: dict[str, tuple[int, int]]
    batch_size: int
    temperature: float
    max_tokens: int
    validation_retries: int
    cache_enabled: bool
    cache_path: Path

    @classmethod
    def from_config(cls, config: dict) -> "IndustrySettings":
        tiers = config.get("tiers")
        if not isinstance(tiers, dict):
            raise ConfigError("config 'tiers' must be a mapping")
        ranges: dict[str, tuple[int, int]] = {}
        for name in TIER_ORDER:
            tier = tiers.get(name)
            bounds = tier.get("score_range") if isinstance(tier, dict) else None
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds)
                or not 1 <= bounds[0] <= bounds[1] <= 10
            ):
                raise ConfigError(f"config tiers.{name}.score_range must be [low, high] within 1-10")
            ranges[name] = (bounds[0], bounds[1])

        llm = config.get("llm")
        scoring = llm.get("scoring") if isinstance(llm, dict) else None
        if not isinstance(scoring, dict):
            raise ConfigError("config llm.scoring must be a mapping")
        batch_size, max_tokens = scoring.get("batch_size"), scoring.get("max_tokens")
        temperature = scoring.get("temperature")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ConfigError("config llm.scoring.batch_size must be a whole number of 1 or more")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ConfigError("config llm.scoring.max_tokens must be a whole number of 1 or more")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
            raise ConfigError("config llm.scoring.temperature must be a number of 0 or more")
        retries = llm.get("validation_retries", 2)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ConfigError("config llm.validation_retries must be a whole number of 0 or more")

        cache = config.get("cache", {})
        if not isinstance(cache, dict):
            raise ConfigError("config 'cache' must be a mapping")
        return cls(
            system_prompt=build_industry_system_prompt(config.get("company", {}), tiers),
            tier_ranges=ranges,
            batch_size=batch_size,
            temperature=float(temperature),
            max_tokens=max_tokens,
            validation_retries=retries,
            cache_enabled=bool(cache.get("enabled", False)),
            cache_path=Path(cache.get("path", ".cache/industry.json")),
        )


# --- cache -----------------------------------------------------------------------------------

def cache_key(lead: Lead) -> str:
    """Identical company, industry and size always get the identical answer."""
    parts = (lead.company or "", lead.industry or "", "" if lead.company_size is None else str(lead.company_size))
    return "|".join(re.sub(r"\s+", " ", part).strip().lower() for part in parts)


class IndustryCache:
    """JSON file of earlier answers. It is discarded when the model or prompt changes."""

    def __init__(self, path: Path, fingerprint: str, enabled: bool = True) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.enabled = enabled
        self._entries: dict[str, dict] = {}
        self._dirty = False
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # no cache yet, or an unreadable one: start empty
        if isinstance(data, dict) and data.get("fingerprint") == self.fingerprint and isinstance(data.get("entries"), dict):
            self._entries = data["entries"]

    def get(self, key: str, lead_id: str, ranges: dict[str, tuple[int, int]]) -> IndustryResult | None:
        entry = self._entries.get(key) if self.enabled else None
        if not isinstance(entry, dict):
            return None
        result, _ = _validate_item({**entry, "lead_id": lead_id}, ranges)
        return result  # a damaged entry fails validation and is simply asked again

    def put(self, key: str, result: IndustryResult) -> None:
        if self.enabled and not result.llm_failed:
            entry = asdict(result)
            for name in ("lead_id", "llm_failed"):
                entry.pop(name)
            self._entries[key] = entry
            self._dirty = True

    def save(self) -> None:
        if not (self.enabled and self._dirty):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"fingerprint": self.fingerprint, "entries": self._entries}, indent=1, sort_keys=True)
        handle, temp_name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temp:
                temp.write(payload)
            os.replace(temp_name, self.path)
        except OSError:
            Path(temp_name).unlink(missing_ok=True)
            log.warning("could not write the industry cache to %s", self.path)
        self._dirty = False


# --- parsing and validation ------------------------------------------------------------------

def _extract_json(raw: str):
    """Parse the reply, tolerating markdown fences and text around the JSON object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found")


def _whole_score(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _validate_item(
    item, ranges: dict[str, tuple[int, int]]
) -> tuple[IndustryResult | None, IndustryResult | None]:
    """Return (valid result, None), or (None, clamped result) when only the score is out of
    range for a valid tier, or (None, None) when the item is unusable."""
    if not isinstance(item, dict) or not isinstance(item.get("lead_id"), str):
        return None, None
    tier, confidence, reason = item.get("tier"), item.get("confidence"), item.get("reason")
    score = _whole_score(item.get("industry_fit"))
    if tier not in ranges or confidence not in CONFIDENCES or score is None:
        return None, None
    if not isinstance(reason, str) or not reason.strip():
        return None, None
    reason = reason.strip()[:MAX_REASON_CHARS]
    low, high = ranges[tier]
    if low <= score <= high:
        return IndustryResult(item["lead_id"], tier, score, confidence, reason), None
    clamped = IndustryResult(item["lead_id"], tier, min(max(score, low), high), confidence, reason)
    return None, clamped


def _fallback(lead_id: str) -> IndustryResult:
    return IndustryResult(lead_id, "unknown", 5, "low", FAILED_REASON, llm_failed=True)


# --- the scorer ------------------------------------------------------------------------------

class IndustryScorer:
    def __init__(self, client: LLMAPIClient, settings: IndustrySettings) -> None:
        self.client = client
        self.settings = settings
        fingerprint = hashlib.sha256(f"{client.model}\n{settings.system_prompt}".encode()).hexdigest()[:16]
        self.cache = IndustryCache(settings.cache_path, fingerprint, settings.cache_enabled)

    def score(self, leads: list[Lead]) -> IndustryScoring:
        """Score every lead. Leads with the same company, industry and size share one answer."""
        groups: dict[str, list[Lead]] = {}
        for lead in leads:
            groups.setdefault(cache_key(lead), []).append(lead)

        scoring = IndustryScoring(results={})
        by_group: dict[str, IndustryResult] = {}
        requests: dict[str, dict] = {}  # representative lead_id -> payload
        owner: dict[str, str] = {}  # representative lead_id -> group key
        for key, members in groups.items():
            representative = members[0]
            cached = self.cache.get(key, representative.lead_id, self.settings.tier_ranges)
            if cached is not None:
                by_group[key] = cached
                scoring.cache_hits += len(members)
                continue
            requests[representative.lead_id] = _payload(representative)
            owner[representative.lead_id] = key

        # Sorted by lead_id so row order never changes which leads share a batch.
        ordered = sorted(requests)
        size = self.settings.batch_size
        try:
            for start in range(0, len(ordered), size):
                batch = {lead_id: requests[lead_id] for lead_id in ordered[start : start + size]}
                for lead_id, result in self._score_batch(batch, scoring).items():
                    by_group[owner[lead_id]] = result
                    self.cache.put(owner[lead_id], result)
        finally:
            self.cache.save()  # keep what succeeded even if a bad API key stops the run

        for key, members in groups.items():
            template = by_group[key]
            for lead in members:
                result = IndustryResult(
                    lead.lead_id, template.tier, template.industry_fit, template.confidence,
                    template.reason, template.llm_failed,
                )
                scoring.results[lead.lead_id] = result
                if result.llm_failed:
                    scoring.failed_ids.append(lead.lead_id)
        return scoring

    def _score_batch(self, batch: dict[str, dict], scoring: IndustryScoring) -> dict[str, IndustryResult]:
        settings = self.settings
        good: dict[str, IndustryResult] = {}
        clamped: dict[str, IndustryResult] = {}
        pending = dict(batch)
        repair = False
        for attempt in range(1 + settings.validation_retries):
            if not pending:
                break
            user_prompt = build_industry_user_prompt(list(pending.values()), repair=repair)
            scoring.api_calls += 1
            try:
                raw = self.client.complete(
                    settings.system_prompt, user_prompt,
                    temperature=settings.temperature, max_tokens=settings.max_tokens,
                )
            except LLMAuthError:
                raise
            except LLMError as exc:
                log.warning("LLM call failed for %d lead(s) after retries: %s", len(pending), exc)
                break  # transport retries are already used up; fall back for what is left
            try:
                data = _extract_json(raw)
                items = data.get("results") if isinstance(data, dict) else data
                if not isinstance(items, list):
                    raise ValueError("reply has no 'results' list")
                repair = False
            except ValueError as exc:
                log.warning("LLM reply was not usable JSON (%s); retrying %d lead(s)", exc, len(pending))
                repair = True
                continue
            for item in items:
                result, clamp = _validate_item(item, settings.tier_ranges)
                lead_id = item.get("lead_id") if isinstance(item, dict) else None
                if not isinstance(lead_id, str) or lead_id not in pending:
                    continue  # unusable, unknown or already-answered id
                if result is not None:
                    good[lead_id] = result
                    del pending[lead_id]
                elif clamp is not None:
                    clamped[lead_id] = clamp
            if pending:
                log.warning("%d of %d lead(s) invalid or missing; retrying", len(pending), len(batch))

        for lead_id in pending:
            if lead_id in clamped:
                log.warning("score for %s was outside its tier range; clamped", lead_id)
                good[lead_id] = clamped[lead_id]
            else:
                log.warning("industry scoring failed for %s; falling back to review", lead_id)
                good[lead_id] = _fallback(lead_id)
        return good


def _payload(lead: Lead) -> dict:
    """Only company, industry and size go to the model (docs/LOW_LEVEL_DESIGN.md 2.3)."""
    return {
        "lead_id": lead.lead_id,
        "company": (lead.company or "(missing)")[:MAX_COMPANY_CHARS],
        "industry": lead.industry or "(missing)",
        "company_size": lead.company_size,
    }
