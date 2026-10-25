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
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from pipeline.config import ConfigError, whole_number
from pipeline.llm_client import LLMAPIClient, LLMAuthError, LLMError, extract_json
from pipeline.models import IndustryItem, IndustryResponse, IndustryResult, Lead, LeadPayload
from pipeline.prompts import TIER_ORDER, build_industry_system_prompt, build_industry_user_prompt

log = logging.getLogger(__name__)

FAILED_REASON = "LLM scoring failed"
TOKENS_PER_LEAD = 80  # reply budget per lead; keeps a large batch from being cut off mid-JSON


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
        batch_size, max_tokens = whole_number(scoring.get("batch_size")), scoring.get("max_tokens")
        temperature = scoring.get("temperature")
        if batch_size is None or batch_size < 1:
            raise ConfigError(
                f"config llm.scoring.batch_size must be a whole number of 1 or more, "
                f"got {scoring.get('batch_size')!r} (set LLM_BATCH_SIZE in .env)"
            )
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


class CacheFile(BaseModel):
    """The on-disk shape of the cache; anything else is treated as no cache."""

    fingerprint: str
    entries: dict[str, dict]


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
            data = CacheFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return  # no cache yet, or an unreadable one: start empty
        if data.fingerprint == self.fingerprint:
            self._entries = data.entries

    def get(self, key: str, lead_id: str, ranges: dict[str, tuple[int, int]]) -> IndustryResult | None:
        entry = self._entries.get(key) if self.enabled else None
        if entry is None:
            return None
        result, _ = _validate_item({**entry, "lead_id": lead_id}, ranges)
        return result  # a damaged entry fails validation and is simply asked again

    def put(self, key: str, result: IndustryResult) -> None:
        if self.enabled and not result.llm_failed:
            self._entries[key] = result.model_dump(exclude={"lead_id", "llm_failed"})
            self._dirty = True

    def save(self) -> None:
        if not (self.enabled and self._dirty):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(CacheFile(fingerprint=self.fingerprint, entries=self._entries).model_dump(), indent=1, sort_keys=True)
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

def _validate_item(
    item, ranges: dict[str, tuple[int, int]]
) -> tuple[IndustryResult | None, IndustryResult | None]:
    """Check one answer with the IndustryItem model.

    Returns (result, None) when valid; (None, clamped result) when only the score is outside
    its tier's range; (None, None) when the item is unusable.
    """
    try:
        valid = IndustryItem.model_validate(item, context={"ranges": ranges})
    except ValidationError as exc:
        errors = exc.errors()
        if all(error["type"] == "score_out_of_range" for error in errors):
            unranged = IndustryItem.model_validate(item)  # the same item, minus the range check
            low, high = ranges[unranged.tier]
            clamped = unranged.model_copy(update={"industry_fit": min(max(unranged.industry_fit, low), high)})
            return None, IndustryResult(**clamped.model_dump())
        return None, None
    return IndustryResult(**valid.model_dump()), None


def _fallback(lead_id: str) -> IndustryResult:
    return IndustryResult(
        lead_id=lead_id, tier="unknown", industry_fit=5, confidence="low", reason=FAILED_REASON, llm_failed=True
    )


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
        requests: dict[str, LeadPayload] = {}  # representative lead_id -> what the model is told
        owner: dict[str, str] = {}  # representative lead_id -> group key
        for key, members in groups.items():
            representative = members[0]
            cached = self.cache.get(key, representative.lead_id, self.settings.tier_ranges)
            if cached is not None:
                by_group[key] = cached
                scoring.cache_hits += len(members)
                continue
            requests[representative.lead_id] = LeadPayload.from_lead(representative)
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
                result = template.model_copy(update={"lead_id": lead.lead_id})
                scoring.results[lead.lead_id] = result
                if result.llm_failed:
                    scoring.failed_ids.append(lead.lead_id)
        return scoring

    def _score_batch(self, batch: dict[str, LeadPayload], scoring: IndustryScoring) -> dict[str, IndustryResult]:
        settings = self.settings
        good: dict[str, IndustryResult] = {}
        clamped: dict[str, IndustryResult] = {}
        pending = dict(batch)
        repair = False
        for attempt in range(1 + settings.validation_retries):
            if not pending:
                break
            user_prompt = build_industry_user_prompt([p.model_dump() for p in pending.values()], repair=repair)
            scoring.api_calls += 1
            try:
                raw = self.client.complete(
                    settings.system_prompt, user_prompt,
                    temperature=settings.temperature,
                    max_tokens=max(settings.max_tokens, TOKENS_PER_LEAD * len(pending)),
                )
            except LLMAuthError:
                raise
            except LLMError as exc:
                log.warning("LLM call failed for %d lead(s) after retries: %s", len(pending), exc)
                break  # transport retries are already used up; fall back for what is left
            try:
                items = IndustryResponse.model_validate(extract_json(raw)).results
                repair = False
            except (ValueError, ValidationError) as exc:  # not JSON, or no "results" list
                log.warning("LLM reply was not usable (%s); retrying %d lead(s)", str(exc).splitlines()[0], len(pending))
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
