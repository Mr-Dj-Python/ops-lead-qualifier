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
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pipeline.config import ConfigError, config_error, parse_count, parse_number
from pipeline.llm_client import LLMAPIClient, LLMAuthError, LLMError, extract_json
from pipeline.models import (
    IndustryItem,
    IndustryProvenance,
    IndustryResponse,
    IndustryResult,
    IndustryScoring,
    Issue,
    Lead,
    LeadPayload,
)
from pipeline.prompts import TIER_ORDER, build_industry_system_prompt, build_industry_user_prompt

log = logging.getLogger(__name__)

FAILED_REASON = "LLM scoring failed"
TOKENS_PER_LEAD = 80  # reply budget per lead; keeps a large batch from being cut off mid-JSON


PATHS = {  # settings field -> the config path a user would edit
    "batch_size": "llm.scoring.batch_size",
    "max_tokens": "llm.scoring.max_tokens",
    "temperature": "llm.scoring.temperature",
    "validation_retries": "llm.validation_retries",
    "tier_ranges": "",  # its messages already name the tier
}


class IndustrySettings(BaseModel):
    """Everything the industry stage needs from config.yaml, validated."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str
    tier_ranges: dict[str, tuple[int, int]]  # tier -> (lowest, highest) allowed score
    batch_size: int = Field(ge=1)
    temperature: float = Field(ge=0)
    max_tokens: int = Field(ge=1)
    validation_retries: int = Field(ge=0)
    cache_enabled: bool
    cache_path: Path

    @field_validator("tier_ranges", mode="before")
    @classmethod
    def _tier_ranges(cls, value) -> dict:
        """Raw input is {tier: score_range as written in config}; every tier must be present."""
        ranges = {}
        for name in TIER_ORDER:
            bounds = value.get(name) if isinstance(value, dict) else None
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds)
                or not 1 <= bounds[0] <= bounds[1] <= 10
            ):
                raise ValueError(f"tiers.{name}.score_range must be [low, high] within 1-10")
            ranges[name] = (bounds[0], bounds[1])
        return ranges

    @field_validator("batch_size", mode="before")
    @classmethod
    def _batch_size(cls, value) -> int:
        return parse_count(value, 1, " (set LLM_BATCH_SIZE in .env)")

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _max_tokens(cls, value) -> int:
        return parse_count(value, 1)

    @field_validator("validation_retries", mode="before")
    @classmethod
    def _validation_retries(cls, value) -> int:
        return parse_count(value, 0)

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value) -> float:
        return parse_number(value, 0)

    @classmethod
    def from_config(cls, config: dict) -> "IndustrySettings":
        tiers = config.get("tiers")
        if not isinstance(tiers, dict):
            raise ConfigError("config 'tiers' must be a mapping")
        llm = config.get("llm")
        scoring = llm.get("scoring") if isinstance(llm, dict) else None
        if not isinstance(scoring, dict):
            raise ConfigError("config llm.scoring must be a mapping")
        cache = config.get("cache", {})
        if not isinstance(cache, dict):
            raise ConfigError("config 'cache' must be a mapping")
        try:
            settings = cls.model_validate(
                {
                    "system_prompt": "",  # built below, once the tiers are known to be valid
                    "tier_ranges": {
                        name: tiers[name].get("score_range") if isinstance(tiers.get(name), dict) else None
                        for name in TIER_ORDER
                    },
                    "batch_size": scoring.get("batch_size"),
                    "temperature": scoring.get("temperature"),
                    "max_tokens": scoring.get("max_tokens"),
                    "validation_retries": llm.get("validation_retries", 2),
                    "cache_enabled": bool(cache.get("enabled", False)),
                    "cache_path": Path(cache.get("path", ".cache/industry.json")),
                }
            )
        except ValidationError as exc:
            raise config_error(exc, PATHS) from exc
        prompt = build_industry_system_prompt(config.get("company", {}), tiers)
        return settings.model_copy(update={"system_prompt": prompt})


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
                for member in members:
                    scoring.provenance[member.lead_id] = IndustryProvenance(source="cache", request_id=None)
                continue
            requests[representative.lead_id] = LeadPayload.from_lead(representative)
            owner[representative.lead_id] = key

        # Sorted by lead_id so row order never changes which leads share a batch.
        ordered = sorted(requests)
        size = self.settings.batch_size
        total_batches = -(-len(ordered) // size)
        log.info(
            "industry fit: %d lead(s), %d unique request(s), %d answered from cache; %d batch(es) of up to %d",
            len(leads), len(requests), scoring.cache_hits, total_batches, size,
        )
        try:
            for number, start in enumerate(range(0, len(ordered), size), start=1):
                batch = {lead_id: requests[lead_id] for lead_id in ordered[start : start + size]}
                for lead_id, result in self._score_batch(batch, scoring, number, total_batches).items():
                    by_group[owner[lead_id]] = result
                    self.cache.put(owner[lead_id], result)
        finally:
            self.cache.save()  # keep what succeeded even if a bad API key stops the run

        for key, members in groups.items():
            template = by_group[key]
            representative = members[0].lead_id
            for lead in members:
                result = template.model_copy(update={"lead_id": lead.lead_id})
                scoring.results[lead.lead_id] = result
                if lead.lead_id != representative and representative in scoring.provenance:
                    shared = scoring.provenance[representative]
                    scoring.provenance[lead.lead_id] = shared.model_copy(update={"request_id": representative})
                if result.llm_failed:
                    scoring.failed_ids.append(lead.lead_id)
        for lead_id in scoring.failed_ids:
            scoring.issues.append(
                Issue(stage="industry", kind="fell_back_to_review", lead_id=lead_id,
                      detail="the model gave no usable industry answer, so the lead goes to review")
            )
        log.info(
            "industry fit done: %d lead(s), %d API call(s), %d validation re-ask(s), %d fell back to review",
            len(scoring.results), scoring.api_calls, scoring.validation_retries, len(scoring.failed_ids),
        )
        return scoring

    def _score_batch(
        self, batch: dict[str, LeadPayload], scoring: IndustryScoring, number: int, total: int
    ) -> dict[str, IndustryResult]:
        settings = self.settings
        good: dict[str, IndustryResult] = {}
        clamped: dict[str, IndustryResult] = {}
        pending = dict(batch)
        attempts = {lead_id: 0 for lead_id in batch}
        repair = False
        scoring.batch_sizes.append(len(batch))
        log.info("industry batch %d/%d: %d lead(s)", number, total, len(batch))
        for attempt in range(1 + settings.validation_retries):
            if not pending:
                break
            if attempt:
                scoring.validation_retries += 1
            for lead_id in pending:
                attempts[lead_id] += 1
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
                scoring.issues.append(
                    Issue(stage="industry", kind="llm_call_failed", detail=f"batch {number}: {exc}")
                )
                break  # transport retries are already used up; fall back for what is left
            try:
                items = IndustryResponse.model_validate(extract_json(raw)).results
                repair = False
            except (ValueError, ValidationError) as exc:  # not JSON, or no "results" list
                log.warning("LLM reply was not usable (%s); retrying %d lead(s)", str(exc).splitlines()[0], len(pending))
                scoring.issues.append(
                    Issue(stage="industry", kind="unusable_reply", detail=f"batch {number}, ask {attempt + 1}: {str(exc).splitlines()[0]}")
                )
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
                scoring.issues.append(
                    Issue(stage="industry", kind="score_clamped", lead_id=lead_id,
                          detail="the score was outside its tier's range after every retry, so it was clamped into the range")
                )
                good[lead_id] = clamped[lead_id]
            else:
                log.warning("industry scoring failed for %s; falling back to review", lead_id)
                good[lead_id] = _fallback(lead_id)
        for lead_id, result in good.items():
            scoring.provenance[lead_id] = IndustryProvenance(
                source="fallback" if result.llm_failed else "llm", request_id=lead_id, batch=number,
                attempts=attempts[lead_id],
            )
        return good
