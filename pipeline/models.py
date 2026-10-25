"""Pydantic models for every record that moves through the pipeline, in order:

    CSV row -> LeadRow -> Lead -> RuleResult / LeadPayload -> (LLM) IndustryResponse ->
    IndustryItem -> IndustryResult -> LeadDecision -> MessagePayload -> (LLM) MessageResponse ->
    MessageItem -> GeneratedMessage

A value that fails a model is rejected at the boundary where it appears, with a message that
names the field, instead of surfacing later as a wrong score. Rules that depend on config or
on the run (placeholder words, the reference date, tier score ranges) are passed in through
pydantic's validation `context`.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticCustomError

from pipeline.message_checks import check_message

Tier = Literal["best", "hard", "none", "unknown"]
Confidence = Literal["high", "medium", "low"]
Decision = Literal["qualified", "review", "rejected"]
Priority = Literal["HIGH", "MEDIUM"]
Variant = Literal["warm_followup", "value_led", "reengagement"]
Score = Annotated[int, Field(ge=1, le=10)]

MAX_COMPANY_CHARS = 100
MAX_REASON_CHARS = 200


def clean_text(value: Any, context: dict | None) -> str | None:
    """Trim, collapse spaces, and turn blanks and placeholders such as 'NA' into None.

    Placeholder words come from the validation context key `missing_values` (lower case).
    """
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    placeholders = (context or {}).get("missing_values", frozenset())
    if not text or text.lower() in placeholders:
        return None
    return text


# --- ingestion -------------------------------------------------------------------------------

class LeadRow(BaseModel):
    """One CSV data row. Validate with context={"missing_values": {...}, "reference_date": date}.

    A bad company_size or last_interaction_date fails its own field, so ingestion can turn it
    into a note and keep the lead. A row with neither name nor company fails as a whole.
    """

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    company: str | None = None
    company_size: int | None = None
    industry: str | None = None
    source: str | None = None
    last_interaction_date: date | None = None

    @field_validator("name", "company", "industry", "source", mode="before")
    @classmethod
    def _text(cls, value: Any, info: ValidationInfo) -> str | None:
        return clean_text(value, info.context)

    @field_validator("company_size", mode="before")
    @classmethod
    def _size(cls, value: Any, info: ValidationInfo) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            if value >= 1:
                return value
            raise ValueError(f"company_size is not a positive whole number: {value!r}")
        text = clean_text(value, info.context)
        if text is None:
            return None
        digits = text.replace(",", "")
        if re.fullmatch(r"\d+", digits) and int(digits) >= 1:
            return int(digits)
        raise ValueError(f"company_size is not a positive whole number: {text!r}")

    @field_validator("last_interaction_date", mode="before")
    @classmethod
    def _date(cls, value: Any, info: ValidationInfo) -> date | None:
        reference = (info.context or {}).get("reference_date") or date.today()
        parsed: date | None = None
        if isinstance(value, date):
            parsed, text = value, value.isoformat()
        else:
            text = clean_text(value, info.context)
            if text is None:
                return None
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                try:
                    parsed = date.fromisoformat(text)
                except ValueError:
                    parsed = None
        if parsed is None:
            raise ValueError(f"last_interaction_date is not a valid YYYY-MM-DD date: {text!r}")
        if parsed > reference:
            raise ValueError(f"last_interaction_date {text} is after the reference date {reference}")
        return parsed

    @model_validator(mode="after")
    def _someone_is_named(self) -> "LeadRow":
        if self.name is None and self.company is None:
            raise ValueError("both name and company are missing")
        return self


class Lead(BaseModel):
    """A cleaned lead with its id. `notes` records values that could not be parsed and `flags`
    marks data problems such as insufficient_data."""

    model_config = ConfigDict(strict=True)  # no "50" for a size; only real types

    lead_id: str = Field(pattern=r"^L\d{3,}$")
    name: str | None = None
    company: str | None = None
    company_size: int | None = Field(default=None, ge=1)
    industry: str | None = None
    source: str | None = None
    last_interaction_date: date | None = None
    notes: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


# --- rule scoring ----------------------------------------------------------------------------

class RuleResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    rule: str
    score: Score
    detail: str  # short reason, reused in the per-lead explanation


# --- LLM request and response ----------------------------------------------------------------

class LeadPayload(BaseModel):
    """What the model is told about a lead: only company, industry and size."""

    model_config = ConfigDict(strict=True)

    lead_id: str
    company: str = Field(max_length=MAX_COMPANY_CHARS)
    industry: str
    company_size: int | None

    @classmethod
    def from_lead(cls, lead: Lead) -> "LeadPayload":
        return cls(
            lead_id=lead.lead_id,
            company=(lead.company or "(missing)")[:MAX_COMPANY_CHARS],
            industry=lead.industry or "(missing)",
            company_size=lead.company_size,
        )


class IndustryResponse(BaseModel):
    """The reply envelope: {"results": [...]}. A bare list is accepted too. Items are checked
    one by one with IndustryItem, so one bad item does not discard its batch mates."""

    results: list[Any]

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_list(cls, data: Any) -> Any:
        return {"results": data} if isinstance(data, list) else data


class IndustryItem(BaseModel):
    """One answer from the model. Validate with context={"ranges": {tier: (low, high)}} to
    also require the score to sit inside its tier's range; a score outside it fails with the
    error type `score_out_of_range`, which the scorer treats as clampable."""

    model_config = ConfigDict(extra="ignore")

    lead_id: str
    tier: Tier
    industry_fit: int
    confidence: Confidence
    reason: str = Field(min_length=1)

    @field_validator("industry_fit", mode="before")
    @classmethod
    def _whole_score(cls, value: Any) -> int:
        # No booleans, no text such as "10"; 9.0 is fine because it is a whole number.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("industry_fit must be a whole number")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("industry_fit must be a whole number")
        return int(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _tidy_reason(cls, value: Any) -> Any:
        return value.strip()[:MAX_REASON_CHARS] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _score_matches_tier(self, info: ValidationInfo) -> "IndustryItem":
        ranges = (info.context or {}).get("ranges")
        if ranges:
            low, high = ranges[self.tier]
            if not low <= self.industry_fit <= high:
                raise PydanticCustomError(
                    "score_out_of_range",
                    "score {score} is outside the {tier} range {low}-{high}",
                    {"score": self.industry_fit, "tier": self.tier, "low": low, "high": high},
                )
        return self


class IndustryResult(BaseModel):
    """A validated industry-fit answer for one lead (or the fallback when scoring failed)."""

    model_config = ConfigDict(frozen=True, strict=True)

    lead_id: str
    tier: Tier
    industry_fit: Score
    confidence: Confidence
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)
    llm_failed: bool = False

    @property
    def caps_at_review(self) -> bool:
        """Stage 4 may not qualify a lead whose industry is unknown, low confidence or failed."""
        return self.tier == "unknown" or self.confidence == "low" or self.llm_failed


# --- decision --------------------------------------------------------------------------------

class LeadDecision(BaseModel):
    """The final record for a lead. Assignments are validated too, so a rank or explanation
    set after ranking cannot put the record in an inconsistent state."""

    model_config = ConfigDict(validate_assignment=True, strict=True)

    lead_id: str
    name: str | None
    company: str | None
    industry_label: str | None  # the industry text from the CSV
    company_size: int | None = Field(ge=1)
    tier: Tier
    confidence: Confidence
    factor_scores: dict[str, Score]
    factor_points: dict[str, float]  # score x weight; adds up to final_score
    final_score: float = Field(ge=1, le=10)
    decision: Decision
    priority: Priority | None  # qualified leads only
    overall_rank: int = Field(ge=0)  # 1 = best fit of all leads; 0 until the leads are ranked
    priority_rank: int | None = Field(ge=1)  # 1 = first qualified lead to contact
    rules_applied: list[str]
    rejection_reason: str | None
    flags: list[str]
    explanation: str = ""

    @model_validator(mode="after")
    def _fields_match_the_decision(self) -> "LeadDecision":
        qualified = self.decision == "qualified"
        if (self.priority is not None) != qualified:
            raise ValueError("priority is set for qualified leads only")
        if self.priority_rank is not None and not qualified:
            raise ValueError("priority_rank is set for qualified leads only")
        if (self.rejection_reason is not None) != (self.decision == "rejected"):
            raise ValueError("rejection_reason is set for rejected leads only")
        return self


# --- messages --------------------------------------------------------------------------------

class MessagePayload(BaseModel):
    """What the model is told about a qualified lead when it writes the email."""

    model_config = ConfigDict(strict=True)

    lead_id: str
    first_name: str
    company: str
    industry: str
    source: str
    variant: Variant


class MessageResponse(BaseModel):
    """The reply envelope: {"messages": [...]}. A bare list is accepted too. Items are checked
    one by one with MessageItem."""

    messages: list[Any]

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_list(cls, data: Any) -> Any:
        return {"messages": data} if isinstance(data, list) else data


class MessageItem(BaseModel):
    """One email from the model. Validate with context={"message_rules": MessageRules(...)} to
    apply every message rule; all problems found are reported together, joined by " | "."""

    model_config = ConfigDict(extra="ignore")

    lead_id: str
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)

    @model_validator(mode="after")
    def _meets_the_message_rules(self, info: ValidationInfo) -> "MessageItem":
        rules = (info.context or {}).get("message_rules")
        if rules is not None:
            problems = check_message(self.subject, self.body, rules)
            if problems:
                raise ValueError(" | ".join(problems))
        return self


class GeneratedMessage(BaseModel):
    """A message that passed every rule, ready for a rep to send."""

    model_config = ConfigDict(frozen=True, strict=True)

    lead_id: str
    variant: Variant
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    word_count: int = Field(ge=1)
