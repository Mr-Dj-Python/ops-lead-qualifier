"""Stage 4: combine the factor scores into one weighted score, decide, and rank.

Inputs are the rule scores (Stage 2) and the industry results (Stage 3). Everything here is
deterministic code. The weighted score is the weighted average of the four factor scores
(the weights add up to 1), so it stays on the same 1-10 scale. See docs/WORKFLOW.md Stage 4.
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pipeline.config import ConfigError, config_error, parse_count, parse_number
from pipeline.models import DecisionSummary, IndustryResult, Lead, LeadDecision, RuleResult

INDUSTRY = "industry"
QUALIFIED, REVIEW, REJECTED = "qualified", "review", "rejected"
FACTOR_LABELS = {INDUSTRY: "Industry fit", "size": "Company size", "source": "Lead source", "recency": "Engagement"}


DECISION_PATHS = {  # settings field -> the config path a user would edit
    "qualified_at": "thresholds.qualified",
    "review_at": "thresholds.review",
    "high_priority_at": "priority.high",
    "stale_days": "caps.stale_days",
    "reject_size_below": "caps.hard_reject_size_below",
    "reject_size_above": "caps.hard_reject_size_above",
}


class DecisionSettings(BaseModel):
    """Weights, thresholds and caps from config.yaml, validated."""

    model_config = ConfigDict(frozen=True)

    weights: dict[str, float]  # factor name -> share of the score, adds up to 1
    qualified_at: float
    review_at: float
    high_priority_at: float
    stale_days: int = Field(ge=0)
    reject_size_below: int = Field(ge=0)
    reject_size_above: int = Field(ge=0)

    @field_validator("qualified_at", "review_at", "high_priority_at", mode="before")
    @classmethod
    def _number(cls, value) -> float:
        return parse_number(value)

    @field_validator("stale_days", "reject_size_below", "reject_size_above", mode="before")
    @classmethod
    def _count(cls, value) -> int:
        return parse_count(value, 0)

    @model_validator(mode="after")
    def _relations(self) -> "DecisionSettings":
        if not 1 <= self.review_at < self.qualified_at <= 10:
            raise ValueError("thresholds must satisfy 1 <= review < qualified <= 10")
        if not self.qualified_at <= self.high_priority_at <= 10:
            raise ValueError("priority.high must be between thresholds.qualified and 10")
        if self.reject_size_below > self.reject_size_above:
            raise ValueError("caps.hard_reject_size_below must not exceed hard_reject_size_above")
        return self

    @classmethod
    def from_config(cls, config: dict, rule_names: list[str]) -> "DecisionSettings":
        weights = config.get("weights")
        if not isinstance(weights, dict):
            raise ConfigError("config 'weights' must be a mapping")
        factors = [INDUSTRY, *rule_names]
        missing = [name for name in factors if name not in weights]
        extra = [name for name in weights if name not in factors]
        if missing:
            raise ConfigError(f"config weights has no entry for: {', '.join(missing)}")
        if extra:
            raise ConfigError(f"config weights names factors that are not scored: {', '.join(extra)}")

        sections = {name: config.get(name) for name in ("thresholds", "priority", "caps")}
        for name, section in sections.items():
            if not isinstance(section, dict):
                raise ConfigError(f"config '{name}' must be a mapping")
        thresholds, priority, caps = sections["thresholds"], sections["priority"], sections["caps"]
        try:
            return cls.model_validate(
                {
                    "weights": dict(weights),
                    "qualified_at": thresholds.get("qualified"),
                    "review_at": thresholds.get("review"),
                    "high_priority_at": priority.get("high"),
                    "stale_days": caps.get("stale_days"),
                    "reject_size_below": caps.get("hard_reject_size_below"),
                    "reject_size_above": caps.get("hard_reject_size_above"),
                }
            )
        except ValidationError as exc:
            raise config_error(exc, DECISION_PATHS) from exc


def decide_leads(
    leads: list[Lead],
    rule_scores: dict[str, dict[str, RuleResult]],
    industry: dict[str, IndustryResult],
    settings: DecisionSettings,
    reference_date: date,
) -> list[LeadDecision]:
    """Decide every lead and return them ranked from best fit to lowest."""
    decisions = [
        _decide(lead, rule_scores[lead.lead_id], industry[lead.lead_id], settings, reference_date) for lead in leads
    ]
    by_id = {lead.lead_id: lead for lead in leads}

    def order(decision: LeadDecision):
        contact = by_id[decision.lead_id].last_interaction_date
        # best score first; then the more recent contact; then lead_id so the order is fixed
        return (-decision.final_score, -(contact.toordinal() if contact else 0), decision.lead_id)

    decisions.sort(key=order)
    priority_rank = 0
    for rank, decision in enumerate(decisions, start=1):
        decision.overall_rank = rank
        if decision.decision == QUALIFIED:
            priority_rank += 1
            decision.priority_rank = priority_rank
    for decision in decisions:
        decision.explanation = _explain(decision, by_id[decision.lead_id], rule_scores[decision.lead_id],
                                        industry[decision.lead_id], settings, reference_date)
    return decisions


def _decide(
    lead: Lead,
    rules: dict[str, RuleResult],
    fit: IndustryResult,
    settings: DecisionSettings,
    reference_date: date,
) -> LeadDecision:
    scores = {INDUSTRY: fit.industry_fit, **{name: result.score for name, result in rules.items()}}
    points = {name: round(scores[name] * settings.weights[name], 2) for name in settings.weights}
    final = round(sum(points.values()), 2)

    if final >= settings.qualified_at:
        decision = QUALIFIED
    elif final >= settings.review_at:
        decision = REVIEW
    else:
        decision = REJECTED
    applied: list[str] = []
    size = lead.company_size

    # A no-fit business of an unrealistic size is rejected whatever its other scores say.
    hard_rejected = False
    if (
        fit.tier == "none"
        and size is not None
        and (size < settings.reject_size_below or size > settings.reject_size_above)
        and decision != REJECTED
    ):
        decision = REJECTED
        hard_rejected = True
        applied.append(
            f"no-fit industry with {size:,} employees (outside {settings.reject_size_below:,}-"
            f"{settings.reject_size_above:,}) is rejected"
        )

    # Uncertain or stale leads can still be reviewed but never qualified.
    if decision == QUALIFIED:
        caps = _caps(lead, fit, settings, reference_date)
        if caps:
            decision = REVIEW
            applied.extend(caps)

    # A lead that could not be scored properly is not rejected on a number that means little:
    # it goes to a person, even when its weighted score is low.
    if decision == REJECTED and not hard_rejected:
        if fit.llm_failed:
            decision = REVIEW
            applied.append("LLM scoring failed, so the lead goes to review instead of being scored")
        elif "insufficient_data" in lead.flags:
            decision = REVIEW
            applied.append("too little data to score; sent to review so the record can be completed")
    if decision == REVIEW and not applied:
        applied.append(f"score is in the review band ({settings.review_at:g} to below {settings.qualified_at:g})")

    return LeadDecision(
        lead_id=lead.lead_id,
        name=lead.name,
        company=lead.company,
        industry_label=lead.industry,
        company_size=size,
        tier=fit.tier,
        confidence=fit.confidence,
        factor_scores=scores,
        factor_points=points,
        final_score=final,
        decision=decision,
        priority=("HIGH" if final >= settings.high_priority_at else "MEDIUM") if decision == QUALIFIED else None,
        overall_rank=0,
        priority_rank=None,
        rules_applied=applied,
        rejection_reason=_rejection_reason(scores, applied) if decision == REJECTED else None,
        flags=sorted(set(lead.flags) | ({"llm_failed"} if fit.llm_failed else set())),
    )


def _caps(lead: Lead, fit: IndustryResult, settings: DecisionSettings, reference_date: date) -> list[str]:
    """Every reason this lead may not be qualified (empty when none applies)."""
    reasons = []
    if fit.llm_failed:
        reasons.append("LLM scoring failed, so the industry is unconfirmed; capped at review")
    elif fit.tier == "none":
        reasons.append('industry tier "none" caps the decision at review')
    elif fit.tier == "unknown":
        reasons.append('industry tier "unknown" caps the decision at review')
    elif fit.confidence == "low":
        reasons.append("low LLM confidence caps the decision at review")
    if "insufficient_data" in lead.flags:
        reasons.append("insufficient data caps the decision at review")
    if lead.last_interaction_date is not None:
        days = (reference_date - lead.last_interaction_date).days
        if days > settings.stale_days:
            reasons.append(f"no contact for {days} days (over {settings.stale_days}) caps the decision at review")
    return reasons


def _rejection_reason(scores: dict[str, int], applied: list[str]) -> str:
    """The hard-reject rule if it fired, otherwise the lowest-scoring factor(s)."""
    if applied:
        return "no-fit industry with size outside the realistic range"
    lowest = min(scores.values())
    return " and ".join(FACTOR_LABELS.get(name, name) for name, score in scores.items() if score == lowest)


def _explain(
    decision: LeadDecision,
    lead: Lead,
    rules: dict[str, RuleResult],
    fit: IndustryResult,
    settings: DecisionSettings,
    reference_date: date,
) -> str:
    who = ", ".join(part for part in (decision.name, decision.company) if part) or decision.lead_id
    facts = [
        f"{lead.company_size:,} people" if lead.company_size is not None else "size not provided",
        lead.industry or "industry not provided",
        lead.source or "source not provided",
    ]
    if lead.last_interaction_date is not None:
        facts.append(f"{max((reference_date - lead.last_interaction_date).days, 0)} days ago")
    verdict = decision.decision.upper() + (f"  Priority: {decision.priority}" if decision.priority else "")
    lines = [f"Lead: {who} ({', '.join(facts)})", f"Score: {decision.final_score:g}/10  Decision: {verdict}"]
    notes = {INDUSTRY: f'"{fit.reason}"', **{name: f"({result.detail})" for name, result in rules.items()}}
    for name, weight in settings.weights.items():
        lines.append(
            f"  {FACTOR_LABELS.get(name, name):<13}({weight * 100:g}%): {decision.factor_scores[name]}/10"
            f" -> {decision.factor_points[name]:.2f} pts  {notes[name]}"
        )
    for rule in decision.rules_applied:
        lines.append(f"  Rule applied: {rule}")
    if decision.flags:
        lines.append(f"  Flags: {', '.join(decision.flags)}")
    return "\n".join(lines)


def add_flag(decision: LeadDecision, flag: str) -> None:
    """Add a flag after ranking (for example message_failed) and keep the explanation in step."""
    if flag in decision.flags:
        return
    decision.flags = sorted({*decision.flags, flag})
    lines = [line for line in decision.explanation.splitlines() if not line.startswith("  Flags: ")]
    lines.append(f"  Flags: {', '.join(decision.flags)}")
    decision.explanation = "\n".join(lines)


def summarize(decisions: list[LeadDecision], settings: DecisionSettings) -> DecisionSummary:
    """Counts, average and range of the weighted score, factor averages, rejection reasons."""
    counts = Counter(decision.decision for decision in decisions)
    scores = [decision.final_score for decision in decisions]
    total = len(decisions)
    return DecisionSummary(
        total=total,
        counts={name: counts.get(name, 0) for name in (QUALIFIED, REVIEW, REJECTED)},
        average_score=round(sum(scores) / total, 2) if total else 0.0,
        highest_score=max(scores, default=0.0),
        lowest_score=min(scores, default=0.0),
        average_factor_scores={
            name: round(sum(d.factor_scores[name] for d in decisions) / total, 2) if total else 0.0
            for name in settings.weights
        },
        rejection_reasons=Counter(d.rejection_reason for d in decisions if d.rejection_reason).most_common(),
    )
