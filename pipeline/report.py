"""Stage 6: the run report (output_report.json), its workflow checks, and the terminal summary.

The report answers three questions about a run:
  * what was decided, and why           (leads, stats, priority queue, sample messages)
  * where each decision came from       (run_info, and a trace on every lead: CSV row, whether the
                                         industry answer came from the model or the cache, which
                                         batch, how many asks, the message attempts)
  * did the whole workflow hold together (telemetry, errors, and `validation`: automated checks on
                                         the run's own output and on run.log)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pipeline.config import ConfigError, config_error, parse_count, parse_number
from pipeline.decide import FACTOR_LABELS, DecisionSettings
from pipeline.industry import IndustrySettings
from pipeline.llm_client import LLMUsage
from pipeline.message_checks import MessageRules, check_message
from pipeline.messages import MessageSettings, first_name
from pipeline.models import (
    Confidence,
    Decision,
    DecisionSummary,
    IndustryScoring,
    IngestResult,
    Issue,
    LeadDecision,
    MessageRun,
    Priority,
    RuleResult,
    Tier,
    Variant,
)
from pipeline.telemetry import LogCounts, StageTiming

SCHEMA_VERSION = 1


# --- settings --------------------------------------------------------------------------------

REPORT_PATHS = {
    "output_path": "report.output_path",
    "log_path": "report.log_path",
    "log_level": "report.log_level",
    "manual_minutes_per_lead": "report.manual_minutes_per_lead",
    "sample_messages": "report.sample_messages",
}


class ReportSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_path: Path
    log_path: Path
    log_level: Literal["DEBUG", "INFO", "WARNING"]
    manual_minutes_per_lead: float = Field(gt=0)  # what deciding one lead by hand costs
    sample_messages: int = Field(ge=1, le=10)  # how many example messages the report carries

    @field_validator("output_path", "log_path", mode="before")
    @classmethod
    def _path(cls, value) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("must be a file path")
        return Path(str(value).strip())

    @field_validator("log_level", mode="before")
    @classmethod
    def _level(cls, value) -> str:
        if not isinstance(value, str) or value.strip().upper() not in ("DEBUG", "INFO", "WARNING"):
            raise ValueError(f"must be DEBUG, INFO or WARNING, got {value!r}")
        return value.strip().upper()

    @field_validator("manual_minutes_per_lead", mode="before")
    @classmethod
    def _minutes(cls, value) -> float:
        number = parse_number(value, 0)
        if number <= 0:
            raise ValueError(f"must be a number above 0, got {value!r}")
        return number

    @field_validator("sample_messages", mode="before")
    @classmethod
    def _samples(cls, value) -> int:
        count = parse_count(value, 1)
        if count > 10:
            raise ValueError(f"must be a whole number from 1 to 10, got {value!r}")
        return count

    @classmethod
    def from_config(cls, config: dict) -> "ReportSettings":
        section = config.get("report")
        if not isinstance(section, dict):
            raise ConfigError("config 'report' must be a mapping")
        try:
            return cls.model_validate(
                {
                    "output_path": section.get("output_path"),
                    "log_path": section.get("log_path"),
                    "log_level": section.get("log_level", "INFO"),
                    "manual_minutes_per_lead": section.get("manual_minutes_per_lead"),
                    "sample_messages": section.get("sample_messages"),
                }
            )
        except ValidationError as exc:
            raise config_error(exc, REPORT_PATHS) from exc


# --- what a run hands to the report ----------------------------------------------------------

class PipelineOutcome(BaseModel):
    """Everything the report is built from: the validated output of every stage, the settings
    they ran with, and the run's own measurements."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    duration_s: float = Field(ge=0)
    input_path: str
    input_sha256: str
    config_sha256: str
    reference_date: date
    provider: str
    model: str
    strict: bool = False
    git_commit: str | None = None
    ingest: IngestResult
    rule_scores: dict[str, dict[str, RuleResult]]
    industry: IndustryScoring
    decisions: list[LeadDecision]
    summary: DecisionSummary
    messages: MessageRun | None = None
    decision_settings: DecisionSettings
    industry_settings: IndustrySettings
    message_settings: MessageSettings | None = None
    report_settings: ReportSettings
    stages: list[StageTiming]
    usage_industry: LLMUsage
    usage_messages: LLMUsage | None = None
    log_counts: LogCounts
    log_path: str | None = None
    secrets: list[str] = Field(default_factory=list, exclude=True)  # never written to the report


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    """The short commit of the code that ran, when run from a git checkout."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


# --- the report ------------------------------------------------------------------------------

class InputInfo(BaseModel):
    path: str
    sha256: str
    rows_read: int = Field(ge=0)
    leads_kept: int = Field(ge=0)
    rows_dropped: int = Field(ge=0)


class RubricInfo(BaseModel):
    """The rubric exactly as it ran, so a report can be read without the config file."""

    weights: dict[str, float]
    qualified_at: float
    review_at: float
    high_priority_at: float
    stale_days: int
    reject_size_below: int
    reject_size_above: int
    tier_score_ranges: dict[str, tuple[int, int]]


class Fingerprints(BaseModel):
    config_sha256: str
    industry_prompt_sha256: str
    message_prompt_sha256: str | None = None


class RunInfo(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    started_at: datetime
    finished_at: datetime
    duration_s: float = Field(ge=0)
    python: str
    git_commit: str | None
    input: InputInfo
    reference_date: date
    provider: str
    model: str
    industry_batch_size: int
    message_batch_size: int | None
    strict: bool
    rubric: RubricInfo
    fingerprints: Fingerprints
    log_file: str | None


class ReasonCount(BaseModel):
    reason: str
    count: int = Field(ge=1)


class ReportStats(BaseModel):
    total: int = Field(ge=0)
    processed: int = Field(ge=0)
    qualified: int = Field(ge=0)
    review: int = Field(ge=0)
    rejected: int = Field(ge=0)
    qualified_pct: float
    review_pct: float
    rejected_pct: float
    average_score: float
    highest_score: float
    lowest_score: float
    average_factor_scores: dict[str, float]
    top_rejection_reasons: list[ReasonCount]
    by_industry_tier: dict[str, int]
    by_variant: dict[str, int]
    messages_written: int = Field(ge=0)
    messages_failed: int = Field(ge=0)
    failed_leads: int = Field(ge=0)  # leads whose industry answer fell back after retries
    api_retries: int = Field(ge=0)  # calls sent again after a rate limit, timeout or server error
    est_manual_minutes_saved: float = Field(ge=0)


class IndustryTrace(BaseModel):
    source: Literal["llm", "cache", "fallback"]
    request_id: str | None
    batch: int | None
    attempts: int


class MessageTrace(BaseModel):
    status: Literal["written", "failed"]
    batch: int
    attempts: int


class LeadTrace(BaseModel):
    source_row: int | None  # the line in the input CSV this lead came from
    notes: list[str]  # values that could not be parsed
    industry: IndustryTrace | None
    message: MessageTrace | None


class LeadMessage(BaseModel):
    variant: Variant
    subject: str
    body: str
    word_count: int = Field(ge=1)


class SampleMessage(LeadMessage):
    lead_id: str
    company: str | None
    priority_rank: int | None


class LeadRecord(BaseModel):
    rank: int = Field(ge=1)  # 1 = best fit of all leads
    lead_id: str
    name: str | None
    company: str | None
    industry: str | None  # the label from the CSV
    company_size: int | None
    source: str | None
    last_interaction_date: date | None
    tier: Tier
    confidence: Confidence
    industry_reason: str
    factor_scores: dict[str, int]
    factor_points: dict[str, float]
    final_score: float
    decision: Decision
    priority: Priority | None
    priority_rank: int | None
    rules_applied: list[str]
    rejection_reason: str | None
    flags: list[str]
    explanation: str
    message: LeadMessage | None
    trace: LeadTrace


class StageLLM(BaseModel):
    batches: int = Field(ge=0)
    batch_sizes: list[int]
    api_calls: int = Field(ge=0)
    validation_retries: int = Field(ge=0)  # extra asks for invalid or missing items
    usage: LLMUsage


class Telemetry(BaseModel):
    stages: list[StageTiming]
    industry: StageLLM
    messages: StageLLM | None
    llm_total: LLMUsage
    cache_hits: int = Field(ge=0)
    log: LogCounts
    log_file: str | None


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class ValidationSummary(BaseModel):
    passed: bool
    checks: list[ValidationCheck]

    @property
    def failed(self) -> list[ValidationCheck]:
        return [check for check in self.checks if not check.passed]


class Report(BaseModel):
    run_info: RunInfo
    stats: ReportStats
    leads: list[LeadRecord]
    priority_queue: list[str]  # lead ids of the qualified leads, first to contact first
    sample_messages: list[SampleMessage]
    errors: list[Issue]  # dropped rows, unparsed values, failed calls and leads sent to review
    telemetry: Telemetry
    validation: ValidationSummary | None = None


# --- building --------------------------------------------------------------------------------

def _sum_usage(*usages: LLMUsage | None) -> LLMUsage:
    total = LLMUsage()
    for usage in usages:
        if usage is None:
            continue
        for name, value in usage.model_dump().items():
            setattr(total, name, getattr(total, name) + value)
    total.latency_s = round(total.latency_s, 6)
    return total


def _stage_llm(batch_sizes: list[int], api_calls: int, retries: int, usage: LLMUsage) -> StageLLM:
    return StageLLM(
        batches=len(batch_sizes), batch_sizes=batch_sizes, api_calls=api_calls,
        validation_retries=retries, usage=usage,
    )


def _issues(outcome: PipelineOutcome) -> list[Issue]:
    found = [Issue(stage="ingest", kind="dropped_row", row=error.row, detail=error.reason) for error in outcome.ingest.errors]
    for lead in outcome.ingest.leads:
        found += [
            Issue(stage="ingest", kind="unparsed_value", lead_id=lead.lead_id, row=lead.source_row, detail=note)
            for note in lead.notes
        ]
    found += outcome.industry.issues
    if outcome.messages is not None:
        found += outcome.messages.issues
    return found


def pick_sample_messages(outcome: PipelineOutcome) -> list[SampleMessage]:
    """Example emails for the report: one of each variant first, then the top-ranked ones."""
    if outcome.messages is None:
        return []
    written = outcome.messages.messages
    ordered = [d for d in outcome.decisions if d.lead_id in written]
    chosen: list[LeadDecision] = []
    seen: set[str] = set()
    for decision in ordered:
        variant = written[decision.lead_id].variant
        if variant not in seen:
            chosen.append(decision)
            seen.add(variant)
    for decision in ordered:
        if len(chosen) >= outcome.report_settings.sample_messages:
            break
        if decision not in chosen:
            chosen.append(decision)
    chosen = sorted(chosen[: outcome.report_settings.sample_messages], key=lambda d: d.overall_rank)
    return [
        SampleMessage(
            lead_id=d.lead_id, company=d.company, priority_rank=d.priority_rank, variant=written[d.lead_id].variant,
            subject=written[d.lead_id].subject, body=written[d.lead_id].body, word_count=written[d.lead_id].word_count,
        )
        for d in chosen
    ]


def build_report(outcome: PipelineOutcome, log_text: str | None = None) -> Report:
    """Assemble the report from a run, then attach the workflow checks."""
    leads = {lead.lead_id: lead for lead in outcome.ingest.leads}
    industry, messages = outcome.industry, outcome.messages
    records = []
    for decision in outcome.decisions:
        lead = leads.get(decision.lead_id)
        if lead is None:  # a decision for a lead that was never ingested; the checks report it
            continue
        fit = industry.results.get(decision.lead_id)
        provenance = industry.provenance.get(decision.lead_id)
        written = messages.messages.get(decision.lead_id) if messages else None
        message_trace = messages.provenance.get(decision.lead_id) if messages else None
        records.append(
            LeadRecord(
                rank=decision.overall_rank, lead_id=decision.lead_id, name=lead.name, company=lead.company,
                industry=lead.industry, company_size=lead.company_size, source=lead.source,
                last_interaction_date=lead.last_interaction_date, tier=decision.tier, confidence=decision.confidence,
                industry_reason=fit.reason if fit else "(no industry answer)", factor_scores=decision.factor_scores, factor_points=decision.factor_points,
                final_score=decision.final_score, decision=decision.decision, priority=decision.priority,
                priority_rank=decision.priority_rank, rules_applied=decision.rules_applied,
                rejection_reason=decision.rejection_reason, flags=decision.flags, explanation=decision.explanation,
                message=None if written is None else LeadMessage(**written.model_dump(exclude={"lead_id"})),
                trace=LeadTrace(
                    source_row=lead.source_row, notes=lead.notes,
                    industry=None if provenance is None else IndustryTrace(**provenance.model_dump()),
                    message=None if message_trace is None else MessageTrace(**message_trace.model_dump(exclude={"variant"})),
                ),
            )
        )

    summary = outcome.summary
    usage_total = _sum_usage(outcome.usage_industry, outcome.usage_messages)
    saved = outcome.report_settings.manual_minutes_per_lead * summary.total - outcome.duration_s / 60
    written_count = len(messages.messages) if messages else 0
    by_variant: dict[str, int] = {}
    for message in (messages.messages.values() if messages else []):
        by_variant[message.variant] = by_variant.get(message.variant, 0) + 1
    by_tier: dict[str, int] = {}
    for decision in outcome.decisions:
        by_tier[decision.tier] = by_tier.get(decision.tier, 0) + 1
    stats = ReportStats(
        total=summary.total, processed=len(outcome.decisions),
        qualified=summary.counts["qualified"], review=summary.counts["review"], rejected=summary.counts["rejected"],
        qualified_pct=round(summary.percent("qualified"), 1), review_pct=round(summary.percent("review"), 1),
        rejected_pct=round(summary.percent("rejected"), 1), average_score=summary.average_score,
        highest_score=summary.highest_score, lowest_score=summary.lowest_score,
        average_factor_scores=summary.average_factor_scores,
        top_rejection_reasons=[ReasonCount(reason=reason, count=count) for reason, count in summary.rejection_reasons],
        by_industry_tier=by_tier, by_variant=by_variant, messages_written=written_count,
        messages_failed=len(messages.failed_ids) if messages else 0, failed_leads=len(industry.failed_ids),
        api_retries=usage_total.retries, est_manual_minutes_saved=round(max(saved, 0.0), 1),
    )

    ds, ins = outcome.decision_settings, outcome.industry_settings
    ms = outcome.message_settings
    report = Report(
        run_info=RunInfo(
            run_id=outcome.run_id, started_at=outcome.started_at, finished_at=outcome.finished_at,
            duration_s=outcome.duration_s, python=platform.python_version(), git_commit=outcome.git_commit,
            input=InputInfo(
                path=outcome.input_path, sha256=outcome.input_sha256, rows_read=outcome.ingest.rows_read,
                leads_kept=len(outcome.ingest.leads), rows_dropped=len(outcome.ingest.errors),
            ),
            reference_date=outcome.reference_date, provider=outcome.provider, model=outcome.model,
            industry_batch_size=ins.batch_size, message_batch_size=None if ms is None else ms.batch_size,
            strict=outcome.strict,
            rubric=RubricInfo(
                weights=ds.weights, qualified_at=ds.qualified_at, review_at=ds.review_at,
                high_priority_at=ds.high_priority_at, stale_days=ds.stale_days,
                reject_size_below=ds.reject_size_below, reject_size_above=ds.reject_size_above,
                tier_score_ranges=ins.tier_ranges,
            ),
            fingerprints=Fingerprints(
                config_sha256=outcome.config_sha256, industry_prompt_sha256=text_sha256(ins.system_prompt),
                message_prompt_sha256=None if ms is None else text_sha256(ms.system_prompt),
            ),
            log_file=outcome.log_path,
        ),
        stats=stats,
        leads=records,
        priority_queue=[d.lead_id for d in outcome.decisions if d.decision == "qualified"],
        sample_messages=pick_sample_messages(outcome),
        errors=_issues(outcome),
        telemetry=Telemetry(
            stages=outcome.stages,
            industry=_stage_llm(industry.batch_sizes, industry.api_calls, industry.validation_retries, outcome.usage_industry),
            messages=None if messages is None else _stage_llm(
                messages.batch_sizes, messages.api_calls, messages.validation_retries,
                outcome.usage_messages or LLMUsage(),
            ),
            llm_total=usage_total, cache_hits=industry.cache_hits, log=outcome.log_counts, log_file=outcome.log_path,
        ),
    )
    report.validation = validate_workflow(outcome, report, log_text)
    return report


# --- checks on the whole workflow ------------------------------------------------------------

def _check(name: str, problems: list[str], ok_detail: str) -> ValidationCheck:
    if problems:
        shown = "; ".join(problems[:5]) + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else "")
        return ValidationCheck(name=name, passed=False, detail=shown)
    return ValidationCheck(name=name, passed=True, detail=ok_detail)


def validate_workflow(outcome: PipelineOutcome, report: Report, log_text: str | None = None) -> ValidationSummary:
    """Check the run against itself: every row and lead accounted for, the arithmetic, the
    ranking, the rules, batching, messages, traces, secrets, and run.log."""
    leads = {lead.lead_id: lead for lead in outcome.ingest.leads}
    decisions = outcome.decisions
    ds, ins = outcome.decision_settings, outcome.industry_settings
    ids = set(leads)
    rule_names = set(ds.weights) - {"industry"}
    checks: list[ValidationCheck] = []

    ingest = outcome.ingest
    checks.append(_check(
        "input_rows_accounted_for",
        [] if ingest.rows_read == len(ingest.leads) + len(ingest.errors)
        else [f"{ingest.rows_read} rows read but {len(ingest.leads)} kept + {len(ingest.errors)} dropped"],
        f"{ingest.rows_read} rows read = {len(ingest.leads)} kept + {len(ingest.errors)} dropped",
    ))
    checks.append(_check(
        "every_lead_scored_by_every_rule",
        [f"{i} has {sorted(outcome.rule_scores.get(i, {}))}" for i in ids if set(outcome.rule_scores.get(i, {})) != rule_names],
        f"{len(ids)} leads x rules {sorted(rule_names)}",
    ))
    checks.append(_check(
        "every_lead_has_an_industry_answer",
        [] if set(outcome.industry.results) == ids else ["industry results do not match the leads"],
        f"{len(outcome.industry.results)} answers",
    ))
    decided = [d.lead_id for d in decisions]
    checks.append(_check(
        "every_lead_decided_exactly_once",
        [] if sorted(decided) == sorted(ids) else [f"{len(decided)} decisions for {len(ids)} leads"],
        f"{len(decided)} decisions",
    ))
    total_weight = sum(ds.weights.values())
    checks.append(_check(
        "weights_add_up_to_one", [] if abs(total_weight - 1) < 1e-6 else [f"weights add up to {total_weight:g}"],
        f"weights {dict(ds.weights)}",
    ))
    out_of_range = [d.lead_id for d in decisions if not (1 <= d.final_score <= 10 and all(1 <= s <= 10 for s in d.factor_scores.values()))]
    checks.append(_check("scores_are_in_range", [f"{i} has a score outside 1-10" for i in out_of_range], "all scores 1-10"))
    off = [d.lead_id for d in decisions if abs(sum(d.factor_points.values()) - d.final_score) > 0.011]
    checks.append(_check("points_add_up_to_final_scores", [f"{i}: points differ from the score" for i in off], "points = score for every lead"))

    problems = []
    if [d.overall_rank for d in decisions] != list(range(1, len(decisions) + 1)):
        problems.append("ranks are not 1..N in order")
    scores = [d.final_score for d in decisions]
    if scores != sorted(scores, reverse=True):
        problems.append("scores are not in descending order")
    checks.append(_check("ranking_is_ordered", problems, "ranks 1..N, scores never increase"))

    qualified = [d for d in decisions if d.decision == "qualified"]
    expected = list(range(1, len(qualified) + 1))
    problems = []
    if [d.priority_rank for d in qualified] != expected:
        problems.append("qualified leads do not have priority ranks 1..Q in order")
    if any(d.priority_rank is not None or d.priority is not None for d in decisions if d.decision != "qualified"):
        problems.append("a lead that is not qualified has a priority")
    if report.priority_queue != [d.lead_id for d in qualified]:
        problems.append("the priority queue does not match the qualified leads")
    checks.append(_check("priority_ranks_are_contiguous", problems, f"{len(qualified)} qualified leads ranked 1..{len(qualified)}"))

    problems = []
    for d in decisions:
        lead, fit = leads.get(d.lead_id), outcome.industry.results.get(d.lead_id)
        if lead is None or fit is None:
            continue  # already reported by the checks on leads and answers above
        stale = lead.last_interaction_date is not None and (outcome.reference_date - lead.last_interaction_date).days > ds.stale_days
        if d.decision == "qualified":
            if d.final_score < ds.qualified_at:
                problems.append(f"{d.lead_id} qualified below {ds.qualified_at:g}")
            if fit.caps_at_review or fit.tier == "none" or "insufficient_data" in lead.flags or stale:
                problems.append(f"{d.lead_id} qualified despite a cap")
        elif d.decision == "rejected":
            hard = any("no-fit industry" in rule for rule in d.rules_applied)
            if d.final_score >= ds.review_at and not hard:
                problems.append(f"{d.lead_id} rejected with a score of {d.final_score:g}")
        else:
            if not (ds.review_at <= d.final_score < ds.qualified_at) and not d.rules_applied:
                problems.append(f"{d.lead_id} in review with no reason")
        if fit.llm_failed and d.decision != "review":
            problems.append(f"{d.lead_id} failed scoring but is {d.decision}")
    checks.append(_check("decisions_follow_the_rules", problems, "thresholds, caps and forced review all hold"))

    recount = {name: sum(1 for d in decisions if d.decision == name) for name in ("qualified", "review", "rejected")}
    checks.append(_check(
        "decision_counts_add_up",
        [] if recount == outcome.summary.counts and sum(recount.values()) == outcome.summary.total
        else [f"summary {outcome.summary.counts} vs recount {recount}"],
        f"{recount}",
    ))

    problems = []
    industry = outcome.industry
    if any(size > ins.batch_size for size in industry.batch_sizes):
        problems.append(f"an industry batch is larger than {ins.batch_size}")
    if industry.api_calls > len(industry.batch_sizes) * (1 + ins.validation_retries):
        problems.append("more industry calls than batches x (1 + re-asks)")
    ms = outcome.message_settings
    if outcome.messages is not None and ms is not None:
        if any(size > ms.batch_size for size in outcome.messages.batch_sizes):
            problems.append(f"a message batch is larger than {ms.batch_size}")
        if outcome.messages.api_calls > len(outcome.messages.batch_sizes) * (1 + ms.validation_retries):
            problems.append("more message calls than batches x (1 + re-asks)")
    calls = industry.api_calls + (outcome.messages.api_calls if outcome.messages else 0)
    checks.append(_check("llm_calls_are_batched", problems, f"{calls} calls for {len(ids)} leads, never one call per lead"))

    if outcome.messages is not None and ms is not None:
        written, failed = outcome.messages.messages, set(outcome.messages.failed_ids)
        qualified_ids = {d.lead_id for d in qualified}
        problems = []
        if set(written) - qualified_ids:
            problems.append("a message was written for a lead that is not qualified")
        missing = qualified_ids - set(written) - failed
        if missing:
            problems.append(f"qualified leads with neither a message nor a failure: {sorted(missing)}")
        checks.append(_check("messages_cover_qualified_leads", problems, f"{len(written)} written + {len(failed)} failed = {len(qualified_ids)} qualified"))
        problems = []
        for lead_id, message in written.items():
            lead = leads.get(lead_id)
            if lead is None:
                continue
            rules = MessageRules(
                first_name=first_name(lead), company=lead.company, industry=lead.industry, sender=ms.sender,
                variant=message.variant, call_minutes=ms.call_minutes, min_words=ms.min_words,
                max_words=ms.max_words, max_subject_words=ms.max_subject_words,
            )
            problems += [f"{lead_id}: {p}" for p in check_message(message.subject, message.body, rules)]
        checks.append(_check("messages_pass_the_rules_again", problems, f"{len(written)} messages re-checked"))

    problems = []
    for record in report.leads:
        if record.trace.source_row is None:
            problems.append(f"{record.lead_id} has no CSV row")
        if record.trace.industry is None:
            problems.append(f"{record.lead_id} has no industry trace")
        if record.decision == "qualified" and outcome.messages is not None and record.trace.message is None:
            problems.append(f"{record.lead_id} has no message trace")
    checks.append(_check("every_lead_is_traceable", problems, "CSV row, industry source and message trace on every lead"))

    problems = []
    if len(industry.failed_ids) != sum(1 for i in industry.issues if i.kind == "fell_back_to_review"):
        problems.append("industry fallbacks and their issues differ")
    if outcome.messages is not None and len(outcome.messages.failed_ids) != sum(1 for i in outcome.messages.issues if i.kind == "message_failed"):
        problems.append("failed messages and their issues differ")
    checks.append(_check("failures_are_all_reported", problems, "every fallback and failed message is in errors"))

    text = report.model_dump_json()
    leaked = [s for s in outcome.secrets if len(s) >= 6 and s in text]
    checks.append(_check("no_secrets_in_report", ["a secret value appears in the report"] if leaked else [], "no API key in the report"))

    if outcome.log_path is not None:
        problems = []
        if log_text is None:
            problems.append("run.log could not be read")
        else:
            if f"run={outcome.run_id}" not in log_text:
                problems.append("the run id is not in run.log")
            for stage in outcome.stages:
                if f"stage {stage.name} finished" not in log_text:
                    problems.append(f"run.log has no 'finished' line for stage {stage.name}")
        checks.append(_check("run_log_matches_the_run", problems, f"run.log has run={outcome.run_id} and every stage's start and end"))

    return ValidationSummary(passed=all(check.passed for check in checks), checks=checks)


# --- output ----------------------------------------------------------------------------------

def write_report(report: Report, path: Path) -> None:
    """Write the report as JSON, replacing any earlier file in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp:
            temp.write(report.model_dump_json(indent=2))
            temp.write("\n")
        os.replace(temp_name, path)
    except OSError:
        Path(temp_name).unlink(missing_ok=True)
        raise


def format_summary(report: Report, report_path: Path, log_path: Path | None) -> str:
    """The closing block of the terminal output."""
    info, stats, tel = report.run_info, report.stats, report.telemetry
    usage = tel.llm_total
    lines = [
        f"Run {info.run_id} finished in {info.duration_s:g}s",
        f"  report:     {report_path}",
        f"  log:        {log_path if log_path else '(none)'}",
        f"  input:      {info.input.path} (sha256 {info.input.sha256[:12]}, {info.input.rows_read} rows)",
        f"  leads:      {info.input.leads_kept} kept, {info.input.rows_dropped} dropped | "
        f"qualified {stats.qualified} ({stats.qualified_pct:g}%), review {stats.review} ({stats.review_pct:g}%), "
        f"rejected {stats.rejected} ({stats.rejected_pct:g}%)",
        f"  score:      average {stats.average_score:g}, range {stats.lowest_score:g} to {stats.highest_score:g}",
    ]
    if report.telemetry.messages is not None:
        lines.append(f"  messages:   {stats.messages_written} written, {stats.messages_failed} without a valid message")
    calls = tel.industry.api_calls + (tel.messages.api_calls if tel.messages else 0)
    lines.append(
        f"  LLM:        {calls} calls (industry {tel.industry.api_calls}"
        + (f", messages {tel.messages.api_calls}" if tel.messages else "")
        + f"), {usage.retries} retries after errors, "
        f"{tel.industry.validation_retries + (tel.messages.validation_retries if tel.messages else 0)} re-asks, "
        f"{usage.prompt_tokens + usage.completion_tokens:,} tokens, {usage.latency_s:.1f}s waiting"
    )
    lines.append("  stages:     " + " | ".join(f"{s.name} {s.duration_s:.2f}s" for s in tel.stages))
    lines.append(
        f"  issues:     {len(report.errors)} in the report; log has {tel.log.warnings} warning(s), {tel.log.errors} error(s)"
    )
    lines.append(f"  time saved: about {stats.est_manual_minutes_saved:,.0f} minutes against deciding each lead by hand")
    validation = report.validation
    if validation is not None:
        if validation.passed:
            lines.append(f"  validation: PASSED ({len(validation.checks)} checks)")
        else:
            lines.append(f"  validation: FAILED ({len(validation.failed)} of {len(validation.checks)} checks)")
            lines += [f"    - {check.name}: {check.detail}" for check in validation.failed]
    return "\n".join(lines)
