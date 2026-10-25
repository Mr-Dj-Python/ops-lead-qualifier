"""Command-line entry point for the lead qualifier.

Usage:
    python main.py --input data/leads_full_scale_100.csv [--strict] [--reference-date YYYY-MM-DD]

Scoring weights are read from .env (WEIGHT_INDUSTRY, WEIGHT_SIZE, WEIGHT_SOURCE and
WEIGHT_RECENCY, referenced from config.yaml); they must add up to 1.

Runs Stage 1 (ingestion) and Stage 2 (rule scoring). With --score-industry it also runs
Stage 3 (LLM industry fit, which needs the API key named by llm.api_key_env in .env, in
batches of LLM_BATCH_SIZE, default 10) and Stage 4 (weighted score, decision and ranking,
printed from best fit to lowest). Adding --messages also writes a personalised email for each
qualified lead (Stage 5, same batch size).

A run with --score-industry also writes the report (output_report.json, Stage 6) and appends to
run.log, checks the whole workflow against itself, and ends with a summary. Exits with status 1
and a message on stderr if the file, config or API key cannot be used, or if a workflow check
fails.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from pipeline.config import ConfigError, load_config, load_env, parse_iso_date
from pipeline.decide import DecisionSettings, add_flag, decide_leads, summarize
from pipeline.industry import IndustryScorer, IndustrySettings
from pipeline.ingest import IngestError, ingest_csv
from pipeline.llm_client import LLMAuthError, create_llm_client
from pipeline.messages import MessageGenerator, MessageSettings
from pipeline.report import (
    PipelineOutcome,
    ReportSettings,
    build_report,
    file_sha256,
    format_summary,
    git_commit,
    write_report,
)
from pipeline.rules import RuleFactory, score_leads
from pipeline.telemetry import RunRecorder

log = logging.getLogger("pipeline.run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify inbound leads from a CSV file.")
    parser.add_argument("--input", required=True, help="path to the leads CSV file")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml (default: config.yaml)")
    parser.add_argument(
        "--env-file",
        help="path to the .env file holding the weights (default: .env)",
    )
    parser.add_argument(
        "--reference-date",
        help="date recency is measured from, YYYY-MM-DD (default: config value, else today)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on any dropped row or unparseable value instead of reporting it",
    )
    parser.add_argument(
        "--score-industry",
        action="store_true",
        help="also score industry fit with the LLM (needs the API key in .env)",
    )
    parser.add_argument(
        "--messages",
        action="store_true",
        help="with --score-industry, also write a first-touch email for each qualified lead",
    )
    parser.add_argument("--output", help="where to write the report (default: report.output_path in config.yaml)")
    parser.add_argument("--log-file", help="where to append the run log (default: report.log_path in config.yaml)")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="with --score-industry, print each lead's score breakdown and the rules applied",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="print each lead's factor scores (size, source, recency, and industry if scored)",
    )
    return parser


def resolve_reference_date(cli_value: str | None, config: dict) -> date:
    text = cli_value or config.get("reference_date")
    if text is None:
        return date.today()
    try:
        return parse_iso_date(str(text))
    except ValueError as exc:
        raise ConfigError(f"invalid reference date: {exc}") from exc


SHORT_NAMES = {"industry": "ind", "size": "size", "source": "src", "recency": "rec"}


def print_ranking(decisions, summary, explain: bool) -> None:
    """Summary lines, then every lead from best fit to lowest."""
    counts = ", ".join(f"{name} {summary.counts[name]} ({summary.percent(name):.0f}%)" for name in summary.counts)
    print(f"Decisions: {counts}")
    print(
        f"Weighted score: average {summary.average_score:g}, "
        f"highest {summary.highest_score:g}, lowest {summary.lowest_score:g}"
    )
    print("Average factor scores: " + ", ".join(f"{n} {v:g}" for n, v in summary.average_factor_scores.items()))
    if summary.rejection_reasons:
        print("Rejection reasons: " + "; ".join(f"{reason} ({count})" for reason, count in summary.rejection_reasons))
    factors = list(summary.average_factor_scores)
    print()
    print(
        f"  {'rank':>4}  {'lead':<5} {'company':<30} {'industry':<20} {'tier':<8}"
        + "".join(f"{SHORT_NAMES.get(name, name[:4]):>5}" for name in factors)
        + f"{'score':>7}  {'decision':<9} priority"
    )
    for d in decisions:
        company = (d.company or d.name or "")[:30]
        priority = f"{d.priority} #{d.priority_rank}" if d.priority else ""
        print(
            f"  {d.overall_rank:>4}  {d.lead_id:<5} {company:<30} {(d.industry_label or '-')[:20]:<20} {d.tier:<8}"
            + "".join(f"{d.factor_scores[name]:>5}" for name in factors)
            + f"{d.final_score:>7g}  {d.decision:<9} {priority}"
        )
    if explain:
        for d in decisions:
            print()
            print(d.explanation)


def print_messages(decisions, written) -> None:
    """Each qualified lead's email in priority order, then any lead left without one."""
    print()
    print(
        f"Messages: {len(written.messages)} written ({written.api_calls} API calls), "
        f"{len(written.failed_ids)} without a valid message"
    )
    for d in decisions:
        message = written.messages.get(d.lead_id)
        if message is None:
            continue
        print()
        print(f"--- {d.priority} #{d.priority_rank}  {d.lead_id}  {d.company or d.name}  [{message.variant}, {message.word_count} words]")
        print(f"Subject: {message.subject}")
        print()
        print(message.body)
    if written.failed_ids:
        print()
        print("No valid message (a rep should write these): " + ", ".join(written.failed_ids))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.messages and not args.score_industry:
        parser.error("--messages needs --score-industry (messages are written for qualified leads)")
    recorder = RunRecorder()
    try:
        env = load_env(args.env_file, required=args.env_file is not None)
        config = load_config(args.config, env=env)
        reference_date = resolve_reference_date(args.reference_date, config)
        report_settings = ReportSettings.from_config(config) if args.score_industry else None
    except (ConfigError, IngestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    log_path = None
    if report_settings is not None:
        log_path = Path(args.log_file) if args.log_file else report_settings.log_path
    level = report_settings.log_level if report_settings else "INFO"

    with recorder.logging_to(log_path, level):
        log.info("run %s started: input=%s reference_date=%s", recorder.run_id, args.input, reference_date)
        try:
            run = run_stages(args, config, env, reference_date, recorder, report_settings, log_path)
        except (ConfigError, IngestError, LLMAuthError) as exc:
            log.error("run failed: %s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1
        report = run["report"]
        print_run(args, config, reference_date, run)
        if report is not None:
            report_path = Path(args.output) if args.output else report_settings.output_path
            print()
            print(format_summary(report, report_path, log_path))
            if report.validation is not None and not report.validation.passed:
                log.error("workflow validation failed: %s", ", ".join(c.name for c in report.validation.failed))
                return 1
    return 0


def run_stages(args, config, env, reference_date, recorder, report_settings, log_path) -> dict:
    """Run the stages in order, timing each, and return everything they produced."""
    with recorder.stage("ingest") as note:
        result = ingest_csv(
            args.input, reference_date=reference_date, missing_values=config["missing_values"], strict=args.strict
        )
        note.items = len(result.leads)
    with recorder.stage("rules") as note:
        scorers = RuleFactory.create_all(config["rules"])
        scores = score_leads(result.leads, scorers, reference_date)
        note.items = len(scores)
    run = {"result": result, "scorers": scorers, "scores": scores, "industry": None, "decisions": None,
           "summary": None, "written": None, "report": None, "outcome": None}
    if not args.score_industry:
        return run

    industry_settings = IndustrySettings.from_config(config)
    decision_settings = DecisionSettings.from_config(config, [scorer.name for scorer in scorers])
    message_settings = MessageSettings.from_config(config) if args.messages else None
    client = create_llm_client(config.get("llm", {}), env)
    log.info("llm provider=%s model=%s", config["llm"]["provider"], client.model)

    before = client.usage.model_copy()
    with recorder.stage("industry") as note:
        industry = IndustryScorer(client, industry_settings).score(result.leads)
        note.items = len(industry.results)
    usage_industry = client.usage.minus(before)
    with recorder.stage("decide") as note:
        decisions = decide_leads(result.leads, scores, industry.results, decision_settings, reference_date)
        summary = summarize(decisions, decision_settings)
        note.items = len(decisions)
        log.info("decisions: %s average_score=%s", summary.counts, summary.average_score)
    written, usage_messages = None, None
    if message_settings is not None:
        before = client.usage.model_copy()
        with recorder.stage("messages") as note:
            written = MessageGenerator(client, message_settings).generate(decisions, result.leads, reference_date)
            note.items = len(written.messages)
        usage_messages = client.usage.minus(before)
        for decision in decisions:
            if decision.lead_id in written.failed_ids:
                add_flag(decision, "message_failed")
    run.update(industry=industry, decisions=decisions, summary=summary, written=written)

    recorder.finish()
    with recorder.stage("report"):
        outcome = PipelineOutcome(
            run_id=recorder.run_id, started_at=recorder.started_at, finished_at=recorder.finished_at,
            duration_s=recorder.duration_s, input_path=args.input, input_sha256=file_sha256(args.input),
            config_sha256=file_sha256(args.config), reference_date=reference_date,
            provider=config["llm"]["provider"], model=client.model, strict=args.strict, git_commit=git_commit(),
            ingest=result, rule_scores=scores, industry=industry, decisions=decisions, summary=summary,
            messages=written, decision_settings=decision_settings, industry_settings=industry_settings,
            message_settings=message_settings, report_settings=report_settings, stages=list(recorder.stages),
            usage_industry=usage_industry, usage_messages=usage_messages, log_counts=recorder.log_counts,
            log_path=None if log_path is None else str(log_path),
            secrets=[env.get(config["llm"].get("api_key_env", ""), "")],
        )
        recorder.flush_log()
        log_text = log_path.read_text(encoding="utf-8") if log_path is not None and log_path.exists() else None
        report = build_report(outcome, log_text)
        report_path = Path(args.output) if args.output else report_settings.output_path
        write_report(report, report_path)
        log.info("report written to %s (validation %s)", report_path, "passed" if report.validation.passed else "FAILED")
    run.update(report=report, outcome=outcome)
    return run


def print_run(args, config, reference_date, run) -> None:
    result, scorers, scores = run["result"], run["scorers"], run["scores"]
    industry, decisions, summary, written = run["industry"], run["decisions"], run["summary"], run["written"]
    weights = ", ".join(f"{name} {value:g}" for name, value in config["weights"].items())
    flagged = sum(1 for lead in result.leads if lead.flags)
    noted = sum(1 for lead in result.leads if lead.notes)
    print(f"Ingested {args.input} (reference date {reference_date})")
    print(f"  weights: {weights}")
    print(f"  rows read:                  {result.rows_read}")
    print(f"  leads kept:                 {len(result.leads)}")
    print(f"  rows dropped:               {len(result.errors)}")
    print(f"  leads with unparsed values: {noted}")
    print(f"  flagged insufficient_data:  {flagged}")
    for error in result.errors:
        print(f"  dropped row {error.row}: {error.reason}")
    print(f"Scored {len(scores)} leads with rules: {', '.join(scorer.name for scorer in scorers)}")
    if industry is not None:
        print(
            f"Industry fit: {len(industry.results)} leads scored "
            f"({industry.api_calls} API calls, {industry.cache_hits} from cache, "
            f"{len(industry.failed_ids)} fell back to review)"
        )
        if industry.failed_ids:
            print(f"  fallback leads: {', '.join(industry.failed_ids)}")
    if decisions is not None:
        print_ranking(decisions, summary, args.explain)
        if written is not None:
            print_messages(decisions, written)
    elif args.show_scores:
        names = [scorer.name for scorer in scorers]
        print(
            "  " + f"{'lead':<6}{'company':<32}{'industry':<22}"
            + "".join(f"{name:>9}" for name in names)
        )
        for lead in result.leads:
            company = (lead.company or lead.name or "")[:30]
            label = (lead.industry or "-")[:20]
            cells = "".join(f"{scores[lead.lead_id][name].score:>9}" for name in names)
            print("  " + f"{lead.lead_id:<6}{company:<32}{label:<22}" + cells)


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)  # the console shows warnings; run.log gets the detail
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    sys.exit(main())
