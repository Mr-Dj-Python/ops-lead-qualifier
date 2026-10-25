"""Command-line entry point for the lead qualifier.

Usage:
    python main.py --input data/leads_full_scale_100.csv [--strict] [--reference-date YYYY-MM-DD]

Scoring weights are read from .env (WEIGHT_INDUSTRY, WEIGHT_SIZE, WEIGHT_SOURCE and
WEIGHT_RECENCY, referenced from config.yaml); they must add up to 1.

Runs Stage 1 (ingestion) and Stage 2 (rule scoring). With --score-industry it also runs
Stage 3 (LLM industry fit, which needs the API key named by llm.api_key_env in .env, in
batches of LLM_BATCH_SIZE, default 10) and Stage 4 (weighted score, decision and ranking,
printed from best fit to lowest). Adding --messages also writes a personalised email for each
qualified lead (Stage 5, same batch size). Exits with status 1 and a message on stderr if the file, config or API key cannot be used.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from pipeline.config import ConfigError, load_config, load_env, parse_iso_date
from pipeline.decide import DecisionSettings, add_flag, decide_leads, summarize
from pipeline.industry import IndustryScorer, IndustrySettings
from pipeline.ingest import IngestError, ingest_csv
from pipeline.llm_client import LLMAuthError, create_llm_client
from pipeline.messages import MessageGenerator, MessageSettings
from pipeline.rules import RuleFactory, score_leads


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
    try:
        env = load_env(args.env_file, required=args.env_file is not None)
        config = load_config(args.config, env=env)
        reference_date = resolve_reference_date(args.reference_date, config)
        result = ingest_csv(
            args.input,
            reference_date=reference_date,
            missing_values=config["missing_values"],
            strict=args.strict,
        )
        scorers = RuleFactory.create_all(config["rules"])
        scores = score_leads(result.leads, scorers, reference_date)
        industry = decisions = summary = written = None
        if args.score_industry:
            settings = IndustrySettings.from_config(config)
            decision_settings = DecisionSettings.from_config(config, [scorer.name for scorer in scorers])
            client = create_llm_client(config.get("llm", {}), env)
            industry = IndustryScorer(client, settings).score(result.leads)
            decisions = decide_leads(result.leads, scores, industry.results, decision_settings, reference_date)
            summary = summarize(decisions, decision_settings)
            if args.messages:
                written = MessageGenerator(client, MessageSettings.from_config(config)).generate(
                    decisions, result.leads, reference_date
                )
                for decision in decisions:
                    if decision.lead_id in written.failed_ids:
                        add_flag(decision, "message_failed")
    except (ConfigError, IngestError, LLMAuthError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    sys.exit(main())
