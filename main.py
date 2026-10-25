"""Command-line entry point for the lead qualifier.

Usage:
    python main.py --input data/leads_full_scale_100.csv [--strict] [--reference-date YYYY-MM-DD]

Scoring weights are read from .env (WEIGHT_INDUSTRY, WEIGHT_SIZE, WEIGHT_SOURCE and
WEIGHT_RECENCY, referenced from config.yaml); they must add up to 1.

Runs Stage 1 (ingestion) and Stage 2 (rule scoring). With --score-industry it also runs
Stage 3 (LLM industry fit), which needs the API key named by llm.api_key_env in .env.
Exits with status 1 and a message on stderr if the file, config or API key cannot be used.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from pipeline.config import ConfigError, load_config, load_env, parse_iso_date
from pipeline.industry import IndustryScorer, IndustrySettings
from pipeline.ingest import IngestError, ingest_csv
from pipeline.llm_client import LLMAuthError, create_llm_client
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        industry = None
        if args.score_industry:
            settings = IndustrySettings.from_config(config)
            client = create_llm_client(config.get("llm", {}), env)
            industry = IndustryScorer(client, settings).score(result.leads)
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
        print(f"  dropped row {error['row']}: {error['reason']}")
    print(f"Scored {len(scores)} leads with rules: {', '.join(scorer.name for scorer in scorers)}")
    if industry is not None:
        print(
            f"Industry fit: {len(industry.results)} leads scored "
            f"({industry.api_calls} API calls, {industry.cache_hits} from cache, "
            f"{len(industry.failed_ids)} fell back to review)"
        )
        if industry.failed_ids:
            print(f"  fallback leads: {', '.join(industry.failed_ids)}")
    if args.show_scores:
        names = [scorer.name for scorer in scorers]
        extra = f"{'fit':>9}  tier" if industry is not None else ""
        print(
            "  " + f"{'lead':<6}{'company':<32}{'industry':<22}"
            + "".join(f"{name:>9}" for name in names) + extra
        )
        for lead in result.leads:
            company = (lead.company or lead.name or "")[:30]
            cells = "".join(f"{scores[lead.lead_id][name].score:>9}" for name in names)
            if industry is not None:
                fit = industry.results[lead.lead_id]
                cells += f"{fit.industry_fit:>9}  {fit.tier}" + ("*" if fit.caps_at_review else "")
            label = (lead.industry or "-")[:20]
            print("  " + f"{lead.lead_id:<6}{company:<32}{label:<22}" + cells)
        if industry is not None:
            print("  (* = capped at review: unknown tier, low confidence or LLM failure)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    sys.exit(main())
