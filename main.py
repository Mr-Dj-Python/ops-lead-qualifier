"""Command-line entry point for the lead qualifier.

Usage:
    python main.py --input data/leads_full_scale_100.csv [--strict] [--reference-date YYYY-MM-DD]

Weights come from config.yaml and can be overridden with WEIGHT_INDUSTRY, WEIGHT_SIZE,
WEIGHT_SOURCE and WEIGHT_RECENCY in the environment or a .env file.

Currently runs Stage 1 (ingestion) and Stage 2 (rule scoring). Exits with status 1 and a
message on stderr if the file or config cannot be used.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from pipeline.config import ConfigError, load_config, load_env, parse_iso_date
from pipeline.ingest import IngestError, ingest_csv
from pipeline.rules import RuleFactory, score_leads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify inbound leads from a CSV file.")
    parser.add_argument("--input", required=True, help="path to the leads CSV file")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml (default: config.yaml)")
    parser.add_argument(
        "--env-file",
        help="path to a .env file with WEIGHT_* overrides (default: .env if it exists)",
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
        "--show-scores",
        action="store_true",
        help="print each lead's rule scores (size, source, recency)",
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
    except (ConfigError, IngestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    overridden = set(config["weights_overridden"])
    weights = ", ".join(
        f"{name} {value:g}" + (" (env)" if name in overridden else "") for name, value in config["weights"].items()
    )
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
    if args.show_scores:
        names = [scorer.name for scorer in scorers]
        print("  " + f"{'lead':<6}{'company':<32}" + "".join(f"{name:>9}" for name in names))
        for lead in result.leads:
            company = (lead.company or lead.name or "")[:30]
            cells = "".join(f"{scores[lead.lead_id][name].score:>9}" for name in names)
            print("  " + f"{lead.lead_id:<6}{company:<32}" + cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
