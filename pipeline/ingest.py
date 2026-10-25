"""Stage 1: read the leads CSV, clean it and validate it.

File-level problems (missing file, bad header, unreadable encoding) raise IngestError and
stop the run. Row-level problems are collected in IngestResult.errors; with strict=True they
raise as well.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REQUIRED_COLUMNS = (
    "name",
    "company",
    "company_size",
    "industry",
    "source",
    "last_interaction_date",
)


class IngestError(Exception):
    """Raised when the CSV cannot be ingested."""


@dataclass
class Lead:
    lead_id: str
    name: str | None
    company: str | None
    company_size: int | None
    industry: str | None
    source: str | None
    last_interaction_date: date | None
    notes: list[str] = field(default_factory=list)  # unparseable values we turned into null
    flags: list[str] = field(default_factory=list)  # for example "insufficient_data"


@dataclass
class IngestResult:
    leads: list[Lead]
    errors: list[dict]  # {"row": <csv line number>, "reason": <text>} for dropped rows
    rows_read: int


def _clean(value: str | None, missing_values: frozenset[str]) -> str | None:
    """Trim, collapse spaces, and turn blanks and placeholders such as 'NA' into None."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text or text.lower() in missing_values:
        return None
    return text


def _parse_size(text: str | None, notes: list[str]) -> int | None:
    if text is None:
        return None
    digits = text.replace(",", "")
    if re.fullmatch(r"\d+", digits) and int(digits) >= 1:
        return int(digits)
    notes.append(f"company_size is not a positive whole number: {text!r}")
    return None


def _parse_date(text: str | None, reference_date: date, notes: list[str]) -> date | None:
    if text is None:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            parsed = date.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed > reference_date:
                notes.append(f"last_interaction_date {text} is after the reference date {reference_date}")
                return None
            return parsed
    notes.append(f"last_interaction_date is not a valid YYYY-MM-DD date: {text!r}")
    return None


def _read_rows(path: Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Return the normalized header and the data rows as (csv line number, fields)."""
    if not path.exists():
        raise IngestError(f"input file not found: {path}")
    if not path.is_file():
        raise IngestError(f"input path is not a file: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if header is None:
                raise IngestError(f"input file is empty: {path}")
            rows = [(reader.line_num, row) for row in reader if any(cell.strip() for cell in row)]
    except UnicodeDecodeError as exc:
        raise IngestError(f"input file is not valid UTF-8 text: {path} ({exc.reason})") from exc
    except csv.Error as exc:
        raise IngestError(f"input file is not valid CSV: {path} ({exc})") from exc
    except OSError as exc:
        raise IngestError(f"cannot read input file {path}: {exc.strerror or exc}") from exc

    header = [cell.strip().lower() for cell in header]
    if len(set(header)) != len(header):
        raise IngestError(f"header has duplicate columns: {header}")
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise IngestError(f"header is missing required columns: {', '.join(missing)}")
    if not rows:
        raise IngestError(f"input file has a header but no data rows: {path}")
    return header, rows


def ingest_csv(
    path: str | Path,
    reference_date: date,
    missing_values: list[str] | None = None,
    strict: bool = False,
) -> IngestResult:
    """Read and clean the leads file.

    Dropped rows (empty name and company, wrong field count, duplicates) are listed in
    errors. With strict=True any dropped row or unparseable value raises IngestError.
    """
    placeholders = frozenset(value.strip().lower() for value in (missing_values or []))
    header, rows = _read_rows(Path(path))
    index = {column: header.index(column) for column in REQUIRED_COLUMNS}

    errors: list[dict] = []
    candidates: list[dict] = []
    for line_number, fields in rows:
        if len(fields) != len(header):
            errors.append({"row": line_number, "reason": f"expected {len(header)} fields, found {len(fields)}"})
            continue
        cleaned = {column: _clean(fields[index[column]], placeholders) for column in REQUIRED_COLUMNS}
        if cleaned["name"] is None and cleaned["company"] is None:
            errors.append({"row": line_number, "reason": "both name and company are missing"})
            continue
        notes: list[str] = []
        candidates.append(
            {
                "row": line_number,
                "name": cleaned["name"],
                "company": cleaned["company"],
                "company_size": _parse_size(cleaned["company_size"], notes),
                "industry": cleaned["industry"],
                "source": cleaned["source"],
                "last_interaction_date": _parse_date(cleaned["last_interaction_date"], reference_date, notes),
                "notes": notes,
            }
        )

    # Duplicates share name and company; keep the most recent interaction (first row on a tie).
    kept: dict[tuple, dict] = {}
    for candidate in candidates:
        key = ((candidate["name"] or "").lower(), (candidate["company"] or "").lower())
        current = kept.get(key)
        if current is None:
            kept[key] = candidate
            continue
        newer = (candidate["last_interaction_date"] or date.min) > (current["last_interaction_date"] or date.min)
        loser, winner = (current, candidate) if newer else (candidate, current)
        kept[key] = winner
        errors.append({"row": loser["row"], "reason": f"duplicate of row {winner['row']} (same name and company)"})

    survivors = sorted(kept.values(), key=lambda candidate: candidate["row"])
    leads = [
        Lead(
            lead_id=f"L{number:03d}",
            name=candidate["name"],
            company=candidate["company"],
            company_size=candidate["company_size"],
            industry=candidate["industry"],
            source=candidate["source"],
            last_interaction_date=candidate["last_interaction_date"],
            notes=candidate["notes"],
        )
        for number, candidate in enumerate(survivors, start=1)
    ]
    for lead in leads:
        gaps = sum(value is None for value in (lead.company_size, lead.source, lead.last_interaction_date))
        if lead.industry is None and gaps >= 2:
            lead.flags.append("insufficient_data")

    errors.sort(key=lambda error: error["row"])
    result = IngestResult(leads=leads, errors=errors, rows_read=len(rows))

    if strict:
        problems = [f"row {e['row']}: {e['reason']}" for e in errors]
        problems += [f"{lead.lead_id}: {note}" for lead in leads for note in lead.notes]
        if problems:
            shown = "; ".join(problems[:5])
            more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
            raise IngestError(f"strict mode: {len(problems)} problem(s) found: {shown}{more}")
    return result
