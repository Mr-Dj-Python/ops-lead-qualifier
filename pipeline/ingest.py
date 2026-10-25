"""Stage 1: read the leads CSV, clean it and validate it.

File-level problems (missing file, bad header, unreadable encoding) raise IngestError and
stop the run. Row-level problems are collected in IngestResult.errors; with strict=True they
raise as well.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from pipeline.models import Lead, LeadRow

REQUIRED_COLUMNS = (
    "name",
    "company",
    "company_size",
    "industry",
    "source",
    "last_interaction_date",
)

# A bad value in one of these fields becomes a note (the lead is kept with the field empty);
# any other validation failure drops the row.
SOFT_FIELDS = ("company_size", "last_interaction_date")


class IngestError(Exception):
    """Raised when the CSV cannot be ingested."""


@dataclass
class IngestResult:
    leads: list[Lead]
    errors: list[dict]  # {"row": <csv line number>, "reason": <text>} for dropped rows
    rows_read: int


def _message(error: dict) -> str:
    """The readable part of a pydantic error ("Value error, " prefix removed)."""
    return error["msg"].removeprefix("Value error, ")


def _validate_row(data: dict, context: dict) -> tuple[LeadRow | None, list[str], str | None]:
    """Validate one row. Returns (row, notes, None) or (None, [], reason) when it must be dropped.

    Unparseable sizes and dates are emptied with a note; everything else that fails drops the row.
    """
    notes: list[str] = []
    for _ in range(2):  # second pass: the row again, with the soft fields emptied
        try:
            return LeadRow.model_validate(data, context=context), notes, None
        except ValidationError as exc:
            errors = exc.errors()
            hard = [e for e in errors if not (e["loc"] and e["loc"][0] in SOFT_FIELDS)]
            if hard or notes:
                reason = "; ".join(_message(e) for e in (hard or errors))
                return None, [], reason
            data = dict(data)
            for error in errors:
                notes.append(_message(error))
                data[error["loc"][0]] = None
    raise AssertionError("unreachable")  # the loop always returns


def _read_rows(path: Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Return the normalized header and the data rows as (csv line number, fields)."""
    if not path.exists():
        raise IngestError(f"input file not found: {path}")
    if not path.is_file():
        raise IngestError(f"input path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise IngestError(f"input file must be a .csv file, got {path.suffix or 'no extension'!r}: {path}")
    try:
        with path.open("rb") as raw:
            if b"\x00" in raw.read(8192):
                raise IngestError(f"input file looks binary, not a CSV text file: {path}")
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
    context = {
        "missing_values": frozenset(value.strip().lower() for value in (missing_values or [])),
        "reference_date": reference_date,
    }
    header, rows = _read_rows(Path(path))
    index = {column: header.index(column) for column in REQUIRED_COLUMNS}

    errors: list[dict] = []
    candidates: list[dict] = []
    for line_number, fields in rows:
        if len(fields) != len(header):
            errors.append({"row": line_number, "reason": f"expected {len(header)} fields, found {len(fields)}"})
            continue
        row, notes, reason = _validate_row({column: fields[index[column]] for column in REQUIRED_COLUMNS}, context)
        if row is None:
            errors.append({"row": line_number, "reason": reason})
            continue
        candidates.append({"row": line_number, "values": row.model_dump(), "notes": notes})

    # Duplicates share name and company; keep the most recent interaction (first row on a tie).
    kept: dict[tuple, dict] = {}
    for candidate in candidates:
        values = candidate["values"]
        key = ((values["name"] or "").lower(), (values["company"] or "").lower())
        current = kept.get(key)
        if current is None:
            kept[key] = candidate
            continue
        newer = (values["last_interaction_date"] or date.min) > (current["values"]["last_interaction_date"] or date.min)
        loser, winner = (current, candidate) if newer else (candidate, current)
        kept[key] = winner
        errors.append({"row": loser["row"], "reason": f"duplicate of row {winner['row']} (same name and company)"})

    survivors = sorted(kept.values(), key=lambda candidate: candidate["row"])
    leads = [
        Lead(lead_id=f"L{number:03d}", notes=candidate["notes"], **candidate["values"])
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
