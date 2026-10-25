from datetime import date
from pathlib import Path

import pytest

from main import main
from pipeline.ingest import IngestError, ingest_csv
from pipeline.models import RowError

HEADER = "name,company,company_size,industry,source,last_interaction_date\n"
MISSING = ["na", "n/a", "null", "none", "unknown", "unknown source", "unknown sector"]
REFERENCE = date(2024, 1, 21)
DATA = Path(__file__).resolve().parent.parent / "data"


def write_csv(tmp_path, body, header=HEADER, name="leads.csv"):
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def ingest(path, **kwargs):
    return ingest_csv(path, reference_date=REFERENCE, missing_values=MISSING, **kwargs)


# --- file-level errors raise ---------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        ingest(tmp_path / "nope.csv")


def test_directory_raises(tmp_path):
    with pytest.raises(IngestError, match="not a file"):
        ingest(tmp_path)


@pytest.mark.parametrize("name", ["leads.txt", "leads.xlsx", "leads.json", "leads"])
def test_non_csv_extension_raises_even_with_csv_content(tmp_path, name):
    path = write_csv(tmp_path, "Ann,Co,10,SaaS,Referral,2024-01-01\n", name=name)
    with pytest.raises(IngestError, match="must be a .csv file"):
        ingest(path)


def test_uppercase_csv_extension_is_accepted(tmp_path):
    path = write_csv(tmp_path, "Ann,Co,10,SaaS,Referral,2024-01-01\n", name="LEADS.CSV")
    assert len(ingest(path).leads) == 1


def test_binary_file_with_csv_extension_raises(tmp_path):
    path = tmp_path / "fake.csv"
    path.write_bytes(b"PK\x03\x04\x00\x00\x08\x00" + bytes(range(256)))
    with pytest.raises(IngestError, match="binary"):
        ingest(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(IngestError, match="empty"):
        ingest(path)


def test_missing_required_column_raises(tmp_path):
    path = write_csv(tmp_path, "A,B,10\n", header="name,company,company_size\n")
    with pytest.raises(IngestError, match="industry, source, last_interaction_date"):
        ingest(path)


def test_duplicate_column_raises(tmp_path):
    path = write_csv(tmp_path, "", header=HEADER.strip() + ",name\n")
    with pytest.raises(IngestError, match="duplicate columns"):
        ingest(path)


def test_header_only_raises(tmp_path):
    with pytest.raises(IngestError, match="no data rows"):
        ingest(write_csv(tmp_path, ""))


def test_non_utf8_raises(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes(HEADER.encode() + "Zoë,Café\xe9,10,SaaS,Referral,2024-01-01\n".encode("latin-1"))
    with pytest.raises(IngestError, match="UTF-8"):
        ingest(path)


# --- cleaning ------------------------------------------------------------------------------

def test_cleans_and_normalizes_fields(tmp_path):
    path = write_csv(tmp_path, "  Alice   Chen ,CloudScale AI,\"1,200\",SaaS,Inbound demo request,2024-01-15\n")
    lead = ingest(path).leads[0]
    assert lead.lead_id == "L001"
    assert lead.name == "Alice Chen"
    assert lead.company_size == 1200
    assert lead.last_interaction_date == date(2024, 1, 15)


def test_header_is_case_insensitive_and_bom_safe(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + HEADER.upper().encode() + b"A,B,10,Retail,Referral,2024-01-01\n")
    assert len(ingest(path).leads) == 1


def test_placeholders_become_missing_without_notes(tmp_path):
    path = write_csv(tmp_path, "Ann,Co,NA,Unknown Sector,Unknown Source,n/a\n")
    lead = ingest(path).leads[0]
    assert (lead.company_size, lead.industry, lead.source, lead.last_interaction_date) == (None, None, None, None)
    assert lead.notes == []


def test_unparseable_values_become_null_with_notes(tmp_path):
    path = write_csv(tmp_path, "Ann,Co,12abc,SaaS,Referral,15/01/2024\nBo,Co2,0,SaaS,Referral,2024-02-30\n")
    first, second = ingest(path).leads
    assert first.company_size is None and first.last_interaction_date is None
    assert len(first.notes) == 2
    assert second.company_size is None and second.last_interaction_date is None


def test_future_date_is_rejected(tmp_path):
    lead = ingest(write_csv(tmp_path, "Ann,Co,10,SaaS,Referral,2024-06-01\n")).leads[0]
    assert lead.last_interaction_date is None
    assert "after the reference date" in lead.notes[0]


# --- row-level problems --------------------------------------------------------------------

def test_row_without_name_and_company_is_dropped(tmp_path):
    result = ingest(write_csv(tmp_path, "NA,,50,SaaS,Referral,2024-01-01\nAnn,Co,10,SaaS,Referral,2024-01-01\n"))
    assert [lead.name for lead in result.leads] == ["Ann"]
    assert result.errors == [RowError(row=2, reason="both name and company are missing")]


def test_company_missing_but_name_present_is_kept(tmp_path):
    lead = ingest(write_csv(tmp_path, "Ann,,50,Consulting,Sales call,2024-01-19\n")).leads[0]
    assert lead.company is None


def test_wrong_field_count_is_dropped(tmp_path):
    result = ingest(write_csv(tmp_path, "Ann,Co,10\nBo,Co2,10,SaaS,Referral,2024-01-01\n"))
    assert len(result.leads) == 1
    assert result.errors[0].row == 2 and "expected 6 fields" in result.errors[0].reason


def test_duplicate_keeps_most_recent_interaction(tmp_path):
    body = (
        "Ann,Co,10,SaaS,Referral,2024-01-01\n"
        "ann,CO,10,SaaS,Referral,2024-01-10\n"
        "Bo,Co2,10,SaaS,Referral,2024-01-05\n"
    )
    result = ingest(write_csv(tmp_path, body))
    assert [lead.lead_id for lead in result.leads] == ["L001", "L002"]
    assert result.leads[0].last_interaction_date == date(2024, 1, 10)
    assert result.errors == [RowError(row=2, reason="duplicate of row 3 (same name and company)")]


def test_insufficient_data_flag(tmp_path):
    result = ingest(write_csv(tmp_path, "James,NA,NA,NA,NA,NA\nAnn,Co,NA,SaaS,NA,NA\n"))
    assert result.leads[0].flags == ["insufficient_data"]
    assert result.leads[1].flags == []  # industry is known


# --- strict mode ---------------------------------------------------------------------------

def test_strict_raises_on_dropped_row(tmp_path):
    path = write_csv(tmp_path, "NA,NA,50,SaaS,Referral,2024-01-01\nAnn,Co,10,SaaS,Referral,2024-01-01\n")
    with pytest.raises(IngestError, match="strict mode"):
        ingest(path, strict=True)


def test_strict_raises_on_unparseable_value(tmp_path):
    with pytest.raises(IngestError, match="company_size"):
        ingest(write_csv(tmp_path, "Ann,Co,lots,SaaS,Referral,2024-01-01\n"), strict=True)


def test_strict_accepts_clean_file(tmp_path):
    assert len(ingest(write_csv(tmp_path, "Ann,Co,10,SaaS,Referral,2024-01-01\n"), strict=True).leads) == 1


# --- provided data files -------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, rows",
    [("leads_full_scale_100.csv", 100), ("leads_training.csv", 29), ("leads_testing.csv", 20)],
)
def test_provided_data_files_ingest(filename, rows):
    result = ingest(DATA / filename)
    assert result.rows_read == rows
    assert len(result.leads) == rows
    assert result.errors == []


def test_testing_file_edge_cases():
    leads = {lead.name: lead for lead in ingest(DATA / "leads_testing.csv").leads}
    assert leads["Test_MissingSize"].company_size is None
    assert leads["Test_MissingDate"].source is None and leads["Test_MissingDate"].last_interaction_date is None
    assert leads["Test_MissingName"].company is None
    assert leads["Test_NoSourceInfo"].source is None  # "Unknown Source" placeholder
    assert leads["Test_UnknownIndustry"].industry is None  # "Unknown Sector" placeholder
    assert leads["Test_NameOnly"].flags == ["insufficient_data"]
    assert [name for name, lead in leads.items() if lead.flags] == ["Test_NameOnly"]


# --- command line --------------------------------------------------------------------------

def test_cli_success_exit_code(capsys):
    assert main(["--input", str(DATA / "leads_training.csv")]) == 0
    assert "leads kept:                 29" in capsys.readouterr().out


def test_cli_error_exit_code_and_message(capsys):
    assert main(["--input", "does-not-exist.csv"]) == 1
    assert "error: input file not found" in capsys.readouterr().err


def test_cli_rejects_non_csv_file(tmp_path, capsys):
    path = write_csv(tmp_path, "Ann,Co,10,SaaS,Referral,2024-01-01\n", name="leads.txt")
    assert main(["--input", str(path)]) == 1
    assert "must be a .csv file" in capsys.readouterr().err


def test_cli_strict_flag_fails_on_problem(tmp_path, capsys):
    path = write_csv(tmp_path, "NA,NA,50,SaaS,Referral,2024-01-01\nAnn,Co,10,SaaS,Referral,2024-01-01\n")
    assert main(["--input", str(path), "--strict"]) == 1
    assert "strict mode" in capsys.readouterr().err
    assert main(["--input", str(path)]) == 0


def test_cli_rejects_bad_reference_date(capsys):
    assert main(["--input", str(DATA / "leads_training.csv"), "--reference-date", "21/01/2024"]) == 1
    assert "invalid reference date" in capsys.readouterr().err


def test_cli_missing_config(capsys):
    assert main(["--input", str(DATA / "leads_training.csv"), "--config", "nope.yaml"]) == 1
    assert "config file not found" in capsys.readouterr().err


def test_each_lead_remembers_its_line_in_the_csv(tmp_path):
    body = "Ann,Co,10,SaaS,Referral,2024-01-01\nNA,NA,10,SaaS,Referral,2024-01-01\nBo,Co2,10,SaaS,Referral,2024-01-01\n"
    leads = ingest_csv(write_csv(tmp_path, body), reference_date=REFERENCE, missing_values=MISSING).leads
    assert [(lead.name, lead.source_row) for lead in leads] == [("Ann", 2), ("Bo", 4)]  # the dropped row 3 is skipped


def test_source_rows_point_at_the_right_line_in_the_provided_files():
    for filename in ("leads_training.csv", "leads_full_scale_100.csv", "leads_testing.csv"):
        lines = (DATA / filename).read_text().splitlines()
        for lead in ingest(DATA / filename).leads:
            assert (lead.company or lead.name) in lines[lead.source_row - 1]
