from datetime import date
from pathlib import Path

import pytest

from main import main
from pipeline import rules
from pipeline.config import ConfigError, load_config
from pipeline.ingest import Lead, ingest_csv
from pipeline.rules import (
    RecencyRule,
    RuleConfigError,
    RuleFactory,
    RuleResult,
    RuleScorer,
    SizeRule,
    SourceRule,
    register_rule,
    score_lead,
    score_leads,
)

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = date(2024, 1, 21)


WEIGHT_ENV = {
    "WEIGHT_INDUSTRY": "0.40",
    "WEIGHT_SIZE": "0.25",
    "WEIGHT_SOURCE": "0.20",
    "WEIGHT_RECENCY": "0.15",
}


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config.yaml", env=WEIGHT_ENV)


@pytest.fixture(scope="module")
def scorers(config):
    return {scorer.name: scorer for scorer in RuleFactory.create_all(config["rules"])}


def make_lead(size=None, source=None, last=None, lead_id="L001"):
    return Lead(
        lead_id=lead_id,
        name="Ann",
        company="Co",
        company_size=size,
        industry="Logistics",
        source=source,
        last_interaction_date=last,
    )


# --- size ----------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "size, expected",
    [
        (1, 1), (9, 1),
        (10, 4), (19, 4),
        (20, 7), (49, 7),
        (50, 10), (300, 10), (500, 10),
        (501, 7), (1000, 7),
        (1001, 4), (5000, 4),
        (5001, 1), (50000, 1),
        (None, 5),
    ],
)
def test_size_bands(scorers, size, expected):
    assert scorers["size"].score(make_lead(size=size), REFERENCE).score == expected


def test_size_bands_cover_every_size(scorers):
    for size in list(range(1, 6000)) + [10**6]:
        scorers["size"].score(make_lead(size=size), REFERENCE)  # raises if a size has no band


def test_size_detail_names_the_band(scorers):
    assert scorers["size"].score(make_lead(size=300), REFERENCE).detail == "300 employees, target range"
    assert "not provided" in scorers["size"].score(make_lead(), REFERENCE).detail


# --- source --------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "source, expected",
    [
        ("Inbound demo request", 10),
        ("Referral", 9),
        ("Sales call", 8),
        ("Webinar attendee", 6),
        ("Content download", 5),
        ("LinkedIn outreach", 3),
        ("  referral ", 9),
        ("REFERRAL", 9),
        ("Cold email", 4),
        (None, 4),
    ],
)
def test_source_scores(scorers, source, expected):
    assert scorers["source"].score(make_lead(source=source), REFERENCE).score == expected


def test_source_detail_for_unrecognized_value(scorers):
    assert "not recognized" in scorers["source"].score(make_lead(source="Cold email"), REFERENCE).detail


# --- recency -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "last, expected",
    [
        (date(2024, 1, 21), 10),  # 0 days
        (date(2024, 1, 7), 10),  # 14 days
        (date(2024, 1, 6), 7),  # 15 days
        (date(2023, 12, 22), 7),  # 30 days
        (date(2023, 12, 21), 4),  # 31 days
        (date(2023, 10, 23), 4),  # 90 days
        (date(2023, 10, 22), 1),  # 91 days
        (date(2022, 6, 15), 1),
        (None, 1),
        (date(2024, 2, 1), 10),  # after the reference date counts as 0 days
    ],
)
def test_recency_bands(scorers, last, expected):
    assert scorers["recency"].score(make_lead(last=last), REFERENCE).score == expected


def test_recency_detail_shows_days(scorers):
    assert scorers["recency"].score(make_lead(last=date(2024, 1, 6)), REFERENCE).detail == "15 days since contact"


# --- worked examples from docs/SCORING_WEIGHTS.md ------------------------------------------

@pytest.mark.parametrize(
    "lead, expected",
    [
        (make_lead(300, "Referral", date(2024, 1, 6)), {"size": 10, "source": 9, "recency": 7}),
        (make_lead(320, "LinkedIn outreach", date(2023, 12, 29)), {"size": 10, "source": 3, "recency": 7}),
        (make_lead(800, "Content download", date(2023, 12, 10)), {"size": 7, "source": 5, "recency": 4}),
    ],
)
def test_scoring_guide_examples(scorers, lead, expected):
    results = score_lead(lead, list(scorers.values()), REFERENCE)
    assert {name: result.score for name, result in results.items()} == expected


# --- factory -------------------------------------------------------------------------------

def test_factory_builds_registered_rules_in_config_order(config):
    built = RuleFactory.create_all(config["rules"])
    assert [type(scorer) for scorer in built] == [SizeRule, SourceRule, RecencyRule]
    assert [scorer.name for scorer in built] == ["size", "source", "recency"]


def test_factory_unknown_rule_lists_available_rules():
    with pytest.raises(RuleConfigError, match="unknown rule 'budget'.*recency.*size.*source"):
        RuleFactory.create("budget", {})


def test_factory_rejects_empty_or_invalid_rules_section():
    for bad in ({}, None, ["size"]):
        with pytest.raises(RuleConfigError, match="non-empty mapping"):
            RuleFactory.create_all(bad)


def test_factory_rejects_non_mapping_settings():
    with pytest.raises(RuleConfigError, match="settings must be a mapping"):
        RuleFactory.create("size", ["bands"])


def test_new_rule_is_added_with_a_class_and_a_config_entry():
    @register_rule("employee_growth")
    class GrowthRule(RuleScorer):
        def __init__(self, settings):
            self.fixed = settings["fixed_score"]

        def score(self, lead, reference_date):
            return RuleResult(rule=self.name, score=self.fixed, detail="growth rule")

    try:
        built = RuleFactory.create_all({"size": {"bands": [{"min": 1, "score": 5}], "missing_score": 5},
                                        "employee_growth": {"fixed_score": 8}})
        results = score_lead(make_lead(size=10), built, REFERENCE)
        assert {name: result.score for name, result in results.items()} == {"size": 5, "employee_growth": 8}
    finally:
        rules._REGISTRY.pop("employee_growth")


def test_registering_a_duplicate_name_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_rule("size")(SizeRule)


# --- invalid rule settings -----------------------------------------------------------------

@pytest.mark.parametrize(
    "name, settings, message",
    [
        ("size", {"missing_score": 5}, "missing setting 'bands'"),
        ("size", {"bands": [], "missing_score": 5}, "non-empty list"),
        ("size", {"bands": [{"score": 5}], "missing_score": 5}, "needs 'min', 'max' or both"),
        ("size", {"bands": [{"min": 10, "max": 5, "score": 5}], "missing_score": 5}, "min greater than max"),
        ("size", {"bands": [{"min": 1, "score": 11}], "missing_score": 5}, "from 1 to 10"),
        ("size", {"bands": [{"min": 1, "score": 5}], "missing_score": 0}, "from 1 to 10"),
        ("size", {"bands": [{"min": -1, "score": 5}], "missing_score": 5}, "0 or more"),
        ("source", {"scores": {}, "unknown_score": 4}, "non-empty mapping"),
        ("source", {"scores": {"referral": "high"}, "unknown_score": 4}, "from 1 to 10"),
        ("source", {"scores": {"referral": 9}}, "missing setting 'unknown_score'"),
        ("recency", {"bands": [{"max_days": 30, "score": 7}, {"max_days": 14, "score": 10}],
                     "older_score": 1, "missing_score": 1}, "strictly increasing"),
        ("recency", {"bands": [{"max_days": 14}], "older_score": 1, "missing_score": 1}, "from 1 to 10"),
        ("recency", {"bands": [{"max_days": 14, "score": 10}], "missing_score": 1}, "missing setting 'older_score'"),
    ],
)
def test_invalid_settings_raise_clear_errors(name, settings, message):
    with pytest.raises(RuleConfigError, match=message):
        RuleFactory.create(name, settings)


def test_size_band_gap_raises_at_scoring_time():
    scorer = RuleFactory.create("size", {"bands": [{"min": 50, "max": 500, "score": 10}], "missing_score": 5})
    with pytest.raises(RuleConfigError, match="no band covers 10 employees"):
        scorer.score(make_lead(size=10), REFERENCE)


# --- config validation ---------------------------------------------------------------------

def test_weights_must_add_up_to_one():
    with pytest.raises(ConfigError, match="add up to 1"):
        load_config(ROOT / "config.yaml", env={**WEIGHT_ENV, "WEIGHT_INDUSTRY": "0.50"})


def test_config_requires_rules(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('weights: {industry: 1}\nmissing_values: ["na"]\n')
    with pytest.raises(ConfigError, match="'rules'"):
        load_config(path, env={})


# --- whole data files ----------------------------------------------------------------------

@pytest.mark.parametrize("filename", ["leads_full_scale_100.csv", "leads_training.csv", "leads_testing.csv"])
def test_every_lead_gets_a_valid_score_from_every_rule(config, scorers, filename):
    leads = ingest_csv(ROOT / "data" / filename, REFERENCE, config["missing_values"]).leads
    all_scores = score_leads(leads, list(scorers.values()), REFERENCE)
    assert set(all_scores) == {lead.lead_id for lead in leads}
    for results in all_scores.values():
        assert set(results) == {"size", "source", "recency"}
        assert all(isinstance(r.score, int) and 1 <= r.score <= 10 and r.detail for r in results.values())


def test_scoring_is_repeatable(config, scorers):
    leads = ingest_csv(ROOT / "data" / "leads_full_scale_100.csv", REFERENCE, config["missing_values"]).leads
    assert score_leads(leads, list(scorers.values()), REFERENCE) == score_leads(leads, list(scorers.values()), REFERENCE)


def test_testing_file_scores_match_the_workflow_edge_cases(config, scorers):
    leads = {l.name: l for l in ingest_csv(ROOT / "data" / "leads_testing.csv", REFERENCE, config["missing_values"]).leads}
    pick = lambda name: {n: r.score for n, r in score_lead(leads[name], list(scorers.values()), REFERENCE).items()}
    assert pick("Test_MissingSize") == {"size": 5, "source": 10, "recency": 10}
    assert pick("Test_StaleInteraction") == {"size": 10, "source": 8, "recency": 1}
    assert pick("Test_NoSourceInfo") == {"size": 10, "source": 4, "recency": 10}
    assert pick("Test_NameOnly") == {"size": 5, "source": 4, "recency": 1}
    assert pick("Test_VeryLargeEnterprise")["size"] == 1


# --- command line --------------------------------------------------------------------------

def test_cli_scores_and_shows_table(capsys):
    assert main(["--input", str(ROOT / "data" / "leads_training.csv"), "--show-scores"]) == 0
    out = capsys.readouterr().out
    assert "Scored 29 leads with rules: size, source, recency" in out
    assert "L001" in out and "source" in out


def row_for(out, company):
    return next(line for line in out.splitlines() if company in line)


def test_score_table_shows_each_leads_industry_label(capsys):
    assert main(["--input", str(ROOT / "data" / "leads_training.csv"), "--show-scores"]) == 0
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if line.lstrip().startswith("lead "))
    assert header.split()[:3] == ["lead", "company", "industry"]
    assert header.split()[-3:] == ["size", "source", "recency"]
    assert "SaaS" in row_for(out, "CloudScale AI")
    assert "Logistics" in row_for(out, "Logistics Global")


def test_score_table_shows_a_long_industry_label_in_full(capsys):
    assert main(["--input", str(ROOT / "data" / "leads_testing.csv"), "--show-scores"]) == 0
    assert "Enterprise Software" in row_for(capsys.readouterr().out, "Global Fortune 500")


def test_score_table_marks_a_missing_industry_with_a_dash(capsys):
    assert main(["--input", str(ROOT / "data" / "leads_testing.csv"), "--show-scores"]) == 0
    line = row_for(capsys.readouterr().out, "Mysterious Startup")  # label was "Unknown Sector"
    assert " - " in line


def test_cli_reports_invalid_rule_config(tmp_path, capsys):
    text = (ROOT / "config.yaml").read_text().replace("older_score: 1", "older_score: 99")
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert main(["--input", str(ROOT / "data" / "leads_training.csv"), "--config", str(path)]) == 1
    assert "from 1 to 10" in capsys.readouterr().err
