import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from pipeline.decide import DecisionSettings, decide_leads
from pipeline.industry import IndustrySettings
from pipeline.message_checks import MessageRules
from pipeline.messages import MessageSettings
from pipeline.llm_client import RetryPolicy
from pipeline.industry import IndustryScorer, IndustrySettings
from pipeline.ingest import IngestError, ingest_csv
from pipeline.config import load_config
from pipeline.models import (
    IndustryItem,
    DecisionSummary,
    IndustryResponse,
    IndustryResult,
    IndustryScoring,
    IngestResult,
    Lead,
    LeadDecision,
    LeadPayload,
    LeadRow,
    MessageRun,
    RowError,
    RuleResult,
    clean_text,
)
from pipeline.rules import RuleFactory, score_leads
from tests.test_decide import DataFileClient
from tests.test_industry import ScriptedClient, item, reply

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = date(2024, 1, 21)
CONTEXT = {"missing_values": frozenset({"na", "n/a", "unknown"}), "reference_date": REFERENCE}
RANGES = {"best": (9, 10), "hard": (4, 6), "none": (1, 3), "unknown": (5, 5)}
ENV = {"WEIGHT_INDUSTRY": "0.4", "WEIGHT_SIZE": "0.25", "WEIGHT_SOURCE": "0.2", "WEIGHT_RECENCY": "0.15"}


def row(**fields):
    base = {"name": "Ann", "company": "Co", "company_size": "50", "industry": "Retail",
            "source": "Referral", "last_interaction_date": "2024-01-15"}
    return LeadRow.model_validate({**base, **fields}, context=CONTEXT)


def error_for(model, data, context=None):
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data, context=context)
    return caught.value.errors()


# --- LeadRow: the CSV row ------------------------------------------------------------------

def test_row_is_cleaned_and_typed():
    parsed = row(name="  Alice   Chen ", company_size="1,200")
    assert (parsed.name, parsed.company_size, parsed.last_interaction_date) == ("Alice Chen", 1200, date(2024, 1, 15))


def test_placeholders_become_none_only_when_the_context_lists_them():
    parsed = row(company_size="NA", industry="Unknown", source="n/a", last_interaction_date="NA")
    assert (parsed.company_size, parsed.industry, parsed.source, parsed.last_interaction_date) == (None,) * 4
    without = LeadRow.model_validate({"name": "Ann", "industry": "NA"})  # no context: nothing is a placeholder
    assert without.industry == "NA"


@pytest.mark.parametrize("value", ["abc", "0", "-5", "1.5", "12abc", "1e3"])
def test_bad_company_size_fails_only_that_field(value):
    errors = error_for(LeadRow, {"name": "Ann", "company_size": value}, CONTEXT)
    assert [e["loc"] for e in errors] == [("company_size",)]
    assert "company_size is not a positive whole number" in errors[0]["msg"]


@pytest.mark.parametrize("value", ["15/01/2024", "2024-1-5", "2024-02-30", "yesterday", "20240115"])
def test_bad_dates_fail_only_that_field(value):
    errors = error_for(LeadRow, {"name": "Ann", "last_interaction_date": value}, CONTEXT)
    assert [e["loc"] for e in errors] == [("last_interaction_date",)]
    assert "not a valid YYYY-MM-DD date" in errors[0]["msg"]


def test_a_date_after_the_reference_date_is_rejected():
    errors = error_for(LeadRow, {"name": "Ann", "last_interaction_date": "2024-01-22"}, CONTEXT)
    assert "after the reference date 2024-01-21" in errors[0]["msg"]
    assert row(last_interaction_date="2024-01-21").last_interaction_date == REFERENCE  # the day itself is fine


def test_python_values_are_accepted_too():
    parsed = row(company_size=300, last_interaction_date=date(2024, 1, 2))
    assert parsed.company_size == 300 and parsed.last_interaction_date == date(2024, 1, 2)
    assert error_for(LeadRow, {"name": "Ann", "company_size": 0}, CONTEXT)
    assert error_for(LeadRow, {"name": "Ann", "company_size": True}, CONTEXT)


def test_a_row_needs_a_name_or_a_company():
    assert row(name="NA").company == "Co"
    assert row(company="NA").name == "Ann"
    errors = error_for(LeadRow, {"name": "NA", "company": ""}, CONTEXT)
    assert errors[0]["loc"] == () and "both name and company are missing" in errors[0]["msg"]


def test_unknown_columns_are_ignored():
    assert row(favourite_colour="blue").name == "Ann"


def test_clean_text():
    assert clean_text("  a   b ", None) == "a b"
    assert clean_text("NA", {"missing_values": frozenset({"na"})}) is None
    assert clean_text(None, None) is None and clean_text("   ", None) is None
    assert clean_text(12, None) == "12"


# --- Lead ----------------------------------------------------------------------------------

def test_lead_id_must_look_like_an_id():
    assert Lead(lead_id="L001").lead_id == "L001" and Lead(lead_id="L1000")
    for bad in ("L1", "001", "l001", "L00A", ""):
        assert error_for(Lead, {"lead_id": bad})


def test_lead_size_must_be_positive():
    assert error_for(Lead, {"lead_id": "L001", "company_size": 0})
    assert error_for(Lead, {"lead_id": "L001", "company_size": "50"})  # strict: text is not a number
    assert error_for(Lead, {"lead_id": "L001", "company_size": True})
    assert Lead(lead_id="L001", company_size=1).company_size == 1


def test_lead_lists_are_not_shared_between_leads():
    first, second = Lead(lead_id="L001"), Lead(lead_id="L002")
    first.flags.append("insufficient_data")
    assert second.flags == [] and second.notes == []


# --- ingestion with the models -------------------------------------------------------------

HEADER = "name,company,company_size,industry,source,last_interaction_date\n"


def ingest(tmp_path, body):
    path = tmp_path / "leads.csv"
    path.write_text(HEADER + body)
    return ingest_csv(path, REFERENCE, ["na", "n/a", "unknown"])


def test_ingestion_returns_lead_models(tmp_path):
    result = ingest(tmp_path, "Ann,Co,50,Retail,Referral,2024-01-15\n")
    assert all(isinstance(lead, Lead) for lead in result.leads)


def test_a_row_with_a_bad_size_and_no_name_or_company_is_dropped_with_that_reason(tmp_path):
    result = ingest(tmp_path, "NA,NA,abc,SaaS,Referral,2024-01-01\nAnn,Co,10,SaaS,Referral,2024-01-01\n")
    assert [lead.name for lead in result.leads] == ["Ann"]
    assert result.errors == [RowError(row=2, reason="both name and company are missing")]


def test_both_soft_fields_bad_gives_two_notes_and_keeps_the_lead(tmp_path):
    lead = ingest(tmp_path, "Ann,Co,lots,SaaS,Referral,soon\n").leads[0]
    assert lead.company_size is None and lead.last_interaction_date is None
    assert len(lead.notes) == 2 and any("company_size" in n for n in lead.notes) and any("YYYY-MM-DD" in n for n in lead.notes)


def test_strict_mode_reports_the_model_message(tmp_path):
    path = tmp_path / "leads.csv"
    path.write_text(HEADER + "Ann,Co,lots,SaaS,Referral,2024-01-01\n")
    with pytest.raises(IngestError, match="company_size is not a positive whole number: 'lots'"):
        ingest_csv(path, REFERENCE, [], strict=True)


# --- RuleResult ----------------------------------------------------------------------------

def test_rule_result_score_must_be_one_to_ten():
    assert RuleResult(rule="size", score=1, detail="d").score == 1
    for bad in (0, 11, -1, 5.5, "7"):
        assert error_for(RuleResult, {"rule": "size", "score": bad, "detail": "d"})


def test_rule_result_is_frozen():
    result = RuleResult(rule="size", score=5, detail="d")
    with pytest.raises(ValidationError):
        result.score = 6


# --- LeadPayload: what the model is told ---------------------------------------------------

def test_payload_is_built_from_a_lead():
    payload = LeadPayload.from_lead(Lead(lead_id="L001", company="Acme", industry="Retail", company_size=50, name="Ann", source="Referral"))
    assert payload.model_dump() == {"lead_id": "L001", "company": "Acme", "industry": "Retail", "company_size": 50}


def test_payload_marks_missing_values_and_truncates_long_names():
    payload = LeadPayload.from_lead(Lead(lead_id="L001", company="x" * 300))
    assert payload.industry == "(missing)" and payload.company_size is None and len(payload.company) == 100
    assert LeadPayload.from_lead(Lead(lead_id="L002")).company == "(missing)"


def test_payload_refuses_an_over_long_company_directly():
    assert error_for(LeadPayload, {"lead_id": "L001", "company": "x" * 101, "industry": "i", "company_size": None})


# --- IndustryResponse and IndustryItem: the model's reply ----------------------------------

def test_reply_envelope_accepts_a_wrapped_or_bare_list():
    assert IndustryResponse.model_validate({"results": [1, 2]}).results == [1, 2]
    assert IndustryResponse.model_validate([{"a": 1}]).results == [{"a": 1}]


@pytest.mark.parametrize("data", [{}, {"answer": []}, {"results": "none"}, {"results": None}, "text", 5, None])
def test_reply_envelope_rejects_other_shapes(data):
    with pytest.raises(ValidationError):
        IndustryResponse.model_validate(data)


def valid_item(**changes):
    return {"lead_id": "L001", "tier": "best", "industry_fit": 10, "confidence": "high", "reason": "Why.", **changes}


def test_valid_item():
    parsed = IndustryItem.model_validate(valid_item(extra_field="ignored"), context={"ranges": RANGES})
    assert (parsed.lead_id, parsed.tier, parsed.industry_fit, parsed.confidence, parsed.reason) == ("L001", "best", 10, "high", "Why.")


@pytest.mark.parametrize(
    "changes",
    [
        {"lead_id": 1}, {"lead_id": ["L001"]}, {"lead_id": None},
        {"tier": "great"}, {"tier": "Best"}, {"tier": None},
        {"confidence": "certain"}, {"confidence": "HIGH"},
        {"industry_fit": "10"}, {"industry_fit": True}, {"industry_fit": None}, {"industry_fit": 9.5}, {"industry_fit": [9]},
        {"reason": ""}, {"reason": "   "}, {"reason": None}, {"reason": 5},
    ],
)
def test_invalid_items(changes):
    assert error_for(IndustryItem, valid_item(**changes), {"ranges": RANGES})


@pytest.mark.parametrize("missing", ["lead_id", "tier", "industry_fit", "confidence", "reason"])
def test_missing_fields_fail(missing):
    data = valid_item()
    del data[missing]
    errors = error_for(IndustryItem, data, {"ranges": RANGES})
    assert errors[0]["type"] == "missing" and errors[0]["loc"] == (missing,)


def test_whole_valued_floats_are_accepted():
    assert IndustryItem.model_validate(valid_item(industry_fit=9.0)).industry_fit == 9


def test_reason_is_trimmed_and_cut_to_200_characters():
    parsed = IndustryItem.model_validate(valid_item(reason="  " + "x" * 500))
    assert parsed.reason == "x" * 200


def test_score_must_sit_inside_its_tier_range_when_ranges_are_given():
    parsed = IndustryItem.model_validate(valid_item(tier="hard", industry_fit=5), context={"ranges": RANGES})
    assert parsed.industry_fit == 5
    errors = error_for(IndustryItem, valid_item(tier="hard", industry_fit=9), {"ranges": RANGES})
    assert [e["type"] for e in errors] == ["score_out_of_range"]
    assert "score 9 is outside the hard range 4-6" in errors[0]["msg"]
    assert error_for(IndustryItem, valid_item(tier="unknown", industry_fit=4), {"ranges": RANGES})


def test_no_range_check_without_ranges():
    assert IndustryItem.model_validate(valid_item(tier="hard", industry_fit=9)).industry_fit == 9


# --- IndustryResult ------------------------------------------------------------------------

def test_result_is_validated_and_frozen():
    result = IndustryResult(lead_id="L001", tier="none", industry_fit=2, confidence="high", reason="r")
    assert result.llm_failed is False
    with pytest.raises(ValidationError):
        result.tier = "best"
    for changes in ({"industry_fit": 0}, {"industry_fit": 11}, {"tier": "great"}, {"confidence": "sure"}, {"reason": ""}, {"reason": "x" * 201}):
        assert error_for(IndustryResult, {"lead_id": "L001", "tier": "none", "industry_fit": 2, "confidence": "high", "reason": "r", **changes})


# --- LeadDecision --------------------------------------------------------------------------

def decision(**changes):
    base = dict(
        lead_id="L001", name="Ann", company="Co", industry_label="Retail", company_size=50, tier="best",
        confidence="high", factor_scores={"industry": 10, "size": 10}, factor_points={"industry": 4.0, "size": 2.5},
        final_score=9.0, decision="qualified", priority="HIGH", overall_rank=0, priority_rank=None,
        rules_applied=[], rejection_reason=None, flags=[],
    )
    return LeadDecision(**{**base, **changes})


def test_a_consistent_decision_is_accepted():
    assert decision().priority == "HIGH"
    assert decision(decision="review", priority=None).priority is None
    assert decision(decision="rejected", priority=None, rejection_reason="Industry fit").rejection_reason


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"priority": None}, "priority is set for qualified leads only"),
        ({"decision": "review"}, "priority is set for qualified leads only"),
        ({"decision": "review", "priority": None, "priority_rank": 1}, "priority_rank is set for qualified leads only"),
        ({"decision": "rejected", "priority": None}, "rejection_reason is set for rejected leads only"),
        ({"rejection_reason": "Industry fit"}, "rejection_reason is set for rejected leads only"),
        ({"final_score": 10.5}, "less than or equal to 10"),
        ({"final_score": 0.5}, "greater than or equal to 1"),
        ({"decision": "maybe"}, "decision"),
        ({"priority": "URGENT"}, "priority"),
        ({"factor_scores": {"industry": 11}}, "less than or equal to 10"),
        ({"company_size": 0}, "greater than or equal to 1"),
        ({"overall_rank": -1}, "greater than or equal to 0"),
    ],
)
def test_inconsistent_decisions_are_rejected(changes, message):
    with pytest.raises(ValidationError, match=message):
        decision(**changes)


def test_values_assigned_after_ranking_are_validated_too():
    record = decision()
    record.overall_rank = 3
    record.priority_rank = 1
    record.explanation = "text"
    assert (record.overall_rank, record.priority_rank, record.explanation) == (3, 1, "text")
    with pytest.raises(ValidationError):
        record.overall_rank = -2
    with pytest.raises(ValidationError):
        record.priority_rank = 0
    with pytest.raises(ValidationError):
        record.final_score = 11


# --- the whole chain -----------------------------------------------------------------------

def test_every_record_in_the_pipeline_is_a_validated_model(tmp_path):
    config = load_config(ROOT / "config.yaml", env=ENV)
    scorers = RuleFactory.create_all(config["rules"])
    leads = ingest_csv(ROOT / "data" / "leads_testing.csv", REFERENCE, config["missing_values"]).leads
    settings = IndustrySettings.from_config(config).model_copy(update={"cache_path": tmp_path / "c.json"})
    industry = IndustryScorer(DataFileClient(config["tiers"]), settings).score(leads)
    rules = score_leads(leads, scorers, REFERENCE)
    decisions = decide_leads(
        leads, rules, industry.results, DecisionSettings.from_config(config, [s.name for s in scorers]), REFERENCE
    )
    assert all(isinstance(lead, Lead) for lead in leads)
    assert all(isinstance(r, RuleResult) for per_lead in rules.values() for r in per_lead.values())
    assert all(isinstance(r, IndustryResult) for r in industry.results.values())
    assert len(decisions) == 20 and all(isinstance(d, LeadDecision) for d in decisions)
    assert all(isinstance(model, BaseModel) for model in (leads[0], decisions[0]))


def test_decisions_serialize_to_json_and_load_back_unchanged(tmp_path):
    config = load_config(ROOT / "config.yaml", env=ENV)
    scorers = RuleFactory.create_all(config["rules"])
    leads = ingest_csv(ROOT / "data" / "leads_training.csv", REFERENCE, config["missing_values"]).leads
    settings = IndustrySettings.from_config(config).model_copy(update={"cache_path": tmp_path / "c.json"})
    industry = IndustryScorer(DataFileClient(config["tiers"]), settings).score(leads)
    decisions = decide_leads(
        leads, score_leads(leads, scorers, REFERENCE), industry.results,
        DecisionSettings.from_config(config, [s.name for s in scorers]), REFERENCE,
    )
    for each in decisions:
        assert LeadDecision.model_validate_json(each.model_dump_json()) == each
    assert json.loads(decisions[0].model_dump_json())["overall_rank"] == 1


def test_a_reply_item_with_the_wrong_tier_case_is_retried(tmp_path):
    config = load_config(ROOT / "config.yaml", env=ENV)
    settings = IndustrySettings.from_config(config).model_copy(update={"cache_path": tmp_path / "c.json"})
    client = ScriptedClient([reply(item("L001", tier="Best")), reply(item("L001"))])
    result = IndustryScorer(client, settings).score([Lead(lead_id="L001", company="Acme", industry="Logistics", company_size=100)])
    assert result.api_calls == 2 and result.results["L001"].tier == "best"


# --- settings and result containers are models, not dataclasses -------------------------------

def test_no_dataclasses_are_left_in_the_pipeline():
    sources = list((ROOT / "pipeline").glob("*.py")) + [ROOT / "main.py"]
    assert [path.name for path in sources if "dataclass" in path.read_text()] == []


@pytest.mark.parametrize("model", [DecisionSettings, IndustrySettings, MessageSettings, MessageRules, RetryPolicy,
                                   IngestResult, IndustryScoring, MessageRun, DecisionSummary, RowError])
def test_every_stage_record_is_a_pydantic_model(model):
    assert issubclass(model, BaseModel)


def decision_settings(**changes):
    base = dict(weights={"industry": 1.0}, qualified_at=7, review_at=5, high_priority_at=8.5,
                stale_days=180, reject_size_below=10, reject_size_above=5000)
    return DecisionSettings(**{**base, **changes})


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"review_at": 8}, "review < qualified"),
        ({"qualified_at": 11}, "review < qualified"),
        ({"high_priority_at": 6}, "priority.high must be between"),
        ({"reject_size_below": 6000}, "must not exceed"),
        ({"stale_days": -1}, "whole number of 0 or more"),
        ({"qualified_at": "high"}, "must be a number"),
        ({"weights": {"industry": "lots"}}, "weights"),
    ],
)
def test_decision_settings_validate_themselves(changes, message):
    with pytest.raises(ValidationError, match=message):
        decision_settings(**changes)


def test_decision_settings_are_frozen_and_keyword_only():
    settings = decision_settings()
    with pytest.raises(ValidationError):
        settings.stale_days = 1
    with pytest.raises(TypeError):
        DecisionSettings({"industry": 1.0}, 7, 5, 8.5, 180, 10, 5000)


def message_settings(config_env=None):
    return MessageSettings.from_config(load_config(ROOT / "config.yaml", env=config_env or ENV))


def test_message_settings_validate_their_fields_directly():
    good = message_settings()
    for changes in ({"batch_size": 0}, {"min_words": 0}, {"temperature": -1}, {"sender": 5}):
        with pytest.raises(ValidationError):
            MessageSettings(**{**good.model_dump(), **changes})
    with pytest.raises(ValidationError, match="min_words must not exceed max_words"):
        MessageSettings(**{**good.model_dump(), "min_words": 200})
    with pytest.raises(ValidationError):
        good.batch_size = 3  # frozen


def test_message_rules_validate_themselves():
    base = dict(first_name="Oscar", company="Acme", industry="Retail", sender="Dana", variant="value_led",
                call_minutes=15, min_words=40, max_words=100, max_subject_words=10)
    assert MessageRules(**base).first_name == "Oscar"
    for changes in ({"variant": "cold_call"}, {"first_name": ""}, {"sender": ""}, {"call_minutes": 0}, {"min_words": 101}):
        with pytest.raises(ValidationError):
            MessageRules(**{**base, **changes})


def test_industry_settings_validate_their_fields_directly():
    config = load_config(ROOT / "config.yaml", env=ENV)
    good = IndustrySettings.from_config(config)
    assert good.batch_size == 10 and good.tier_ranges["best"] == (9, 10)
    for changes in ({"batch_size": 0}, {"temperature": -1}, {"max_tokens": 0}, {"validation_retries": -1}):
        with pytest.raises(ValidationError):
            IndustrySettings(**{**good.model_dump(), **changes})
    with pytest.raises(ValidationError, match="tiers.best.score_range"):
        IndustrySettings(**{**good.model_dump(), "tier_ranges": {"best": [10, 9]}})


def test_row_error_and_ingest_result_are_validated():
    assert RowError(row=2, reason="x").row == 2
    for bad in ({"row": 0, "reason": "x"}, {"row": 2, "reason": ""}, {"row": None, "reason": "x"}):
        with pytest.raises(ValidationError):
            RowError(**bad)
    with pytest.raises(ValidationError):
        IngestResult(leads=[], errors=[], rows_read=-1)
    with pytest.raises(ValidationError):
        IngestResult(leads=["not a lead"], errors=[], rows_read=0)


def test_run_counters_are_validated_when_they_change():
    scoring = IndustryScoring(results={})
    scoring.api_calls += 1
    scoring.failed_ids.append("L001")
    assert scoring.api_calls == 1 and scoring.failed_ids == ["L001"]
    with pytest.raises(ValidationError):
        scoring.api_calls = -1
    run = MessageRun(messages={})
    run.api_calls += 2
    assert run.api_calls == 2
    with pytest.raises(ValidationError):
        run.api_calls = "many"


def test_decision_summary_percent():
    summary = DecisionSummary(
        total=4, counts={"qualified": 1, "review": 2, "rejected": 1}, average_score=5.0,
        highest_score=9.0, lowest_score=2.0, average_factor_scores={"industry": 5.0},
    )
    assert summary.percent("review") == 50.0
    assert DecisionSummary(total=0, counts={"qualified": 0, "review": 0, "rejected": 0}, average_score=0,
                           highest_score=0, lowest_score=0, average_factor_scores={}).percent("qualified") == 0.0
    with pytest.raises(ValidationError):
        DecisionSummary(total=1, counts={"maybe": 1}, average_score=1, highest_score=1, lowest_score=1,
                        average_factor_scores={})
