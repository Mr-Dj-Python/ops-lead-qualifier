import json
import random
import re
from datetime import date
from pathlib import Path

import pytest

import main
from pipeline.config import ConfigError, load_config
from pipeline.decide import (
    QUALIFIED,
    REJECTED,
    REVIEW,
    DecisionSettings,
    add_flag,
    decide_leads,
    summarize,
)
from pipeline.industry import IndustryResult
from pipeline.ingest import Lead, ingest_csv
from pipeline.llm_client import LLMError
from pipeline.rules import RuleFactory, RuleResult, score_leads
from tests.test_industry import RulesClient, ScriptedClient, item, reply

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = date(2024, 1, 21)
ENV = {"WEIGHT_INDUSTRY": "0.4", "WEIGHT_SIZE": "0.25", "WEIGHT_SOURCE": "0.2", "WEIGHT_RECENCY": "0.15"}


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config.yaml", env=ENV)


@pytest.fixture(scope="module")
def scorers(config):
    return RuleFactory.create_all(config["rules"])


@pytest.fixture(scope="module")
def settings(config, scorers):
    return DecisionSettings.from_config(config, [scorer.name for scorer in scorers])


def only_industry(settings):
    """Weights that make the weighted score equal the industry score, to hit thresholds exactly."""
    return DecisionSettings(weights={"industry": 1.0, "size": 0.0, "source": 0.0, "recency": 0.0}, qualified_at=settings.qualified_at, review_at=settings.review_at, high_priority_at=settings.high_priority_at, stale_days=settings.stale_days, reject_size_below=settings.reject_size_below, reject_size_above=settings.reject_size_above)


def make_lead(lead_id="L001", size=100, last=date(2024, 1, 15), flags=(), company="Acme", industry="Logistics"):
    return Lead(
        lead_id=lead_id, name="Ann", company=company, company_size=size, industry=industry,
        source="Referral", last_interaction_date=last, flags=list(flags),
    )


def fit(lead_id="L001", tier="best", score=10, confidence="high", failed=False):
    return IndustryResult(
        lead_id=lead_id, tier=tier, industry_fit=score, confidence=confidence, reason="Because.", llm_failed=failed
    )


def rules(size=10, source=10, recency=10):
    return {
        "size": RuleResult(rule="size", score=size, detail="size detail"),
        "source": RuleResult(rule="source", score=source, detail="source detail"),
        "recency": RuleResult(rule="recency", score=recency, detail="recency detail"),
    }


def decide(lead, industry, settings, rule_scores=None):
    out = decide_leads([lead], {lead.lead_id: rule_scores or rules()}, {lead.lead_id: industry}, settings, REFERENCE)
    return out[0]


# --- settings ------------------------------------------------------------------------------

def test_settings_come_from_config(settings):
    assert settings.weights == {"industry": 0.4, "size": 0.25, "source": 0.2, "recency": 0.15}
    assert (settings.qualified_at, settings.review_at, settings.high_priority_at) == (7.0, 5.0, 8.5)
    assert (settings.stale_days, settings.reject_size_below, settings.reject_size_above) == (180, 10, 5000)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["weights"].pop("size"), "no entry for: size"),
        (lambda c: c["weights"].update(budget=0.1), "not scored: budget"),
        (lambda c: c.update(thresholds={"qualified": 5, "review": 7}), "review < qualified"),
        (lambda c: c.update(thresholds={"qualified": 7, "review": "low"}), "thresholds.review must be a number"),
        (lambda c: c.update(thresholds={"qualified": 11, "review": 5}), "review < qualified"),
        (lambda c: c.update(priority={"high": 6}), "priority.high"),
        (lambda c: c["caps"].update(stale_days=-1), "caps.stale_days"),
        (lambda c: c["caps"].update(hard_reject_size_below=6000), "must not exceed"),
        (lambda c: c.pop("caps"), "'caps' must be a mapping"),
        (lambda c: c.pop("thresholds"), "'thresholds' must be a mapping"),
    ],
)
def test_invalid_decision_settings_are_rejected(config, mutate, message):
    broken = json.loads(json.dumps(config))
    mutate(broken)
    with pytest.raises(ConfigError, match=message):
        DecisionSettings.from_config(broken, ["size", "source", "recency"])


# --- weighted score ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lead, fit_result, rule_scores, expected_score, expected_decision",
    [
        # the worked examples in docs/SCORING_WEIGHTS.md
        (make_lead(size=300, last=date(2024, 1, 6)), fit(tier="best", score=10), rules(10, 9, 7), 9.35, QUALIFIED),
        (make_lead(size=320, last=date(2023, 12, 29)), fit(tier="hard", score=5), rules(10, 3, 7), 6.15, REVIEW),
        (make_lead(size=800, last=date(2023, 12, 10)), fit(tier="none", score=2), rules(7, 5, 4), 4.15, REJECTED),
    ],
)
def test_scoring_guide_examples(settings, lead, fit_result, rule_scores, expected_score, expected_decision):
    result = decide(lead, fit_result, settings, rule_scores)
    assert result.final_score == expected_score and result.decision == expected_decision


def test_points_add_up_to_the_final_score(settings):
    result = decide(make_lead(), fit(score=9), settings, rules(7, 8, 4))
    assert result.factor_points == {"industry": 3.6, "size": 1.75, "source": 1.6, "recency": 0.6}
    assert round(sum(result.factor_points.values()), 2) == result.final_score == 7.55


def test_weighted_score_is_the_weighted_average_on_the_same_scale(settings):
    result = decide(make_lead(), fit(score=10), settings, rules(10, 10, 10))
    assert result.final_score == 10
    result = decide(make_lead(), fit(tier="none", score=1), settings, rules(1, 1, 1))
    assert result.final_score == 1


def test_changed_weights_change_the_score(settings):
    heavier_size = DecisionSettings(weights={"industry": 0.1, "size": 0.7, "source": 0.1, "recency": 0.1}, qualified_at=7.0, review_at=5.0, high_priority_at=8.5, stale_days=180, reject_size_below=10, reject_size_above=5000)
    base = decide(make_lead(), fit(score=10), settings, rules(1, 10, 10)).final_score
    shifted = decide(make_lead(), fit(score=10), heavier_size, rules(1, 10, 10)).final_score
    assert base == 7.75 and shifted == 3.7


# --- thresholds and priority ---------------------------------------------------------------

@pytest.mark.parametrize("score, decision", [(10, QUALIFIED), (7, QUALIFIED), (6, REVIEW), (5, REVIEW), (4, REJECTED), (1, REJECTED)])
def test_threshold_boundaries(settings, score, decision):
    result = decide(make_lead(), fit(tier="best", score=score), only_industry(settings))
    assert result.final_score == score and result.decision == decision


@pytest.mark.parametrize("score, priority", [(10, "HIGH"), (9, "HIGH"), (8, "MEDIUM"), (7, "MEDIUM")])
def test_priority_labels_for_qualified_leads(settings, score, priority):
    assert decide(make_lead(), fit(score=score), only_industry(settings)).priority == priority


def test_only_qualified_leads_get_a_priority(settings):
    for score in (6, 3):
        result = decide(make_lead(), fit(score=score), only_industry(settings))
        assert result.priority is None and result.priority_rank is None


# --- caps and overrides --------------------------------------------------------------------

@pytest.mark.parametrize(
    "lead, fit_result, rule_text",
    [
        (make_lead(), fit(tier="none", score=3), 'tier "none"'),
        (make_lead(), fit(tier="unknown", score=5, confidence="low"), 'tier "unknown"'),
        (make_lead(), fit(tier="best", score=10, confidence="low"), "low LLM confidence"),
        (make_lead(flags=["insufficient_data"]), fit(), "insufficient data"),
        (make_lead(last=date(2023, 7, 1)), fit(), "no contact for 204 days (over 180)"),
        (make_lead(), fit(tier="unknown", score=5, confidence="low", failed=True), "LLM scoring failed"),
    ],
)
def test_caps_stop_a_high_score_from_qualifying(settings, lead, fit_result, rule_text):
    strong = DecisionSettings(weights={"industry": 0.0, "size": 0.5, "source": 0.5, "recency": 0.0}, qualified_at=7.0, review_at=5.0, high_priority_at=8.5, stale_days=180, reject_size_below=10, reject_size_above=5000)
    result = decide(lead, fit_result, strong)
    assert result.final_score == 10 and result.decision == REVIEW and result.priority is None
    assert any(rule_text in rule for rule in result.rules_applied)


def test_stale_cap_boundary(settings):
    at_limit = decide(make_lead(last=date(2023, 7, 25)), fit(), settings)  # exactly 180 days
    past_limit = decide(make_lead(last=date(2023, 7, 24)), fit(), settings)  # 181 days
    assert (at_limit.decision, past_limit.decision) == (QUALIFIED, REVIEW)


def test_a_missing_contact_date_is_not_treated_as_stale(settings):
    assert decide(make_lead(last=None), fit(), settings).decision == QUALIFIED


def test_all_applicable_caps_are_listed(settings):
    lead = make_lead(flags=["insufficient_data"], last=date(2022, 6, 15))
    result = decide(lead, fit(tier="none", score=3), settings)
    assert len(result.rules_applied) == 3 and result.decision == REVIEW


def test_caps_never_raise_a_decision(settings):
    result = decide(make_lead(), fit(tier="none", score=2), only_industry(settings))
    assert result.decision == REJECTED and result.rules_applied == [] and result.rejection_reason == "Industry fit"


@pytest.mark.parametrize("size", [1, 9, 5001, 50000])
def test_no_fit_industry_with_unrealistic_size_is_rejected_whatever_the_score(settings, size):
    strong = DecisionSettings(weights={"industry": 0.0, "size": 0.0, "source": 0.5, "recency": 0.5}, qualified_at=7.0, review_at=5.0, high_priority_at=8.5, stale_days=180, reject_size_below=10, reject_size_above=5000)
    result = decide(make_lead(size=size), fit(tier="none", score=3), strong)
    assert result.final_score == 10 and result.decision == REJECTED
    assert "no-fit industry" in result.rules_applied[0]
    assert result.rejection_reason == "no-fit industry with size outside the realistic range"


@pytest.mark.parametrize("size", [10, 50, 5000, None])
def test_no_fit_industry_with_ordinary_or_unknown_size_is_not_hard_rejected(settings, size):
    strong = DecisionSettings(weights={"industry": 0.0, "size": 0.0, "source": 0.5, "recency": 0.5}, qualified_at=7.0, review_at=5.0, high_priority_at=8.5, stale_days=180, reject_size_below=10, reject_size_above=5000)
    assert decide(make_lead(size=size), fit(tier="none", score=3), strong).decision == REVIEW


def test_a_better_industry_with_tiny_size_is_not_hard_rejected(settings):
    strong = DecisionSettings(weights={"industry": 0.0, "size": 0.0, "source": 0.5, "recency": 0.5}, qualified_at=7.0, review_at=5.0, high_priority_at=8.5, stale_days=180, reject_size_below=10, reject_size_above=5000)
    assert decide(make_lead(size=5), fit(tier="best", score=10), strong).decision == QUALIFIED


def test_a_failed_lead_goes_to_review_even_with_a_low_score(settings):
    failed = fit(tier="unknown", score=5, confidence="low", failed=True)
    result = decide(make_lead(), failed, settings, rules(1, 1, 1))
    assert result.final_score < settings.review_at and result.decision == REVIEW
    assert "llm_failed" in result.flags and "LLM scoring failed" in result.rules_applied[0]


def test_a_lead_with_too_little_data_goes_to_review_even_with_a_low_score(settings):
    result = decide(make_lead(flags=["insufficient_data"]), fit(tier="unknown", score=5, confidence="low"), settings, rules(5, 4, 1))
    assert result.final_score == 4.2 and result.decision == REVIEW and result.rejection_reason is None
    assert "too little data" in result.rules_applied[0]


def test_review_by_score_band_says_so(settings):
    result = decide(make_lead(), fit(tier="hard", score=5), settings, rules(7, 7, 7))
    assert result.final_score == 6.2 and result.decision == REVIEW
    assert "review band (5 to below 7)" in result.rules_applied[0]


# --- rejection reasons ---------------------------------------------------------------------

def test_rejection_reason_is_the_lowest_scoring_factor(settings):
    result = decide(make_lead(), fit(tier="hard", score=4), settings, rules(1, 5, 5))
    assert result.decision == REJECTED and result.rejection_reason == "Company size"


def test_tied_lowest_factors_are_all_named(settings):
    result = decide(make_lead(), fit(tier="hard", score=4), settings, rules(1, 9, 1))
    assert result.rejection_reason == "Company size and Engagement"


# --- ranking -------------------------------------------------------------------------------

def ranked_batch(settings):
    leads = [
        make_lead("L001", last=date(2024, 1, 10)),
        make_lead("L002", last=date(2024, 1, 20)),
        make_lead("L003", last=date(2024, 1, 20)),
        make_lead("L004", last=None),
        make_lead("L005", last=date(2024, 1, 5)),
        make_lead("L006", last=date(2024, 1, 1)),
    ]
    scores = {"L001": 9, "L002": 9, "L003": 9, "L004": 9, "L005": 6, "L006": 2}
    industry = {i: fit(i, "best" if s > 4 else "hard", s) for i, s in scores.items()}
    rule_scores = {lead.lead_id: rules() for lead in leads}
    return leads, rule_scores, industry


def test_leads_are_ordered_best_fit_first_with_ties_broken_by_recent_contact(settings):
    leads, rule_scores, industry = ranked_batch(settings)
    result = decide_leads(leads, rule_scores, industry, only_industry(settings), REFERENCE)
    assert [d.lead_id for d in result] == ["L002", "L003", "L001", "L004", "L005", "L006"]
    assert [d.overall_rank for d in result] == [1, 2, 3, 4, 5, 6]
    scores = [d.final_score for d in result]
    assert scores == sorted(scores, reverse=True)


def test_priority_rank_counts_only_qualified_leads_in_order(settings):
    leads, rule_scores, industry = ranked_batch(settings)
    result = decide_leads(leads, rule_scores, industry, only_industry(settings), REFERENCE)
    assert [(d.lead_id, d.priority_rank) for d in result if d.decision == QUALIFIED] == [
        ("L002", 1), ("L003", 2), ("L001", 3), ("L004", 4),
    ]
    assert [d.priority_rank for d in result if d.decision != QUALIFIED] == [None, None]


def test_input_order_does_not_change_the_ranking(settings):
    leads, rule_scores, industry = ranked_batch(settings)
    expected = [d.lead_id for d in decide_leads(leads, rule_scores, industry, only_industry(settings), REFERENCE)]
    for seed in range(5):
        shuffled = leads[:]
        random.Random(seed).shuffle(shuffled)
        got = decide_leads(shuffled, rule_scores, industry, only_industry(settings), REFERENCE)
        assert [d.lead_id for d in got] == expected


# --- explanation ---------------------------------------------------------------------------

def test_explanation_lists_every_factor_with_its_points(settings):
    lead = make_lead(size=50, last=date(2024, 1, 15), industry="SaaS")
    lead.name, lead.company = "Alice Chen", "CloudScale AI"
    result = decide(lead, fit(tier="none", score=3, confidence="high"), settings)
    lines = result.explanation.splitlines()
    assert lines[0] == "Lead: Alice Chen, CloudScale AI (50 people, SaaS, Referral, 6 days ago)"
    assert lines[1] == "Score: 7.2/10  Decision: REVIEW"
    assert lines[2].startswith("  Industry fit (40%): 3/10 -> 1.20 pts") and '"Because."' in lines[2]
    assert lines[3].startswith("  Company size (25%): 10/10 -> 2.50 pts") and "(size detail)" in lines[3]
    assert lines[4].startswith("  Lead source  (20%): 10/10 -> 2.00 pts")
    assert lines[5].startswith("  Engagement   (15%): 10/10 -> 1.50 pts")
    assert 'Rule applied: industry tier "none" caps the decision at review' in lines[6]


def test_explanation_shows_priority_and_flags(settings):
    result = decide(make_lead(), fit(), settings)
    assert "Decision: QUALIFIED  Priority: HIGH" in result.explanation
    flagged = decide(make_lead(flags=["insufficient_data"]), fit(), settings)
    assert "Flags: insufficient_data" in flagged.explanation


def test_explanation_copes_with_missing_fields(settings):
    lead = Lead(lead_id="L001")
    result = decide(lead, fit(tier="unknown", score=5, confidence="low"), settings)
    first = result.explanation.splitlines()[0]
    assert first == "Lead: L001 (size not provided, industry not provided, source not provided)"


def test_explanation_points_always_add_up_to_the_score(settings):
    for size_score in (1, 4, 7, 10):
        result = decide(make_lead(), fit(score=9), settings, rules(size_score, 6, 4))
        listed = [float(line.split("->")[1].split("pts")[0]) for line in result.explanation.splitlines() if "->" in line]
        assert round(sum(listed), 2) == result.final_score


def test_add_flag_updates_the_flags_and_the_explanation(settings):
    result = decide(make_lead(), fit(), settings)
    assert "Flags:" not in result.explanation
    add_flag(result, "message_failed")
    assert result.flags == ["message_failed"] and result.explanation.endswith("  Flags: message_failed")
    add_flag(result, "message_failed")  # again: nothing changes
    add_flag(result, "llm_failed")
    assert result.flags == ["llm_failed", "message_failed"]
    assert result.explanation.count("Flags:") == 1 and result.explanation.endswith("Flags: llm_failed, message_failed")


# --- summary -------------------------------------------------------------------------------

def test_summary_counts_average_range_and_reasons(settings):
    leads, rule_scores, industry = ranked_batch(settings)
    decisions = decide_leads(leads, rule_scores, industry, only_industry(settings), REFERENCE)
    summary = summarize(decisions, only_industry(settings))
    assert summary.total == 6
    assert summary.counts == {QUALIFIED: 4, REVIEW: 1, REJECTED: 1}
    assert summary.average_score == round((9 * 4 + 6 + 2) / 6, 2)
    assert (summary.highest_score, summary.lowest_score) == (9, 2)
    assert summary.average_factor_scores["industry"] == round((9 * 4 + 6 + 2) / 6, 2)
    assert summary.rejection_reasons == [("Industry fit", 1)]
    assert round(summary.percent(QUALIFIED), 1) == 66.7


def test_summary_of_nothing_is_zero(settings):
    summary = summarize([], settings)
    assert summary.total == 0 and summary.average_score == 0 and summary.percent(QUALIFIED) == 0


# --- the provided data files ---------------------------------------------------------------

class DataFileClient(RulesClient):
    """RulesClient, but labels the way a sensible model would for two labels the config lacks."""

    def default(self, user_prompt):
        results = []
        for entry in json.loads(user_prompt.split("<leads>")[1].split("</leads>")[0]):
            label = entry["industry"]
            if label in ("Enterprise Software", "Technology"):
                results.append(item(entry["lead_id"], "none", 2, "high"))
            elif label in self.tiers["best"]["industries"]:
                results.append(item(entry["lead_id"], "best", 10, "high"))
            elif label in self.tiers["hard"]["industries"]:
                results.append(item(entry["lead_id"], "hard", 5, "high"))
            elif label in self.tiers["none"]["industries"]:
                results.append(item(entry["lead_id"], "none", 2, "high"))
            else:
                results.append(item(entry["lead_id"], "unknown", 5, "low"))
        return reply(*results)


def run_file(filename, config, scorers, settings, tmp_path):
    from pipeline.industry import IndustryScorer, IndustrySettings

    leads = ingest_csv(ROOT / "data" / filename, REFERENCE, config["missing_values"]).leads
    industry_settings = IndustrySettings.from_config(config).model_copy(update={"cache_path": tmp_path / "c.json"})
    industry = IndustryScorer(DataFileClient(config["tiers"]), industry_settings).score(leads)
    scores = score_leads(leads, scorers, REFERENCE)
    return leads, decide_leads(leads, scores, industry.results, settings, REFERENCE)


@pytest.mark.parametrize("filename, count", [("leads_full_scale_100.csv", 100), ("leads_training.csv", 29), ("leads_testing.csv", 20)])
def test_data_files_rank_every_lead_consistently(config, scorers, settings, tmp_path, filename, count):
    _, decisions = run_file(filename, config, scorers, settings, tmp_path)
    assert [d.overall_rank for d in decisions] == list(range(1, count + 1))
    scores = [d.final_score for d in decisions]
    assert scores == sorted(scores, reverse=True)
    for d in decisions:
        assert 1 <= d.final_score <= 10 and d.explanation
        assert round(sum(d.factor_points.values()), 2) == d.final_score
        if d.decision == QUALIFIED:
            assert d.final_score >= settings.qualified_at and d.tier not in ("none", "unknown") and d.confidence != "low"
        if d.decision == REJECTED:
            assert d.rejection_reason


def test_testing_file_matches_the_expected_results_in_the_workflow(config, scorers, settings, tmp_path):
    leads, decisions = run_file("leads_testing.csv", config, scorers, settings, tmp_path)
    by_name = {lead.name: d for lead, d in zip(sorted(leads, key=lambda x: x.lead_id), sorted(decisions, key=lambda x: x.lead_id))}
    expected = {
        "Test_MissingSize": (REVIEW, 5.55),
        "Test_VeryLargeEnterprise": (REJECTED, 4.55),
        "Test_Solo": (REJECTED, 3.15),
        "Test_StaleInteraction": (REVIEW, 8.25),
        "Test_RecentEngagement": (REVIEW, 6.8),
        "Test_UnknownIndustry": (REVIEW, 6.3),
        "Test_VeryRecentSignup": (REJECTED, 4.55),
        "Test_MultipleInteractions": (REVIEW, 6.8),
        "Test_NoSourceInfo": (QUALIFIED, 8.8),
        "Test_HighValueMatch": (REVIEW, 5.3),
        "Test_MissingName": (REVIEW, 6.4),
        "Test_PartialData": (REJECTED, 4.35),
        "Test_WeekOldInteraction": (QUALIFIED, 9.2),
        "Test_MidMarketSoft": (REJECTED, 4.3),
        "Test_StartupShutdown": (REJECTED, 3.0),
        "Test_GovernmentBuyer": (REJECTED, 4.9),
        "Test_SMBAverage": (QUALIFIED, 10),
    }
    for name, (decision, score) in expected.items():
        assert (by_name[name].decision, by_name[name].final_score) == (decision, score), name
    assert by_name["Test_NameOnly"].decision == REVIEW and "insufficient_data" in by_name["Test_NameOnly"].flags
    assert by_name["Test_MissingDate"].decision != QUALIFIED
    assert by_name["Test_LowValueNiche"].decision == REJECTED
    assert by_name["Test_NoSourceInfo"].priority == "HIGH" and by_name["Test_SMBAverage"].priority_rank == 1
    assert "no contact for 585 days" in by_name["Test_StaleInteraction"].explanation


# --- command line --------------------------------------------------------------------------

@pytest.fixture
def run_cli(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in ENV.items()))

    def run(client, filename="leads_full_scale_100.csv", *extra):
        monkeypatch.setattr(main, "create_llm_client", lambda llm_config, env: client)
        return main.main(
            ["--input", str(ROOT / "data" / filename), "--config", str(ROOT / "config.yaml"),
             "--env-file", str(env_file), "--score-industry", *extra]
        )

    return run


def test_cli_prints_the_ranking_best_fit_first(run_cli, config, capsys):
    client = RulesClient(config["tiers"])
    assert run_cli(client) == 0
    out = capsys.readouterr().out
    assert "Decisions: qualified" in out and "Weighted score: average" in out
    assert "Average factor scores: industry" in out and "Rejection reasons:" in out
    lines = [line for line in out.splitlines() if line.strip()[:1].isdigit()]
    rows = [line.split() for line in lines]
    assert len(rows) == 100 and [int(r[0]) for r in rows] == list(range(1, 101))
    scores = [float(re.search(r"\s(\d+(?:\.\d+)?)\s+(?:qualified|review|rejected)\b", line).group(1)) for line in lines]
    assert len(scores) == 100 and scores == sorted(scores, reverse=True)
    assert "HIGH #1" in out


def test_cli_calls_the_llm_in_batches_of_ten_by_default(run_cli, config):
    client = RulesClient(config["tiers"])
    assert run_cli(client) == 0
    sizes = [len(json.loads(p.split("<leads>")[1].split("</leads>")[0])) for p in client.prompts]
    assert sizes == [10] * 10


def test_cli_batch_size_follows_the_env_file(run_cli, config, tmp_path):
    (tmp_path / ".env").write_text((tmp_path / ".env").read_text() + "LLM_BATCH_SIZE=25\n")
    client = RulesClient(config["tiers"])
    assert run_cli(client) == 0
    sizes = [len(json.loads(p.split("<leads>")[1].split("</leads>")[0])) for p in client.prompts]
    assert sizes == [25, 25, 25, 25]


def test_cli_explain_prints_every_breakdown(run_cli, config, capsys):
    assert run_cli(RulesClient(config["tiers"]), "leads_testing.csv", "--explain") == 0
    out = capsys.readouterr().out
    assert out.count("Lead: ") == 20 and "Rule applied:" in out and "pts" in out


def test_cli_sends_failed_leads_to_review(run_cli, capsys):
    assert run_cli(ScriptedClient([LLMError("down")] * 10), "leads_training.csv") == 0
    out = capsys.readouterr().out
    assert "fell back to review" in out and "rejected 0" in out and "review 29" in out
