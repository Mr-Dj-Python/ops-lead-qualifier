import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import main
from pipeline.config import ConfigError, load_config
from pipeline.llm_client import LLMError
from pipeline.models import GeneratedMessage
from pipeline.report import (
    Report,
    ReportSettings,
    ValidationCheck,
    ValidationSummary,
    build_report,
    file_sha256,
    format_summary,
    git_commit,
    text_sha256,
    validate_workflow,
    write_report,
)
from tests.test_messages import EverythingClient

ROOT = Path(__file__).resolve().parent.parent
ENV = {"WEIGHT_INDUSTRY": "0.4", "WEIGHT_SIZE": "0.25", "WEIGHT_SOURCE": "0.2", "WEIGHT_RECENCY": "0.15",
       "SENDER_NAME": "Dana, OpsFlow", "LLM_API_KEY": "sk-test-secret-123456"}
CHECKS = [
    "input_rows_accounted_for", "every_lead_scored_by_every_rule", "every_lead_has_an_industry_answer",
    "every_lead_decided_exactly_once", "weights_add_up_to_one", "scores_are_in_range",
    "points_add_up_to_final_scores", "ranking_is_ordered", "priority_ranks_are_contiguous",
    "decisions_follow_the_rules", "decision_counts_add_up", "llm_calls_are_batched",
    "messages_cover_qualified_leads", "messages_pass_the_rules_again", "every_lead_is_traceable",
    "failures_are_all_reported", "no_secrets_in_report", "run_log_matches_the_run",
]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config.yaml", env=ENV)


@pytest.fixture
def run(tmp_path, monkeypatch, config):
    """Run the CLI in a temp folder against a fake model; returns what it produced."""
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in ENV.items()))
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    captured = {}
    real_build = main.build_report

    def spy(outcome, log_text=None):
        captured["outcome"], captured["log_text"] = outcome, log_text
        return real_build(outcome, log_text)

    monkeypatch.setattr(main, "build_report", spy)

    def go(client=None, *extra, filename="leads_full_scale_100.csv"):
        client = client or EverythingClient(config["tiers"])
        monkeypatch.setattr(main, "create_llm_client", lambda llm_config, env: client)
        code = main.main(
            ["--input", str(ROOT / "data" / filename), "--config", str(ROOT / "config.yaml"),
             "--env-file", str(env_file), "--score-industry", "--messages", *extra]
        )
        report_path = tmp_path / "output_report.json"
        return SimpleNamespace(
            code=code, client=client, dir=tmp_path, outcome=captured.get("outcome"), log_text=captured.get("log_text"),
            report=json.loads(report_path.read_text()) if report_path.exists() else None,
        )

    return go


def tampered(result):
    return result.outcome.model_copy(deep=True)


def failed_names(outcome, log_text):
    return {check.name for check in build_report(outcome, log_text).validation.failed}


# --- the report file -------------------------------------------------------------------------

def test_the_report_has_every_section(run):
    result = run()
    assert result.code == 0
    assert list(result.report) == ["run_info", "stats", "leads", "priority_queue", "sample_messages", "errors", "telemetry", "validation"]


def test_the_report_file_loads_back_into_the_report_model(run):
    result = run()
    loaded = Report.model_validate_json((result.dir / "output_report.json").read_text())
    assert len(loaded.leads) == 100 and loaded.validation.passed


def test_run_info_identifies_the_run_and_its_input(run):
    result = run()
    info = result.report["run_info"]
    data = ROOT / "data" / "leads_full_scale_100.csv"
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", info["run_id"])
    assert info["input"] == {"path": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                             "rows_read": 100, "leads_kept": 100, "rows_dropped": 0}
    assert info["reference_date"] == "2024-01-21" and info["provider"] == "openai"
    assert info["industry_batch_size"] == 10 and info["message_batch_size"] == 10
    assert info["schema_version"] == 1 and info["strict"] is False and info["log_file"] == "run.log"
    assert info["finished_at"] >= info["started_at"]


def test_run_info_carries_the_rubric_as_it_ran(run):
    rubric = run().report["run_info"]["rubric"]
    assert rubric["weights"] == {"industry": 0.4, "size": 0.25, "source": 0.2, "recency": 0.15}
    assert (rubric["qualified_at"], rubric["review_at"], rubric["high_priority_at"]) == (7.0, 5.0, 8.5)
    assert (rubric["stale_days"], rubric["reject_size_below"], rubric["reject_size_above"]) == (180, 10, 5000)
    assert rubric["tier_score_ranges"]["best"] == [9, 10]


def test_fingerprints_tie_the_report_to_the_config_and_prompts(run, config):
    result = run()
    prints = result.report["run_info"]["fingerprints"]
    assert prints["config_sha256"] == hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()
    assert prints["industry_prompt_sha256"] == text_sha256(result.outcome.industry_settings.system_prompt)
    assert prints["message_prompt_sha256"] == text_sha256(result.outcome.message_settings.system_prompt)
    assert prints["industry_prompt_sha256"] != prints["message_prompt_sha256"]


def test_stats_match_the_ranked_table(run):
    stats = run().report["stats"]
    assert (stats["total"], stats["processed"]) == (100, 100)
    assert (stats["qualified"], stats["review"], stats["rejected"]) == (22, 58, 20)
    assert (stats["qualified_pct"], stats["review_pct"], stats["rejected_pct"]) == (22.0, 58.0, 20.0)
    assert (stats["average_score"], stats["highest_score"], stats["lowest_score"]) == (6.15, 10.0, 2.55)
    assert stats["top_rejection_reasons"][0] == {"reason": "Industry fit", "count": 16}
    assert sum(stats["by_industry_tier"].values()) == 100
    assert stats["messages_written"] == 22 and stats["messages_failed"] == 0 and sum(stats["by_variant"].values()) == 22
    assert stats["failed_leads"] == 0 and stats["api_retries"] == 0
    assert stats["est_manual_minutes_saved"] == pytest.approx(1000, abs=1)  # 100 leads x 10 minutes


def test_every_lead_is_in_the_report_best_fit_first(run):
    leads = run().report["leads"]
    assert [lead["rank"] for lead in leads] == list(range(1, 101))
    scores = [lead["final_score"] for lead in leads]
    assert scores == sorted(scores, reverse=True)
    top = leads[0]
    assert top["decision"] == "qualified" and top["priority_rank"] == 1 and "Industry fit" in top["explanation"]
    assert top["message"]["body"].startswith("Hi ") and top["industry_reason"]


def test_the_priority_queue_lists_the_qualified_leads_in_order(run):
    report = run().report
    qualified = [lead["lead_id"] for lead in report["leads"] if lead["decision"] == "qualified"]
    assert report["priority_queue"] == qualified and len(qualified) == 22


def test_sample_messages_cover_each_variant_and_stay_within_the_limit(run):
    samples = run().report["sample_messages"]
    assert 3 <= len(samples) <= 5
    assert {"warm_followup", "value_led", "reengagement"} <= {s["variant"] for s in samples}
    assert [s["priority_rank"] for s in samples] == sorted(s["priority_rank"] for s in samples)
    assert all(40 <= s["word_count"] <= 100 and s["body"].endswith("Dana, OpsFlow") for s in samples)


# --- traceability ----------------------------------------------------------------------------

def test_every_lead_traces_back_to_its_csv_row(run):
    result = run()
    rows = (ROOT / "data" / "leads_full_scale_100.csv").read_text().splitlines()
    for lead in result.report["leads"]:
        row = lead["trace"]["source_row"]
        assert lead["company"] in rows[row - 1]  # that line of the CSV really is this lead


def test_the_trace_says_where_the_industry_answer_came_from(run):
    leads = run().report["leads"]
    for lead in leads:
        trace = lead["trace"]["industry"]
        assert trace == {"source": "llm", "request_id": lead["lead_id"], "batch": trace["batch"], "attempts": 1}
        assert 1 <= trace["batch"] <= 10


def test_the_trace_says_which_message_batch_wrote_each_email(run):
    leads = run().report["leads"]
    traced = [lead["trace"]["message"] for lead in leads if lead["decision"] == "qualified"]
    assert all(t == {"status": "written", "batch": t["batch"], "attempts": 1} for t in traced)
    assert sorted({t["batch"] for t in traced}) == [1, 2, 3]
    assert all(lead["trace"]["message"] is None for lead in leads if lead["decision"] != "qualified")


def test_a_second_run_shows_the_cache_in_the_trace(run):
    run()
    second = run()  # the first run left .cache/industry.json in the temp folder
    assert all(lead["trace"]["industry"]["source"] == "cache" and lead["trace"]["industry"]["batch"] is None
               for lead in second.report["leads"])
    assert second.report["telemetry"]["cache_hits"] == 100 and second.report["telemetry"]["industry"]["api_calls"] == 0


# --- telemetry -------------------------------------------------------------------------------

def test_telemetry_shows_batching_and_llm_use(run):
    tel = run().report["telemetry"]
    assert tel["industry"]["batch_sizes"] == [10] * 10 and tel["industry"]["api_calls"] == 10
    assert tel["messages"]["batch_sizes"] == [10, 10, 2] and tel["messages"]["api_calls"] == 3
    assert tel["industry"]["usage"]["requests"] == 10 and tel["messages"]["usage"]["requests"] == 3
    total = tel["llm_total"]
    assert total["requests"] == 13 and total["retries"] == 0 and total["failed_calls"] == 0
    assert [s["name"] for s in tel["stages"]] == ["ingest", "rules", "industry", "decide", "messages"]
    assert tel["stages"][0]["items"] == 100 and tel["stages"][-1]["items"] == 22
    assert tel["log"] == {"warnings": 0, "errors": 0} and tel["log_file"] == "run.log"


def test_a_failing_model_shows_up_in_telemetry_errors_and_the_trace(run, config):
    client = EverythingClient(config["tiers"], [LLMError("provider is down")] * 2)
    result = run(client)
    assert result.code == 0 and result.report["validation"]["passed"]  # the run is still consistent
    report = result.report
    kinds = [e["kind"] for e in report["errors"]]
    assert kinds.count("llm_call_failed") == 2 and kinds.count("fell_back_to_review") == 20
    assert report["stats"]["failed_leads"] == 20 and report["telemetry"]["llm_total"]["failed_calls"] == 2
    fallbacks = [lead for lead in report["leads"] if lead["trace"]["industry"]["source"] == "fallback"]
    assert len(fallbacks) == 20 and all(lead["decision"] == "review" and "llm_failed" in lead["flags"] for lead in fallbacks)
    assert report["telemetry"]["log"]["warnings"] >= 20


def test_failed_messages_are_reported_and_flagged(run, config):
    client = EverythingClient(config["tiers"])
    original = client.default
    client.default = lambda prompt: "{}" if prompt.startswith("Write one email") else original(prompt)
    result = run(client)
    report = result.report
    assert result.code == 0 and report["validation"]["passed"]
    assert report["stats"]["messages_written"] == 0 and report["stats"]["messages_failed"] == 22
    assert [e["kind"] for e in report["errors"]].count("message_failed") == 22
    qualified = [lead for lead in report["leads"] if lead["decision"] == "qualified"]
    assert all("message_failed" in lead["flags"] and lead["trace"]["message"]["status"] == "failed" and lead["message"] is None for lead in qualified)
    assert report["sample_messages"] == []


def test_dropped_rows_and_unparsed_values_are_reported(run, tmp_path):
    messy = ROOT / "data" / "leads_testing.csv"
    result = run(None, filename="leads_testing.csv")
    assert result.code == 0 and result.report["run_info"]["input"]["path"] == str(messy)
    assert result.report["validation"]["passed"]
    insufficient = [lead for lead in result.report["leads"] if "insufficient_data" in lead["flags"]]
    assert len(insufficient) == 1 and insufficient[0]["decision"] == "review"


# --- the workflow checks ---------------------------------------------------------------------

def test_a_good_run_passes_every_check(run):
    result = run()
    validation = result.report["validation"]
    assert validation["passed"] is True
    assert [c["name"] for c in validation["checks"]] == CHECKS
    assert all(c["passed"] and c["detail"] for c in validation["checks"])


def break_rows(o):
    o.ingest.rows_read += 1


def drop_a_rule_score(o):
    del o.rule_scores[o.ingest.leads[0].lead_id]["size"]


def drop_an_industry_answer(o):
    del o.industry.results[o.ingest.leads[0].lead_id]


def duplicate_a_decision(o):
    o.decisions.append(o.decisions[0])


def change_the_weights(o):
    o.decision_settings = o.decision_settings.model_copy(update={"weights": {"industry": 0.5, "size": 0.25, "source": 0.2, "recency": 0.15}})


def score_out_of_range(o):
    o.decisions[0].__dict__["final_score"] = 11.0


def points_off(o):
    o.decisions[0].__dict__["factor_points"] = {**o.decisions[0].factor_points, "industry": 0.0}


def swap_ranks(o):
    o.decisions[0], o.decisions[1] = o.decisions[1], o.decisions[0]


def wrong_priority_rank(o):
    o.decisions[0].__dict__["priority_rank"] = 5


def qualify_below_threshold(o):
    o.decisions[0].__dict__["final_score"] = o.decision_settings.qualified_at - 1


def reject_a_good_score(o):
    lead = next(d for d in o.decisions if d.decision == "review")
    lead.__dict__.update(decision="rejected", rejection_reason="x", rules_applied=[])


def wrong_counts(o):
    o.summary.counts["qualified"] += 1


def oversized_batch(o):
    o.industry.batch_sizes[0] = o.industry_settings.batch_size + 5


def too_many_calls(o):
    o.industry.api_calls = 500


def lose_a_message(o):
    del o.messages.messages[next(iter(o.messages.messages))]


def message_for_unqualified(o):
    other = next(d for d in o.decisions if d.decision != "qualified")
    sample = next(iter(o.messages.messages.values()))
    o.messages.messages[other.lead_id] = sample.model_copy(update={"lead_id": other.lead_id})


def bad_message_text(o):
    lead_id = next(iter(o.messages.messages))
    o.messages.messages[lead_id] = o.messages.messages[lead_id].model_copy(update={"body": "Hi there! Buy now!"})


def lose_the_csv_row(o):
    o.ingest.leads[0].source_row = None


def lose_industry_provenance(o):
    del o.industry.provenance[o.ingest.leads[0].lead_id]


def unreported_fallback(o):
    o.industry.failed_ids.append(o.ingest.leads[0].lead_id)


def unreported_message_failure(o):
    o.messages.failed_ids.append("L999")


@pytest.mark.parametrize(
    "tamper, check",
    [
        (break_rows, "input_rows_accounted_for"),
        (drop_a_rule_score, "every_lead_scored_by_every_rule"),
        (drop_an_industry_answer, "every_lead_has_an_industry_answer"),
        (duplicate_a_decision, "every_lead_decided_exactly_once"),
        (change_the_weights, "weights_add_up_to_one"),
        (score_out_of_range, "scores_are_in_range"),
        (points_off, "points_add_up_to_final_scores"),
        (swap_ranks, "ranking_is_ordered"),
        (wrong_priority_rank, "priority_ranks_are_contiguous"),
        (qualify_below_threshold, "decisions_follow_the_rules"),
        (reject_a_good_score, "decisions_follow_the_rules"),
        (wrong_counts, "decision_counts_add_up"),
        (oversized_batch, "llm_calls_are_batched"),
        (too_many_calls, "llm_calls_are_batched"),
        (lose_a_message, "messages_cover_qualified_leads"),
        (message_for_unqualified, "messages_cover_qualified_leads"),
        (bad_message_text, "messages_pass_the_rules_again"),
        (lose_the_csv_row, "every_lead_is_traceable"),
        (lose_industry_provenance, "every_lead_is_traceable"),
        (unreported_fallback, "failures_are_all_reported"),
        (unreported_message_failure, "failures_are_all_reported"),
    ],
)
def test_each_check_catches_the_damage_it_is_for(run, tamper, check):
    result = run()
    outcome = tampered(result)
    tamper(outcome)
    report = build_report(outcome, result.log_text)
    assert check in {c.name for c in report.validation.failed}, report.validation.model_dump()
    assert report.validation.passed is False


def test_a_secret_in_the_report_is_caught(run):
    result = run()
    outcome = tampered(result)
    outcome.secrets = ["Dana, OpsFlow"]  # text that does appear in every message
    assert "no_secrets_in_report" in failed_names(outcome, result.log_text)
    outcome.secrets = ["sk-not-in-the-report"]
    assert "no_secrets_in_report" not in failed_names(outcome, result.log_text)


def test_the_real_api_key_never_reaches_the_report_or_the_log(run):
    result = run()
    assert "sk-test-secret-123456" not in (result.dir / "output_report.json").read_text()
    assert "sk-test-secret-123456" not in (result.dir / "run.log").read_text()
    assert result.outcome.secrets == ["sk-test-secret-123456"]
    assert "secrets" not in result.outcome.model_dump()


def test_the_run_log_check_needs_the_run_id_and_every_stage(run):
    result = run()
    outcome = result.outcome
    assert "run_log_matches_the_run" not in failed_names(outcome, result.log_text)
    assert "run_log_matches_the_run" in failed_names(outcome, "nothing useful in here")
    assert "run_log_matches_the_run" in failed_names(outcome, None)
    no_decide = result.log_text.replace("stage decide finished", "stage decide ")
    assert "run_log_matches_the_run" in failed_names(outcome, no_decide)


def test_no_log_check_when_there_is_no_log_file(run):
    outcome = tampered(run())
    outcome.log_path = None
    names = [c.name for c in build_report(outcome, None).validation.checks]
    assert "run_log_matches_the_run" not in names and len(names) == len(CHECKS) - 1


def test_validate_workflow_can_be_called_on_its_own(run):
    result = run()
    report = build_report(result.outcome, result.log_text)
    again = validate_workflow(result.outcome, report, result.log_text)
    assert again.passed and [c.name for c in again.checks] == CHECKS


def test_validation_summary_lists_the_failed_checks():
    summary = ValidationSummary(passed=False, checks=[
        ValidationCheck(name="a", passed=True, detail="ok"), ValidationCheck(name="b", passed=False, detail="bad")])
    assert [c.name for c in summary.failed] == ["b"]


# --- the CLI: files, exit code, summary ------------------------------------------------------

def test_the_cli_writes_the_report_and_the_log_and_prints_a_summary(run, capsys):
    result = run()
    out = capsys.readouterr().out
    assert (result.dir / "output_report.json").exists() and (result.dir / "run.log").exists()
    summary = out[out.index("Run "):]
    for fragment in ("report:     output_report.json", "log:        run.log", "leads:      100 kept, 0 dropped",
                     "qualified 22 (22%)", "messages:   22 written", "13 calls (industry 10, messages 3)",
                     "stages:     ingest", "time saved:", "validation: PASSED (18 checks)"):
        assert fragment in summary


def test_the_run_log_tells_the_story_of_the_run(run):
    result = run()
    log = (result.dir / "run.log").read_text()
    run_id = result.report["run_info"]["run_id"]
    assert all(f"run={run_id}" in line for line in log.splitlines())
    for fragment in ("started: input=", "stage ingest started", "ingest: 100 row(s) read", "industry batch 10/10",
                     "stage decide finished", "message batch 3/3", "messages done: 22 written", "report written to",
                     "validation passed"):
        assert fragment in log


def test_output_and_log_file_options_override_the_config(run, tmp_path):
    result = run(None, "--output", str(tmp_path / "out" / "r.json"), "--log-file", str(tmp_path / "logs" / "x.log"))
    assert (tmp_path / "out" / "r.json").exists() and (tmp_path / "logs" / "x.log").exists()
    assert not (tmp_path / "output_report.json").exists() and not (tmp_path / "run.log").exists()
    assert result.code == 0


def test_runs_append_to_the_same_log_with_their_own_ids(run):
    run()
    run()
    log = (Path.cwd() / "run.log").read_text()
    ids = set(re.findall(r"run=(\S+)", log))
    assert len(ids) == 2 and log.count("started: input=") == 2


def test_a_failed_workflow_check_makes_the_run_exit_1(run, monkeypatch, capsys):
    real = main.build_report

    def broken(outcome, log_text=None):
        report = real(outcome, log_text)
        report.validation = ValidationSummary(passed=False, checks=[ValidationCheck(name="demo_check", passed=False, detail="boom")])
        return report

    monkeypatch.setattr(main, "build_report", broken)
    result = run()
    out = capsys.readouterr().out
    assert result.code == 1 and "validation: FAILED (1 of 1 checks)" in out and "demo_check: boom" in out
    assert result.report is not None  # the report is still written so it can be inspected
    assert "workflow validation failed: demo_check" in (result.dir / "run.log").read_text()


def test_a_bad_api_key_is_logged_and_exits_1_without_a_report(run, config, capsys):
    from pipeline.llm_client import LLMAuthError

    client = EverythingClient(config["tiers"], [LLMAuthError("OpenAI rejected the API key (401)")])
    result = run(client)
    assert result.code == 1 and result.report is None
    assert "error: OpenAI rejected the API key" in capsys.readouterr().err
    assert "run failed: OpenAI rejected the API key" in (result.dir / "run.log").read_text()


def test_no_files_are_written_without_score_industry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main.main(["--input", str(ROOT / "data" / "leads_training.csv"), "--config", str(ROOT / "config.yaml")]) == 0
    assert list(tmp_path.iterdir()) == []


# --- settings and helpers --------------------------------------------------------------------

def test_report_settings_come_from_config(config):
    settings = ReportSettings.from_config(config)
    assert (settings.output_path, settings.log_path, settings.log_level) == (Path("output_report.json"), Path("run.log"), "INFO")
    assert (settings.manual_minutes_per_lead, settings.sample_messages) == (10, 5)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("output_path",), "", "report.output_path"),
        (("log_path",), None, "report.log_path"),
        (("log_level",), "LOUD", "report.log_level"),
        (("manual_minutes_per_lead",), 0, "report.manual_minutes_per_lead"),
        (("manual_minutes_per_lead",), "ten", "report.manual_minutes_per_lead"),
        (("sample_messages",), 0, "report.sample_messages"),
        (("sample_messages",), 11, "report.sample_messages"),
    ],
)
def test_invalid_report_settings_name_the_config_path(config, path, value, message):
    broken = json.loads(json.dumps(config))
    broken["report"][path[0]] = value
    with pytest.raises(ConfigError, match=message):
        ReportSettings.from_config(broken)


def test_report_section_is_required(config):
    with pytest.raises(ConfigError, match="'report'"):
        ReportSettings.from_config({k: v for k, v in config.items() if k != "report"})


def test_report_settings_are_frozen(config):
    with pytest.raises(ValidationError):
        ReportSettings.from_config(config).sample_messages = 3


def test_file_and_text_hashes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello")
    assert file_sha256(path) == hashlib.sha256(b"hello").hexdigest() == text_sha256("hello")


def test_git_commit_is_a_short_hash_or_none():
    commit = git_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{4,40}", commit)


def test_write_report_replaces_the_file_in_one_step_and_makes_folders(run, tmp_path):
    result = run()
    report = Report.model_validate(result.report)
    target = tmp_path / "deep" / "dir" / "report.json"
    write_report(report, target)
    write_report(report, target)  # overwrite
    assert json.loads(target.read_text())["run_info"]["run_id"] == result.report["run_info"]["run_id"]
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]  # no temp files left


def test_format_summary_reports_failed_checks(run):
    result = run()
    report = build_report(result.outcome, result.log_text)
    report.validation = ValidationSummary(passed=False, checks=[ValidationCheck(name="x", passed=False, detail="why")])
    text = format_summary(report, Path("r.json"), None)
    assert "log:        (none)" in text and "validation: FAILED (1 of 1 checks)" in text and "- x: why" in text
