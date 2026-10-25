import logging
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline.llm_client import (
    LLMAPIClient,
    LLMAuthError,
    LLMError,
    LLMTransientError,
    LLMUsage,
    OpenAIClient,
    RetryPolicy,
)
from pipeline.telemetry import LogCounts, RunRecorder, StageTiming

PACKAGE = logging.getLogger("pipeline")


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def recorder(clock=None, run_id=None):
    clock = clock or FakeClock()
    return RunRecorder(now=lambda: datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc), clock=clock, run_id=run_id), clock


# --- run id and stage timing -----------------------------------------------------------------

def test_run_id_has_a_timestamp_and_a_random_part_and_differs_between_runs():
    first, _ = recorder()
    second, _ = recorder()
    assert re.fullmatch(r"20261025T120000Z-[0-9a-f]{6}", first.run_id)
    assert first.run_id != second.run_id
    assert recorder(run_id="fixed")[0].run_id == "fixed"


def test_stages_are_timed_in_order_with_item_counts():
    run, clock = recorder()
    with run.stage("ingest") as note:
        clock.advance(1.5)
        note.items = 100
    with run.stage("rules"):
        clock.advance(0.25)
    assert run.stages == [StageTiming(name="ingest", duration_s=1.5, items=100), StageTiming(name="rules", duration_s=0.25, items=None)]
    clock.advance(2)
    assert run.duration_s == 3.75


def test_a_stage_that_fails_is_still_recorded_and_the_error_propagates():
    run, clock = recorder()
    with pytest.raises(RuntimeError):
        with run.stage("industry"):
            clock.advance(0.5)
            raise RuntimeError("boom")
    assert run.stages == [StageTiming(name="industry", duration_s=0.5, items=None)]


def test_finish_stamps_the_end_time():
    run, _ = recorder()
    assert run.finished_at is None
    run.finish()
    assert run.finished_at == datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)


def test_stage_timing_is_validated():
    with pytest.raises(ValidationError):
        StageTiming(name="", duration_s=1)
    with pytest.raises(ValidationError):
        StageTiming(name="x", duration_s=-1)


# --- run.log ---------------------------------------------------------------------------------

def test_every_log_line_carries_the_run_id(tmp_path):
    path = tmp_path / "run.log"
    run, _ = recorder(run_id="RUN-1")
    with run.logging_to(path):
        with run.stage("ingest") as note:
            note.items = 3
            logging.getLogger("pipeline.ingest").info("hello from ingest")
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and all("run=RUN-1" in line for line in lines)
    assert "stage ingest started" in lines[0]
    assert "pipeline.ingest: hello from ingest" in lines[1]
    assert "stage ingest finished in" in lines[2] and "items=3" in lines[2]
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO ", lines[0])


def test_the_log_is_appended_so_several_runs_keep_their_history(tmp_path):
    path = tmp_path / "logs" / "run.log"  # the folder is created
    for run_id in ("A", "B"):
        run, _ = recorder(run_id=run_id)
        with run.logging_to(path):
            logging.getLogger("pipeline.x").info("a line")
    text = path.read_text()
    assert text.count("run=A") == 1 and text.count("run=B") == 1


def test_warnings_and_errors_are_counted(tmp_path):
    run, _ = recorder()
    with run.logging_to(tmp_path / "run.log"):
        logger = logging.getLogger("pipeline.test")
        logger.info("fine")
        logger.warning("careful")
        logger.warning("careful again")
        logger.error("broken")
    assert run.log_counts == LogCounts(warnings=2, errors=1)


def test_counting_works_without_a_log_file():
    run, _ = recorder()
    with run.logging_to(None):
        logging.getLogger("pipeline.test").warning("careful")
    assert run.log_counts == LogCounts(warnings=1, errors=0) and run.log_path is None


def test_handlers_and_level_are_restored_afterwards(tmp_path):
    handlers_before, level_before = list(PACKAGE.handlers), PACKAGE.level
    run, _ = recorder()
    with run.logging_to(tmp_path / "run.log", "DEBUG"):
        assert len(PACKAGE.handlers) == len(handlers_before) + 2
        assert PACKAGE.level == logging.DEBUG
    assert PACKAGE.handlers == handlers_before and PACKAGE.level == level_before


def test_handlers_are_restored_even_when_the_run_fails(tmp_path):
    handlers_before = list(PACKAGE.handlers)
    run, _ = recorder()
    with pytest.raises(RuntimeError):
        with run.logging_to(tmp_path / "run.log"):
            raise RuntimeError("boom")
    assert PACKAGE.handlers == handlers_before


def test_the_log_level_filters_the_file(tmp_path):
    path = tmp_path / "run.log"
    run, _ = recorder()
    with run.logging_to(path, "WARNING"):
        logging.getLogger("pipeline.test").info("quiet")
        logging.getLogger("pipeline.test").warning("loud")
    assert "quiet" not in path.read_text() and "loud" in path.read_text()


def test_debug_lines_appear_only_at_debug_level(tmp_path):
    for level, expected in (("INFO", False), ("DEBUG", True)):
        path = tmp_path / f"{level}.log"
        run, _ = recorder()
        with run.logging_to(path, level):
            logging.getLogger("pipeline.test").debug("fine detail")
        assert ("fine detail" in path.read_text()) is expected


def test_messages_from_other_libraries_are_not_in_the_run_log(tmp_path):
    path = tmp_path / "run.log"
    run, _ = recorder()
    with run.logging_to(path):
        logging.getLogger("some.other.library").warning("not ours")
        logging.getLogger("pipeline.ours").info("ours")
    assert "not ours" not in path.read_text() and "ours" in path.read_text()


# --- LLM usage telemetry in the client -------------------------------------------------------

class ScriptedClient(LLMAPIClient):
    def __init__(self, script):
        super().__init__(RetryPolicy(max_attempts=3, base_delay_s=0, max_delay_s=0), lambda s: None)
        self.script = list(script)

    @classmethod
    def from_config(cls, llm_config, api_key, retry_policy, sleep):
        raise NotImplementedError

    @property
    def model(self):
        return "scripted"

    def _request(self, system_prompt, user_prompt, temperature, max_tokens):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ask(client):
    return client.complete("s", "u", temperature=0, max_tokens=1)


def test_a_clean_call_counts_one_request():
    client = ScriptedClient(["{}"])
    ask(client)
    assert (client.usage.requests, client.usage.transient_failures, client.usage.retries, client.usage.failed_calls) == (1, 0, 0, 0)
    assert client.usage.latency_s >= 0


def test_retried_calls_count_failures_and_retries():
    client = ScriptedClient([LLMTransientError("429"), LLMTransientError("503"), "{}"])
    ask(client)
    assert (client.usage.requests, client.usage.transient_failures, client.usage.retries, client.usage.failed_calls) == (1, 2, 2, 0)


def test_a_call_given_up_on_counts_as_failed():
    client = ScriptedClient([LLMTransientError("a"), LLMTransientError("b"), LLMTransientError("c")])
    with pytest.raises(LLMTransientError):
        ask(client)
    assert (client.usage.requests, client.usage.transient_failures, client.usage.retries, client.usage.failed_calls) == (0, 3, 2, 1)


@pytest.mark.parametrize("error", [LLMAuthError("bad key"), LLMError("bad request")])
def test_calls_that_cannot_be_retried_count_as_failed(error):
    client = ScriptedClient([error])
    with pytest.raises(LLMError):
        ask(client)
    assert (client.usage.failed_calls, client.usage.retries) == (1, 0)


def test_usage_difference_gives_the_cost_of_one_stage():
    client = ScriptedClient(["{}", LLMTransientError("429"), "{}", "{}"])
    ask(client)
    before = client.usage.model_copy()
    ask(client)
    ask(client)
    spent = client.usage.minus(before)
    assert (spent.requests, spent.transient_failures, spent.retries) == (2, 1, 1)
    assert client.usage.requests == 3  # the client's own totals are untouched


def test_usage_is_validated():
    with pytest.raises(ValidationError):
        LLMUsage(requests=-1)
    with pytest.raises(ValidationError):
        LLMUsage(latency_s=-0.1)


def test_openai_tokens_are_added_up():
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    client = OpenAIClient(api_key="x", model="m", sdk_client=sdk, retry_policy=RetryPolicy(max_attempts=1))
    client.complete("s", "u", temperature=0, max_tokens=1)
    client.complete("s", "u", temperature=0, max_tokens=1)
    assert (client.usage.prompt_tokens, client.usage.completion_tokens, client.usage.requests) == (240, 60, 2)


def test_a_reply_without_usage_information_adds_no_tokens():
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    client = OpenAIClient(api_key="x", model="m", sdk_client=sdk)
    client.complete("s", "u", temperature=0, max_tokens=1)
    assert (client.usage.prompt_tokens, client.usage.completion_tokens) == (0, 0)
