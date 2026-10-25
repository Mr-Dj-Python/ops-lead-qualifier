from types import SimpleNamespace

import httpx2
import openai
import pytest

from pipeline import llm_client
from pipeline.config import ConfigError
from pipeline.llm_client import (
    LLMAPIClient,
    LLMAuthError,
    LLMError,
    LLMTransientError,
    OpenAIClient,
    RetryPolicy,
    create_llm_client,
    register_provider,
)

LLM_CONFIG = {
    "provider": "openai",
    "model": "test-model",
    "api_key_env": "LLM_API_KEY",
    "timeout_s": 5,
    "retries": {"max_attempts": 3, "base_delay_s": 1, "max_delay_s": 10},
}


class ScriptedClient(LLMAPIClient):
    """Returns or raises the next item of `script` on each call."""

    def __init__(self, script, policy=None):
        self.sleeps = []
        super().__init__(policy or RetryPolicy(max_attempts=4, base_delay_s=2, max_delay_s=30), self.sleeps.append)
        self.script = list(script)
        self.calls = 0

    @classmethod
    def from_config(cls, llm_config, api_key, retry_policy, sleep):
        raise NotImplementedError

    @property
    def model(self):
        return "scripted"

    def _request(self, system_prompt, user_prompt, temperature, max_tokens):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ask(client):
    return client.complete("system", "user", temperature=0, max_tokens=10)


# --- retry and backoff (shared by every provider) ------------------------------------------

def test_success_makes_one_call_and_does_not_sleep():
    client = ScriptedClient(["{}"])
    assert ask(client) == "{}"
    assert client.calls == 1 and client.sleeps == []


def test_transient_errors_are_retried_with_growing_delays():
    client = ScriptedClient([LLMTransientError("429"), LLMTransientError("503"), "ok"])
    assert ask(client) == "ok"
    assert client.calls == 3
    assert 2 <= client.sleeps[0] < 4  # base 2 plus jitter below 2
    assert 4 <= client.sleeps[1] < 6  # doubled


def test_gives_up_after_max_attempts_and_raises_the_last_error():
    client = ScriptedClient([LLMTransientError(f"fail {n}") for n in range(4)])
    with pytest.raises(LLMTransientError, match="fail 3"):
        ask(client)
    assert client.calls == 4 and len(client.sleeps) == 3


def test_delay_is_capped_at_max_delay():
    policy = RetryPolicy(max_attempts=5, base_delay_s=10, max_delay_s=15)
    client = ScriptedClient([LLMTransientError("x")] * 4 + ["ok"], policy)
    ask(client)
    assert max(client.sleeps) == 15


def test_retry_after_from_the_provider_is_honored():
    client = ScriptedClient([LLMTransientError("429", retry_after=20), "ok"])
    ask(client)
    assert client.sleeps == [20]


def test_auth_errors_are_never_retried():
    client = ScriptedClient([LLMAuthError("bad key"), "ok"])
    with pytest.raises(LLMAuthError):
        ask(client)
    assert client.calls == 1 and client.sleeps == []


def test_other_errors_are_not_retried():
    client = ScriptedClient([LLMError("bad request"), "ok"])
    with pytest.raises(LLMError, match="bad request"):
        ask(client)
    assert client.calls == 1


def test_retry_policy_rejects_nonsense():
    with pytest.raises(ConfigError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ConfigError):
        RetryPolicy(base_delay_s=-1)


# --- OpenAI client -------------------------------------------------------------------------

def http_error(cls, status, headers=None):
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("boom", response=response, body=None)


def fake_sdk(reply=None, error=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return reply

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return sdk, calls


def openai_client(sdk):
    sleeps = []
    client = OpenAIClient(
        api_key="unused", model="test-model", sdk_client=sdk,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=1, max_delay_s=5), sleep=sleeps.append,
    )
    return client, sleeps


def reply_with(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_openai_request_shape():
    sdk, calls = fake_sdk(reply_with('{"results": []}'))
    client, _ = openai_client(sdk)
    assert client.complete("SYS", "USR", temperature=0, max_tokens=123) == '{"results": []}'
    assert calls == [
        {
            "model": "test-model",
            "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}],
            "temperature": 0,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]
    assert client.model == "test-model"


def test_openai_empty_content_returns_empty_text():
    client, _ = openai_client(fake_sdk(reply_with(None))[0])
    assert client.complete("s", "u", temperature=0, max_tokens=1) == ""


def test_openai_no_choices_is_transient():
    client, _ = openai_client(fake_sdk(SimpleNamespace(choices=[]))[0])
    with pytest.raises(LLMTransientError, match="no choices"):
        client.complete("s", "u", temperature=0, max_tokens=1)


@pytest.mark.parametrize(
    "error, expected",
    [
        (http_error(openai.AuthenticationError, 401), LLMAuthError),
        (http_error(openai.PermissionDeniedError, 403), LLMAuthError),
        (http_error(openai.RateLimitError, 429), LLMTransientError),
        (http_error(openai.InternalServerError, 500), LLMTransientError),
        (openai.APITimeoutError(request=httpx2.Request("POST", "https://x")), LLMTransientError),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), LLMTransientError),
        (http_error(openai.BadRequestError, 400), LLMError),
    ],
)
def test_openai_errors_are_translated(error, expected):
    client, _ = openai_client(fake_sdk(error=error)[0])
    with pytest.raises(expected) as caught:
        client._request("s", "u", 0, 1)
    assert type(caught.value) is expected


def test_openai_rate_limit_retry_after_header_is_used():
    error = http_error(openai.RateLimitError, 429, {"retry-after": "7"})
    client, sleeps = openai_client(fake_sdk(error=error)[0])
    with pytest.raises(LLMTransientError):
        client.complete("s", "u", temperature=0, max_tokens=1)
    assert sleeps == [7.0]  # one retry, delay raised to the header value


def test_openai_auth_error_message_does_not_include_the_key():
    client, _ = openai_client(fake_sdk(error=http_error(openai.AuthenticationError, 401))[0])
    with pytest.raises(LLMAuthError) as caught:
        client.complete("s", "u", temperature=0, max_tokens=1)
    assert "unused" not in str(caught.value) and "401" in str(caught.value)


# --- provider factory ----------------------------------------------------------------------

def test_create_llm_client_builds_the_configured_provider():
    client = create_llm_client(LLM_CONFIG, {"LLM_API_KEY": "sk-test"})
    assert isinstance(client, OpenAIClient)
    assert client.model == "test-model"
    assert client.retry_policy == RetryPolicy(max_attempts=3, base_delay_s=1, max_delay_s=10)


def test_missing_api_key_names_the_variable():
    with pytest.raises(ConfigError, match="add LLM_API_KEY to .env"):
        create_llm_client(LLM_CONFIG, {})
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        create_llm_client(LLM_CONFIG, {"LLM_API_KEY": "   "})


def test_unknown_provider_lists_available_ones():
    with pytest.raises(ConfigError, match="unknown llm provider 'claude'.*openai"):
        create_llm_client({**LLM_CONFIG, "provider": "claude"}, {"LLM_API_KEY": "k"})


@pytest.mark.parametrize(
    "change, message",
    [
        ({"api_key_env": ""}, "api_key_env"),
        ({"model": ""}, "llm.model"),
        ({"retries": {"max_attempts": 0}}, "max_attempts"),
        ({"retries": {"base_delay_s": "soon"}}, "retries is invalid"),
    ],
)
def test_invalid_llm_config_is_rejected(change, message):
    with pytest.raises(ConfigError, match=message):
        create_llm_client({**LLM_CONFIG, **change}, {"LLM_API_KEY": "k"})


def test_another_provider_can_be_swapped_in_with_one_class_and_a_config_value():
    @register_provider("echo")
    class EchoClient(LLMAPIClient):
        def __init__(self, model, retry_policy, sleep):
            super().__init__(retry_policy, sleep)
            self._model = model

        @classmethod
        def from_config(cls, llm_config, api_key, retry_policy, sleep):
            return cls(llm_config["model"], retry_policy, sleep)

        @property
        def model(self):
            return self._model

        def _request(self, system_prompt, user_prompt, temperature, max_tokens):
            return user_prompt

    try:
        client = create_llm_client({**LLM_CONFIG, "provider": "echo", "model": "echo-1"}, {"LLM_API_KEY": "k"})
        assert isinstance(client, LLMAPIClient) and client.model == "echo-1"
        assert client.complete("s", "hello", temperature=0, max_tokens=1) == "hello"
    finally:
        llm_client._PROVIDERS.pop("echo")


def test_registering_a_duplicate_provider_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_provider("openai")(OpenAIClient)
