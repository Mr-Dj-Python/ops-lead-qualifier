"""Provider-neutral LLM connection.

LLMAPIClient is the only thing the rest of the pipeline knows about. A provider is one
subclass that implements `_request` (one call, errors translated to the exceptions below) and
is registered with @register_provider. Switching provider, for example from OpenAI to Claude, is
a new subclass plus `llm.provider` in config.yaml; retry and backoff are shared and live here.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

from pipeline.config import ConfigError

log = logging.getLogger(__name__)


class LLMError(Exception):
    """A call failed and retrying the same request will not help."""


class LLMTransientError(LLMError):
    """Rate limit, timeout, connection problem or server error: worth retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """Bad or missing credentials. The whole run must stop; never retried."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 2.0
    max_delay_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigError("llm retries max_attempts must be at least 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ConfigError("llm retries delays must be 0 or more")


class LLMAPIClient(ABC):
    """One connection to an LLM provider, asked for JSON text."""

    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep

    @classmethod
    @abstractmethod
    def from_config(
        cls, llm_config: dict, api_key: str, retry_policy: RetryPolicy, sleep: Callable[[float], None]
    ) -> "LLMAPIClient":
        """Build the client from the `llm` section of config.yaml and the API key."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name in use; it is part of the scoring cache fingerprint."""

    @abstractmethod
    def _request(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        """Make one call and return the reply text.

        Raise LLMAuthError for credential problems, LLMTransientError for rate limits,
        timeouts, connection and server errors, and LLMError for anything else.
        """

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float, max_tokens: int) -> str:
        """Ask for a JSON reply, retrying transient failures with exponential backoff."""
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return self._request(system_prompt, user_prompt, temperature, max_tokens)
            except LLMTransientError as exc:
                if attempt == policy.max_attempts:
                    raise
                backoff = policy.base_delay_s * 2 ** (attempt - 1) + random.uniform(0, policy.base_delay_s)
                delay = max(min(backoff, policy.max_delay_s), exc.retry_after or 0.0)
                log.warning("LLM call failed (%s); retry %d of %d in %.1fs", exc, attempt, policy.max_attempts - 1, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises


# --- providers -------------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMAPIClient]] = {}


def register_provider(name: str):
    """Class decorator that makes a client available to create_llm_client under `name`."""

    def decorator(cls: type[LLMAPIClient]) -> type[LLMAPIClient]:
        if name in _PROVIDERS:
            raise ValueError(f"provider {name!r} is already registered")
        _PROVIDERS[name] = cls
        return cls

    return decorator


@register_provider("openai")
class OpenAIClient(LLMAPIClient):
    """OpenAI chat completions in JSON mode."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        sdk_client=None,
    ) -> None:
        super().__init__(retry_policy, sleep)
        self._model = model
        if sdk_client is None:
            import openai  # imported here so other providers do not need the package

            # The SDK's own retries are off: LLMAPIClient.complete owns retry and backoff.
            sdk_client = openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._sdk = sdk_client

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    def from_config(
        cls, llm_config: dict, api_key: str, retry_policy: RetryPolicy, sleep: Callable[[float], None]
    ) -> "OpenAIClient":
        return cls(
            api_key=api_key,
            model=llm_config["model"],
            timeout_s=float(llm_config.get("timeout_s", 30)),
            retry_policy=retry_policy,
            sleep=sleep,
        )

    def _request(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        import openai

        try:
            response = self._sdk.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMAuthError(f"OpenAI rejected the API key ({exc.status_code})") from exc
        except openai.RateLimitError as exc:
            raise LLMTransientError("OpenAI rate limit", _retry_after(exc)) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise LLMTransientError(f"OpenAI connection problem: {type(exc).__name__}") from exc
        except openai.InternalServerError as exc:
            raise LLMTransientError(f"OpenAI server error ({exc.status_code})", _retry_after(exc)) from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        if not response.choices:
            raise LLMTransientError("OpenAI returned no choices")
        return response.choices[0].message.content or ""


def _retry_after(exc) -> float | None:
    """Seconds from a Retry-After header, if the SDK exception carries one."""
    try:
        return max(float(exc.response.headers.get("retry-after")), 0.0)
    except (AttributeError, TypeError, ValueError):
        return None


def extract_json(raw: str):
    """Parse the reply, tolerating markdown fences and text around the JSON object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found")


def create_llm_client(
    llm_config: dict,
    env: Mapping[str, str],
    sleep: Callable[[float], None] = time.sleep,
) -> LLMAPIClient:
    """Build the client named by `llm.provider`, reading the API key from the env variable
    named by `llm.api_key_env` (the key itself is never in config.yaml)."""
    provider = llm_config.get("provider")
    try:
        cls = _PROVIDERS[provider]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise ConfigError(f"unknown llm provider {provider!r}; available: {known}") from None
    key_name = llm_config.get("api_key_env")
    if not isinstance(key_name, str) or not key_name:
        raise ConfigError("config llm.api_key_env must name the environment variable holding the API key")
    api_key = env.get(key_name, "").strip()
    if not api_key:
        raise ConfigError(f"LLM API key not set: add {key_name} to .env")
    if not isinstance(llm_config.get("model"), str) or not llm_config["model"]:
        raise ConfigError("config llm.model must name the model to use")
    retries = llm_config.get("retries", {})
    try:
        policy = RetryPolicy(
            max_attempts=int(retries.get("max_attempts", 4)),
            base_delay_s=float(retries.get("base_delay_s", 2)),
            max_delay_s=float(retries.get("max_delay_s", 30)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config llm.retries is invalid: {exc}") from exc
    return cls.from_config(llm_config, api_key, policy, sleep)
