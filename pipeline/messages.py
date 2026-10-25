"""Stage 5: write a personalised first-touch email for each qualified lead, in batches.

Flow per batch (docs/WORKFLOW.md Stage 5, docs/LOW_LEVEL_DESIGN.md section 5): ask -> check every
email against the message rules -> re-ask only the leads whose email failed, telling the model
what was wrong (at most `validation_retries` times) -> a lead whose email still fails keeps its
qualified decision but gets no message and is flagged `message_failed`. A bad API key stops the
run; other API failures only cost the affected leads their message.
"""
from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pipeline.config import ConfigError, config_error, parse_count, parse_number
from pipeline.llm_client import LLMAPIClient, LLMAuthError, LLMError, extract_json
from pipeline.message_checks import MessageRules
from pipeline.models import (
    GeneratedMessage,
    Lead,
    LeadDecision,
    MessageItem,
    MessageRun,
    MessagePayload,
    MessageResponse,
)
from pipeline.prompts import build_message_system_prompt, build_message_user_prompt

log = logging.getLogger(__name__)

TOKENS_PER_MESSAGE = 250  # reply budget per lead; keeps a large batch from being cut off mid-JSON
NO_REPLY = "no message was returned for this lead"


MESSAGE_PATHS = {  # settings field -> the config path a user would edit
    "sender": "messages.sender_name",
    "call_minutes": "messages.call_minutes",
    "min_words": "messages.min_words",
    "max_words": "messages.max_words",
    "max_subject_words": "messages.max_subject_words",
    "reengagement_after_days": "messages.reengagement_after_days",
    "warm_sources": "messages.warm_sources",
    "batch_size": "llm.messaging.batch_size",
    "max_tokens": "llm.messaging.max_tokens",
    "temperature": "llm.messaging.temperature",
    "validation_retries": "llm.validation_retries",
}


class MessageSettings(BaseModel):
    """Everything the message stage needs from config.yaml, validated."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str
    sender: str  # the line under "Best regards,"
    call_minutes: int = Field(ge=1)
    min_words: int = Field(ge=1)
    max_words: int = Field(ge=1)
    max_subject_words: int = Field(ge=1)
    reengagement_after_days: int = Field(ge=1)
    warm_sources: frozenset[str]  # lower case
    batch_size: int = Field(ge=1)
    temperature: float = Field(ge=0)
    max_tokens: int = Field(ge=1)
    validation_retries: int = Field(ge=0)

    @field_validator("sender", mode="before")
    @classmethod
    def _sender(cls, value) -> str:
        if not isinstance(value, str) or not value.strip() or "\n" in value.strip():
            raise ValueError("must be a single line of text (set SENDER_NAME in .env)")
        return value.strip()

    @field_validator(
        "call_minutes", "min_words", "max_words", "max_subject_words", "reengagement_after_days", mode="before"
    )
    @classmethod
    def _positive_count(cls, value) -> int:
        return parse_count(value, 1)

    @field_validator("batch_size", mode="before")
    @classmethod
    def _batch_size(cls, value) -> int:
        return parse_count(value, 1, " (set LLM_BATCH_SIZE in .env)")

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _max_tokens(cls, value) -> int:
        return parse_count(value, 1)

    @field_validator("validation_retries", mode="before")
    @classmethod
    def _validation_retries(cls, value) -> int:
        return parse_count(value, 0)

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value) -> float:
        return parse_number(value, 0)

    @field_validator("warm_sources", mode="before")
    @classmethod
    def _warm_sources(cls, value) -> frozenset:
        if not isinstance(value, (list, tuple, set, frozenset)) or not all(isinstance(v, str) for v in value):
            raise ValueError("must be a list of text values")
        return frozenset(source.strip().lower() for source in value)

    @model_validator(mode="after")
    def _word_range(self) -> "MessageSettings":
        if self.min_words > self.max_words:
            raise ValueError("messages.min_words must not exceed max_words")
        return self

    @classmethod
    def from_config(cls, config: dict) -> "MessageSettings":
        messages, llm = config.get("messages"), config.get("llm")
        if not isinstance(messages, dict):
            raise ConfigError("config 'messages' must be a mapping")
        sending = llm.get("messaging") if isinstance(llm, dict) else None
        if not isinstance(sending, dict):
            raise ConfigError("config llm.messaging must be a mapping")
        try:
            settings = cls.model_validate(
                {
                    "system_prompt": "",  # built below, once everything else is known to be valid
                    "sender": messages.get("sender_name"),
                    "call_minutes": messages.get("call_minutes"),
                    "min_words": messages.get("min_words"),
                    "max_words": messages.get("max_words"),
                    "max_subject_words": messages.get("max_subject_words"),
                    "reengagement_after_days": messages.get("reengagement_after_days"),
                    "warm_sources": messages.get("warm_sources"),
                    "batch_size": sending.get("batch_size"),
                    "temperature": sending.get("temperature"),
                    "max_tokens": sending.get("max_tokens"),
                    "validation_retries": llm.get("validation_retries", 2),
                }
            )
        except ValidationError as exc:
            raise config_error(exc, MESSAGE_PATHS) from exc
        prompt = build_message_system_prompt(
            config.get("company", {}),
            {
                "sender_name": settings.sender,
                "call_minutes": settings.call_minutes,
                "min_words": settings.min_words,
                "max_words": settings.max_words,
                "max_subject_words": settings.max_subject_words,
            },
        )
        return settings.model_copy(update={"system_prompt": prompt})


def choose_variant(lead: Lead, settings: MessageSettings, reference_date: date) -> str:
    """First match wins: no contact for over a month, then demo request or referral, then the rest."""
    if lead.last_interaction_date is not None:
        if (reference_date - lead.last_interaction_date).days > settings.reengagement_after_days:
            return "reengagement"
    if lead.source is not None and lead.source.strip().lower() in settings.warm_sources:
        return "warm_followup"
    return "value_led"


def first_name(lead: Lead) -> str:
    """The first word of the name for the greeting, or "there" when there is none."""
    words = (lead.name or "").split()
    name = words[0].strip(".,;:!?\"'()[]") if words else ""
    return name or "there"


class MessageGenerator:
    def __init__(self, client: LLMAPIClient, settings: MessageSettings) -> None:
        self.client = client
        self.settings = settings

    def generate(self, decisions: list[LeadDecision], leads: list[Lead], reference_date: date) -> MessageRun:
        """Write an email for every qualified lead, in priority order."""
        by_id = {lead.lead_id: lead for lead in leads}
        qualified = [by_id[d.lead_id] for d in decisions if d.decision == "qualified"]
        run = MessageRun(messages={})
        size = self.settings.batch_size
        for start in range(0, len(qualified), size):
            batch = qualified[start : start + size]
            self._write_batch(batch, reference_date, run)
        return run

    def _write_batch(self, batch: list[Lead], reference_date: date, run: MessageRun) -> None:
        settings = self.settings
        payloads: dict[str, MessagePayload] = {}
        rules: dict[str, MessageRules] = {}
        for lead in batch:
            variant = choose_variant(lead, settings, reference_date)
            payloads[lead.lead_id] = MessagePayload(
                lead_id=lead.lead_id, first_name=first_name(lead), company=lead.company or "(missing)",
                industry=lead.industry or "(missing)", source=lead.source or "(missing)", variant=variant,
            )
            rules[lead.lead_id] = MessageRules(
                first_name=first_name(lead), company=lead.company, industry=lead.industry, sender=settings.sender,
                variant=variant, call_minutes=settings.call_minutes, min_words=settings.min_words,
                max_words=settings.max_words, max_subject_words=settings.max_subject_words,
            )

        pending = dict(payloads)
        problems: dict[str, str] = {}
        repair = False
        for attempt in range(1 + settings.validation_retries):
            if not pending:
                break
            prompt = build_message_user_prompt(
                [p.model_dump() for p in pending.values()], problems=problems or None, repair=repair
            )
            run.api_calls += 1
            try:
                raw = self.client.complete(
                    settings.system_prompt, prompt,
                    temperature=settings.temperature,
                    max_tokens=max(settings.max_tokens, TOKENS_PER_MESSAGE * len(pending)),
                )
            except LLMAuthError:
                raise
            except LLMError as exc:
                log.warning("LLM call failed for %d message(s) after retries: %s", len(pending), exc)
                break
            try:
                items = MessageResponse.model_validate(extract_json(raw)).messages
                repair = False
            except (ValueError, ValidationError) as exc:
                log.warning("LLM reply was not usable (%s); retrying %d message(s)", str(exc).splitlines()[0], len(pending))
                repair = True
                continue

            answered: set[str] = set()
            problems = {}
            for item in items:
                lead_id = item.get("lead_id") if isinstance(item, dict) else None
                if not isinstance(lead_id, str) or lead_id not in pending or lead_id in answered:
                    continue
                answered.add(lead_id)
                try:
                    valid = MessageItem.model_validate(item, context={"message_rules": rules[lead_id]})
                except ValidationError as exc:
                    problems[lead_id] = "; ".join(
                        e["msg"].removeprefix("Value error, ").replace(" | ", "; ") for e in exc.errors()
                    )
                    continue
                run.messages[lead_id] = GeneratedMessage(
                    lead_id=lead_id, variant=payloads[lead_id].variant, subject=valid.subject.strip(),
                    body=valid.body.replace("\r\n", "\n").strip(), word_count=len(valid.body.split()),
                )
                del pending[lead_id]
            for lead_id in pending:
                problems.setdefault(lead_id, NO_REPLY)
            if pending:
                log.warning("%d of %d message(s) failed the rules; retrying", len(pending), len(batch))

        for lead_id in pending:
            log.warning("no valid message for %s after retries; the lead keeps its qualified decision", lead_id)
            run.failed_ids.append(lead_id)
