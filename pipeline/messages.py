"""Stage 5: write a personalised first-touch email for each qualified lead, in batches.

Flow per batch (docs/WORKFLOW.md Stage 5, docs/LOW_LEVEL_DESIGN.md section 5): ask -> check every
email against the message rules -> re-ask only the leads whose email failed, telling the model
what was wrong (at most `validation_retries` times) -> a lead whose email still fails keeps its
qualified decision but gets no message and is flagged `message_failed`. A bad API key stops the
run; other API failures only cost the affected leads their message.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from pydantic import ValidationError

from pipeline.config import ConfigError, whole_number
from pipeline.llm_client import LLMAPIClient, LLMAuthError, LLMError, extract_json
from pipeline.message_checks import MessageRules
from pipeline.models import (
    GeneratedMessage,
    Lead,
    LeadDecision,
    MessageItem,
    MessagePayload,
    MessageResponse,
)
from pipeline.prompts import build_message_system_prompt, build_message_user_prompt

log = logging.getLogger(__name__)

TOKENS_PER_MESSAGE = 250  # reply budget per lead; keeps a large batch from being cut off mid-JSON
NO_REPLY = "no message was returned for this lead"


@dataclass(frozen=True)
class MessageSettings:
    system_prompt: str
    sender: str
    call_minutes: int
    min_words: int
    max_words: int
    max_subject_words: int
    reengagement_after_days: int
    warm_sources: frozenset[str]
    batch_size: int
    temperature: float
    max_tokens: int
    validation_retries: int

    @classmethod
    def from_config(cls, config: dict) -> "MessageSettings":
        messages, llm = config.get("messages"), config.get("llm")
        if not isinstance(messages, dict):
            raise ConfigError("config 'messages' must be a mapping")
        sender = messages.get("sender_name")
        if not isinstance(sender, str) or not sender.strip() or "\n" in sender:
            raise ConfigError("config messages.sender_name must be a single line of text (set SENDER_NAME in .env)")
        counts = {}
        for key in ("call_minutes", "min_words", "max_words", "max_subject_words", "reengagement_after_days"):
            value = whole_number(messages.get(key))
            if value is None or value < 1:
                raise ConfigError(f"config messages.{key} must be a whole number of 1 or more")
            counts[key] = value
        if counts["min_words"] > counts["max_words"]:
            raise ConfigError("config messages.min_words must not exceed max_words")
        warm = messages.get("warm_sources")
        if not isinstance(warm, list) or not all(isinstance(source, str) for source in warm):
            raise ConfigError("config messages.warm_sources must be a list of text values")

        sending = llm.get("messaging") if isinstance(llm, dict) else None
        if not isinstance(sending, dict):
            raise ConfigError("config llm.messaging must be a mapping")
        batch_size, max_tokens = whole_number(sending.get("batch_size")), whole_number(sending.get("max_tokens"))
        temperature = sending.get("temperature")
        if batch_size is None or batch_size < 1:
            raise ConfigError(
                f"config llm.messaging.batch_size must be a whole number of 1 or more, "
                f"got {sending.get('batch_size')!r} (set LLM_BATCH_SIZE in .env)"
            )
        if max_tokens is None or max_tokens < 1:
            raise ConfigError("config llm.messaging.max_tokens must be a whole number of 1 or more")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
            raise ConfigError("config llm.messaging.temperature must be a number of 0 or more")
        retries = whole_number(llm.get("validation_retries", 2))
        if retries is None or retries < 0:
            raise ConfigError("config llm.validation_retries must be a whole number of 0 or more")

        merged = {**messages, **counts, "sender_name": sender.strip()}
        return cls(
            system_prompt=build_message_system_prompt(config.get("company", {}), merged),
            sender=sender.strip(),
            call_minutes=counts["call_minutes"],
            min_words=counts["min_words"],
            max_words=counts["max_words"],
            max_subject_words=counts["max_subject_words"],
            reengagement_after_days=counts["reengagement_after_days"],
            warm_sources=frozenset(source.strip().lower() for source in warm),
            batch_size=batch_size,
            temperature=float(temperature),
            max_tokens=max_tokens,
            validation_retries=retries,
        )


@dataclass
class MessageRun:
    messages: dict[str, GeneratedMessage]  # keyed by lead_id; only leads that got a valid email
    api_calls: int = 0
    failed_ids: list[str] = field(default_factory=list)  # qualified leads left without a message


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
