"""The rules every generated message must pass (docs/LOW_LEVEL_DESIGN.md section 5).

`check_message` returns every problem it finds as a plain sentence. The sentences double as
feedback in the retry prompt, so each one says what to fix. No pydantic or LLM code in here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VARIANTS = ("warm_followup", "value_led", "reengagement")
SIGN_OFF = "Best regards,"
MISSING = "(missing)"

# Pressure, filler and sales-speak that would undo the tone. Matched as whole words or phrases.
AVOID_PHRASES = (
    "hope this email finds you well", "hope this finds you well", "just following up", "following up again",
    "circling back", "touching base", "last chance", "limited time", "act now", "urgent", "urgently",
    "asap", "don't miss", "final reminder", "guarantee", "guaranteed", "game-changer", "game changer",
    "revolutionary", "synergy", "score", "qualified", "ai-driven",
)
_AVOID = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(p) for p in AVOID_PHRASES) + r")(?![\w-])", re.IGNORECASE)
_OPT_OUT = re.compile(r"\b(?:rather|prefer) not (?:to )?hear\b", re.IGNORECASE)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_PLACEHOLDER = re.compile(r"\{[^}]*\}|\[[^\]]*\]|<[^>]*>")
_MARKDOWN = re.compile(r"\*\*|__|`|^#{1,6}\s", re.MULTILINE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_MONEY_OR_PERCENT = re.compile(r"[$€£¥%]")


@dataclass(frozen=True)
class MessageRules:
    first_name: str
    company: str | None  # None or "(missing)" when unknown
    industry: str | None
    sender: str  # the line after "Best regards,"
    variant: str
    call_minutes: int
    min_words: int
    max_words: int
    max_subject_words: int


def check_message(subject: str, body: str, rules: MessageRules) -> list[str]:
    """Every way this message breaks the rules (an empty list means it is good)."""
    problems: list[str] = []
    subject = subject.strip()
    text = body.replace("\r\n", "\n").strip()
    both = f"{subject}\n{text}"

    # Subject and length
    if not subject:
        problems.append("the subject is empty")
    elif len(subject.split()) > rules.max_subject_words:
        problems.append(f"the subject has {len(subject.split())} words; use at most {rules.max_subject_words}")
    words = len(text.split())
    if not rules.min_words <= words <= rules.max_words:
        problems.append(f"the body has {words} words; it must have {rules.min_words} to {rules.max_words}")

    # Shape: greeting, 2-3 short paragraphs, sign-off, separated by blank lines
    blocks = ["\n".join(line.rstrip() for line in block.split("\n")) for block in re.split(r"\n\s*\n", text)]
    if len(blocks) not in (4, 5):
        problems.append(
            f"the body has {len(blocks)} blocks separated by blank lines; use the greeting, then 2 or 3 short "
            "paragraphs, then the sign-off"
        )
    greeting = f"Hi {rules.first_name},"
    if blocks[0] != greeting:
        problems.append(f'the body must start with exactly "{greeting}" followed by a blank line')
    sign_off = f"{SIGN_OFF}\n{rules.sender}"
    if blocks[-1] != sign_off:
        problems.append(f'the body must end with "{SIGN_OFF}" and then "{rules.sender}" on the next line')

    # One suggested action, in the paragraph just before the sign-off
    questions = text.count("?")
    if questions != 1:
        problems.append(f"the body has {questions} question marks; it needs exactly one, the suggested action")
    elif len(blocks) >= 2 and "?" not in blocks[-2]:
        problems.append("the question must be in the last paragraph before the sign-off")

    # Tone and cleanliness
    if "!" in both:
        problems.append("remove exclamation marks")
    if _EMOJI.search(both):
        problems.append("remove emojis")
    if _MARKDOWN.search(both) or text.lower().startswith("subject:"):
        problems.append("use plain text only: no markdown and no subject line inside the body")
    if _PLACEHOLDER.search(both):
        problems.append("remove placeholders, brackets and template markers")
    for match in sorted({m.group(0).lower() for m in _AVOID.finditer(both)}):
        problems.append(f'remove the phrase "{match}"')

    # Personalised, and nothing invented
    company = None if rules.company in (None, MISSING) else rules.company
    industry_words = [w for w in re.findall(r"[a-z]{4,}", (rules.industry or "").lower())]
    lowered = text.lower()
    if (company or industry_words) and not (
        (company and company.lower() in lowered) or any(word in lowered for word in industry_words)
    ):
        named = " or ".join(part for part in (company, rules.industry if industry_words else None) if part)
        problems.append(f"mention the lead's company or industry ({named})")
    allowed = {str(rules.call_minutes)} | set(_NUMBER.findall(f"{rules.company or ''} {rules.industry or ''}"))
    invented = sorted(set(_NUMBER.findall(both)) - allowed)
    if invented:
        problems.append(f"remove numbers that are not in the lead's details: {', '.join(invented)}")
    if _MONEY_OR_PERCENT.search(both):
        problems.append("remove percentages and currency amounts")

    # What each variant must contain
    if rules.variant == "warm_followup" and "demo" not in lowered:
        problems.append('a warm follow-up must offer a demo (use the word "demo")')
    if rules.variant == "value_led" and "overview" not in lowered:
        problems.append('a value-led email must offer a short overview (use the word "overview")')
    if rules.variant == "reengagement" and not _OPT_OUT.search(text):
        problems.append('a re-engagement email must end its last paragraph with "If you\'d rather not hear from us, just let me know."')
    return problems
