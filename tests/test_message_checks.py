import dataclasses
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.config import load_config
from pipeline.message_checks import MessageRules, check_message
from pipeline.models import GeneratedMessage, MessageItem, MessagePayload, MessageResponse
from pipeline.prompts import build_message_system_prompt, message_examples

ROOT = Path(__file__).resolve().parent.parent
ENV = {"WEIGHT_INDUSTRY": "0.4", "WEIGHT_SIZE": "0.25", "WEIGHT_SOURCE": "0.2", "WEIGHT_RECENCY": "0.15"}
SENDER = "Dana, OpsFlow"

WARM = (
    "Hi Oscar,\n\nThanks for connecting with us through a referral. It's good to meet the team at "
    "Warehouse Automation. OpsFlow helps logistics teams keep inventory, warehouse and fleet activity "
    "visible in one place, so fewer shipments slip through the cracks.\n\nIf it would be useful, I can "
    "walk you through a short demo. Would a 15-minute call sometime next week suit you? No worries if "
    "the timing isn't right.\n\nBest regards,\nDana, OpsFlow"
)


def rules(**changes):
    base = MessageRules(
        first_name="Oscar", company="Warehouse Automation", industry="Logistics", sender=SENDER,
        variant="warm_followup", call_minutes=15, min_words=40, max_words=100, max_subject_words=10,
    )
    return dataclasses.replace(base, **changes)


def check(body=WARM, subject="Following up on your referral", **changes):
    return check_message(subject, body, rules(**changes))


def has(problems, text):
    return any(text in problem for problem in problems)


# --- the agreed sample passes, and the prompt's own examples pass --------------------------

def test_the_agreed_sample_has_no_problems():
    assert check() == []


def test_every_example_in_the_prompt_passes_every_rule():
    config = load_config(ROOT / "config.yaml", env=ENV)
    for payload, subject, body in message_examples("OpsFlow", SENDER, 15):
        rule = MessageRules(
            first_name=payload["first_name"], company=payload["company"], industry=payload["industry"],
            sender=SENDER, variant=payload["variant"], call_minutes=15, min_words=40, max_words=100,
            max_subject_words=10,
        )
        assert check_message(subject, body, rule) == [], payload["variant"]
        assert 60 <= len(body.split()) <= 90  # the length the prompt asks for


def test_example_lengths_sit_inside_the_agreed_range():
    for _, _, body in message_examples("OpsFlow", SENDER, 15):
        assert 40 <= len(body.split()) <= 100


def test_the_rendered_prompt_embeds_valid_json_examples():
    config = load_config(ROOT / "config.yaml", env=ENV)
    prompt = build_message_system_prompt(config["company"], {**config["messages"], "sender_name": SENDER})
    outputs = [line[len("Output: "):] for line in prompt.splitlines() if line.startswith("Output: ")]
    assert len(outputs) == 3
    for text in outputs:
        message = json.loads(text)["messages"][0]
        assert message["body"].startswith("Hi ") and message["body"].endswith(SENDER)


# --- length --------------------------------------------------------------------------------

def test_length_limits():
    short = "Hi Oscar,\n\nWarehouse Automation logistics demo?\n\nBest regards,\nDana, OpsFlow"
    assert has(check(short), "words; it must have 40 to 100")
    long_body = WARM.replace("No worries if the timing isn't right.", "No worries if the timing isn't right. " + "word " * 40)
    assert has(check(long_body), "words; it must have 40 to 100")


def test_the_word_range_comes_from_the_rules():
    assert not has(check(min_words=10, max_words=200), "words; it must have")
    assert has(check(max_words=50), "it must have 40 to 50")


def test_subject_rules():
    assert has(check(subject=""), "subject is empty")
    assert has(check(subject="   "), "subject is empty")
    assert has(check(subject="one two three four five six seven eight nine ten eleven"), "subject has 11 words")
    assert check(subject="one two three four five six seven eight nine ten") == []
    assert has(check(subject="Quick question!"), "exclamation")


# --- shape: greeting, paragraphs, sign-off, one question -----------------------------------

@pytest.mark.parametrize(
    "old, new, message",
    [
        ("Hi Oscar,", "Hello Oscar,", 'start with exactly "Hi Oscar,"'),
        ("Hi Oscar,", "Hi Oscar", 'start with exactly "Hi Oscar,"'),
        ("Hi Oscar,", "Hi oscar,", 'start with exactly "Hi Oscar,"'),
        ("Best regards,\nDana, OpsFlow", "Kind regards,\nDana, OpsFlow", 'end with "Best regards,"'),
        ("Best regards,\nDana, OpsFlow", "Best regards,\nSomeone Else", 'end with "Best regards,"'),
        ("Best regards,\nDana, OpsFlow", "Best regards, Dana, OpsFlow", 'end with "Best regards,"'),
        ("Best regards,\nDana, OpsFlow", "Best regards,\nDana, OpsFlow\n\nP.S. Call me.", 'end with "Best regards,"'),
    ],
)
def test_greeting_and_sign_off_must_match_exactly(old, new, message):
    assert has(check(WARM.replace(old, new)), message)


def test_trailing_spaces_and_windows_line_endings_are_tolerated():
    assert check(WARM.replace("\n", " \n")) == []
    assert check(WARM.replace("\n", "\r\n")) == []


def test_paragraph_count():
    one_paragraph = WARM.replace("\n\nIf it would be useful", " If it would be useful")
    assert has(check(one_paragraph), "3 blocks")
    four_paragraphs = WARM.replace(". It's good", ".\n\nIt's good").replace("\n\nIf it would", "\n\nSo be it.\n\nIf it would")
    assert has(check(four_paragraphs), "blocks separated by blank lines")
    three_paragraphs = WARM.replace(". It's good", ".\n\nIt's good")
    assert not has(check(three_paragraphs), "blocks separated")


def test_exactly_one_question_in_the_last_paragraph():
    assert has(check(WARM.replace("suit you?", "suit you.")), "0 question marks")
    assert has(check(WARM.replace("No worries if", "Does that work? No worries if")), "2 question marks")
    moved = WARM.replace("Would a 15-minute call sometime next week suit you? ", "").replace(
        "It's good to meet", "Could we talk? It's good to meet"
    )
    assert has(check(moved), "last paragraph before the sign-off")


# --- tone and cleanliness ------------------------------------------------------------------

def test_no_exclamation_marks_emojis_markdown_or_placeholders():
    assert has(check(WARM.replace("good to meet", "great to meet!")), "exclamation")
    assert has(check(WARM + " 🙂"), "emojis")
    assert has(check(WARM.replace("logistics teams", "**logistics** teams")), "plain text only")
    assert has(check("Subject: Hello\n\n" + WARM), "plain text only")
    assert has(check(WARM.replace("Oscar", "[Name]", 1)), "placeholders")
    assert has(check(WARM.replace("team at", "team at {company} at")), "placeholders")
    assert has(check(WARM.replace("team at", "team at <b>")), "placeholders")


@pytest.mark.parametrize(
    "phrase",
    ["I hope this email finds you well", "Just following up", "Circling back", "last chance", "Limited time",
     "ACT NOW", "this is urgent", "ASAP", "Don't miss out", "We guarantee results", "a real game-changer", "synergy"],
)
def test_pressure_and_filler_phrases_are_rejected(phrase):
    assert has(check(WARM.replace("No worries if the timing isn't right.", f"{phrase}.")), "remove the phrase")


def test_words_that_merely_contain_a_banned_word_are_fine():
    assert not has(check(WARM.replace("shipments", "scorecards and shipments")), "remove the phrase")
    assert not has(check(WARM.replace("shipments", "unqualified shipments")), "remove the phrase")


def test_internal_words_are_rejected():
    assert has(check(WARM.replace("useful", "a qualified lead")), '"qualified"')
    assert has(check(WARM.replace("useful", "high score")), '"score"')


# --- personalised and nothing invented -----------------------------------------------------

def test_company_or_industry_must_be_mentioned():
    generic = WARM.replace("Warehouse Automation", "your team").replace("logistics", "operations")
    assert has(check(generic), "mention the lead's company or industry (Warehouse Automation or Logistics)")
    assert not has(check(WARM.replace("Warehouse Automation", "your team")), "mention the lead's company")  # industry word is there
    assert not has(check(WARM.replace("logistics", "operations")), "mention the lead's company")  # company is there


def test_a_missing_company_falls_back_to_the_industry():
    no_company = WARM.replace("Warehouse Automation", "your team")
    assert not has(check(no_company, company="(missing)"), "mention the lead's company")
    assert has(check(no_company.replace("logistics", "operations"), company="(missing)"), "mention the lead's company or industry (Logistics)")
    assert not has(check(no_company, company=None), "mention the lead's company")


def test_nothing_to_mention_means_no_personalisation_check():
    plain = WARM.replace("Warehouse Automation", "your team").replace("logistics", "operations")
    assert not has(check(plain, company=None, industry=None), "mention the lead's company")
    assert not has(check(plain, company="(missing)", industry="AI"), "mention the lead's company")  # no word of 4+ letters


def test_only_the_call_length_and_numbers_from_the_lead_are_allowed():
    assert not has(check(), "numbers")
    assert has(check(WARM.replace("short demo", "short demo that saves 30 hours")), "numbers that are not in the lead's details: 30")
    assert has(check(WARM.replace("15-minute", "20-minute")), "numbers that are not in the lead's details: 20")
    assert not has(check(WARM.replace("Warehouse Automation", "Fortune 500 Co"), company="Fortune 500 Co"), "numbers")
    assert has(check(WARM.replace("short demo", "short demo in 2 steps")), "numbers that are not in the lead's details: 2")
    assert has(check(WARM.replace("short demo", "short demo with 3.5x savings")), "3.5")


def test_percentages_and_money_are_rejected():
    assert has(check(WARM.replace("short demo", "short demo and save $")), "percentages and currency")
    assert has(check(WARM.replace("short demo", "short demo and cut costs by half%")), "percentages and currency")


# --- variants ------------------------------------------------------------------------------

def test_warm_followup_offers_a_demo():
    assert has(check(WARM.replace("demo", "walkthrough")), 'use the word "demo"')


def test_value_led_offers_an_overview():
    value = (
        "Hi Oscar,\n\nThanks for downloading our content. Many logistics teams find stock and delivery tracking "
        "slow when it relies on spreadsheets. OpsFlow brings inventory, warehouse and fleet visibility together "
        "to make planning simpler for companies like Warehouse Automation.\n\nWould it help if I sent a short "
        "overview, or would a quick call be easier? No pressure either way.\n\nBest regards,\nDana, OpsFlow"
    )
    assert check(value, variant="value_led") == []
    assert has(check(value.replace("overview", "summary"), variant="value_led"), 'use the word "overview"')


def test_reengagement_ends_with_the_opt_out():
    re_engage = (
        "Hi Oscar,\n\nIt's been a little while since we last spoke, so I wanted to check in. OpsFlow helps "
        "logistics teams keep inventory, warehouse and fleet activity visible in one place, which can make "
        "planning easier.\n\nIf stock and supply tracking is still on your radar at Warehouse Automation, would "
        "a quick 15-minute call sometime in the next couple of weeks be useful? If you'd rather not hear from "
        "us, just let me know.\n\nBest regards,\nDana, OpsFlow"
    )
    assert check(re_engage, variant="reengagement") == []
    assert has(check(re_engage.replace("If you'd rather not hear from us, just let me know.", "No worries if not."), variant="reengagement"), "rather not hear")
    assert not has(check(re_engage.replace("rather", "prefer"), variant="reengagement"), "rather not hear")


def test_several_problems_are_reported_together():
    bad = "Hello,\n\nBuy now! Limited time offer, save 50%.\n\nCheers"
    problems = check(bad, subject="")
    assert len(problems) >= 6


# --- the models around the rules -----------------------------------------------------------

def item(**changes):
    return {"lead_id": "L001", "subject": "Following up on your referral", "body": WARM, **changes}


def test_message_item_passes_with_rules_and_reports_all_problems_when_it_fails():
    context = {"message_rules": rules()}
    assert MessageItem.model_validate(item(), context=context).lead_id == "L001"
    with pytest.raises(ValidationError) as caught:
        MessageItem.model_validate(item(body="Hello there!"), context=context)
    message = caught.value.errors()[0]["msg"].removeprefix("Value error, ")
    assert " | " in message and "words; it must have" in message and "exclamation" in message


def test_message_item_without_rules_only_checks_the_shape():
    assert MessageItem.model_validate(item(body="anything")).body == "anything"
    for bad in ({"subject": ""}, {"body": ""}, {"subject": 5}, {"lead_id": 1}):
        with pytest.raises(ValidationError):
            MessageItem.model_validate(item(**bad))
    with pytest.raises(ValidationError):
        MessageItem.model_validate({"lead_id": "L001", "subject": "s"})


def test_message_response_envelope():
    assert MessageResponse.model_validate({"messages": [1]}).messages == [1]
    assert MessageResponse.model_validate([{"a": 1}]).messages == [{"a": 1}]
    for data in ({}, {"results": []}, {"messages": "x"}, None):
        with pytest.raises(ValidationError):
            MessageResponse.model_validate(data)


def test_message_payload_and_generated_message_are_strict():
    payload = MessagePayload(lead_id="L001", first_name="Oscar", company="Acme", industry="Retail", source="Referral", variant="warm_followup")
    assert payload.variant == "warm_followup"
    with pytest.raises(ValidationError):
        MessagePayload(lead_id="L001", first_name="Oscar", company="Acme", industry="Retail", source="Referral", variant="cold_call")
    message = GeneratedMessage(lead_id="L001", variant="value_led", subject="s", body="b", word_count=1)
    with pytest.raises(ValidationError):
        message.subject = "other"
    with pytest.raises(ValidationError):
        GeneratedMessage(lead_id="L001", variant="value_led", subject="s", body="b", word_count="1")
