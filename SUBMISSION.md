# Lead Intelligence System: Assignment Submission

**Repository (final submission):** https://github.com/Mr-Dj-Python/ops-lead-qualifier
Branch `main` is the submission. Start with `README.md`; the sample output is `output_report.json`.

| | |
|---|---|
| What it is | A command-line tool that qualifies inbound leads, ranks them, writes a personalised first email for the qualified ones, and reports every decision with its reasoning |
| Stack | Python, OpenAI (`gpt-4o-mini`) for judgment, Pydantic for validation, pytest |
| Live result on the 100-lead file | 21 qualified, 59 review, 20 rejected, in 57.6 s, using 17 API calls and no failed calls; 21 emails written; all 18 built-in workflow checks passed |
| Tests | 681 passing, none need the network |
| Delivered | Source code, the three provided datasets, `output_report.json`, `README.md`, and design documents in `docs/` |

---

## 1. The problem and how I read it

The brief: a mid-market SaaS company gets about 1,200 inbound leads a month and qualifies only about 60 (5%). Sales spends 8-12 minutes per decision and suspects it loses good leads. Build a tool that qualifies leads against a rubric I design, scores and ranks them, writes personalised outreach, and gives the sales team a report they can act on, using an LLM, batches, and real error handling.

Two things in the assignment materials shaped everything:
- **There is no ground truth.** The expected-outcomes note says leads have no "correct" answer, and grades on **consistency, defensibility, correctness against my own rubric, and clarity**. So the real work is a rubric I can defend and apply identically to every lead, not matching an answer key.
- **"How you solve it matters more than polish."** The brief asks for stated assumptions, reasoning, and an honest account of what broke. This document and the README are written to that.

## 2. What was built

```
leads.csv -> 1 Ingest and clean -> 2 Rule scores -> 3 LLM industry fit -> 4 Decide and rank
          -> 5 Personalised emails -> 6 Report (output_report.json + run.log + summary)
```

| Stage | What it does | Code |
|---|---|---|
| 1 Ingest | Reads the CSV, treats `NA`/blank as missing, drops unusable rows and lists them, de-duplicates, gives each lead an id and remembers its CSV line | `pipeline/ingest.py` |
| 2 Rules | Scores company size, lead source and recency from bands in config (a factory pattern, so a new attribute is one class) | `pipeline/rules.py` |
| 3 Industry fit | The LLM scores industry fit in batches of 10; invalid or missing items are asked again, then fall back to review | `pipeline/industry.py`, `llm_client.py`, `prompts.py` |
| 4 Decide | Weighted score, thresholds, caps, rank, priority label and a plain-text explanation for each lead | `pipeline/decide.py` |
| 5 Messages | One email per qualified lead in a fixed format, three variants, checked against strict rules | `pipeline/messages.py`, `message_checks.py` |
| 6 Report | Report, traceability, telemetry, run log, and 18 checks that verify the run against itself | `pipeline/report.py`, `telemetry.py` |

The rubric, in one paragraph: I assume the company sells **operations software** (inventory, warehouse and fleet visibility, planning), so a good lead is a physical-goods business of **50-500 employees** with a **real buying signal** and **recent contact**. Score = 40% industry fit (LLM) + 25% size + 20% source + 15% recency, each 1-10. **7 and above qualified, 5 to 7 review, below 5 rejected.** Full detail: `docs/SCORING_WEIGHTS.md` (for the business team), `docs/WORKFLOW.md`, `docs/LOW_LEVEL_DESIGN.md`.

## 3. The steps I took

I worked in small, tested pieces, one branch and one pull request per task. The design documents came first, so each task had a spec.

| Step | Pull request | What it delivered |
|---|---|---|
| 0 | (design docs) | Read the brief and the notes; wrote the workflow, a business-facing scoring guide, and a low-level design with the exact attributes and prompts |
| 1 | #1 | The CLI and CSV ingestion, with errors that stop the run on bad files and `--strict` for bad rows |
| 2 | #2 | Rule scoring with a factory; the weights moved into `.env` so they can change without code edits |
| 3 | #3 | LLM industry-fit scoring behind a swappable `LLMAPIClient`: batches, retries, validation, cache, fallback to review |
| 4 | #4 | Weighted score, thresholds, caps, ranking and explanations; the batch size default set to 10 via `.env` |
| 5 | #5 | Pydantic models for every record from CSV row to LLM reply |
| 6 | #6 | Personalised emails: fixed format, three variants, the five hard rules enforced by validation |
| 7 | #7 | Replaced the remaining dataclasses with Pydantic models, as validation should live in the models |
| 8 | #8 | Stage 6: report, traceability, telemetry, run log, 18 workflow checks |
| 9 | #9 | README and the sample report from a live run |

After each task I ran the tests, and for risky logic I broke the code on purpose to confirm the tests catch it. At the end I reviewed the delivery against the assignment documents and a live run on the real data, and wrote down the gaps (section 8).

## 4. Decisions and why

| Decision | Why | Trade-off |
|---|---|---|
| **Product assumption: operations software** | The brief names no product, so "fit" had to be defined. This option gives a clear spread of best, hard and no-fit industries in the data | SaaS leads, most of the sample, count as no-fit (see the Alice Chen example below) |
| **Four factors, weights 40/25/20/15** | Industry separates buyers from non-buyers, so it weighs most; size next; source shows intent; recency is lowest because a follow-up can fix it | The weights are assumptions, not learned from conversions |
| **Code scores what is numeric; the LLM scores only industry fit and writes emails** | Same input must always give the same score. Arithmetic and explanations built by code cannot drift or contradict the decision | One LLM judgment (industry) carries 40% of the score |
| **Leads the tool is unsure about go to review, never a guess** | The brief asks for a human in the loop. An unknown industry, low model confidence, thin data, or no contact for 180+ days can never qualify; a failed LLM answer goes to review, not rejected | The review bucket is large (section 8) |
| **A no-fit industry with an unrealistic size is rejected outright** | Strong source and recency should not push a wrong-industry lead into the queue | Needs the model's tier to be right |
| **Industry prompt generated from config** | The tier table the model sees and the ranges the validator enforces come from one place, so they cannot disagree | |
| **An "unknown" tier and a confidence field** | The model is allowed to say it does not know, instead of inventing a tier | |
| **Two retry layers** | Transport problems (rate limits, timeouts) retry with backoff inside the client. Invalid or missing answers re-ask only the bad items, and for messages the retry lists exactly what was wrong | More calls when the model breaks rules |
| **Batches of 10, identical leads share one request, answers cached** | Never one call per lead; fewer calls; repeat runs give identical results. The cache is discarded if the model or prompt changes | |
| **A provider-neutral client class** | Moving to another LLM is one new class and two config values | Only OpenAI was tested live |
| **Pydantic models at every boundary, strict where data is internal** | A bad value is stopped where it enters and named by field. Strict mode stops `"7"` from silently becoming 7 | Slightly more code |
| **Fixed email format and three variants chosen by code** | Consistent tone, length and one low-pressure ask. Variant (warm follow-up, value-led, re-engagement) is picked from source and recency, not by the model | The format is strict, so the model needs re-asks |
| **Five hard rules for emails, enforced by validation** | No invented facts, no scores or analysis talk, no filler openers, follow the variant, and treat lead fields as data, not instructions | Word lists catch common phrasings, not all |
| **A failed email leaves the lead qualified with a flag** | A rep can write it by hand; one bad email never blocks a run | |
| **Report with traceability, telemetry, a run log and 18 self-checks** | With no ground truth, the best proof of a sound run is that it is consistent with itself and can be traced to its input | |

**The Alice Chen example.** The expected-outcomes note shows Alice Chen (SaaS, 50 people, inbound demo, 6 days ago) as a qualified lead. Under my rubric she scores 6.8 and goes to **review**, because I defined the product as operations software. The note says no lead is inherently right or wrong, so I kept the rubric and made sure every lead's explanation says exactly why. A SaaS vendor would define "qualified" differently, and I say so.

## 5. How I addressed "no ground truth"

| The note grades | What I did |
|---|---|
| Correctness (scores follow my rubric) | Size, source and recency are deterministic code driven by config; the rubric is copied into every report |
| Consistency | Temperature 0, cached answers, a fixed ranking order, and tests that shuffle the input. A second run on the 100-lead file reproduced the identical 21/59/20 split |
| Defensibility | Every lead has a points breakdown, the model's one-line industry reason, and any rule that capped or forced the decision. Explanations are built by code, so they cannot contradict the score |
| Clarity | One-page README, a business-facing scoring guide, and the rubric stated in one paragraph |

## 6. Evidence

Live run on `data/leads_full_scale_100.csv` (`output_report.json`):

| Measure | Result |
|---|---|
| Leads processed | 100 of 100, 0 dropped, no crashes |
| Decisions | 21 qualified, 59 review, 20 rejected |
| Time | 57.6 s for 100 leads |
| LLM use | 17 calls: 12 for industry in 8 batches (71 leads asked, 29 reused from the cache of an earlier run), 5 for emails in 3 batches. About 27.7k input and 4.9k output tokens. No failed calls |
| Re-asks needed | 4 industry and 2 email re-asks; every item was fixed, none fell back |
| Emails | 21 written, all different, 51-73 words, 7 of each variant, the opt-out line only in re-engagement |
| Self-checks | All 18 passed; no API key in the report or the log |

Other evidence:
- **Real model against my expectations:** on 9 leads I had pre-classified (competitor vendors, a distributor labelled Healthcare, clear buyers and non-buyers) the model agreed on 9 of 9.
- **Tests:** 681, including a test for each of the 18 checks that damages the run output in 21 ways and confirms the damage is caught, and tests that run all three datasets (including the messy 20-lead file) end to end with a fake model.
- **Cost of the live run:** about 32,600 tokens in total and under a minute of waiting, against roughly 1,000 minutes of deciding the same 100 leads by hand at 10 minutes each.

## 7. What went wrong and what I changed

| What happened | Fix |
|---|---|
| YAML read an unquoted `null` as an empty value, so a missing-value word vanished from the config | Values quoted and the config loader rejects non-text entries |
| Pydantic quietly turned the text `"7"` into the number 7 | Internal models are strict |
| The email prompt's "write line breaks as `\n`" reached the model as a real line break | Fixed and covered by a test |
| The real model broke the email rules on most first attempts | Retries now tell it exactly what was wrong; every email passed after one re-ask |
| My own checker crashed with a `KeyError` on the inconsistency it should report | It now returns a failed check; a test covers each case |
| The first live report showed 0 industry calls because everything came from the cache | The final sample shows a mix, and the telemetry says which is which |
| A review of the first draft found no README and a sample report from the wrong file | Added both |
| Dataclasses were used for settings and results | Replaced with Pydantic models and added a test that fails if one comes back |

## 8. Honest gaps and what I would do next

I would fix these first, in this order:
1. **Too many leads go to review.** 59 of 100, and 48 of those are clear no-fit industries the model was sure about. Review should mean "unsure". Rejecting confident no-fit leads would give roughly 21 qualified, 11 review, 68 rejected. The report's "minutes saved" (999) also counts the review leads, which still need a person; counting only decided leads gives about 409.
2. **LinkedIn-outreach leads get a wrong email opener.** The value-led email can say "thank you for reaching out", but we contacted them. It needs its own cold-outreach case.
3. **Fewer first-attempt failures.** The model needs re-asks often (4 of 7 full industry batches, both full email batches). The log does not yet record which rule it broke, which blocks tuning the prompt.
4. **The name rule can lift an industry one tier.** Commercial Cleaning, labelled Business Services, qualified at 7.55. Tighten the rule or explain it.
5. **Not covered live:** the messy testing file was verified with automated tests and a fake model, and the live runs were on the training and 100-lead files. Only OpenAI was tested live.

## 9. How to run and check it

```bash
git clone https://github.com/Mr-Dj-Python/ops-lead-qualifier && cd ops-lead-qualifier
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest                                              # 681 tests, no API key needed
python main.py --input data/leads_training.csv      # ingestion and rule scoring, no API key
```
For the full run, create a `.env` as shown in `README.md` (API key and weights), then:
```bash
python main.py --input data/leads_full_scale_100.csv --score-industry --messages
```
It writes `output_report.json` and appends to `run.log`; the closing summary lists the 18 checks and whether they passed.

| Where | What |
|---|---|
| `README.md` | One-page overview, run steps, rubric, limitations |
| `output_report.json` | The sample report from the live 100-lead run |
| `docs/WORKFLOW.md`, `docs/LOW_LEVEL_DESIGN.md`, `docs/SCORING_WEIGHTS.md` | Design, exact prompts and attributes, business guide |
| `pipeline/`, `main.py`, `config.yaml`, `tests/` | Code, settings, tests |
| `data/`, `outcome/`, `problem-statement/` | The provided datasets and assignment documents |
