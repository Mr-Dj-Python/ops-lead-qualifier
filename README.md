# OpsFlow Lead Qualifier

A command-line tool that reads a CSV of inbound leads, decides **qualified / review / rejected** for each, ranks them, writes a personalised first email for the qualified ones, and produces a report a sales team can act on. An LLM (OpenAI) makes the judgment calls; everything else is plain, deterministic Python.

## Run it
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Create `.env` in the project root (git-ignored, never commit it):
```
LLM_API_KEY=your-openai-key
WEIGHT_INDUSTRY=0.40      # the four weights must add up to 1
WEIGHT_SIZE=0.25
WEIGHT_SOURCE=0.20
WEIGHT_RECENCY=0.15
LLM_BATCH_SIZE=10         # optional: leads per API call (default 10)
SENDER_NAME=Dana, OpsFlow # optional: signs the emails (default "The OpsFlow Team")
```
```bash
python main.py --input data/leads_full_scale_100.csv --score-industry --messages
pytest                    # 681 tests, no network needed
```
Each run writes `output_report.json` and appends to `run.log`. Without `--score-industry` only ingestion and rule scoring run (no API key needed). Other flags: `--explain` (breakdown per lead), `--strict`, `--output`, `--log-file`, `--reference-date`. The data is dated 2023-24, so recency is measured from 2024-01-21 (set in `config.yaml`).

## How it works
CSV, cleaned and validated (bad rows dropped and listed) → rule scores for size, source and recency → LLM scores industry fit in batches (retry, then fall back to review) → weighted score, decision and rank → LLM writes one email per qualified lead, checked against fixed rules → report. Detail: `docs/WORKFLOW.md`, `docs/LOW_LEVEL_DESIGN.md`.

## The rubric
I assume OpsFlow sells operations software (inventory, warehouse and fleet visibility, planning), so a good lead is a physical-goods business of 50-500 employees with a real buying signal and recent contact. Score = 40% industry fit (LLM) + 25% size + 20% lead source + 15% recency, each 1-10; **7 and above qualified, 5 to 7 review, below 5 rejected**. A no-fit or unclear industry, low model confidence, thin data, or no contact for 180+ days can never qualify, and a lead the model fails to score goes to review (`docs/SCORING_WEIGHTS.md` explains it for the business team).

## Result on the provided data
Live run on the 100-lead file (`gpt-4o-mini`): 21 qualified, 59 review, 20 rejected in 57.6 s, 17 API calls (never one per lead), no failed calls, all 18 built-in workflow checks passed. Every lead has a points breakdown and a trace to its CSV row.

## Assumptions and choices
- No ground truth is provided, so the goal is one rubric applied identically and explained per lead: temperature 0, cached answers, arithmetic and explanations built by code, not the LLM.
- Under this rubric SaaS companies are not buyers, so a SaaS lead such as Alice Chen with perfect size and source lands in review (6.8). A SaaS vendor would define "qualified" differently.
- Every record, from CSV row to LLM reply, is a Pydantic model. Alerting is the report's `errors`, warnings in the log and console, and exit status 1 on a bad key, bad config or failed check.

## What broke and changed
Pydantic quietly turned `"7"` into 7, so internal models are strict. The prompt's `\n` instruction reached the model as a real line break (fixed). The model often broke message rules on its first try, so retries now list exactly what was wrong; every message passed after one re-ask. The checker once crashed on the inconsistency it should report (hardened).

## Known limitations
- 59 of 100 leads go to review, 48 of them clear no-fit industries the model was sure about. Review should mean "unsure"; tightening this is the first change I would make.
- The model needs re-asks often (4 of 7 industry batches, both full message batches), which adds calls and time.
- LinkedIn-outreach leads get a value-led email whose opener can be wrong (we contacted them, not the reverse).
- The name rule can lift a label one tier (Commercial Cleaning qualified at 7.55).
- Weights and thresholds are assumptions, not learned from conversions; email wording is not cached, so it changes between runs; only OpenAI was tested live (the client is swappable in `pipeline/llm_client.py`).
