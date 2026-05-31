# CLAUDE.md

Guidance for Claude Code when working on the BusinessSentimentAnalysis repo.

## Project overview

A stockholder sentiment market analysis report generator for a data mining course project. The pipeline ingests Reddit posts about Chipotle, Starbucks, and Tesla, runs LDA topic modeling and RoBERTa-based emotion scoring, aggregates the results into a structured brief, and uses the Gemini API to generate a PDF report aimed at executives at the analyzed company.

Group project, 1-2 week timeline. Repo: https://github.com/wzooom/BusinessSentimentAnalysis

The course requirement is to use something learned in class — LDA satisfies this. LDA is not negotiable.

## Companies analyzed

- Chipotle
- Starbucks
- Tesla

Each run targets one company. The pipeline should be parameterized so the same code handles all three.

## Pipeline stages

```
ingest -> preprocess -> sentiment -> lda -> aggregate -> brief -> report
```

1. **ingest** — Pull Reddit posts via PRAW. Supports both historical pulls and ongoing collection. Writes to `data/raw/`.
2. **preprocess** — Clean text, deduplicate, language filter, lemmatize. Produces two parallel outputs: a lemmatized version for LDA and a cleaned-but-unlemmatized version for sentiment and quote extraction. Writes to `data/processed/`.
3. **sentiment** — Score every post with `SamLowe/roberta-base-go_emotions-onnx`. Runs once on Kelvin's local machine; output is committed to the repo so other group members don't need to run the model. Writes to `data/scored/`.
4. **lda** — Fit LDA on the lemmatized corpus, sweep k to pick best coherence, produce topic-word and document-topic distributions. Writes to `data/topics/`.
5. **aggregate** — Compute per-topic weighted emotion distributions, time series (weekly), source breakdowns, week-over-week deltas, topic co-occurrence, spike detection, representative quotes. Writes to `data/aggregated/`.
6. **brief** — Assemble structured JSON brief: labeled topics with volume, emotion distributions, deltas, peaks, co-occurrences, quotes, plus short Gemini-generated topic descriptions. Writes to `data/briefs/`.
7. **report** — Render the brief into the templated PDF using Gemini for prose sections. Writes to `reports/`.

## Data sources

- **Reddit:** PRAW. No X data (paid API was a blocker).
- Both historical and ongoing collection should be supported.
- Volume target: enough posts per company to produce analysis comparable to existing published market analyses. Tune as needed — don't over-engineer for scale.

## Tech stack

- Python only
- PRAW for Reddit ingestion
- `SamLowe/roberta-base-go_emotions-onnx` for emotion scoring (note: this is an emotion model with 28 labels, not a 3-class sentiment model — the aggregation stage needs to handle multi-label emotion distributions, not just pos/neu/neg)
- LDA via gensim or scikit-learn (Claude's choice — gensim has better coherence scoring built in, prefer it unless there's a reason not to)
- Parquet for tabular data, JSON for the brief
- ReportLab or WeasyPrint for PDF generation (Claude's choice)
- Charting library is Claude's choice — matplotlib is the safe default
- Gemini API (free tier) for topic descriptions and report prose

## Storage layout

```
data/
  raw/           reddit_<company>_<date>.parquet
  processed/     posts_clean_<company>.parquet
                 posts_lemmatized_<company>.parquet
  scored/        posts_scored_<company>.parquet
  topics/        lda_model_<company>.pkl
                 posts_with_topics_<company>.parquet
  aggregated/    topic_signals_<company>.parquet
  briefs/        brief_<company>_<date>.json
reports/
  report_<company>_<date>.pdf
```

Parquet files commit to the repo. The scored output is the boundary — once Kelvin runs sentiment and commits the scored parquet, other group members work downstream without touching RoBERTa.

## Code organization

Modular files, one per pipeline stage. This is a hard requirement to prevent merge conflicts in a group setting.

```
src/
  ingest.py
  preprocess.py
  sentiment.py
  lda.py
  aggregate.py
  brief.py
  report.py
  config.py       # paths, company list, model names, Gemini config
  io_utils.py     # shared parquet read/write helpers
tests/
  test_ingest.py
  test_preprocess.py
  ...
```

Each stage module exposes a clear entry function (e.g. `run_preprocess(company: str) -> Path`) that takes a company name and returns the output path. Stages are independently runnable.

## Gemini API usage

**Free tier constraints to design around:**
- Gemini 2.5 Flash-Lite: 15 RPM, 1,000 RPD, 250K TPM (most generous, use this as default)
- Gemini 2.5 Flash: 10 RPM, 250 RPD
- Gemini 2.5 Pro: 5 RPM, 50 RPD (avoid except for final report generation if quality demands it)
- Free tier prompts may be used by Google for model training — do not send any private or sensitive data. Reddit posts are public so this is fine, but flag it if other data types get added later.

**Two Gemini touchpoints:**
1. **Topic descriptions** — Short description of each topic before it's referenced in analysis. Generated once per topic per run. Use Flash-Lite. ~20 topics per run = well within daily limits.
2. **Report prose** — Generate the executive-facing sections by filling the template with synthesized analysis. Use Flash or Flash-Lite depending on quality needs.

**Implement basic retry-with-backoff for 429 errors.** The pipeline should not crash on a transient rate limit.

## Report format

- **Format:** PDF
- **Audience:** Executives at the analyzed company (Chipotle, Starbucks, or Tesla leadership)
- **Structure:** Templated. One group member is building the template; report.py fills in the dynamic content.
- **Tone:** Executive-appropriate. Concise, action-oriented, business-focused. Not academic.
- **Topic handling:** Each topic gets a short Gemini-generated description the first time it appears, before being used in analysis.
- **Charts:** Yes. Sentiment/emotion trends over time per topic, topic volume comparisons, etc.

## Code style

- No long AI-generated comments. Comments should be sparse and only where they add real value (non-obvious logic, course-specific reasoning, gotchas).
- No verbose docstrings explaining what's obvious from the function signature.
- Type hints are fine but not required for trivial functions.
- Don't over-engineer. This is a course project, not production code.

## Testing

- pytest, with tests for every module.
- Tests should cover the actual logic, not trivial pass-throughs.
- Use small fixture datasets in `tests/fixtures/` rather than hitting Reddit or Gemini during tests.
- Mock the Gemini API and PRAW calls in tests.

## What's out of scope

- **No UI.** Pipeline runs from the CLI. Do not build a web app, dashboard, or notebook interface.
- **No X/Twitter data.** Reddit only.
- **No deployment, no Docker, no CI/CD setup.** Local runs and a shared repo are enough.
- **No real-time streaming infrastructure.** "Live" means re-running ingestion to pull recent posts, not a continuous stream.

## Working with this codebase

- When adding a new pipeline stage or modifying an existing one, keep the input/output parquet contracts stable. Other modules depend on column names.
- The scored dataset is the most expensive artifact to regenerate. Don't invalidate it without a good reason.
- When in doubt about library choices, pick the simpler option. This is a 1-2 week project for a class.
- The course requirement is satisfied by LDA. Don't replace it with BERTopic or other alternatives, even if they'd produce better results.
