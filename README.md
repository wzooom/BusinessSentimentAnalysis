# BusinessSentimentAnalysis
## Key Achievements

* **Decoupled Data Pipeline Architecture:** Designed an asynchronous, file-isolated pipeline where each stage (`ingest` through `report`) communicates exclusively via disk-serialized state (CSV/JSON). This enables independent execution, caching, and simple local debugging.
* **Granular 28-Class Emotion Modeling:** Leveraged an optimized ONNX runtime implementation of `RoBERTa-base-go_emotions` to process text data across 28 distinct emotional dimensions, bypassing generic sentiment binaries to extract true investor psychology (e.g., anxiety, pride, realization).
* **Automated Topic Optimization:** Integrated a dynamic Latent Dirichlet Allocation (LDA) sweep ($K=4\dots10$) using `gensim`, automatically evaluating and selecting optimal topic distributions using the $c_v$ coherence score.
* **Executive Report Creation:** Created a reporting system combining dynamic `matplotlib` charting with the `Gemini API` for contextual topic labeling, culminating in automated, executive ready PDF generation via `reportlab`.

---

## Pipeline

ingest → preprocess → sentiment → lda → aggregate → brief → report

Stages communicate **only through files on disk** — each is independently runnable and re-runnable. The pipeline is parameterized by `--company <name>`.

| Stage | What it does | Output / Artifact |
|---|---|---|
| `ingest` | Paginated StockTwits API pull (resumable, 429-backoff) | `data/raw/` |
| `preprocess` | Cleans raw CSV into specialized `bert_text` + `lda_text` columns | `data/processed/` |
| `sentiment` | RoBERTa go_emotions scoring (28 emotions) | Combined-with-emotions CSV |
| `lda` | gensim LDA, sweeps K=4..10, picks best by `c_v` coherence | `data/topics/` |
| `aggregate` | Compresses per-post output into per-topic signals | `data/aggregated/` |
| `brief` | Gemini topic labels + executive descriptions (cached) | `data/briefs/` |
| `report` | Templated executive PDF (Gemini summaries + matplotlib charts) | `reports/` |

See [CLAUDE.md](CLAUDE.md) for the full design, pipeline stages, storage layout, and conventions.

## Setup

```bash
git clone https://github.com/wzooom/BusinessSentimentAnalysis.git
cd BusinessSentimentAnalysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then fill in Reddit + Gemini credentials
```

## Running a stage

```bash
python main.py <stage> --company <chipotle|starbucks|tesla>
```

Stages: `ingest`, `preprocess`, `sentiment`, `lda`, `aggregate`, `brief`, `report`.

## Tests

```bash
pytest
```
