# BusinessSentimentAnalysis

Customer sentiment market analysis report generator for CSCI 185. Pulls Reddit posts about Chipotle, Starbucks, and Tesla, runs LDA + RoBERTa emotion scoring, and generates an executive PDF brief via Gemini.

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
