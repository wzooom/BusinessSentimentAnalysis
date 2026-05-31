from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import models
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from src.config import DATA_DIR, TOPICS_DIR, ensure_data_dirs


INPUT_CSV   = DATA_DIR / "stocktwits_all_messages_clean_combined_with_emotions.csv"
COMPANY_COL = "company"

TEXT_COL = "lda_text"
DATE_COL = "created_datetime"
ID_COL   = "post_idx"

K_RANGE      = range(4, 11)
PASSES       = 10
ITERATIONS   = 400
RANDOM_STATE = 42

NO_BELOW = 5
NO_ABOVE = 0.5
KEEP_N   = 20000

EMOTION_COLS = [
    "emotion_admiration", "emotion_amusement", "emotion_anger", "emotion_annoyance",
    "emotion_approval", "emotion_caring", "emotion_confusion", "emotion_curiosity",
    "emotion_desire", "emotion_disappointment", "emotion_disapproval", "emotion_disgust",
    "emotion_embarrassment", "emotion_excitement", "emotion_fear", "emotion_gratitude",
    "emotion_grief", "emotion_joy", "emotion_love", "emotion_nervousness",
    "emotion_neutral", "emotion_optimism", "emotion_pride", "emotion_realization",
    "emotion_relief", "emotion_remorse", "emotion_sadness", "emotion_surprise",
]

CARRY_COLS = [
    "post_idx",
    "company", "source", "ticker",
    "created_datetime", "sentiment_label",
    "bert_text",
]


def load_corpus(company: str) -> pd.DataFrame:
    company_norm = company.strip().lower()
    df = pd.read_csv(INPUT_CSV)
    df = df[df[COMPANY_COL].str.lower() == company_norm]
    df = df[df[TEXT_COL].notna()].copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.split().str.len() > 0]
    if df.empty:
        raise ValueError(f"no rows for {company!r} in {INPUT_CSV}")
    df = df.reset_index(drop=True)
    df["post_idx"] = [f"{company_norm}_{i:05d}" for i in range(len(df))]
    return df


def tokenize(df: pd.DataFrame) -> list[list[str]]:
    return [t.split() for t in df[TEXT_COL]]


def build_bow(tokens: list[list[str]]) -> tuple[Dictionary, list]:
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
    if len(dictionary) == 0:
        raise ValueError("vocabulary empty after filter_extremes; lower NO_BELOW")
    corpus = [dictionary.doc2bow(t) for t in tokens]
    return dictionary, corpus


def fit_lda(corpus, dictionary: Dictionary, k: int) -> models.LdaModel:
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=PASSES,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        chunksize=2000,
        minimum_probability=0.0,
        eval_every=None,
    )


def coherence(model: models.LdaModel, tokens, dictionary: Dictionary) -> float:
    return CoherenceModel(
        model=model, texts=tokens, dictionary=dictionary, coherence="c_v"
    ).get_coherence()


def sweep_k(corpus, dictionary: Dictionary, tokens):
    max_k = min(max(K_RANGE), max(2, len(tokens) // 5))
    k_values = [k for k in K_RANGE if k <= max_k]
    if not k_values:
        k_values = [2]

    best = None
    log: list[tuple[int, float]] = []
    for k in k_values:
        m = fit_lda(corpus, dictionary, k)
        cv = coherence(m, tokens, dictionary)
        log.append((k, cv))
        print(f"  K={k} c_v={cv:.3f}")
        if best is None or cv > best[1]:
            best = (m, cv, k)
    return best[0], best[2], log


def doc_topic_matrix(model: models.LdaModel, corpus) -> np.ndarray:
    k = model.num_topics
    rows = []
    for bow in corpus:
        dist = dict(model.get_document_topics(bow, minimum_probability=0.0))
        rows.append([dist.get(i, 0.0) for i in range(k)])
    return np.asarray(rows, dtype=float)


def topic_terms(model: models.LdaModel, topn: int = 15) -> dict[int, list[tuple[str, float]]]:
    out: dict[int, list[tuple[str, float]]] = {}
    for tid, terms in model.show_topics(num_topics=-1, num_words=topn, formatted=False):
        out[int(tid)] = [(w, float(p)) for w, p in terms]
    return out


def persist(
    company: str,
    model: models.LdaModel,
    dictionary: Dictionary,
    df: pd.DataFrame,
    topic_mat: np.ndarray,
    terms_map: dict,
    sweep_log: list[tuple[int, float]],
) -> Path:
    k = topic_mat.shape[1]
    missing_emo = [c for c in EMOTION_COLS if c not in df.columns]
    if missing_emo:
        raise ValueError(f"input CSV missing expected emotion columns: {missing_emo}")
    carry = [c for c in CARRY_COLS if c in df.columns] + EMOTION_COLS
    out_df = df[carry].copy()
    for i in range(k):
        out_df[f"topic_{i}"] = topic_mat[:, i]
    out_df["dominant_topic"] = topic_mat.argmax(axis=1)
    out_df["dominant_topic_prob"] = topic_mat.max(axis=1)

    csv_path  = TOPICS_DIR / f"posts_with_topics_{company}.csv"
    model_path = TOPICS_DIR / f"lda_model_{company}.pkl"
    dict_path  = TOPICS_DIR / f"lda_dictionary_{company}.pkl"
    json_path  = TOPICS_DIR / f"lda_topics_{company}.json"

    out_df.to_csv(csv_path, index=False)
    model.save(str(model_path))
    dictionary.save(str(dict_path))
    with open(json_path, "w") as f:
        json.dump(
            {
                "best_k": k,
                "sweep": [[int(kk), float(cv)] for kk, cv in sweep_log],
                "topics": {str(t): [[w, p] for w, p in terms] for t, terms in terms_map.items()},
            },
            f,
            indent=2,
        )
    return csv_path


def run_lda(company: str) -> Path:
    ensure_data_dirs()
    print(f"[lda:{company}] loading corpus")
    df = load_corpus(company)
    tokens = tokenize(df)
    dictionary, corpus = build_bow(tokens)
    print(f"[lda:{company}] docs={len(tokens)} vocab={len(dictionary)}")

    model, best_k, log = sweep_k(corpus, dictionary, tokens)
    print(f"[lda:{company}] best K={best_k}")

    terms = topic_terms(model)
    for tid, items in terms.items():
        print(f"  Topic {tid}: " + ", ".join(w for w, _ in items))

    mat = doc_topic_matrix(model, corpus)
    out = persist(company, model, dictionary, df, mat, terms, log)
    print(f"[lda:{company}] wrote {out}")
    return out
