from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AGGREGATED_DIR, TOPICS_DIR, ensure_data_dirs


INPUT_CSV_TMPL   = TOPICS_DIR / "posts_with_topics_{company}.csv"
TOPICS_JSON_TMPL = TOPICS_DIR / "lda_topics_{company}.json"
OUTPUT_JSON_TMPL = AGGREGATED_DIR / "topic_signals_{company}.json"

ID_COL, DATE_COL, TEXT_COL = "post_idx", "created_datetime", "bert_text"
SRC_COL, SENT_COL          = "source", "sentiment_label"
DOM_COL, DOM_PROB          = "dominant_topic", "dominant_topic_prob"

CONFIDENCE_MIN   = 0.40
BUCKET_TZ        = "America/New_York"
WEEK_RULE        = "W-MON"
WEEKLY_MIN_WEEKS = 3

SPIKE_WINDOW     = 7
SPIKE_Z          = 2.5
SPIKE_MIN_DAYS   = 14

COOC_TOP_PAIRS   = 5
COOC_MIN_SCORE   = 0.05

QUOTES_PER_TOPIC      = 5
QUOTE_MAX_CHARS       = 280
QUOTE_LEN_RANGE       = (20, 600)
QUOTE_EMOTION_BUCKETS = ["positive", "negative", "curious", "neutral_other"]

TOPN_EMOTIONS    = 5
TOPN_WORDS_CARRY = 10

EMOTION_COLS = [
    "emotion_admiration", "emotion_amusement", "emotion_anger", "emotion_annoyance",
    "emotion_approval", "emotion_caring", "emotion_confusion", "emotion_curiosity",
    "emotion_desire", "emotion_disappointment", "emotion_disapproval", "emotion_disgust",
    "emotion_embarrassment", "emotion_excitement", "emotion_fear", "emotion_gratitude",
    "emotion_grief", "emotion_joy", "emotion_love", "emotion_nervousness",
    "emotion_neutral", "emotion_optimism", "emotion_pride", "emotion_realization",
    "emotion_relief", "emotion_remorse", "emotion_sadness", "emotion_surprise",
]
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "excitement", "gratitude",
    "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}
POSITIVE_EMOTION_COLS = [f"emotion_{e}" for e in POSITIVE_EMOTIONS]
NEGATIVE_EMOTION_COLS = [f"emotion_{e}" for e in NEGATIVE_EMOTIONS]
NEUTRAL_IDX = EMOTION_COLS.index("emotion_neutral")


def load_inputs(company: str):
    csv_path  = Path(str(INPUT_CSV_TMPL).format(company=company))
    json_path = Path(str(TOPICS_JSON_TMPL).format(company=company))
    if not csv_path.exists():
        raise FileNotFoundError(f"run_lda first: {csv_path} missing")
    if not json_path.exists():
        raise FileNotFoundError(f"run_lda first: {json_path} missing")
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        topics_json = json.load(f)
    return df, topics_json


def prepare(df: pd.DataFrame):
    n_total = len(df)
    df = df.copy()
    df["dt"] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
    bad_dates = int(df["dt"].isna().sum())
    df = df.dropna(subset=["dt"])
    df_kept = df[df[DOM_PROB] >= CONFIDENCE_MIN].copy()
    dropped_low_conf = len(df) - len(df_kept)
    df_kept["date"] = df_kept["dt"].dt.tz_convert(BUCKET_TZ).dt.date
    if df_kept.empty:
        meta = {
            "total_docs": n_total, "kept_docs": 0,
            "dropped_low_conf": dropped_low_conf, "bad_dates": bad_dates,
            "date_min": None, "date_max": None, "span_days": 0,
        }
        return df_kept, meta
    date_min, date_max = df_kept["date"].min(), df_kept["date"].max()
    span_days = (date_max - date_min).days
    meta = {
        "total_docs": n_total,
        "kept_docs": len(df_kept),
        "dropped_low_conf": dropped_low_conf,
        "bad_dates": bad_dates,
        "date_min": date_min,
        "date_max": date_max,
        "span_days": span_days,
    }
    return df_kept, meta


def topic_emotions(df_kept: pd.DataFrame, t: int) -> dict | None:
    w = df_kept[f"topic_{t}"].to_numpy()
    if w.sum() <= 0:
        return None
    E = df_kept[EMOTION_COLS].to_numpy()
    mean = (w[:, None] * E).sum(0) / w.sum()
    mean_dict = {c: float(v) for c, v in zip(EMOTION_COLS, mean)}
    neutral_share = float(mean[NEUTRAL_IDX])
    non_neut = sorted(
        [(c, float(v)) for c, v in zip(EMOTION_COLS, mean) if c != "emotion_neutral"],
        key=lambda x: -x[1],
    )
    top_non_neutral = non_neut[:TOPN_EMOTIONS]
    pos = sum(v for c, v in zip(EMOTION_COLS, mean) if c[len("emotion_"):] in POSITIVE_EMOTIONS)
    neg = sum(v for c, v in zip(EMOTION_COLS, mean) if c[len("emotion_"):] in NEGATIVE_EMOTIONS)
    return {
        "mean": mean_dict,
        "top_non_neutral": top_non_neutral,
        "neutral_share": neutral_share,
        "dominant_non_neutral": top_non_neutral[0][0] if top_non_neutral else None,
        "valence": {
            "positive": float(pos),
            "negative": float(neg),
            "ambiguous": float(max(0.0, 1.0 - neutral_share - pos - neg)),
        },
    }


def topic_daily(df_topic: pd.DataFrame, date_min, date_max) -> list[dict]:
    if df_topic.empty or date_min is None:
        return []
    idx = pd.date_range(date_min, date_max, freq="D").date
    counts = df_topic.groupby("date").size().reindex(idx, fill_value=0)
    probs = df_topic.groupby("date")[DOM_PROB].mean().reindex(idx)
    return [
        {
            "date": str(d),
            "count": int(c),
            "mean_dom_prob": None if pd.isna(p) else round(float(p), 4),
        }
        for d, c, p in zip(idx, counts, probs)
    ]


def _topic_weekly_buckets(df_topic: pd.DataFrame, date_max):
    if df_topic.empty:
        return []
    grouped = (
        df_topic.set_index("dt")
        .tz_convert(BUCKET_TZ)
        .groupby(pd.Grouper(freq=WEEK_RULE))
    )
    out = []
    for week_start, sub in grouped:
        if sub.empty:
            continue
        wk_date = week_start.date()
        is_partial = (wk_date + timedelta(days=6)) > date_max
        emo_mean = sub[EMOTION_COLS].mean()
        non_neut = emo_mean.drop("emotion_neutral").sort_values(ascending=False)
        top_emotion = str(non_neut.index[0]) if not non_neut.empty else None
        out.append({
            "week_start": str(wk_date),
            "count": int(len(sub)),
            "top_emotion": top_emotion,
            "is_partial": bool(is_partial),
        })
    return out


def topic_wow(weekly_records: list[dict]) -> dict | None:
    full = [w for w in weekly_records if not w["is_partial"]]
    if len(full) < 2:
        return None
    prev_w, this_w = full[-2], full[-1]
    prev_c, this_c = prev_w["count"], this_w["count"]
    pct = (this_c - prev_c) / max(prev_c, 1)
    return {
        "this_week_start": this_w["week_start"],
        "prev_week_start": prev_w["week_start"],
        "this_count": this_c,
        "prev_count": prev_c,
        "pct_change": round(float(pct), 4),
        "dominant_emotion_now": this_w["top_emotion"],
        "dominant_emotion_prev": prev_w["top_emotion"],
        "emotion_shifted": this_w["top_emotion"] != prev_w["top_emotion"],
    }


def detect_spikes(daily_records: list[dict], span_days: int) -> list[dict]:
    if span_days < SPIKE_MIN_DAYS or not daily_records:
        return []
    idx = pd.to_datetime([r["date"] for r in daily_records])
    s = pd.Series([r["count"] for r in daily_records], index=idx)
    wd_mask = s.index.dayofweek < 5
    s_wd = s[wd_mask]
    if s_wd.empty:
        return []
    mu = s_wd.shift(1).rolling(SPIKE_WINDOW, min_periods=5).mean()
    sd = s_wd.shift(1).rolling(SPIKE_WINDOW, min_periods=5).std(ddof=0).replace(0, np.nan)
    z = (s_wd - mu) / sd
    spikes = []
    for d, zv in z.items():
        if pd.notna(zv) and zv >= SPIKE_Z:
            spikes.append({
                "date": str(d.date()),
                "count": int(s_wd.loc[d]),
                "baseline_mean": round(float(mu.loc[d]), 3),
                "baseline_std": round(float(sd.loc[d]), 3),
                "z": round(float(zv), 3),
            })
    return spikes


def cooccurrence(df_kept: pd.DataFrame, k: int) -> dict:
    if df_kept.empty or k < 2:
        return {"mode": "soft_within_doc_cosine",
                "note": "cosine over soft topic-prob columns; diagonal omitted",
                "top_pairs": []}
    P = df_kept[[f"topic_{i}" for i in range(k)]].to_numpy(dtype=float)
    G = P.T @ P
    norm = np.sqrt(np.diag(G).clip(min=1e-12))
    M = G / np.outer(norm, norm)
    pairs = sorted(
        ((i, j, float(M[i, j])) for i in range(k) for j in range(i + 1, k)),
        key=lambda x: -x[2],
    )
    top = [
        {"a": int(i), "b": int(j), "score": round(s, 4)}
        for i, j, s in pairs if s >= COOC_MIN_SCORE
    ][:COOC_TOP_PAIRS]
    return {
        "mode": "soft_within_doc_cosine",
        "note": "cosine over soft topic-prob columns; diagonal omitted",
        "top_pairs": top,
    }


def _clean_quote_text(s: str) -> str:
    s = html.unescape(str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _row_to_quote(row, bucket: str) -> dict:
    text = _clean_quote_text(row[TEXT_COL])
    if len(text) > QUOTE_MAX_CHARS:
        text = text[: QUOTE_MAX_CHARS - 1].rstrip() + "…"
    sent = row.get(SENT_COL)
    if isinstance(sent, float) and pd.isna(sent):
        sent = None
    elif isinstance(sent, str) and sent.strip() == "":
        sent = None
    return {
        "post_idx": str(row[ID_COL]),
        "date": str(row["date"]),
        "bucket": bucket,
        "sentiment_label": sent,
        "dom_prob": round(float(row[DOM_PROB]), 4),
        "text": text,
    }


def pick_quotes(df_topic: pd.DataFrame) -> list[dict]:
    if df_topic.empty:
        return []
    sub = df_topic.copy()
    sub["_clean"] = sub[TEXT_COL].astype(str).map(_clean_quote_text)
    lo, hi = QUOTE_LEN_RANGE
    sub = sub[sub["_clean"].str.len().between(lo, hi)]
    if sub.empty:
        return []
    sub["pos_score"] = sub[POSITIVE_EMOTION_COLS].sum(axis=1)
    sub["neg_score"] = sub[NEGATIVE_EMOTION_COLS].sum(axis=1)
    sub["curi_score"] = sub["emotion_curiosity"] + sub["emotion_confusion"]
    sub["bucket"] = np.select(
        [sub["pos_score"] > 0.3, sub["neg_score"] > 0.3, sub["curi_score"] > 0.2],
        ["positive", "negative", "curious"],
        default="neutral_other",
    )

    picked: list[dict] = []
    seen_ids: set[str] = set()
    for bucket in QUOTE_EMOTION_BUCKETS:
        cand = sub[sub["bucket"] == bucket].sort_values(DOM_PROB, ascending=False)
        for _, row in cand.iterrows():
            if row[ID_COL] in seen_ids:
                continue
            picked.append(_row_to_quote(row, bucket))
            seen_ids.add(row[ID_COL])
            break
        if len(picked) >= QUOTES_PER_TOPIC:
            return picked

    fallback = sub.sort_values(DOM_PROB, ascending=False)
    for _, row in fallback.iterrows():
        if len(picked) >= QUOTES_PER_TOPIC:
            break
        if row[ID_COL] in seen_ids:
            continue
        picked.append(_row_to_quote(row, str(row["bucket"])))
        seen_ids.add(row[ID_COL])
    return picked


def _by_source(df_topic: pd.DataFrame) -> list[dict]:
    if df_topic.empty:
        return []
    total = len(df_topic)
    out = []
    for src, sub in df_topic.groupby(SRC_COL):
        out.append({
            "source": str(src),
            "volume": int(len(sub)),
            "share": round(len(sub) / total, 4),
        })
    out.sort(key=lambda r: -r["volume"])
    return out


def _by_sentiment_label(df_topic: pd.DataFrame) -> dict:
    if df_topic.empty:
        return {"bullish": 0, "bearish": 0, "unlabeled": 0, "labeled_bullish_share": None}
    lbl = df_topic[SENT_COL].astype("object").where(df_topic[SENT_COL].notna(), "")
    lbl = lbl.astype(str).str.strip().str.lower()
    bullish = int((lbl == "bullish").sum())
    bearish = int((lbl == "bearish").sum())
    unlabeled = int(len(df_topic) - bullish - bearish)
    denom = max(bullish + bearish, 1)
    share = bullish / denom if (bullish + bearish) > 0 else None
    return {
        "bullish": bullish,
        "bearish": bearish,
        "unlabeled": unlabeled,
        "labeled_bullish_share": None if share is None else round(share, 4),
    }


def build_payload(company: str, df_kept: pd.DataFrame, topics_json: dict, meta: dict) -> dict:
    k = int(topics_json.get("best_k", len(topics_json.get("topics", {}))))
    weekly_emitted = False
    spikes_emitted = False
    n_weeks_observed = 0

    if not df_kept.empty:
        weekly_buckets_all = df_kept.set_index("dt").tz_convert(BUCKET_TZ).resample(WEEK_RULE)
        n_weeks_observed = int(sum(1 for _, s in weekly_buckets_all if len(s) > 0))
        weekly_emitted = n_weeks_observed >= WEEKLY_MIN_WEEKS
        spikes_emitted = meta["span_days"] >= SPIKE_MIN_DAYS

    topics_payload: list[dict] = []
    topic_words_map = topics_json.get("topics", {})

    for t_str, words in topic_words_map.items():
        t = int(t_str)
        df_topic = df_kept[df_kept[DOM_COL] == t] if not df_kept.empty else df_kept
        volume = int(len(df_topic))
        share_kept = (volume / meta["kept_docs"]) if meta["kept_docs"] else 0.0
        mean_dom_prob = float(df_topic[DOM_PROB].mean()) if volume else None

        emotions = topic_emotions(df_kept, t) if not df_kept.empty else None
        daily = topic_daily(df_topic, meta["date_min"], meta["date_max"])
        weekly = _topic_weekly_buckets(df_topic, meta["date_max"]) if weekly_emitted else []
        wow = topic_wow(weekly) if weekly_emitted else None
        spikes = detect_spikes(daily, meta["span_days"]) if spikes_emitted else []
        quotes = pick_quotes(df_topic)

        topics_payload.append({
            "topic_id": t,
            "top_words": [[w, round(float(p), 4)] for w, p in words[:TOPN_WORDS_CARRY]],
            "volume": volume,
            "share_kept": round(share_kept, 4),
            "mean_dom_prob": None if mean_dom_prob is None else round(mean_dom_prob, 4),
            "emotions": emotions,
            "by_source": _by_source(df_topic),
            "sentiment_label": _by_sentiment_label(df_topic),
            "daily": daily,
            "weekly": weekly,
            "wow": wow,
            "spikes": spikes,
            "quotes": quotes,
        })

    topics_payload.sort(key=lambda x: -x["volume"])

    cooc = cooccurrence(df_kept, k)

    lda_meta = {"best_k": k}
    sweep = topics_json.get("sweep") or []
    for kk, cv in sweep:
        if int(kk) == k:
            lda_meta["best_cv"] = round(float(cv), 4)
            break

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    return {
        "company": company,
        "generated_at": generated_at,
        "config": {
            "confidence_min": CONFIDENCE_MIN,
            "bucket_tz": BUCKET_TZ,
            "week_rule": WEEK_RULE,
            "spike_z": SPIKE_Z,
            "spike_window": SPIKE_WINDOW,
        },
        "run": {
            "k": k,
            "total_docs": meta["total_docs"],
            "kept_docs": meta["kept_docs"],
            "dropped_low_conf": meta["dropped_low_conf"],
            "bad_dates": meta["bad_dates"],
            "date_min": None if meta["date_min"] is None else str(meta["date_min"]),
            "date_max": None if meta["date_max"] is None else str(meta["date_max"]),
            "span_days": meta["span_days"],
            "n_weeks_observed": n_weeks_observed,
            "weekly_emitted": weekly_emitted,
            "spikes_emitted": spikes_emitted,
            "lda": lda_meta,
        },
        "topics": topics_payload,
        "cooccurrence": cooc,
    }


def persist(payload: dict, company: str) -> Path:
    out = Path(str(OUTPUT_JSON_TMPL).format(company=company))
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def run_aggregate(company: str) -> Path:
    ensure_data_dirs()
    print(f"[aggregate:{company}] loading inputs")
    df, topics_json = load_inputs(company)
    df_kept, meta = prepare(df)
    print(
        f"[aggregate:{company}] total={meta['total_docs']} kept={meta['kept_docs']} "
        f"dropped_low_conf={meta['dropped_low_conf']} bad_dates={meta['bad_dates']} "
        f"span_days={meta['span_days']}"
    )
    if df_kept.empty:
        print(f"[aggregate:{company}] WARNING: no docs survived confidence filter")
    payload = build_payload(company, df_kept, topics_json, meta)
    out = persist(payload, company)
    print(f"[aggregate:{company}] wrote {out}")
    return out
