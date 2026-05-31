from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from src.config import AGGREGATED_DIR, BRIEFS_DIR, ensure_data_dirs, get_env

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:
    raise ImportError(
        "google-genai not installed. Run: pip install google-genai"
    ) from e


INPUT_TMPL  = AGGREGATED_DIR / "topic_signals_{company}.json"
OUTPUT_TMPL = BRIEFS_DIR / "brief_{company}_{date}.json"
CACHE_DIR   = BRIEFS_DIR / ".cache"
CACHE_TMPL  = CACHE_DIR / "labels_{company}.json"

GEMINI_MODEL    = "gemini-2.5-flash-lite"
API_KEY_VAR     = "GEMINI_API_KEY"
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3
RETRY_BASE_S    = 5.0

LABEL_MIN_WORDS      = 3
LABEL_MAX_WORDS      = 6
TOP_WORDS_FOR_PROMPT = 10
QUOTES_FOR_PROMPT    = 3
DESCRIPTION_MAX_WORDS = 60
TEMPERATURE          = 0.3

COMPANY_DISPLAY = {
    "chipotle":  "Chipotle (CMG)",
    "starbucks": "Starbucks (SBUX)",
    "tesla":     "Tesla (TSLA)",
}


def _pretty_emotion(name: str | None) -> str:
    if not name:
        return "n/a"
    return name.removeprefix("emotion_")


def load_signals(company: str) -> dict:
    path = Path(str(INPUT_TMPL).format(company=company))
    if not path.exists():
        raise FileNotFoundError(f"run_aggregate first: {path} missing")
    with open(path) as f:
        return json.load(f)


def load_cache(company: str) -> dict:
    path = Path(str(CACHE_TMPL).format(company=company))
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[brief:{company}] cache file corrupt; ignoring")
        return {}


def save_cache(company: str, cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(str(CACHE_TMPL).format(company=company))
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def hash_topic(top_words) -> str:
    words = sorted([w for w, _ in top_words[:TOP_WORDS_FOR_PROMPT]])
    h = hashlib.sha256(json.dumps(words).encode("utf-8")).hexdigest()
    return h[:16]


def build_prompt(company: str, topic: dict) -> str:
    display = COMPANY_DISPLAY.get(company, company.title())
    top_words = topic.get("top_words", [])[:TOP_WORDS_FOR_PROMPT]
    top_words_csv = ", ".join(w for w, _ in top_words)

    emo = topic.get("emotions") or {}
    dominant = _pretty_emotion(emo.get("dominant_non_neutral"))
    neutral_share = emo.get("neutral_share", 0.0) or 0.0
    valence = emo.get("valence") or {}
    pos = valence.get("positive", 0.0) or 0.0
    neg = valence.get("negative", 0.0) or 0.0

    volume = topic.get("volume", 0)
    share = topic.get("share_kept", 0.0) or 0.0

    lines = [
        f"You are labeling a discovery topic from {volume} StockTwits posts about {display}.",
        "",
        f"LDA top words: {top_words_csv}",
        f"Most prominent expressed feeling: {dominant}",
        f"Neutral share: {neutral_share*100:.0f}%",
        f"Valence: positive {pos*100:.0f}%, negative {neg*100:.0f}%",
        f"Volume: {volume} posts ({share*100:.1f}% of kept discussion)",
    ]

    wow = topic.get("wow")
    if wow:
        pct = wow.get("pct_change", 0.0) or 0.0
        prev = _pretty_emotion(wow.get("dominant_emotion_prev"))
        now = _pretty_emotion(wow.get("dominant_emotion_now"))
        shifted = wow.get("emotion_shifted")
        shift_note = (
            f"dominant feeling shifted from {prev} to {now}"
            if shifted else f"dominant feeling remained {now}"
        )
        sign = "+" if pct >= 0 else ""
        lines.append(f"Week-over-week: {sign}{pct*100:.1f}% volume change; {shift_note}.")

    quotes = topic.get("quotes") or []
    if quotes:
        lines.append("")
        lines.append("Example posts (representative, lightly cleaned):")
        for i, q in enumerate(quotes[:QUOTES_FOR_PROMPT], 1):
            text = (q.get("text") or "").replace("\n", " ").strip()
            lines.append(f"{i}. {text}")

    lines.extend([
        "",
        "Return JSON with two fields:",
        f'- "label": a {LABEL_MIN_WORDS}-{LABEL_MAX_WORDS} word noun phrase suitable as a chart title. '
        'Be specific to the actual content. Example good: "EV pricing concerns". '
        'Example bad: "Discussion", "Comments", "General chatter".',
        '- "description": 1-2 sentences explaining what the topic is about and how '
        'investors/customers are reacting. Reference the prominent non-neutral feeling, NOT "neutral".',
        "",
        "Constraints:",
        '- Do not describe the dominant emotion as "neutral".',
        "- Do not use generic labels.",
        f"- Keep label under {LABEL_MAX_WORDS} words; keep description under {DESCRIPTION_MAX_WORDS} words total.",
    ])
    return "\n".join(lines)


def _label_schema():
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "label": genai_types.Schema(type=genai_types.Type.STRING),
            "description": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["label", "description"],
    )


def call_gemini(client, prompt: str) -> dict:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_label_schema(),
        temperature=TEMPERATURE,
        http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
    )
    delay = RETRY_BASE_S
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            return json.loads(resp.text)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            print(f"  gemini attempt {attempt+1} failed: {type(e).__name__}: {e}; backing off {delay}s")
            time.sleep(delay)
            delay *= 2
    assert last_err is not None
    raise last_err


def label_topic(client, company: str, topic: dict, cache: dict) -> tuple[dict, str]:
    h = hash_topic(topic.get("top_words", []))
    if h in cache:
        entry = cache[h]
        return {"label": entry["label"], "description": entry["description"]}, "cache_hit"

    prompt = build_prompt(company, topic)
    try:
        pair = call_gemini(client, prompt)
    except Exception as e:
        print(f"  topic {topic.get('topic_id')} failed after retries: {type(e).__name__}: {e}")
        return {"label": f"Topic {topic.get('topic_id')}",
                "description": "Label generation failed."}, "failure"

    label = (pair.get("label") or "").strip()
    description = (pair.get("description") or "").strip()
    if not label or not description:
        return {"label": f"Topic {topic.get('topic_id')}",
                "description": "Label generation failed."}, "failure"

    cache[h] = {
        "label": label,
        "description": description,
        "model": GEMINI_MODEL,
        "cached_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "topic_id": topic.get("topic_id"),
        "top_words_preview": [w for w, _ in (topic.get("top_words") or [])[:5]],
    }
    return {"label": label, "description": description}, "cache_miss"


def assemble_brief(company: str, signals: dict, labeled_topics: list[dict],
                   gemini_stats: dict) -> dict:
    now = datetime.now().astimezone()
    return {
        "company": company,
        "brief_date": now.date().isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        "gemini": {
            "model": GEMINI_MODEL,
            **gemini_stats,
        },
        "source_signals_path": str(Path(str(INPUT_TMPL).format(company=company))),
        "config": signals.get("config", {}),
        "run":    signals.get("run", {}),
        "topics": labeled_topics,
        "cooccurrence": signals.get("cooccurrence", {}),
    }


def persist(brief: dict, company: str) -> Path:
    date_str = brief["brief_date"].replace("-", "")
    out = Path(str(OUTPUT_TMPL).format(company=company, date=date_str))
    with open(out, "w") as f:
        json.dump(brief, f, indent=2)
    return out


def _make_client() -> "genai.Client":
    api_key = get_env(API_KEY_VAR)
    if not api_key:
        raise RuntimeError(
            f"set {API_KEY_VAR} in .env (see .env.example) — get a free key at "
            f"https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def run_brief(company: str) -> Path:
    ensure_data_dirs()
    print(f"[brief:{company}] loading signals")
    signals = load_signals(company)
    cache = load_cache(company)

    topics_in = signals.get("topics", [])
    if not topics_in:
        print(f"[brief:{company}] WARNING: no topics in signals")
        brief = assemble_brief(company, signals, [], {
            "topic_calls": 0, "cache_hits": 0, "cache_misses": 0, "failures": 0,
        })
        out = persist(brief, company)
        print(f"[brief:{company}] wrote {out}")
        return out

    client = _make_client()
    labeled: list[dict] = []
    hits = misses = failures = empty = 0
    for topic in topics_in:
        if topic.get("volume", 0) == 0:
            empty += 1
            labeled.append({
                **topic,
                "label": f"Topic {topic.get('topic_id')} (no posts)",
                "description": "",
            })
            continue
        pair, status = label_topic(client, company, topic, cache)
        if status == "cache_hit":
            hits += 1
        elif status == "cache_miss":
            misses += 1
        else:
            failures += 1
        labeled.append({**topic, **pair})
        print(f"  topic {topic.get('topic_id')} [{status}]: {pair['label']}")

    save_cache(company, cache)

    stats = {
        "topic_calls": misses + failures,
        "cache_hits": hits,
        "cache_misses": misses,
        "failures": failures,
        "empty_topics_skipped": empty,
    }
    brief = assemble_brief(company, signals, labeled, stats)
    out = persist(brief, company)
    print(f"[brief:{company}] wrote {out}; stats={stats}")
    return out
