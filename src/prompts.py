import json
from datetime import date, datetime, timedelta

from src.style import strip_emotion, company_kind, display_name


TOPIC_SYSTEM_PROMPT = """\
You are a financial-communications analyst writing ONE section of an executive \
sentiment report. Each section covers a single theme found in public StockTwits \
posts about one company.

READER: a senior executive at the company. Non-technical. Write in a concise, \
businesslike register. Never use data-science jargon — do not write "topic", \
"cluster", "LDA", "coherence", "model", or "K". Say "theme" or describe the \
subject matter directly.

INPUT: a JSON fact block for ONE theme. Every figure is already computed. Use the \
provided numbers verbatim; never recalculate, estimate, round differently, or \
invent any figure.

LENGTH & SHAPE: write exactly 2-3 sentences, 45-80 words, in this content order:
1. What the theme is about and its scale — use `volume` and `share_pct` (its \
share of the conversation).
2. The most important dynamic, when the data is present: the week-over-week change \
(`wow.pct_change_pct`) and, if `wow.emotion_shifted` is true, the pivot from \
`wow.emotion_prev` to `wow.emotion_now`; and/or the largest spike stated as \
`spike.multiple`x its normal level.
3. When `quote` is present, close by anchoring with that quote VERBATIM in quotation \
marks, attributed only as "a customer on StockTwits" or "an investor on StockTwits" \
(use `quote.kind`).

DATA RULES (these prevent embarrassing errors):
- EMOTION: neutral always dominates the emotion mass and is never newsworthy. \
Never say the theme "feels neutral" or name neutral as the main emotion. Refer to \
`dominant_non_neutral` / `top_non_neutral` only, and phrase it as "the most \
prominent expressed feeling", not "the theme is optimistic".
- BULL/BEAR: if `bullish_share_pct` is present, you may state it as the share of \
posts that tagged a stance. If `labeled_total` < 30, caveat it ("among the small \
share who tagged a position…"). If `company_kind` is "operating", mention bull/bear \
only briefly or omit it — these readers don't think in bull/bear terms. If \
`company_kind` is "investor", treat it as a primary signal. If the field is absent, \
omit stance language entirely.
- QUOTES: reproduce `quote.text` exactly, in quotation marks. Never paraphrase, \
never merge two quotes, never attribute to a named person. If no quote is given, \
write the section without one.
- COMPETITORS: names in `top_words` such as Cava or Sweetgreen are real signal — \
surface them when relevant. The company's OWN name never appears in `top_words`; \
its absence is meaningless, never remark on it.
- MISSING FIELDS: if a field is absent or null (common when a company has no \
time-series data yet), simply omit that element. Never write "no data", "N/A", \
"not available", or otherwise note an absence.

OUTPUT: only the section prose. No heading, no list, no preamble, no emojis.

--- EXAMPLE 1 (operating company, full time series) ---
INPUT:
{"label":"Pre-earnings positioning split","company_kind":"operating","volume":219,\
"share_pct":19.0,"top_words":["week","good","today","cava","earnings"],\
"dominant_non_neutral":"optimism","top_non_neutral":["optimism","curiosity","admiration"],\
"bullish_share_pct":78,"labeled_total":100,\
"wow":{"pct_change_pct":64,"emotion_shifted":true,"emotion_prev":"admiration","emotion_now":"confusion"},\
"spike":{"date":"April 22","multiple":4.6},\
"quote":{"text":"$CMG puts or calls? What's the sentiment for earnings. I think we drop.","kind":"investor"}}
OUTPUT:
Pre-earnings positioning is the largest theme in the conversation at roughly 19% of \
posts, split between optimism on technical levels and unease about fundamentals, \
with rival Cava recurring as a comparison point. Volume rose 64% week-over-week and \
the prevailing feeling pivoted from admiration to confusion, while a single day on \
April 22 drew 4.6x its normal posting — a clear surge of attention into earnings. \
As one investor on StockTwits put it, "$CMG puts or calls? What's the sentiment for \
earnings. I think we drop."

--- EXAMPLE 2 (investor company, NO time series) ---
INPUT:
{"label":"Delivery-number debate","company_kind":"investor","volume":64,"share_pct":22.0,\
"top_words":["delivery","q2","numbers","guidance","wait"],"dominant_non_neutral":"curiosity",\
"top_non_neutral":["curiosity","optimism"],"bullish_share_pct":41,"labeled_total":38,\
"wow":null,"spike":null,\
"quote":{"text":"Delivery miss already priced in or not? Nobody agrees.","kind":"investor"}}
OUTPUT:
Debate over upcoming delivery numbers is a leading theme at about 22% of the \
conversation, with curiosity the most prominent expressed feeling as traders await \
Q2 guidance. Sentiment is roughly balanced — 41% of posts that tagged a stance were \
bullish — pointing to genuine disagreement rather than conviction. One investor on \
StockTwits captured it: "Delivery miss already priced in or not? Nobody agrees."
"""


SUMMARY_SYSTEM_PROMPT = """\
You are a financial-communications analyst writing the opening EXECUTIVE SUMMARY of \
a sentiment report for one company, built from public StockTwits posts.

READER: a senior executive at the company. Non-technical. Concise, businesslike. \
No data-science jargon ("topic", "cluster", "LDA", "model", "K"); say "theme".

INPUT: a compact fact block for the whole report — company, date range, post count, \
the leading themes in order of volume, the largest week-over-week shift, the biggest \
spike, the overall valence balance, and (for investor companies) the bullish tilt. \
Every figure is pre-computed; use it verbatim and never recalculate or invent.

LENGTH & SHAPE: ONE paragraph, ~120 words (100-140), in roughly this order:
1. Open by naming the evidence base exactly: "Based on N StockTwits posts from \
<start> to <end>, ...". Then the headline — what dominates the conversation (lead \
with the first-listed, highest-volume themes) and the overall tone.
2. The most significant movement: the largest week-over-week volume change and any \
emotion pivot, plus the most extreme spike as a multiple of its normal level.
3. The valence balance, and for investor companies the bullish/bearish tilt.
4. A final action-oriented sentence: one concrete, data-justified implication for \
the executive (e.g. rising confusion into earnings -> a communications opportunity). \
Do not overstate beyond what the figures support.

RULES:
- Neutral emotion dominates every theme and is never newsworthy; speak only to the \
most prominent expressed (non-neutral) feelings.
- If `has_timeseries` is false, OMIT all trend, spike, and week-over-week language \
and instead summarize the standing picture (leading themes, tone, valence). Never \
note the absence of data.
- Themes are already ranked by importance (volume order); respect that order.
- No emojis, no marketing language, no headings, no lists. Output only the paragraph.

--- EXAMPLE (operating company, full data) ---
INPUT:
{"company":"Chipotle","company_kind":"operating","n_posts":1155,"date_start":"Feb 4, 2026",\
"date_end":"May 28, 2026","has_timeseries":true,\
"themes":[{"label":"Pre-earnings positioning split","share_pct":19,"feeling":"optimism"},\
{"label":"Food-safety memories resurfacing","share_pct":14,"feeling":"disgust"},\
{"label":"Cava competitive comparison","share_pct":11,"feeling":"curiosity"}],\
"biggest_wow":{"theme":"Pre-earnings positioning split","pct_change_pct":64,"emotion_shifted":true,"emotion_now":"confusion"},\
"top_spike":{"theme":"Food-safety memories resurfacing","multiple":4.6,"date":"April 22"},\
"valence":{"positive_pct":42,"negative_pct":31,"ambiguous_pct":27},"bullish_share_pct":71}
OUTPUT:
Based on 1,155 StockTwits posts from Feb 4 to May 28, 2026, the conversation is led \
by pre-earnings positioning (about 19% of posts), recurring food-safety memories \
(14%), and comparisons to Cava (11%) — an engaged but cautious audience. Momentum is \
concentrated in the pre-earnings theme, which rose 64% week-over-week as the mood \
shifted toward confusion, while food-safety chatter spiked to 4.6 times its normal \
level on April 22. Overall tone leans modestly positive (42% positive against 31% \
negative). With confusion building into earnings and old food-safety concerns \
resurfacing, communications should pre-empt both narratives with clear, proactive \
messaging rather than waiting to react.
"""


def _pct(x, digits=0):
    if x is None:
        return None
    return round(x * 100, digits) if digits else int(round(x * 100))


def _fmt_date(d):
    if d is None:
        return None
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    return d.strftime("%b %-d, %Y") if hasattr(d, "strftime") else str(d)


def _labeled_total(topic):
    sl = topic.get("sentiment_label") or {}
    return int(sl.get("bullish", 0)) + int(sl.get("bearish", 0))


def _pick_quote(topic, kind_default):
    quotes = topic.get("quotes") or []
    if not quotes:
        return None
    best = max(quotes, key=lambda q: q.get("dom_prob", 0.0))
    text = (best.get("text") or "").strip()
    if not text:
        return None
    has_stance = best.get("sentiment_label") in ("Bullish", "Bearish")
    kind = "investor" if has_stance else kind_default
    return {"text": text, "kind": kind}


def build_topic_factblock(topic: dict, company: str) -> dict:
    kind = company_kind(company)
    kind_default = "investor" if kind == "investor" else "customer"

    emo = topic.get("emotions", {}) or {}
    top_nn = [strip_emotion(e) for e, _ in (emo.get("top_non_neutral") or [])][:3]
    dom_nn = strip_emotion(emo.get("dominant_non_neutral") or "")

    fb = {
        "label": topic.get("label", ""),
        "company_kind": kind,
        "volume": topic.get("volume", 0),
        "share_pct": _pct(topic.get("share_kept"), digits=1),
        "top_words": [w for w, _ in (topic.get("top_words") or [])][:6],
        "dominant_non_neutral": dom_nn,
        "top_non_neutral": top_nn,
    }

    sl = topic.get("sentiment_label") or {}
    labeled_total = _labeled_total(topic)
    if labeled_total > 0 and sl.get("labeled_bullish_share") is not None:
        fb["bullish_share_pct"] = _pct(sl["labeled_bullish_share"])
        fb["labeled_total"] = labeled_total

    wow = topic.get("wow")
    if wow:
        fb["wow"] = {
            "pct_change_pct": _pct(wow.get("pct_change")),
            "emotion_shifted": bool(wow.get("emotion_shifted")),
            "emotion_prev": strip_emotion(wow.get("dominant_emotion_prev") or ""),
            "emotion_now": strip_emotion(wow.get("dominant_emotion_now") or ""),
        }
    else:
        fb["wow"] = None

    spikes = topic.get("spikes") or []
    if spikes:
        top = max(spikes, key=lambda s: s.get("z", 0))
        base = top.get("baseline_mean") or 0
        mult = round(top["count"] / base, 1) if base else None
        fb["spike"] = {"date": _fmt_date(top.get("date")), "multiple": mult}
    else:
        fb["spike"] = None

    fb["quote"] = _pick_quote(topic, kind_default)
    return fb


def build_summary_factblock(brief: dict) -> dict:
    company = brief.get("company", "")
    kind = company_kind(company)
    run = brief.get("run", {}) or {}
    topics = brief.get("topics", []) or []

    has_ts = bool(run.get("weekly_emitted"))

    themes = []
    for t in topics[:4]:
        emo = t.get("emotions", {}) or {}
        themes.append({
            "label": t.get("label", ""),
            "share_pct": _pct(t.get("share_kept")),
            "feeling": strip_emotion(emo.get("dominant_non_neutral") or ""),
        })

    fb = {
        "company": brief_display_name(company),
        "company_kind": kind,
        "n_posts": run.get("kept_docs", run.get("total_docs", 0)),
        "date_start": _fmt_date(run.get("date_min")),
        "date_end": _fmt_date(run.get("date_max")),
        "has_timeseries": has_ts,
        "themes": themes,
    }

    if has_ts:
        best_wow, best_mag = None, -1
        for t in topics:
            w = t.get("wow")
            if not w or w.get("pct_change") is None:
                continue
            mag = abs(w["pct_change"])
            if mag > best_mag:
                best_mag, best_wow = mag, (t, w)
        if best_wow:
            t, w = best_wow
            fb["biggest_wow"] = {
                "theme": t.get("label", ""),
                "pct_change_pct": _pct(w.get("pct_change")),
                "emotion_shifted": bool(w.get("emotion_shifted")),
                "emotion_now": strip_emotion(w.get("dominant_emotion_now") or ""),
            }

        best_spk, best_z = None, -1
        for t in topics:
            for s in (t.get("spikes") or []):
                if s.get("z", 0) > best_z:
                    base = s.get("baseline_mean") or 0
                    best_z = s["z"]
                    best_spk = {
                        "theme": t.get("label", ""),
                        "multiple": round(s["count"] / base, 1) if base else None,
                        "date": _fmt_date(s.get("date")),
                    }
        if best_spk:
            fb["top_spike"] = best_spk

    pos = neg = amb = wsum = 0.0
    for t in topics:
        v = t.get("volume", 0)
        val = (t.get("emotions", {}) or {}).get("valence", {}) or {}
        pos += val.get("positive", 0) * v
        neg += val.get("negative", 0) * v
        amb += val.get("ambiguous", 0) * v
        wsum += v
    if wsum:
        total = pos + neg + amb
        if total:
            fb["valence"] = {
                "positive_pct": int(round(pos / total * 100)),
                "negative_pct": int(round(neg / total * 100)),
                "ambiguous_pct": int(round(amb / total * 100)),
            }

    if kind == "investor":
        b = sum(int((t.get("sentiment_label") or {}).get("bullish", 0)) for t in topics)
        be = sum(int((t.get("sentiment_label") or {}).get("bearish", 0)) for t in topics)
        if (b + be) > 0:
            fb["bullish_share_pct"] = int(round(b / (b + be) * 100))

    return fb


def brief_display_name(company: str) -> str:
    return display_name(company)


def factblock_to_user_turn(fact_block: dict) -> str:
    return json.dumps(fact_block, ensure_ascii=False, separators=(",", ":"))
