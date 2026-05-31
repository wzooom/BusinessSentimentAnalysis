import json
import os
import sys
import tempfile
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, Table, TableStyle, PageBreak,
                                KeepTogether, HRFlowable)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

from src.style import (PALETTE, TYPE, FONT_BODY, FONT_BOLD, PAGE_MARGIN, CONTENT_WIDTH,
                       display_name, company_kind, strip_emotion)
from src import charts
from src import prompts
from src.gemini_client import GeminiClient
from src.config import BRIEFS_DIR, REPORTS_DIR, ensure_data_dirs


def _styles():
    ss = getSampleStyleSheet()
    ink, soft = HexColor(PALETTE["ink"]), HexColor(PALETTE["ink_soft"])
    accent = HexColor(PALETTE["accent"])

    def mk(name, **kw):
        kw.setdefault("fontName", FONT_BODY)
        kw.setdefault("textColor", ink)
        return ParagraphStyle(name, parent=ss["Normal"], **kw)

    return {
        "cover_title": mk("cover_title", fontName=FONT_BOLD, fontSize=TYPE["cover_title"],
                          leading=TYPE["cover_title"] * 1.1, textColor=accent),
        "cover_sub": mk("cover_sub", fontSize=TYPE["cover_sub"], leading=TYPE["cover_sub"] * 1.4,
                        textColor=soft),
        "h1": mk("h1", fontName=FONT_BOLD, fontSize=TYPE["h1"], leading=TYPE["h1"] * 1.2,
                 textColor=accent, spaceBefore=4, spaceAfter=8),
        "h2": mk("h2", fontName=FONT_BOLD, fontSize=TYPE["h2"], leading=TYPE["h2"] * 1.25,
                 textColor=ink, spaceBefore=14, spaceAfter=2),
        "scale": mk("scale", fontSize=TYPE["caption"], leading=TYPE["caption"] * 1.3,
                    textColor=soft, spaceAfter=6),
        "body": mk("body", fontSize=TYPE["body"], leading=TYPE["body"] * 1.45,
                   alignment=TA_LEFT, spaceAfter=8),
        "quote": mk("quote", fontSize=TYPE["quote"], leading=TYPE["quote"] * 1.4,
                    textColor=ink, leftIndent=10, rightIndent=10),
        "quote_attr": mk("quote_attr", fontSize=TYPE["caption"], textColor=soft,
                         leftIndent=10, spaceBefore=2),
        "method": mk("method", fontSize=TYPE["caption"] + 0.5, leading=TYPE["caption"] * 1.6,
                     textColor=soft),
    }


class _Doc(BaseDocTemplate):
    def __init__(self, path, footer_text, **kw):
        super().__init__(path, pagesize=letter,
                         leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
                         topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN + 12, **kw)
        self.footer_text = footer_text
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                            onPage=self._footer)])

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT_BODY, TYPE["footer"])
        canvas.setFillColor(HexColor(PALETTE["muted"]))
        canvas.setStrokeColor(HexColor(PALETTE["rule"]))
        y = self.bottomMargin - 6
        canvas.line(self.leftMargin, y + 10, self.leftMargin + self.width, y + 10)
        canvas.drawString(self.leftMargin, y, self.footer_text)
        canvas.drawRightString(self.leftMargin + self.width, y, f"Page {doc.page}")
        canvas.restoreState()


def _img(path, max_w=CONTENT_WIDTH):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = min(max_w, iw)
    return Image(path, width=w, height=w * ih / iw)


def _quote_block(q, S):
    text = (q.get("text") or "").strip()
    if not text:
        return None
    stance = q.get("sentiment_label")
    who = "an investor" if stance in ("Bullish", "Bearish") else "a customer"
    body = Paragraph(f'“{text}”', S["quote"])
    attr = Paragraph(f"— {who} on StockTwits", S["quote_attr"])
    tbl = Table([[body], [attr]], colWidths=[CONTENT_WIDTH])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(PALETTE["panel"])),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, HexColor(PALETTE["accent_2"])),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
    ]))
    return tbl


def _select_pull_quotes(topic, n=2):
    quotes = sorted(topic.get("quotes") or [],
                    key=lambda q: q.get("dom_prob", 0), reverse=True)
    seen, out = set(), []
    for q in quotes:
        t = (q.get("text") or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(q)
        if len(out) >= n:
            break
    return out


def generate_report(brief_path, out_path=None, client=None, chart_dir=None):
    with open(brief_path, "r", encoding="utf-8") as f:
        brief = json.load(f)

    company = brief.get("company", "company")
    name = display_name(company)
    run = brief.get("run", {}) or {}
    topics = brief.get("topics", []) or []
    n_posts = run.get("kept_docs", run.get("total_docs", 0))
    d0 = prompts._fmt_date(run.get("date_min"))
    d1 = prompts._fmt_date(run.get("date_max"))
    has_ts = bool(run.get("weekly_emitted"))

    client = client or GeminiClient()
    chart_dir = chart_dir or tempfile.mkdtemp(prefix="sentiment_charts_")
    os.makedirs(chart_dir, exist_ok=True)
    S = _styles()

    summary_fb = prompts.build_summary_factblock(brief)
    summary_prose = client.generate(
        prompts.SUMMARY_SYSTEM_PROMPT,
        prompts.factblock_to_user_turn(summary_fb),
        kind="summary")

    volume_png = charts.topic_volume_bar(
        topics, os.path.join(chart_dir, "volume.png"))

    story = []

    story += [Spacer(1, 2.0 * inch),
              Paragraph(f"{name}", S["cover_title"]),
              Spacer(1, 6),
              Paragraph("Customer &amp; Investor Sentiment Report", S["cover_sub"]),
              Spacer(1, 18),
              Paragraph(f"Reporting period: {d0} – {d1}", S["cover_sub"]),
              Paragraph(f"Based on {n_posts:,} public StockTwits posts", S["cover_sub"]),
              Paragraph(f"Prepared {prompts._fmt_date(run.get('date_max'))}", S["cover_sub"]),
              PageBreak()]

    story += [Paragraph("Executive Summary", S["h1"]),
              HRFlowable(width="100%", thickness=1, color=HexColor(PALETTE["rule"]),
                         spaceBefore=2, spaceAfter=10),
              Paragraph(summary_prose, S["body"]),
              Spacer(1, 10),
              _img(volume_png),
              Spacer(1, 6)]

    story.append(Paragraph("Themes in the Conversation", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor(PALETTE["rule"]),
                            spaceBefore=2, spaceAfter=4))

    for i, topic in enumerate(topics, 1):
        topic_fb = prompts.build_topic_factblock(topic, company)
        narrative = client.generate(
            prompts.TOPIC_SYSTEM_PROMPT,
            prompts.factblock_to_user_turn(topic_fb),
            kind="topic")

        share = topic.get("share_kept", 0) * 100
        vol = topic.get("volume", 0)
        scale = f"{vol:,} posts · {share:.0f}% of conversation"

        ts_png = (charts.weekly_timeseries(
            topic, os.path.join(chart_dir, f"ts_{i}.png")) if has_ts else None)
        fp_png = charts.emotion_fingerprint(
            topic, os.path.join(chart_dir, f"fp_{i}.png"))

        head = [Paragraph(f"{i}. {topic.get('label', f'Theme {i}')}", S["h2"]),
                Paragraph(scale, S["scale"])]
        if ts_png:
            head.append(_img(ts_png))
        story.append(KeepTogether(head))

        story += [Paragraph(narrative, S["body"])]
        if fp_png:
            story += [_img(fp_png), Spacer(1, 4)]

        for q in _select_pull_quotes(topic, n=2):
            blk = _quote_block(q, S)
            if blk:
                story += [blk, Spacer(1, 6)]

        story.append(Spacer(1, 6))

    story += [PageBreak(),
              Paragraph("Methodology &amp; Notes", S["h1"]),
              HRFlowable(width="100%", thickness=1, color=HexColor(PALETTE["rule"]),
                         spaceBefore=2, spaceAfter=10)]
    story += _methodology_flowables(brief, S, n_posts, d0, d1, has_ts)

    if out_path is None:
        ensure_data_dirs()
        date_compact = (run.get("date_max", "") or "").replace("-", "")
        out_path = str(REPORTS_DIR / f"report_{company}_{date_compact}.pdf")
    footer = f"{name} Sentiment Report · {d0}–{d1} · Source: StockTwits"
    doc = _Doc(out_path, footer_text=footer)
    doc.build(story)
    return out_path


def _methodology_flowables(brief, S, n_posts, d0, d1, has_ts):
    cfg = brief.get("config", {}) or {}
    run = brief.get("run", {}) or {}
    conf = cfg.get("confidence_min", 0.4)

    lines = [
        f"This report summarizes {n_posts:,} public posts about "
        f"{display_name(brief.get('company',''))} on StockTwits between {d0} and {d1}. "
        "Posts were grouped into recurring discussion themes, and each post was scored "
        "for emotional tone across 28 categories using an automated language model.",

        "Theme narratives and the executive summary are generated from the underlying "
        "figures by an automated assistant and reviewed for accuracy against the source "
        "metrics. All quotations are reproduced verbatim from public posts; individual "
        "users are never identified.",

        f"Posts below a confidence threshold of {conf:.0%} were excluded. "
        "Bullish/bearish figures reflect only the posts where an author tagged a stance, "
        "and are noted as a minority signal where that sample is small. Emotional tone is "
        "reported using the most prominent expressed (non-neutral) feelings; a large, "
        "uninformative neutral share is excluded by design.",
    ]
    if has_ts:
        spike_z = cfg.get("spike_z", 2.5)
        lines.append(
            "Week-over-week comparisons use the two most recent complete calendar weeks "
            "(Eastern Time). Spikes flag single days whose volume is unusually high "
            f"relative to the prior week's daily average (threshold: {spike_z} standard "
            "deviations), and are reported as a multiple of the normal level.")
    else:
        lines.append(
            "Trend, week-over-week, and spike analyses are omitted from this edition "
            "because the available history is too short to support them; they will appear "
            "once a longer collection window is in place.")

    co = (brief.get("cooccurrence", {}) or {}).get("top_pairs") or []
    if not co:
        lines.append("No significant cross-theme overlap was detected; the themes are "
                     "largely distinct.")

    out = []
    for ln in lines:
        out.append(Paragraph(ln, S["method"]))
        out.append(Spacer(1, 6))
    return out


def _latest_brief_for(company: str) -> Path:
    pattern = f"brief_{company}_*.json"
    candidates = sorted(BRIEFS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"no brief found for {company!r} (looked for {BRIEFS_DIR}/{pattern}); "
            f"run_brief first")
    return candidates[-1]


def run_report(company: str) -> Path:
    ensure_data_dirs()
    brief_path = _latest_brief_for(company)
    print(f"[report:{company}] using brief {brief_path}")
    out = generate_report(str(brief_path))
    print(f"[report:{company}] wrote {out}")
    return Path(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.report <brief.json> [out.pdf]")
        sys.exit(1)
    out = generate_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"wrote {out}")
