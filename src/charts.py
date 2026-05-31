from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from src.style import (PALETTE, CHART_PT, FONT_BODY, emotion_color, strip_emotion,
                       VALENCE_COLOR, EMOTION_VALENCE)

DPI = 200


def apply_base_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": CHART_PT["tick"],
        "axes.titlesize": CHART_PT["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": CHART_PT["label"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["ink_soft"],
        "axes.titlecolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink_soft"],
        "ytick.color": PALETTE["ink_soft"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["rule"],
        "grid.linewidth": 0.8,
        "figure.facecolor": PALETTE["white"],
        "axes.facecolor": PALETTE["white"],
        "savefig.facecolor": PALETTE["white"],
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,
    })


apply_base_style()


def _save(fig, outpath):
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def _truncate(label, n=34):
    return label if len(label) <= n else label[: n - 1] + "…"


def topic_volume_bar(topics, outpath, width_in=6.6, bar_h=0.42):
    labels = [_truncate(t.get("label", f"Theme {i+1}")) for i, t in enumerate(topics)]
    vols = [t.get("volume", 0) for t in topics]

    labels, vols = labels[::-1], vols[::-1]
    height = max(2.2, len(vols) * (bar_h + 0.18) + 0.8)
    fig, ax = plt.subplots(figsize=(width_in, height))

    bars = ax.barh(labels, vols, color=PALETTE["accent"], height=bar_h, zorder=3)
    ax.set_title("Conversation volume by theme", loc="left", pad=12)
    ax.set_xlabel("Posts")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    vmax = max(vols) if vols else 1
    for bar, v in zip(bars, vols):
        ax.text(v + vmax * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", ha="left",
                fontsize=CHART_PT["annot"], color=PALETTE["ink_soft"])
    ax.set_xlim(0, vmax * 1.12)
    return _save(fig, outpath)


def _week_start(date_str, week_rule_monday=True):
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return d - timedelta(days=d.weekday())


def weekly_timeseries(topic, outpath, width_in=6.6, height_in=2.7):
    weekly = topic.get("weekly") or []
    if len(weekly) < 2:
        return None

    xs = [datetime.strptime(w["week_start"], "%Y-%m-%d").date() for w in weekly]
    ys = [w.get("count", 0) for w in weekly]
    marker_colors = [emotion_color(w.get("top_emotion", "neutral")) for w in weekly]
    partial = [bool(w.get("is_partial")) for w in weekly]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot(xs, ys, color=PALETTE["accent_2"], linewidth=1.8, zorder=2)

    for x, y, c, p in zip(xs, ys, marker_colors, partial):
        ax.scatter([x], [y], s=42, zorder=3,
                   facecolor=(PALETTE["white"] if p else c),
                   edgecolor=c, linewidth=1.6)

    ax.set_title("Weekly volume and prevailing feeling", loc="left", pad=10)
    ax.set_ylabel("Posts / week")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    spikes = topic.get("spikes") or []
    week_lookup = {x: y for x, y in zip(xs, ys)}
    annotated = set()
    for s in spikes:
        ws = _week_start(s["date"])
        if ws in week_lookup and ws not in annotated:
            base = s.get("baseline_mean") or 0
            mult = f"{s['count'] / base:.1f}x" if base else "spike"
            ax.annotate(mult, xy=(ws, week_lookup[ws]),
                        xytext=(0, 16), textcoords="offset points",
                        ha="center", fontsize=CHART_PT["annot"],
                        fontweight="bold", color=VALENCE_COLOR["negative"],
                        arrowprops=dict(arrowstyle="-", color=VALENCE_COLOR["negative"],
                                        linewidth=1.0))
            annotated.add(ws)

    _add_valence_legend(ax)
    ax.margins(y=0.22)
    return _save(fig, outpath)


def _add_valence_legend(ax):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=VALENCE_COLOR[v], markeredgecolor=VALENCE_COLOR[v],
               label=v.capitalize())
        for v in ("positive", "negative", "ambiguous")
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=CHART_PT["annot"], ncol=3, handletextpad=0.3,
              columnspacing=1.1, bbox_to_anchor=(0, 1.0))


def emotion_fingerprint(topic, outpath, width_in=6.6, top_n=6):
    emo = topic.get("emotions", {}) or {}
    pairs = (emo.get("top_non_neutral") or [])[:top_n]
    if not pairs:
        return None

    names = [strip_emotion(e).capitalize() for e, _ in pairs][::-1]
    vals = [v * 100 for _, v in pairs][::-1]
    colors = [emotion_color(n) for n in names]

    height = max(1.9, len(names) * 0.40 + 0.9)
    fig, ax = plt.subplots(figsize=(width_in, height))
    bars = ax.barh(names, vals, color=colors, height=0.55, zorder=3)
    ax.set_title("Most prominent expressed feelings", loc="left", pad=10)
    ax.set_xlabel("Share of emotional signal (%)")
    ax.grid(axis="y", visible=False)

    vmax = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(v + vmax * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left",
                fontsize=CHART_PT["annot"], color=PALETTE["ink_soft"])
    ax.set_xlim(0, vmax * 1.18)
    ax.tick_params(labelsize=CHART_PT["tick"])
    return _save(fig, outpath)
