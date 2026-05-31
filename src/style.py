PALETTE = {
    "ink":        "#1F2933",
    "ink_soft":   "#52606D",
    "muted":      "#9AA5B1",
    "rule":       "#E4E7EB",
    "panel":      "#F5F7FA",
    "accent":     "#1D3557",
    "accent_2":   "#457B9D",
    "white":      "#FFFFFF",
}

VALENCE_COLOR = {
    "positive":  "#2A9D8F",
    "negative":  "#D1495B",
    "ambiguous": "#E6A817",
    "neutral":   "#ADB5BD",
}

EMOTION_VALENCE = {
    "admiration": "positive", "amusement": "positive", "approval": "positive",
    "caring": "positive", "desire": "positive", "excitement": "positive",
    "gratitude": "positive", "joy": "positive", "love": "positive",
    "optimism": "positive", "pride": "positive", "relief": "positive",
    "anger": "negative", "annoyance": "negative", "disappointment": "negative",
    "disapproval": "negative", "disgust": "negative", "embarrassment": "negative",
    "fear": "negative", "grief": "negative", "nervousness": "negative",
    "remorse": "negative", "sadness": "negative",
    "confusion": "ambiguous", "curiosity": "ambiguous",
    "realization": "ambiguous", "surprise": "ambiguous",
    "neutral": "neutral",
}


def strip_emotion(name: str) -> str:
    return name[len("emotion_"):] if name.startswith("emotion_") else name


def emotion_color(name: str) -> str:
    base = strip_emotion(name).lower()
    return VALENCE_COLOR[EMOTION_VALENCE.get(base, "neutral")]


FONT_BODY = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_MONO = "Courier"

TYPE = {
    "cover_title":  30,
    "cover_sub":    13,
    "h1":           18,
    "h2":           13.5,
    "body":         10.5,
    "caption":      8.5,
    "quote":        10.5,
    "footer":       7.5,
}

CHART_PT = {
    "title":   12,
    "label":   10,
    "tick":     9,
    "annot":    8.5,
}

COMPANY_KIND = {
    "chipotle":  "operating",
    "starbucks": "operating",
    "tesla":     "investor",
}

DISPLAY_NAME = {
    "chipotle":  "Chipotle",
    "starbucks": "Starbucks",
    "tesla":     "Tesla",
}


def company_kind(company: str) -> str:
    return COMPANY_KIND.get(company.lower(), "operating")


def display_name(company: str) -> str:
    return DISPLAY_NAME.get(company.lower(), company.title())


PAGE_MARGIN = 54
CONTENT_WIDTH = 612 - 2 * PAGE_MARGIN
