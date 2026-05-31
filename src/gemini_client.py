import os
import random
import re
import time

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    _SDK = "google-genai"
except ImportError:
    genai = None
    _SDK = None


MODEL_FLASH_LITE = "gemini-2.5-flash-lite"
MODEL_FLASH = "gemini-2.5-flash"

MAX_OUTPUT_TOKENS = {
    "summary": 400,
    "topic": 220,
}

_RETRYABLE = {429, 500, 502, 503, 504}
_RETRY_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s")


class GeminiClient:
    def __init__(self, api_key: str | None = None, default_model: str = MODEL_FLASH_LITE,
                 temperature: float = 0.25, max_retries: int = 6):
        if genai is None:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai")
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        self.client = genai.Client(api_key=key)
        self.default_model = default_model
        self.temperature = temperature
        self.max_retries = max_retries

    def _raw_generate(self, model, system_prompt, user_text, max_tokens):
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=max_tokens,
            top_p=0.9,
        )
        resp = self.client.models.generate_content(
            model=model, contents=user_text, config=cfg)
        return (resp.text or "").strip()

    @staticmethod
    def _status_of(err) -> int | None:
        for attr in ("code", "status_code", "http_status"):
            v = getattr(err, attr, None)
            if isinstance(v, int):
                return v
        m = re.search(r"\b(4\d\d|5\d\d)\b", str(err))
        return int(m.group(1)) if m else None

    @staticmethod
    def _hinted_delay(err) -> float | None:
        m = _RETRY_DELAY_RE.search(str(err))
        return float(m.group(1)) if m else None

    def generate(self, system_prompt: str, user_text: str, *, kind: str = "topic",
                 model: str | None = None) -> str:
        model = model or self.default_model
        max_tokens = MAX_OUTPUT_TOKENS.get(kind, 256)
        last_err = None

        for attempt in range(self.max_retries):
            try:
                out = self._raw_generate(model, system_prompt, user_text, max_tokens)
                if out:
                    return out
                last_err = RuntimeError("empty response")
            except Exception as err:
                status = self._status_of(err)
                if status is not None and status not in _RETRYABLE:
                    raise
                last_err = err

            if attempt < self.max_retries - 1:
                backoff = min(2 ** attempt, 32) + random.uniform(0, 1)
                hinted = self._hinted_delay(last_err) if last_err else None
                time.sleep(max(backoff, hinted or 0))

        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} attempts: {last_err}")
