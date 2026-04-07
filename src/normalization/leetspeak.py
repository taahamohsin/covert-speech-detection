"""
Leetspeak normalizer: three-pass text normalization.
Pass 1: strip interstitial punctuation (h.a.t.e → hate)
Pass 2: decode leet character substitutions (1→l, 3→e, etc.)
Pass 3: mark asterisk-censored words (f**k → f<CENSORED>k)
"""

import re

import pandas as pd

LEET_MAP: dict[str, str] = {
    "1": "l",
    "3": "e",
    "4": "a",
    "0": "o",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "+": "t",
}

# URL pattern — skip normalization inside URLs entirely
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Matches sequences of single chars separated by punctuation/spaces:
# e.g. "h.a.t.e", "h a t e", "f-u-c-k"
# Requires at least 3 separated single chars.
# Excludes matches preceded by two or more letters (mid-word context).
_INTERSTITIAL_RE = re.compile(
    r"(?<![a-zA-Z]{2})"           # not preceded by 2+ letters
    r"(?<!\w\.\w)"                 # not inside abbreviation like e.g.
    r"\b([a-zA-Z])"               # first single letter at word boundary
    r"(?:[.\-_])([a-zA-Z])"       # separator (NOT space) + second letter
    r"(?:(?:[.\-_])[a-zA-Z])+"   # one or more additional sep+letter pairs
    r"\b",
    re.IGNORECASE,
)

# Matches asterisk-censored words: one or more leading alpha + asterisks + optional trailing alpha
_ASTERISK_RE = re.compile(r"\b([a-zA-Z]+)\*+([a-zA-Z]*)\b")


def _mask_urls(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Replace URLs with placeholders and return the masked text + restore map."""
    placeholders = []
    masked = text
    offset = 0
    for m in _URL_RE.finditer(text):
        start, end = m.start() + offset, m.end() + offset
        placeholder = f"\x00URL{len(placeholders)}\x00"
        placeholders.append((start, start + len(placeholder), m.group(0)))
        masked = masked[:start] + placeholder + masked[end:]
        offset += len(placeholder) - (m.end() - m.start())
    return masked, placeholders


def _restore_urls(text: str, placeholders: list[tuple[int, int, str]]) -> str:
    """Restore original URLs from placeholders."""
    for i, (_, _, original) in enumerate(placeholders):
        text = text.replace(f"\x00URL{i}\x00", original)
    return text


def strip_interstitial_punctuation(text: str) -> str:
    """
    Collapse single letters separated by punctuation into words.
    h.a.t.e → hate, f-u-c-k → fuck
    Space-separated single letters are NOT collapsed (too many false positives).
    Requires ≥3 separated chars. Skips URLs.
    """
    if not text:
        return text
    masked, placeholders = _mask_urls(text)

    def _collapse(m: re.Match) -> str:
        return re.sub(r"[.\-_]", "", m.group(0))

    result = _INTERSTITIAL_RE.sub(_collapse, masked)
    return _restore_urls(result, placeholders)


def decode_leet(text: str) -> str:
    """
    Substitute leet characters with their alpha equivalents.
    Only fires when ALL of these conditions hold:
    - The character is in LEET_MAP
    - Both the previous AND next character are alphabetic
      (requires both neighbors to be alpha, not just one, to avoid URLs/IDs)
    Skips URLs entirely.
    """
    if not text:
        return text

    masked, placeholders = _mask_urls(text)
    chars = list(masked)
    result = []
    for i, ch in enumerate(chars):
        if ch in LEET_MAP:
            prev_alpha = i > 0 and chars[i - 1].isalpha()
            next_alpha = i < len(chars) - 1 and chars[i + 1].isalpha()
            # Require BOTH neighbors to be alpha (stricter than before)
            if prev_alpha and next_alpha:
                result.append(LEET_MAP[ch])
                continue
        result.append(ch)
    return _restore_urls("".join(result), placeholders)


def decode_asterisk_censoring(text: str) -> str:
    """
    Replace asterisk spans with <CENSORED> marker.
    f**k → f<CENSORED>k, n***er → n<CENSORED>er
    Preserves leading/trailing anchor characters for API detection.
    """
    if not text:
        return text
    return _ASTERISK_RE.sub(lambda m: f"{m.group(1)}<CENSORED>{m.group(2)}", text)


def normalize_leetspeak(text: str) -> str:
    """Full leetspeak normalization: interstitial → leet decode → asterisk censoring."""
    if not text:
        return text
    text = strip_interstitial_punctuation(text)
    text = decode_leet(text)
    text = decode_asterisk_censoring(text)
    return text


def normalize_leetspeak_series(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Apply normalize_leetspeak element-wise.
    Returns (normalized_series, changed_bool_series).
    """
    normalized = series.apply(normalize_leetspeak)
    changed = normalized != series
    return normalized, changed
