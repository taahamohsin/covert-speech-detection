"""
Dogwhistle lexicon matcher: flag-only, never rewrites text.
Loads dogwhistles.csv, pre-compiles word-boundary regex patterns,
and matches against normalized text.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class LexiconEntry:
    term: str
    category: str
    hateful_meaning: str
    source: str


@dataclass
class LexiconMatch:
    term: str
    category: str
    hateful_meaning: str
    start: int
    end: int


def load_lexicon(path: str) -> list[LexiconEntry]:
    """Load dogwhistle lexicon from CSV. Skips rows with empty term."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Lexicon not found: {path}")
    entries = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row.get("term", "").strip()
            if not term:
                continue
            entries.append(LexiconEntry(
                term=term,
                category=row.get("category", "").strip(),
                hateful_meaning=row.get("hateful_meaning", "").strip(),
                source=row.get("source", "").strip(),
            ))
    return entries


def build_pattern_index(
    entries: list[LexiconEntry],
) -> list[tuple[re.Pattern, LexiconEntry]]:
    """Pre-compile word-boundary regex for each lexicon entry."""
    index = []
    for entry in entries:
        pattern = re.compile(
            r"\b" + re.escape(entry.term) + r"\b",
            flags=re.IGNORECASE,
        )
        index.append((pattern, entry))
    return index


def match_dogwhistles(
    text: str,
    pattern_index: list[tuple[re.Pattern, LexiconEntry]],
) -> list[LexiconMatch]:
    """Run all patterns against text. Returns all matches found."""
    if not text:
        return []
    matches = []
    for pattern, entry in pattern_index:
        for m in pattern.finditer(text):
            matches.append(LexiconMatch(
                term=entry.term,
                category=entry.category,
                hateful_meaning=entry.hateful_meaning,
                start=m.start(),
                end=m.end(),
            ))
    return matches


def flag_series(
    series: pd.Series,
    lexicon_path: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Match dogwhistles across a pandas Series.
    Loads lexicon and builds pattern index once.

    Returns:
        flags_series: comma-separated matched terms per row ("" if none)
        matched_any:  bool Series — True if any dogwhistle matched
    """
    entries = load_lexicon(lexicon_path)
    pattern_index = build_pattern_index(entries)

    def _flag(text: str) -> str:
        matches = match_dogwhistles(str(text), pattern_index)
        if not matches:
            return ""
        seen = []
        for m in matches:
            if m.term not in seen:
                seen.append(m.term)
        return ",".join(seen)

    flags = series.apply(_flag)
    matched_any = flags != ""
    return flags, matched_any
