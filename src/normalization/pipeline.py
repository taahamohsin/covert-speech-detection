"""
Component 3: Normalization pipeline.
Chains homoglyph resolution → leetspeak decoding → dogwhistle flagging.
Overwrites the `text` column so detection scripts need zero changes.

Usage:
  python3 src/normalization/pipeline.py \
    --input data/raw/reddit_raw_20260406_011406.csv \
    --output data/processed/reddit_normalized.csv
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from homoglyphs import normalize_homoglyphs_series
from leetspeak import normalize_leetspeak_series
from lexicon import flag_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    total_rows: int
    homoglyph_changed: int
    leetspeak_changed: int
    either_changed: int
    dogwhistle_flagged: int
    both_changed_and_flagged: int

    def __str__(self) -> str:
        return (
            f"\n=== Normalization Stats ===\n"
            f"  Total rows:              {self.total_rows}\n"
            f"  Homoglyph changes:       {self.homoglyph_changed} ({100 * self.homoglyph_changed / self.total_rows:.1f}%)\n"
            f"  Leetspeak changes:       {self.leetspeak_changed} ({100 * self.leetspeak_changed / self.total_rows:.1f}%)\n"
            f"  Either changed:          {self.either_changed} ({100 * self.either_changed / self.total_rows:.1f}%)\n"
            f"  Dogwhistle flagged:      {self.dogwhistle_flagged} ({100 * self.dogwhistle_flagged / self.total_rows:.1f}%)\n"
            f"  Changed + flagged:       {self.both_changed_and_flagged}\n"
        )


def run(
    input_path: str,
    output_path: str,
    lexicon_path: str = "data/lexicons/dogwhistles.csv",
) -> PipelineStats:
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError(f"Input CSV has no 'text' column: {input_path}")

    log.info("Loaded %d rows from %s", len(df), input_path)
    df["text"] = df["text"].fillna("").astype(str)
    df["_original_text"] = df["text"]

    # Pass 1: homoglyph normalization
    log.info("Pass 1: homoglyph normalization ...")
    hg_normalized, hg_changed = normalize_homoglyphs_series(df["text"])
    df["_hg_text"] = hg_normalized

    # Pass 2: leetspeak normalization (operates on homoglyph-normalized text)
    log.info("Pass 2: leetspeak normalization ...")
    leet_normalized, leet_changed = normalize_leetspeak_series(df["_hg_text"])
    df["normalized_text"] = leet_normalized

    # Pass 3: dogwhistle flagging (operates on fully normalized text)
    log.info("Pass 3: dogwhistle lexicon matching ...")
    flags, matched_any = flag_series(df["normalized_text"], lexicon_path)
    df["dogwhistle_flags"] = flags

    # Compute normalization_changed vs original
    df["normalization_changed"] = df["normalized_text"] != df["_original_text"]

    # Overwrite text column for detection script compatibility
    df["text"] = df["normalized_text"]

    # Drop internal columns
    df = df.drop(columns=["_original_text", "_hg_text"])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d rows to %s", len(df), output_path)

    stats = PipelineStats(
        total_rows=len(df),
        homoglyph_changed=int(hg_changed.sum()),
        leetspeak_changed=int(leet_changed.sum()),
        either_changed=int(df["normalization_changed"].sum()),
        dogwhistle_flagged=int(matched_any.sum()),
        both_changed_and_flagged=int((df["normalization_changed"] & matched_any).sum()),
    )
    log.info("%s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Reddit text for hate speech detection."
    )
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to write normalized CSV")
    parser.add_argument(
        "--lexicon",
        default="data/lexicons/dogwhistles.csv",
        help="Path to dogwhistles.csv",
    )
    args = parser.parse_args()
    run(args.input, args.output, args.lexicon)


if __name__ == "__main__":
    main()
