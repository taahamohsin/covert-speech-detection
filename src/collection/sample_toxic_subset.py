"""
Sample a high-signal subset from the toxic-targeted collection.

The raw toxic corpus (~70k items) was collected by searching Arctic Shift for
dogwhistle terms, but not every comment in a hateful thread is itself hateful.
Running 70k items through the APIs would be wasteful and rate-limited.

This script produces an adversarial subset that should actually exercise the
detection gap + normalization layer:

  Stratum A — dogwhistle-flagged (primary signal):
    items where lexicon matched a known covert hate term
  Stratum B — normalization-changed (orthographic evasion):
    items where homoglyph/leetspeak normalization altered the text
  Stratum C — banned-sub control:
    random items from banned subs that match NEITHER A nor B
    (sanity check: do the APIs catch novel hate we didn't pattern-match?)

Selection is NOT based on any API output, so there's no circularity when we
later measure detection rates on the result.

Usage:
    python3 src/collection/sample_toxic_subset.py \
        --input data/processed/reddit_toxic_normalized.csv \
        --output data/processed/reddit_toxic_sample.csv \
        [--control-size 500]
"""

import argparse
import csv
import logging
import random
from pathlib import Path

csv.field_size_limit(10_000_000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42


def is_normalized(row: dict) -> bool:
    return row.get("normalization_changed", "").strip().lower() == "true"


def has_flag(row: dict) -> bool:
    return bool(row.get("dogwhistle_flags", "").strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--control-size",
        type=int,
        default=500,
        help="Random items from banned subs that match no heuristic (default: 500)",
    )
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    log.info("Loaded %d rows from %s", len(rows), args.input)

    # Stratum A: dogwhistle-flagged
    stratum_a = [r for r in rows if has_flag(r)]
    # Stratum B: normalization-changed only (not already in A)
    stratum_b = [r for r in rows if is_normalized(r) and not has_flag(r)]
    # Stratum C candidates: banned subs, no flag, no normalization change
    stratum_c_candidates = [
        r for r in rows
        if r.get("subreddit_type", "") == "banned"
        and not has_flag(r)
        and not is_normalized(r)
    ]

    rng = random.Random(RANDOM_STATE)
    k = min(args.control_size, len(stratum_c_candidates))
    stratum_c = rng.sample(stratum_c_candidates, k)

    # Tag each row with stratum so we can break results down later
    for r in stratum_a:
        r["sample_stratum"] = "A_dogwhistle"
    for r in stratum_b:
        r["sample_stratum"] = "B_normalized"
    for r in stratum_c:
        r["sample_stratum"] = "C_banned_control"

    combined = stratum_a + stratum_b + stratum_c

    # Dedup on item_id (A and B are mutually exclusive by construction; C by filter)
    seen: set[str] = set()
    deduped = []
    for r in combined:
        iid = r.get("item_id", "")
        if iid and iid not in seen:
            seen.add(iid)
            deduped.append(r)

    out_fields = list(fieldnames) + ["sample_stratum"] if "sample_stratum" not in fieldnames else list(fieldnames)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(deduped)

    log.info("--- Sample composition ---")
    log.info("  Stratum A (dogwhistle-flagged):    %d", len(stratum_a))
    log.info("  Stratum B (normalization-changed): %d", len(stratum_b))
    log.info("  Stratum C (banned control):        %d", len(stratum_c))
    log.info("  Total (deduped):                   %d", len(deduped))
    log.info("  Saved -> %s", args.output)


if __name__ == "__main__":
    main()
