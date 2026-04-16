"""
Targeted annotation sample for covert hate speech detection.

Draws from the RAW dataset but uses normalization metadata (dogwhistle_flags,
normalization_changed) to stratify, ensuring coded/evasive content is represented.

Strata:
  F — Dogwhistle flagged (excluding 'based'-only):   15 items
  G — Normalization changed text (leetspeak/homoglyphs): 15 items
  H — 'based'-only flagged (likely benign, for contrast): 10 items
  I — Unflagged polarized (random baseline):           5 items
  J — Unflagged control subreddit (baseline):          5 items
  Total:                                              50 items

Usage:
  python3 src/annotation/export_covert.py
"""

import pandas as pd
from pathlib import Path

RAW_PERSPECTIVE = "data/processed/perspective_scored.csv"
RAW_OPENAI = "data/processed/openai_scored.csv"
NORMALIZED = "data/processed/reddit_normalized.csv"
OUTPUT_PATH = "data/annotations/annotation_sample_covert.csv"
RANDOM_STATE = 99

STRATA_N = {"F": 15, "G": 15, "H": 10, "I": 5, "J": 5}


def main():
    # Load raw scored data (annotators see raw text)
    p = pd.read_csv(RAW_PERSPECTIVE)
    o = pd.read_csv(RAW_OPENAI)

    df = p.copy()
    for col in ["openai_flagged", "openai_hate", "openai_harassment"]:
        df[col] = o[col].values

    # Load normalization metadata
    norm = pd.read_csv(NORMALIZED)
    df["dogwhistle_flags"] = norm["dogwhistle_flags"].values
    df["normalization_changed"] = norm["normalization_changed"].values

    # Build strata masks
    has_flags = df["dogwhistle_flags"].notna() & (df["dogwhistle_flags"] != "")
    based_only = has_flags & (df["dogwhistle_flags"].str.strip() == "based")
    meaningful_flags = has_flags & ~based_only
    changed = df["normalization_changed"] == True
    polarized = df["subreddit_type"] == "polarized"

    # Exclude items already in the original 75-item sample
    original = pd.read_csv("data/annotations/annotation_sample.csv")
    original_ids = set(original["item_id"])
    not_in_original = ~df["item_id"].isin(original_ids)

    control = df["subreddit_type"] == "control"

    conditions = {
        "F": meaningful_flags & not_in_original,
        "G": changed & ~has_flags & not_in_original,  # changed but no dogwhistle flag
        "H": based_only & not_in_original,
        "I": ~has_flags & ~changed & polarized & not_in_original,
        "J": ~has_flags & ~changed & control & not_in_original,
    }

    samples = []
    for stratum, mask in conditions.items():
        pool = df[mask]
        n = STRATA_N[stratum]
        if len(pool) < n:
            print(
                f"WARNING: stratum {stratum} only has {len(pool)} items (wanted {n}) — taking all"
            )
            n = len(pool)
        sample = pool.sample(n=n, random_state=RANDOM_STATE)
        sample = sample.copy()
        sample["stratum"] = stratum
        samples.append(sample)
        print(f"  Stratum {stratum}: {n} items sampled from {len(pool)} eligible")

    out = (
        pd.concat(samples)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Output columns — raw text for annotators, no normalized text
    out_cols = [
        "item_id",
        "item_type",
        "subreddit",
        "subreddit_type",
        "text",
        "score",
        "url",
        "perspective_flagged",
        "perspective_toxicity",
        "perspective_identity_attack",
        "openai_flagged",
        "openai_hate",
        "openai_harassment",
        "stratum",
        "annotator_1_hate_class",
        "annotator_1_evasion_strategy",
        "annotator_2_hate_class",
        "annotator_2_evasion_strategy",
    ]
    for col in [
        "annotator_1_hate_class",
        "annotator_1_evasion_strategy",
        "annotator_2_hate_class",
        "annotator_2_evasion_strategy",
    ]:
        out[col] = ""

    out = out[out_cols]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    # Annotator-facing version: no API scores or strata (to avoid bias)
    annotator_out = out[["item_id", "item_type", "subreddit", "text", "url"]].copy()
    annotator_out["hate_class"] = ""
    annotator_out["evasion_strategy"] = ""
    annotator_path = OUTPUT_PATH.replace(".csv", "_annotator.csv")
    annotator_out.to_csv(annotator_path, index=False)
    print(f"\nSaved annotator-facing file ({len(annotator_out)} items) to {annotator_path}")

    print(f"Saved {len(out)} items to {OUTPUT_PATH}")
    print(f"Stratum distribution:\n{out['stratum'].value_counts().sort_index()}")
    print(f"\nLabel options:")
    print("  hate_class: Overt Hate | Covert Hate | Borderline/Ambiguous | Not Hateful")
    print(
        "  evasion_strategy: leetspeak | dogwhistle | deliberate_misspelling | "
        "emoji_substitution | unicode_homoglyph | punctuation_insertion | none"
    )


if __name__ == "__main__":
    main()
