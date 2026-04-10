"""
Stratified annotation sample export.

Strata:
  A — Both flagged (perspective + openai):        13 items
  B — Perspective only:                            7 items
  C — OpenAI only:                                20 items
  D — Neither flagged, polarized subreddit:       25 items
  E — Neither flagged, control subreddit:         10 items
  Total:                                          75 items

Usage:
  python3 src/annotation/export.py
"""

import pandas as pd
from pathlib import Path

PERSPECTIVE_PATH = "data/processed/perspective_scored.csv"
OPENAI_PATH = "data/processed/openai_scored.csv"
OUTPUT_PATH = "data/annotations/annotation_sample.csv"
RANDOM_STATE = 42

STRATA_N = {"A": 13, "B": 7, "C": 20, "D": 25, "E": 10}


def main():
    p = pd.read_csv(PERSPECTIVE_PATH)
    o = pd.read_csv(OPENAI_PATH)

    # Merge on index position — both files are derived from the same raw CSV
    df = p.copy()
    for col in ["openai_flagged", "openai_hate", "openai_harassment"]:
        df[col] = o[col].values

    # Assign strata
    pf = df["perspective_flagged"].fillna(False).astype(bool)
    of = df["openai_flagged"].fillna(False).astype(bool)
    polarized = df["subreddit_type"] == "polarized"

    conditions = {
        "A": pf & of,
        "B": pf & ~of,
        "C": ~pf & of,
        "D": ~pf & ~of & polarized,
        "E": ~pf & ~of & ~polarized,
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

    # Select output columns
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

    # Annotator-facing version: only what annotators need, no API scores
    annotator_cols = [
        "item_id", "item_type", "subreddit", "text", "url",
        "annotator_1_hate_class", "annotator_1_evasion_strategy",
        "annotator_2_hate_class", "annotator_2_evasion_strategy",
    ]
    annotator_out = out[annotator_cols]
    annotator_path = OUTPUT_PATH.replace(".csv", "_annotator.csv")
    annotator_out.to_csv(annotator_path, index=False)
    print(f"Saved annotator-facing file ({len(annotator_out)} items) to {annotator_path}")

    print(f"\nSaved {len(out)} items to {OUTPUT_PATH}")
    print(f"Stratum distribution:\n{out['stratum'].value_counts().sort_index()}")
    print(f"\nLabel options:")
    print("  hate_class: Overt Hate | Covert Hate | Borderline/Ambiguous | Not Hateful")
    print(
        "  evasion_strategy: leetspeak | dogwhistle | deliberate_misspelling | "
        "emoji_substitution | unicode_homoglyph | punctuation_insertion | none"
    )


if __name__ == "__main__":
    main()
