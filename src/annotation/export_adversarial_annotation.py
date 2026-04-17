"""
Export a 150-item stratified annotation sample from the adversarial corpus.

Sampling plan (150 items total):
  - Stratum A, flagged by neither API:       60 items  (detection gap)
  - Stratum A, flagged by OpenAI only:       30 items  (OpenAI agreement check)
  - Stratum A, flagged by Perspective only:  15 items  (Perspective agreement check)
  - Stratum A, flagged by both APIs:         15 items  (true positive baseline)
  - Stratum C, flagged by neither API:       30 items  (not-hateful baseline)

Outputs:
  data/annotations/annotation_sample_adversarial.csv  — full version with API scores (for adjudication)
  data/annotations/annotation_sample_adversarial_annotator.csv  — annotator-facing (no scores, no strata)

Usage:
  python3 src/annotation/export_adversarial_annotation.py
"""

import pandas as pd
from pathlib import Path

INPUT = "data/processed/detection_results_final.csv"
OUTPUT_FULL = "data/annotations/annotation_sample_adversarial.csv"
OUTPUT_ANNOTATOR = "data/annotations/annotation_sample_adversarial_annotator.csv"
RANDOM_STATE = 42

PLAN = [
    # (label, stratum, openai_flagged, perspective_flagged, n)
    ("neither",        "A_dogwhistle",    False, False, 60),
    ("openai_only",    "A_dogwhistle",    True,  False, 30),
    ("perspective_only","A_dogwhistle",   False, True,  15),
    ("both",           "A_dogwhistle",    True,  True,  15),
    ("control_neither", "C_banned_control", False, False, 30),
]


def main():
    df = pd.read_csv(INPUT)

    # Require both APIs to have scored the item
    df = df.dropna(subset=["openai_flagged", "perspective_flagged"])
    df["openai_flagged"] = df["openai_flagged"].astype(bool)
    df["perspective_flagged"] = df["perspective_flagged"].astype(bool)

    # Drop items with no usable text
    df = df[df["text"].str.strip().str.len() >= 50].copy()

    samples = []
    for label, stratum, openai_flag, persp_flag, n in PLAN:
        pool = df[
            (df["sample_stratum"] == stratum)
            & (df["openai_flagged"] == openai_flag)
            & (df["perspective_flagged"] == persp_flag)
        ]
        if len(pool) < n:
            print(f"WARNING: {label} pool has only {len(pool)} items (wanted {n}), taking all")
            n = len(pool)
        sample = pool.sample(n=n, random_state=RANDOM_STATE).copy()
        sample["agreement_condition"] = label
        samples.append(sample)
        print(f"  {label}: {len(sample)} items (pool: {len(pool)})")

    out = (
        pd.concat(samples)
        .sample(frac=1, random_state=RANDOM_STATE)  # shuffle so annotators can't infer strata
        .reset_index(drop=True)
    )
    out.index = out.index + 1  # 1-based item numbers for annotators

    print(f"\nTotal: {len(out)} items")
    print(f"Agreement condition distribution:\n{out['agreement_condition'].value_counts()}")

    # Full version — includes API scores and agreement condition for adjudication
    full_cols = [
        "item_id", "item_type", "subreddit", "text", "url",
        "openai_flagged", "openai_hate", "openai_harassment",
        "perspective_flagged", "perspective_toxicity", "perspective_identity_attack",
        "sample_stratum", "agreement_condition",
    ]
    full_out = out[[c for c in full_cols if c in out.columns]]

    Path(OUTPUT_FULL).parent.mkdir(parents=True, exist_ok=True)
    full_out.to_csv(OUTPUT_FULL, index_label="item_num")
    print(f"\nSaved full version ({len(full_out)} items) to {OUTPUT_FULL}")

    # Annotator-facing version — text and response fields only, no scores
    annotator_cols = ["item_id", "item_type", "subreddit", "text", "url"]
    annotator_out = out[[c for c in annotator_cols if c in out.columns]].copy()
    annotator_out["hate_class"] = ""
    annotator_out["evasion_strategy"] = ""
    annotator_out["notes"] = ""
    annotator_out.to_csv(OUTPUT_ANNOTATOR, index_label="item_num")
    print(f"Saved annotator-facing version ({len(annotator_out)} items) to {OUTPUT_ANNOTATOR}")

    print("\nLabel options:")
    print("  hate_class:       Overt Hate | Covert Hate | Borderline/Ambiguous | Not Hateful")
    print("  evasion_strategy: dogwhistle | leetspeak | deliberate_misspelling |")
    print("                    emoji_substitution | unicode_homoglyph | punctuation_insertion | none")


if __name__ == "__main__":
    main()
