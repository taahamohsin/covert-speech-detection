"""
Curated 50-item annotation sample for covert hate speech detection (v2).

Uses multi-signal analysis (OpenAI hate/harassment scores, Perspective toxicity,
dogwhistle flags, normalization metadata) to select items where hate speech is
coded, dog-whistled, or uses evasion strategies — not overt slurs.

Target composition: ~80% covert hate, ~20% not hateful (baseline).

Strata:
  F — Antisemitic / conspiracy coded:       6 items
  G — Racist / xenophobic coded:           11 items
  H — Islamophobic coded:                   4 items
  I — Misogynistic / incel coded:           8 items
  J — Anti-LGBTQ / transphobic coded:       7 items
  K — Mixed coded hate / evasion:           4 items
  L — Not hateful (control baseline):      10 items
  Total:                                   50 items

Usage:
  python3 src/annotation/export_covert_v2.py
"""

import pandas as pd
from pathlib import Path

RAW = "data/raw/reddit_raw_20260406_011406.csv"
PERSPECTIVE = "data/processed/perspective_scored.csv"
OPENAI = "data/processed/openai_scored.csv"
OUTPUT_PATH = "data/annotations/annotation_sample_covert_v2.csv"
RANDOM_STATE = 42

# Manually curated item IDs, grouped by covert hate category
COVERT_HATE_IDS = {
    # F — Antisemitic / conspiracy coded
    "F": [
        "kghwfnw",  # "jews overrepresentation... replaced president with a jew"
        "kukjwgp",  # "jews destroyed my house and homework"
        "kgdm6ng",  # "don't kill all Jews?"
        "kg56duk",  # "Which jews, I only know of the La Li Lu Le Lo"
        "18zg5z0",  # "jewish colonizer"
        "18vtdfe",  # "great reset... genocide a nation"
    ],
    # G — Racist / xenophobic coded
    "G": [
        "kg52533",  # "israelis smarter, americans dumber"
        "kfutg3h",  # "chinese are week... inferior"
        "kg2qizt",  # "Native Americans zero claim to moon"
        "18yqqtv",  # "If Indian Or Romani, You Can Be Racist"
        "kgcz7ag",  # "pickpocketed by those gypsy fucks"
        "kg0pdf3",  # "stop letting these come to UK... civilised world"
        "khmdj0w",  # "racist Huwite Women" (coded spelling)
        "kg55zp1",  # "DEI... lower bar... black academic"
        "kfy8t7v",  # "turn US into 3rd world"
        "18zl58g",  # "Macacos" (racial slur)
        "18wkup6",  # "too white for that"
    ],
    # H — Islamophobic coded
    "H": [
        "kr3l34g",  # "Islam is fake"
        "kwdwhcf",  # "Mohammed spoke with lucifer"
        "18w5yid",  # "Islam created by Satan"
        "18zh30t",  # "Islam respects women, bigot!"
    ],
    # I — Misogynistic / incel coded
    "I": [
        "kg5t181",  # "white women are social terrorists"
        "kgm2jis",  # "sex with men because I hate women"
        "kg6z78k",  # "send women back to Stone Age"
        "kfyohuk",  # "stop listening to women"
        "kg8oz2r",  # "chicken heads... Females deserve less"
        "kfy44d9",  # "weak coomers"
        "kfsvee4",  # "Soyboy trying to play nice with women"
        "kgrgihp",  # "ruin franchises... root of feminism"
    ],
    # J — Anti-LGBTQ / transphobic coded
    "J": [
        "kgrwtdr",  # "hates women = epitome of being gay"
        "kg5dnke",  # "put a chick in it and make it gay"
        "18wioo1",  # "Trans rights don't exist"
        "18w36ly",  # "Censored for exposing transgenderism"
        "18w1h3l",  # "running tab of transgender crimes"
        "kg05cdu",  # "none of those Gays!"
        "kfv394a",  # "transvestite and transgender is different"
    ],
    # K — Mixed coded hate / evasion techniques
    "K": [
        "kiv6cf0",  # "colored/female/lgbtq+" anti-progressive coded
        "kfyqulf",  # "BLM and Antifa skulls cracked"
        "kg51opx",  # leetspeak evasion "m0r0ns" (normalization changed)
        "kgdig9g",  # anti-Palestinian dismissal
    ],
}

# L — Not hateful (baseline: 5 control, 5 benign polarized)
NOT_HATEFUL_IDS = [
    # Control subreddits
    "18vv9ca",  # AskHistorians — countries question
    "18w6x4r",  # AskHistorians — ocean trade ships
    "18w3hyo",  # todayilearned — pole dancing
    "18w6pfc",  # todayilearned — Carly Simon
    "kg9m9r6",  # explainlikeimfive — ADHD
    # Benign polarized
    "kg8wygv",  # Firearms — optic question
    "18yxln1",  # Firearms — love my hammers
    "kgler29",  # PCM — taxing endowments
    "192lifl",  # KotakuInAction — Neil Druckmann at Golden Globes
    "18vqa4a",  # PublicFreakout — robbery attempt news
]


def main():
    raw = pd.read_csv(RAW)
    p = pd.read_csv(PERSPECTIVE)
    o = pd.read_csv(OPENAI)

    df = raw.copy()
    for col in ["perspective_toxicity", "perspective_identity_attack", "perspective_flagged"]:
        df[col] = p[col]
    for col in ["openai_flagged", "openai_hate", "openai_harassment"]:
        df[col] = o[col]

    # Build the sample
    samples = []

    # Covert hate strata
    for stratum, ids in COVERT_HATE_IDS.items():
        subset = df[df["item_id"].isin(ids)].copy()
        subset["stratum"] = stratum
        found = set(subset["item_id"])
        missing = set(ids) - found
        if missing:
            print(f"WARNING: stratum {stratum} missing IDs: {missing}")
        print(f"  Stratum {stratum}: {len(subset)} items")
        samples.append(subset)

    # Not hateful baseline
    baseline = df[df["item_id"].isin(NOT_HATEFUL_IDS)].copy()
    baseline["stratum"] = "L"
    found = set(baseline["item_id"])
    missing = set(NOT_HATEFUL_IDS) - found
    if missing:
        print(f"WARNING: stratum L missing IDs: {missing}")
    print(f"  Stratum L (not hateful): {len(baseline)} items")
    samples.append(baseline)

    out = (
        pd.concat(samples)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Output columns
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
    ]
    out = out[[c for c in out_cols if c in out.columns]]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    # Annotator-facing version: no API scores or strata
    annotator_out = out[["item_id", "item_type", "subreddit", "text", "url"]].copy()
    annotator_out["hate_class"] = ""
    annotator_out["evasion_strategy"] = ""
    annotator_path = OUTPUT_PATH.replace(".csv", "_annotator.csv")
    annotator_out.to_csv(annotator_path, index=False)

    print(f"\nSaved {len(out)} items to {OUTPUT_PATH}")
    print(f"Saved annotator-facing file ({len(annotator_out)} items) to {annotator_path}")
    print(f"\nStratum distribution:\n{out['stratum'].value_counts().sort_index()}")

    covert_count = sum(len(ids) for ids in COVERT_HATE_IDS.values())
    total = covert_count + len(NOT_HATEFUL_IDS)
    print(f"\nTarget composition: {covert_count}/{total} covert hate ({100*covert_count/total:.0f}%)")
    print(f"                    {len(NOT_HATEFUL_IDS)}/{total} not hateful ({100*len(NOT_HATEFUL_IDS)/total:.0f}%)")

    print(f"\nLabel options:")
    print("  hate_class: Overt Hate | Covert Hate | Borderline/Ambiguous | Not Hateful")
    print(
        "  evasion_strategy: leetspeak | dogwhistle | deliberate_misspelling | "
        "emoji_substitution | unicode_homoglyph | punctuation_insertion | none"
    )


if __name__ == "__main__":
    main()
