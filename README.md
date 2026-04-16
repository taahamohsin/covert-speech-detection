# Between the Lines: Measuring the Detection Gap for Covert Hate Speech on Reddit

NYU Tandon - CS-GY 9223 (Trust & Safety, Spring 2026)
Instructor: Dr. Rosanna Bellini

---

## Overview

This project audits how well off-the-shelf content moderation APIs detect covert hate speech on Reddit - content that uses dogwhistles, coded language, leetspeak, Unicode homoglyphs, and other evasion strategies to avoid automated detection.

We are not building a new classifier. We are:
1. Measuring the detection gap between two leading APIs on adversarially-sampled Reddit content
2. Evaluating whether a custom preprocessing normalization layer can close that gap
3. Building a human-annotated gold standard to compute precision, recall, and F1 against

---

## Research Questions

1. What proportion of covert hate speech in polarized Reddit communities is missed by Google's Perspective API and OpenAI's Moderation API?
2. Which evasion strategies are most effective at evading current tools?
3. Can a custom preprocessing layer (dogwhistle lexicon + text normalization) significantly reduce the detection gap?

---

## Team

- Taaha Bin Mohsin (tb3486) - Pipeline development, normalization layer, API integration, cloud scoring infrastructure, evaluation framework
- Amory Gao (axg212) - Data collection, sampling strategy, statistical analysis, visualizations
- Mohamed El Atoubi (me2890) - Research design, dogwhistle lexicon curation, annotation codebook, lead annotator, report writing

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `config/config.yaml` and fill in your API keys before running any scripts. Contact Taaha at taahabmohsin@hotmail.com if you believe you should have access to the credentials but do not.

---

## Data

### Original Corpus
8,720 posts and comments collected from 10 subreddits (6 polarized, 4 control) via the Arctic Shift public archive. Scored with Perspective and OpenAI Moderation APIs pre- and post-normalization. Results in `data/processed/reddit_normalized.csv`.

### Adversarial Corpus
The original corpus turned out to be nearly entirely benign - "hot" and "top" Reddit posts that had already survived platform automoderation contained very little covert hate speech. To get meaningful detection gap results, we built a targeted adversarial corpus by keyword-searching 30+ dogwhistle terms across polarized and banned subreddits via Arctic Shift.

- 6,462 items across three strata:
  - A_dogwhistle (5,622 items) - matched dogwhistle lexicon
  - B_normalized (340 items) - normalization changed the text
  - C_banned_control (500 items) - random banned-sub content, no heuristic match
- 87% of items hit the dogwhistle lexicon vs. 1.9% in the original corpus
- All four detection conditions run on this corpus
- Final artifact: `data/processed/detection_results_final.csv` (6,462 rows, 39 columns)

---

## Pipeline

### Step 1 - Data Collection

Original corpus:
```bash
python3 src/collection/collect_arctic_shift.py --config config/config.yaml
```

Adversarial corpus (dogwhistle-targeted):
```bash
python3 src/collection/collect_toxic_targeted.py
python3 src/collection/sample_toxic_subset.py
```

---

### Step 2 - Normalization

Preprocesses text before re-running through APIs:
- `homoglyphs.py` - resolves ~150 Unicode codepoints (Cyrillic, Greek, fullwidth Latin) to ASCII
- `leetspeak.py` - three-pass normalizer: strips interstitial punctuation, decodes leet substitutions, converts asterisk-censored words to `<CENSORED>`
- `lexicon.py` - dogwhistle lexicon matcher (~50 seed terms from Mendelsohn et al. and ADL Hate Symbols Database)
- `pipeline.py` - chains all three, adds `normalized_text`, `dogwhistle_flags`, `normalization_changed` columns

```bash
python3 src/normalization/pipeline.py \
  --input data/raw/<file>.csv \
  --output data/processed/<file>.csv
```

Finding: Only 0.8% of original corpus items had text changed by normalization. The corpus evades detection semantically, not orthographically.

---

### Step 3 - API Detection

Perspective API (TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, THREAT - threshold: 0.7):
```bash
python3 src/detection/perspective.py \
  --input data/processed/<file>.csv \
  --output data/processed/<scored_file>.csv
```

OpenAI Moderation API (hate, harassment, self-harm categories):
```bash
python3 src/detection/openai_sync.py \
  --input data/processed/<file>.csv \
  --scored data/processed/<scored_file>.csv \
  --rate 0.3
```

Both scripts are resumable - already-scored rows are skipped on restart.

Cloud scoring: All four conditions on the adversarial corpus were run via Modal cloud compute using `modal_score.py`, `modal_perspective.py`, and `modal_score_chunk.py` to avoid keeping a laptop open overnight. Results were checkpointed to a Modal Volume every 50-100 rows and merged back locally via `merge_chunks.py`.

---

## Key Results

All results are on the adversarial corpus (6,462 items). Full data in `data/processed/detection_results_final.csv`.

| Condition | Flagged |
|-----------|---------|
| OpenAI pre-norm | 2,421 / 6,462 (37.5%) |
| OpenAI post-norm | 2,426 / 6,462 (37.5%) |
| Perspective pre-norm | 343 / 6,348 (5.4%) |
| Perspective post-norm | 341 / 6,348 (5.4%) |

Normalization lift: +0.1pp (OpenAI), 0.0pp (Perspective) - effectively zero.

API agreement (on 6,348 items scored by both):
- Flagged by both: 304 (4.8%)
- OpenAI only: 2,108 (33.2%)
- Perspective only: 39 (0.6%)
- Neither: 3,897 (61.4%)

By stratum:
| Stratum | Items | OpenAI | Perspective |
|---------|-------|--------|-------------|
| A - dogwhistle-flagged | 5,622 | 38.9% | 5.2% |
| B - normalization-changed | 340 | 34.1% | 6.9% |
| C - banned sub control | 500 | 24.0% | 6.5% |

---

## Central Finding

Our original hypothesis was that normalization would lift detection rates by fixing character-substitution evasion. The data showed the opposite: normalization has essentially zero effect because the corpus evades detection semantically, not orthographically. Posts use plain-text dogwhistles ("jogger", "globalists", "great replacement") that are spelled correctly - classifiers miss them because they lack the cultural context to interpret the coded meaning, not because the text is obfuscated.

61.4% of adversarially-sampled dogwhistle content was missed by both APIs. This is the detection gap. It cannot be closed with preprocessing alone. Platforms need human-in-the-loop escalation for dogwhistle-flagged content and API providers should consider adding coded-language subscales (e.g., `identity_attack_coded`) trained on implicit rather than explicit hate.

---

## Annotation (In Progress)

### Status
- Round 1 (75 items, original corpus): kappa = 0.00 - calibration failure, sample was too benign
- Round 2 (50 items, original corpus, stratified by target group): Annotator B complete, Annotator A pending

### Next steps
Rounds 1 and 2 drew from the original corpus (1.9% dogwhistle hit rate), which is too benign for meaningful calibration. The next annotation round must draw from the adversarial corpus (`detection_results_final.csv`). `src/annotation/export_covert_v2.py` currently exports from the original corpus - a new export script targeting the adversarial corpus is needed before Round 3 can begin.

Target: 100-200 annotated items stratified by API agreement condition:

| Condition | Items | Purpose |
|-----------|-------|---------|
| Flagged by neither API (stratum A) | ~40 | Detection gap - cases both APIs miss |
| Flagged by OpenAI only (stratum A) | ~20 | OpenAI agreement check |
| Flagged by Perspective only (stratum A) | ~10 | Perspective agreement check |
| Flagged by both (stratum A) | ~10 | True positive baseline |
| Banned sub control, neither (stratum C) | ~20 | Not-hateful baseline |

Once kappa >= 0.6 is achieved, precision/recall/F1 will be computed for all four detection conditions against the gold standard, broken down by hate speech type (overt vs. covert) and evasion strategy.

---

## Key References

- Gorwa et al. (2020) - Algorithmic content moderation
- Mendelsohn et al. (2023) - Dogwhistle glossary (300+ terms)
- Sasse et al. (2023) - Emergent dogwhistles
- Gröndahl et al. (2018) - Adversarial attacks on toxicity classifiers
- Thomas et al. (2021) - Hate & harassment taxonomy
