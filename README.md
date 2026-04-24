# Speaking in Code: Automated Detection of Dogwhistle Language and Covert Hate Speech on Reddit

**NYU - CS-GY 9223 (Trust & Safety, Spring 2026)**

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

- Taaha Bin Mohsin
- Amory Gao
- Mohamed El Atoubi

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `config/config.yaml` and fill in your API credentials before running any scripts.

> **Note:** Raw data and scored CSVs are not committed to this repository for data privacy and Reddit ToS-compliance reasons (see `.gitignore`). Run the collection and detection scripts to reproduce them.

---

## Current Status

| Component | Status |
|---|---|
| Data Collection (8,720 items) | Complete |
| Perspective API scoring | Complete (8,720/8,720) |
| OpenAI Moderation API scoring - raw text | Complete (8,720/8,720) |
| Normalization pipeline | Complete |
| OpenAI Moderation API scoring - normalized text | Complete |
| Dogwhistle lexicon (~50 terms) | Complete |
| Annotation codebook | Complete |
| Human annotation - 150-item adversarial sample | Complete |
| Inter-annotator agreement (kappa) | Complete (0.622 post-calibration, 1.000 post-adjudication) |
| Detection gap analysis | Complete |
| Evaluation (precision/recall/F1) | Complete |

---

## Data

### Original Corpus
8,720 posts and comments collected from 10 subreddits (6 polarized, 4 control) via the Arctic Shift public archive. Scored with Perspective and OpenAI Moderation APIs pre- and post-normalization. Results in `data/processed/reddit_normalized.csv`.

### Adversarial Corpus
The original corpus turned out to be nearly entirely benign - "hot" and "top" Reddit posts that had already survived platform automoderation contained very little covert hate speech. To get meaningful detection gap results, we built a targeted adversarial corpus by keyword-searching 30+ dogwhistle terms across polarized and banned subreddits via Arctic Shift.

- 6,462 items across three strata:
  - A_dogwhistle (5,622 items) - matched dogwhistle lexicon
  - B_normalized (340 items) - normalization changed the text
  - C_banned_control (500 items) - control group, no heuristic match
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

### Step 4 - Human Annotation

150 items were sampled from the adversarial corpus, stratified by API agreement condition:

| Condition | Items | Purpose |
|---|---|---|
| Flagged by neither API (stratum A) | 60 | Detection gap cases |
| Flagged by OpenAI only (stratum A) | 30 | OpenAI agreement check |
| Flagged by Perspective only (stratum A) | 15 | Perspective agreement check |
| Flagged by both (stratum A) | 15 | True positive baseline |
| Control group, neither (stratum C) | 30 | Not-hateful baseline |

Two annotators labeled each item independently across four categories: Overt Hate, Covert Hate, Borderline/Ambiguous, Not Hateful. Inter-annotator kappa reached 0.622 after a terminology calibration session. Remaining disagreements were adjudicated by a third reviewer. Final kappa: 1.000 across all 150 items.

Export script: `src/annotation/export_adversarial_annotation.py`

---

### Step 5 - Evaluation

```bash
python3 src/evaluation/metrics.py
```

Computes precision, recall, and F1 for all four detection conditions against the gold standard. Borderline/Ambiguous items are excluded from primary metrics. Results are broken down by hate type (overt vs. covert) and evasion strategy.

---

## Key Results

### Detection gap at scale (adversarial corpus, 6,462 items)

| Condition | Flagged |
|---|---|
| OpenAI pre-norm | 2,421 / 6,462 (37.5%) |
| OpenAI post-norm | 2,426 / 6,462 (37.5%) |
| Perspective pre-norm | 343 / 6,462 (5.3%) |
| Perspective post-norm | 341 / 6,462 (5.3%) |

Normalization lift: +0.1pp (OpenAI), 0.0pp (Perspective).

API agreement:
- Flagged by both: 304 (4.8%)
- OpenAI only: 2,108 (33.2%)
- Perspective only: 39 (0.6%)
- Neither: 3,897 (61.4%)

By stratum:
| Stratum | Items | OpenAI | Perspective |
|---|---|---|---|
| A - dogwhistle-flagged | 5,622 | 38.9% | 5.2% |
| B - normalization-changed | 340 | 34.1% | 6.9% |
| C - control group | 500 | 24.0% | 6.5% |

### Evaluation against gold standard (134 items, Borderline/Ambiguous excluded)

| Condition | Precision | Recall | F1 |
|---|---|---|---|
| OpenAI pre-norm | 0.500 | 0.556 | 0.526 |
| OpenAI post-norm | 0.500 | 0.556 | 0.526 |
| Perspective pre-norm | 0.519 | 0.389 | 0.444 |
| Perspective post-norm | 0.519 | 0.389 | 0.444 |

By hate type (pre-norm):
| | OpenAI Recall | OpenAI F1 | Perspective Recall | Perspective F1 |
|---|---|---|---|---|
| Overt Hate (n=12) | 0.917 | 0.512 | 0.500 | 0.387 |
| Covert Hate (n=24) | 0.375 | 0.340 | 0.333 | 0.356 |

By evasion strategy (OpenAI pre-norm):
| Strategy | Recall | F1 | n |
|---|---|---|---|
| None (overt slurs) | 1.000 | 1.000 | 7 |
| Leetspeak | 1.000 | 1.000 | 1 |
| Dogwhistle | 0.444 | 0.585 | 27 |
| Deliberate misspelling | 0.000 | 0.000 | 1 |

---

## Central Finding

Our original hypothesis was that normalization would lift detection rates by fixing character-substitution evasion. The data showed the opposite: normalization has essentially zero effect because the corpus evades detection semantically, not orthographically. Posts use plain-text dogwhistles ("jogger", "globalists", "great replacement") that are spelled correctly - classifiers miss them because they lack the cultural context to interpret the coded meaning, not because the text is obfuscated.

61.4% of adversarially-sampled dogwhistle content was missed by both APIs. OpenAI catches 91.7% of overt hate but only 37.5% of covert hate. The detection gap is specifically in dogwhistle content: OpenAI recall on dogwhistle items is 44.4% vs. 100% on items with no evasion strategy.

---

## Policy Recommendations

1. Platforms should treat dogwhistle lexicon matches as an independent escalation signal routing content to human review. 61.4% of missed content would have been surfaced this way.
2. API providers should add a coded-language subscale (e.g. `hate_implicit`) trained on dogwhistle and incel-coded content, not just explicit slurs.
3. Orthographic normalization preprocessing is not worth deploying alone. Resources are better spent on lexicon curation and human review pipelines.
4. Platforms should apply stricter human review thresholds in communities with known histories of coordinated coded hate rather than relying solely on automated scores.

---

## Key References

- Gorwa et al. (2020) - Algorithmic content moderation
- Mendelsohn et al. (2023) - Dogwhistle glossary (300+ terms)
- Sasse et al. (2023) - Emergent dogwhistles
- Gröndahl et al. (2018) - Adversarial attacks on toxicity classifiers
- Thomas et al. (2021) - Hate & harassment taxonomy
