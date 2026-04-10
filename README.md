# Speaking in Code: Automated Detection of Dogwhistle Language and Covert Hate Speech on Reddit

**NYU - CS-GY 9223 (Trust & Safety, Spring 2026)**

**Instructor:** Dr. Rosanna Bellini

---

## Overview

This project audits how well off-the-shelf content moderation APIs detect covert hate speech on Reddit — content that uses dogwhistles, leetspeak, Unicode homoglyphs, and other evasion strategies to avoid automated detection.

This is not a new classifier. We are:
1. Measuring the detection gap between human-annotated ground truth and API outputs
2. Building a preprocessing normalization layer to close that gap

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
| OpenAI Moderation API scoring - normalized text | 84.9% complete (7,404/8,720) |
| Dogwhistle lexicon (~50 terms) | Complete |
| Annotation codebook | Complete |
| Human annotation - pilot round (75 items) | Complete (calibration failure; see below) |
| Human annotation - revised sample (50 items) | Ready for re-annotation |
| Detection gap analysis | Pending gold standard |
| Evaluation (precision/recall/F1) | Pending gold standard |

### Preliminary Detection Gap

On the raw corpus (8,720 items, no normalization applied):

| | Perspective API | OpenAI Moderation |
|---|---|---|
| All items | 2.1% flagged (186) | 9.6% flagged (835) |
| Polarized subreddits only | 3.1% flagged (172/5,588) | 13.3% flagged (742/5,588) |

702 items are flagged by OpenAI but missed by Perspective. Only 53 are flagged by Perspective but missed by OpenAI.

---

## Pipeline

### Component 1 - Data Collection

Collects posts and top-level comments from polarized and control subreddits via [Arctic Shift](https://arctic-shift.photon-reddit.com) (no API key required). Drop-in replacement with PRAW once credentials are approved.

```bash
python3 src/collection/collect_arctic_shift.py --config config/config.yaml
```

Output: `data/raw/reddit_raw_<timestamp>.csv`

**Subreddits:**

| Type | Communities | Items |
|---|---|---|
| Polarized | r/conspiracy, r/Firearms, r/PoliticalCompassMemes, r/KotakuInAction, r/Conservative, r/PublicFreakout | 5,588 |
| Control | r/AskHistorians, r/science, r/explainlikeimfive, r/todayilearned | 3,132 |

---

### Component 2 - API Baseline Detection

**Google Perspective API** — TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, THREAT scores (flag threshold: 0.7):

```bash
python3 src/detection/perspective.py \
  --input data/raw/<file>.csv \
  --output data/processed/perspective_scored.csv
```

**OpenAI Moderation API** (Batch API) — hate, harassment, violence, illicit categories:

```bash
# Initial scoring
python3 src/detection/openai_mod.py \
  --input data/raw/<file>.csv \
  --output data/processed/openai_scored.csv

# Retry failed/null rows
python3 src/detection/openai_retry.py \
  --scored data/processed/openai_scored.csv \
  --input data/raw/<file>.csv
```

---

### Component 3 - Normalization Layer

Three-pass preprocessing pipeline applied before re-running APIs:

- `homoglyphs.py` — resolves ~150 Unicode codepoints (Cyrillic, Greek, fullwidth Latin) to ASCII equivalents
- `leetspeak.py` — decodes leetspeak (`1→l`, `3→e`, `0→o`), strips interstitial punctuation (`h.a.t.e→hate`), replaces asterisk censoring with `<CENSORED>` placeholder
- `lexicon.py` — flags dogwhistle terms from `data/lexicons/dogwhistles.csv`; never rewrites text
- `pipeline.py` — chains all three; adds `normalized_text`, `dogwhistle_flags`, `normalization_changed` columns

```bash
# Step 1: Normalize
python3 src/normalization/pipeline.py \
  --input data/raw/<file>.csv \
  --output data/processed/reddit_normalized.csv

# Step 2: Re-score normalized text
python3 src/detection/perspective.py \
  --input data/processed/reddit_normalized.csv \
  --output data/processed/perspective_normalized_scored.csv

python3 src/detection/openai_mod.py \
  --input data/processed/reddit_normalized.csv \
  --output data/processed/openai_normalized_scored.csv
```

---

### Component 4 - Human Annotation

Two annotators independently label items along two dimensions:
- **Hate class:** Overt Hate | Covert Hate | Borderline/Ambiguous | Not Hateful
- **Evasion strategy:** leetspeak | dogwhistle | deliberate_misspelling | emoji_substitution | unicode_homoglyph | punctuation_insertion | none *(only applies to Covert Hate)*

```bash
# Export original 75-item stratified sample
python3 src/annotation/export.py

# Export revised 50-item covert-hate-targeted sample
python3 src/annotation/export_covert_v2.py
```

**Annotation status:** A pilot round (75 items) produced Cohen's κ = 0.00 due to calibration failure — one annotator classified all items as "Not Hateful." The sample was redesigned to target ~80% covert hate prevalence using multi-signal API scoring. Re-annotation with calibration session is pending.

---

### Evaluation

Computes precision, recall, and F1 against human-annotated gold standard across four conditions:
- Perspective alone
- OpenAI alone
- Perspective + normalization
- OpenAI + normalization

```bash
python3 src/evaluation/metrics.py \
  --annotations data/annotations/<file>.csv \
  --predictions data/processed/<file>.csv
```
