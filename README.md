# Between the Lines: Measuring the Detection Gap for Covert Hate Speech on Reddit

**NYU - CS-GY 9223 (Trust & Safety, Spring 2026)**
**Instructor:** Dr. Rosanna Bellini

---

## Overview

This project audits how well off-the-shelf content moderation APIs detect covert hate speech on Reddit - content that uses dogwhistles, leetspeak, emoji encoding, Unicode homoglyphs, and other evasion strategies to avoid automated detection.

Importantly, this is not a new classifier. We are:
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

Copy `config/config.yaml` and fill in your API credentials before running any scripts. Contact Taaha (taahabmohsin@hotmail.com) if you need access to credentials.

---

## Current Status

| Component | Status |
|---|---|
| Component 1 — Data Collection | Complete (8,720 items across 10 subreddits) |
| Component 2 — API Baseline Detection | In progress |
| Component 3 — Normalization Layer | Not started |
| Human Annotation | Not started |
| Evaluation | Not started |

---

## Pipeline

### Component 1 — Data Collection

Collects posts and top-level comments from polarized and control subreddits via [Arctic Shift](https://arctic-shift.photon-reddit.com) (no API key required). Drop-in replacement with PRAW once credentials are approved.

```bash
python3 src/collection/collect_arctic_shift.py --config config/config.yaml
```

Output: `data/raw/reddit_raw_<timestamp>.csv` and `.json`

**Subreddits:**

| Type | Communities |
|---|---|
| Polarized | r/conspiracy, r/Firearms, r/PoliticalCompassMemes, r/KotakuInAction, r/Conservative, r/PublicFreakout |
| Control | r/AskHistorians, r/science, r/explainlikeimfive, r/todayilearned |

---

### Component 2 - API Baseline Detection

**Google Perspective API** (service account auth) — TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, THREAT scores (flag threshold: 0.7):
```bash
python3 src/detection/perspective.py \
  --input data/raw/<file>.csv \
  --output data/processed/perspective_scored.csv
```

**OpenAI Moderation API** (Batch API) — hate, harassment, violence, illicit categories:
```bash
# Submit batch job (handles rate limits automatically)
python3 src/detection/openai_mod.py \
  --input data/raw/<file>.csv \
  --output data/processed/openai_scored.csv

# Parse results and retry failures
python3 src/detection/parse_and_retry_batch.py \
  --input data/raw/<file>.csv \
  --output-file batch_<id>_output.jsonl \
  --error-file batch_<id>_error.jsonl \
  --state data/processed/openai_state.json
```

---

### Component 3 — Normalization Layer

Preprocesses text before re-running through APIs:
- Leetspeak decoder (`1→l`, `3→e`, `0→o`)
- Unicode homoglyph resolver
- Dogwhistle lexicon lookup (300+ terms)
- Pattern rules: asterisk censoring, emoji substitutions, deliberate misspellings

```bash
python3 src/normalization/pipeline.py \
  --input data/raw/<file>.csv \
  --output data/processed/normalized.csv
```

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

---

## Key References

- Gorwa et al. (2020) — Algorithmic content moderation
- Mendelsohn et al. (2023) — Dogwhistle glossary (300+ terms)
- Sasse et al. (2023) — Emergent dogwhistles
- Gröndahl et al. (2018) — Adversarial attacks on toxicity classifiers
- Thomas et al. (2021) — Hate & harassment taxonomy
