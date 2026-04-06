# Between the Lines: Measuring the Detection Gap for Covert Hate Speech on Reddit

**NYU Tandon - CS-GY 9223 (Trust & Safety, Spring 2026)**
**Instructor:** Dr. Rosanna Bellini

---

## Overview

This project audits how well off-the-shelf content moderation APIs detect covert hate speech on Reddit - content that uses dogwhistles, leetspeak, emoji encoding, Unicode homoglyphs, and other evasion strategies to avoid automated detection.

It is not a new classifier. Instead, it:
1. Measures the detection gap between human-annotated ground truth and API outputs
2. Builds a preprocessing normalization layer to close that gap

---

## Research Questions

1. What proportion of covert hate speech in polarized Reddit communities is missed by Google's Perspective API and OpenAI's Moderation API?
2. Which evasion strategies are most effective at evading current tools?
3. Can a custom preprocessing layer (dogwhistle lexicon + text normalization) significantly reduce the detection gap?

---

## Team

- Taaha Bin Mohsin - MS CS
- Amory Gao - BS/MS CS
- Mohamed El Atoubi - PhD Cybersecurity

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `config/config.yaml` and fill in your API keys before running any scripts. Contact Taaha at taahabmohsin@hotmail.com if you believe you should have access to the credentials but do not.

---

## Pipeline

### Component 1 - Data Collection

Collects posts and top-level comments from polarized and control subreddits.

**Interim (no credentials needed):**
```bash
python src/collection/collect_arctic_shift.py --config config/config.yaml
```
Uses the [Arctic Shift](https://arctic-shift.photon-reddit.com) public Reddit archive — no API key required.

**When PRAW credentials are approved:**
```bash
python src/collection/collect_reddit.py --config config/config.yaml
```

Output: `data/raw/reddit_raw_<timestamp>.csv` and `.json`

---

### Component 2 - API Baseline Detection

Passes collected items through:
- **Google Perspective API** — toxicity, identity_attack, insult, threat (threshold: 0.7)
- **OpenAI Moderation API** — hate, harassment, self-harm categories

```bash
python src/detection/perspective.py --input data/raw/<file>.csv
python src/detection/openai_mod.py --input data/raw/<file>.csv
```

---

### Component 3 - Normalization Layer

Preprocesses text before re-running through APIs:
- Leetspeak decoder (`1→l`, `3→e`, `0→o`)
- Unicode homoglyph resolver
- Dogwhistle lexicon lookup (300+ terms)
- Pattern rules: asterisk censoring, emoji substitutions, deliberate misspellings

```bash
python src/normalization/pipeline.py --input data/raw/<file>.csv --output data/processed/<file>.csv
```

---

### Evaluation

Computes precision, recall, and F1 against human-annotated gold standard across four conditions:
- Perspective alone
- OpenAI alone
- Perspective + normalization
- OpenAI + normalization

```bash
python src/evaluation/metrics.py --annotations data/annotations/<file>.csv --predictions data/processed/<file>.csv
```

---

## Key References

- Gorwa et al. (2020) - Algorithmic content moderation
- Mendelsohn et al. (2023) - Dogwhistle glossary (300+ terms)
- Sasse et al. (2023) - Emergent dogwhistles
- Gröndahl et al. (2018) - Adversarial attacks on toxicity classifiers
- Thomas et al. (2021) - Hate & harassment taxonomy
