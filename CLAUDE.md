# Between the Lines: Measuring the Detection Gap for Covert Hate Speech on Reddit

## Project Overview
This is a Trust & Safety audit project for NYU Tandon (CS-UY 3943 / CS-GY 9223, Spring 2026, Prof. Bellini). We are measuring how well off-the-shelf content moderation APIs detect **covert hate speech** — dogwhistles, leetspeak, emoji encoding, Unicode homoglyphs, and coded in-group terminology — on Reddit.

We are NOT building a new hate speech classifier. We are **auditing existing ones** and building a **preprocessing normalization layer** to close the detection gap.

## Team
- **Taaha Bin Mohsin** (tb3486) — Pipeline development, normalization layer, API integration, evaluation framework
- **Amory Gao** (axg212) — Data collection (PRAW scripts), sampling strategy, statistical analysis, visualizations
- **Mohamed El Atoubi** (me2890) — Research design, dogwhistle lexicon curation, annotation codebook, lead annotator, report writing

## Research Questions
1. What proportion of covert hate speech in polarized Reddit communities is missed by leading off-the-shelf detection tools, specifically OpenAI's Moderation API and a RoBERTa-based hate speech classifier (facebook/roberta-hate-speech-dynabench)?
2. What are the most common evasion strategies, and which are most effective at evading current tools?
3. Can a custom preprocessing layer (dogwhistle lexicon + text normalization + pattern-matching) significantly reduce the detection gap?

## Architecture — Three Components

### Component 1: Data Collection (PRAW)
- Collect posts + top-level comments from polarized subreddits using PRAW
- Also collect from well-moderated control subreddits (e.g., r/AskHistorians, r/science)
- Sort by "hot", "top", and "controversial" to capture engagement range
- Target: ~2,000–3,000 raw posts/comments across all subreddits
- Per item: title, body text, score, upvote_ratio, comment count, awards, timestamp, author (anonymized), subreddit
- Rate limit: 100 requests/min (Reddit free tier)
- Store as structured JSON/CSV

### Component 2: Off-the-Shelf Baselines (Red-Teaming Phase)
- Pass every collected item through:
  - **Google Perspective API** (service account auth): returns TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, THREAT scores (0–1). Flag threshold: 0.7. Script: `src/detection/perspective.py`
  - **OpenAI Moderation API**: classifies across hate, hate/threatening, harassment, self-harm categories. Free with API key. Script: `src/detection/openai_mod.py`
- Record all scores/predictions per item
- Compare against human-annotated gold standard to measure detection gap

### Component 3: Custom Normalization + Lexicon Layer (Mitigation Phase)
Three sub-components, implemented across four modules in `src/normalization/`:

1. **`homoglyphs.py`** — Unicode homoglyph resolver. Hardcoded map of ~150 codepoints (Cyrillic, Greek, fullwidth Latin, mathematical symbols) → ASCII equivalents. Applies `unicodedata.normalize("NFC")` first to canonicalize composed characters. Must run before leetspeak decoding.

2. **`leetspeak.py`** — Three-pass normalizer applied in order:
   - `strip_interstitial_punctuation`: collapses `h.a.t.e` → `hate` (requires ≥3 separated chars to avoid destroying "e.g.", "U.S.")
   - `decode_leet`: substitutes `1→l, 3→e, 4→a, 0→o, 5→s, 7→t, @→a, $→s` only when adjacent to alpha chars (avoids mangling dates/IDs)
   - `decode_asterisk_censoring`: `f**k` → `f<CENSORED>k` (preserves anchor chars for API detection without guessing the word)

3. **`lexicon.py`** — Dogwhistle flag-only matcher. Loads `data/lexicons/dogwhistles.csv` (schema: `term, category, hateful_meaning, source`). Pre-compiles `\b<term>\b` regex patterns at load time. Returns comma-separated matched terms per item — never rewrites text.

4. **`pipeline.py`** — Chains all three in order (homoglyphs → leetspeak → lexicon). Adds three columns to output CSV:
   - `normalized_text` — text after homoglyph + leetspeak normalization
   - `dogwhistle_flags` — comma-separated matched terms (`""` if none)
   - `normalization_changed` — bool
   Overwrites `text` column with `normalized_text` so detection scripts (`perspective.py`, `openai_mod.py`) need zero changes.

**Dogwhistle lexicon** (`data/lexicons/dogwhistles.csv`): ~50 seed terms from Mendelsohn et al. (2023) and ADL Hate Symbols Database covering white nationalist (1488, ZOG, great replacement, white genocide, groyper, honkler), racial (jogger, dindu, apes), antisemitic (globalists, juice, skypes), misogynist (femoid, roastie, AWFL), and LGBTQ-targeting (groomer) coded vocabulary.

**Invocation:**
```bash
# Step 1: Normalize
conda run python3 src/normalization/pipeline.py \
  --input data/raw/reddit_raw_20260406_011406.csv \
  --output data/processed/reddit_normalized.csv

# Step 2: Re-run detection on normalized text (same scripts, no changes)
conda run python3 src/detection/perspective.py \
  --input data/processed/reddit_normalized.csv \
  --output data/processed/perspective_post_norm.csv

conda run python3 src/detection/openai_mod.py \
  --input data/processed/reddit_normalized.csv \
  --output data/processed/openai_post_norm.csv
```

After normalization, compare detection rates before vs. after across all four conditions.

### Human Annotation (Gold Standard)
- 2 annotators independently label a stratified subset of 100–200 items
- Label dimensions:
  - Hate speech class: "Overt Hate", "Covert Hate", "Borderline/Ambiguous", "Not Hateful"
  - Evasion strategy: leetspeak, dogwhistle/coded term, deliberate misspelling, emoji substitution, Unicode homoglyph, punctuation insertion, none
- Borderline/Ambiguous items excluded from primary precision/recall, reported separately

## Evaluation
- Precision, recall, F1 for each approach against gold standard
- Broken down by hate speech type (overt vs. covert) and by evasion strategy
- Four conditions: Perspective alone, OpenAI alone, Perspective+normalization, OpenAI+normalization

## Tech Stack
- Python 3.10+
- PRAW (Reddit API) / Arctic Shift (interim)
- Google Perspective API (service account auth)
- OpenAI Moderation API
- Pandas for data management
- Standard ML evaluation metrics (sklearn)

## Key References
- Gorwa et al. (2020) — Algorithmic content moderation
- Mendelsohn et al. (2023) — Dogwhistle glossary (300+ terms)
- Sasse et al. (2023) — Emergent dogwhistles
- Gröndahl et al. (2018) — Adversarial attacks on toxicity classifiers
- Thomas et al. (2021) — Hate & harassment taxonomy

## Current Status
- Project plan and methodology are finalized
- Directory structure and config scaffolded
- Component 1 in progress (data collection)
- Reddit API access pending approval (ticket submitted via support.reddithelp.com)
- **Using Arctic Shift as interim data source** until PRAW credentials are approved

## Data Collection: Arctic Shift (Interim)
Reddit now requires pre-approval for API access. While waiting, we collect via the Arctic Shift public archive:
- Base URL: `https://arctic-shift.photon-reddit.com`
- No API key or auth required
- Endpoints: `/api/posts/search`, `/api/comments/search`
- Script: `src/collection/collect_arctic_shift.py`
- Output schema is identical to the planned PRAW output for downstream compatibility
- Fields unavailable in Arctic Shift (`upvote_ratio`, `awards`) are set to `None`

Once PRAW credentials are approved, `src/collection/collect_reddit.py` can be run as a drop-in replacement.

## File Structure (Planned)
```
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml          # API keys, subreddit lists, thresholds
├── src/
│   ├── collection/
│   │   └── collect_reddit.py # PRAW data collection
│   ├── detection/
│   │   ├── dynabench.py      # HuggingFace RoBERTa Dynabench classifier
│   │   └── openai_mod.py     # OpenAI Moderation API wrapper
│   ├── normalization/
│   │   ├── leetspeak.py      # Leetspeak decoder
│   │   ├── homoglyphs.py     # Unicode homoglyph resolver
│   │   ├── lexicon.py        # Dogwhistle lexicon lookup
│   │   └── pipeline.py       # Combined normalization pipeline
│   ├── evaluation/
│   │   └── metrics.py        # Precision, recall, F1 computation
│   └── annotation/
│       └── export.py         # Export samples for human annotation
├── data/
│   ├── raw/                  # Raw collected Reddit data
│   ├── processed/            # Normalized data
│   ├── annotations/          # Human annotations
│   └── lexicons/             # Dogwhistle glossaries
└── notebooks/
    └── analysis.ipynb        # Statistical analysis and visualizations
```