# Midpoint Report: Required Changes to Intention Document

This document catalogs every change needed to convert the Intention Document into
a complete Mid-Point Report, incorporating: the midpoint spec requirements, the
professor's feedback on the intention document (GroupC.pdf), and actual project
progress as of April 9, 2026.

---

## 1. Title and Header

**Change:** Update title.

- **From:** "Between the Lines: Measuring the Detection Gap for Covert Hate Speech on Reddit"
- **To:** "Speaking in Code: Automated Detection of Dogwhistle Language and Covert Hate Speech on Reddit"
  *(Professor's suggested title, GroupC.pdf)*

---

## 2. Document Structure Overhaul

The midpoint spec (MidSemester_Spring26.pdf) requires a **different section structure** than the intention document. The document needs to be reorganized into three required sections:

| Remove (Intention Doc) | Replace With (Midpoint) |
|---|---|
| §1 Project Options | *(drop entirely)* |
| §2 Project Summary | → Becomes **Introduction** |
| §3 Connection to T&S Themes | → Absorbed into **Introduction** and **Related Work** |
| §4 Measurement Plan | → Becomes **Research Plan & Current Status** |
| §5 Anticipated Challenges | → Absorbed into **Research Plan & Current Status** (Next Steps subsection) |
| §6 Expected Outcomes | → Abbreviated at end of **Introduction** (Project Outcomes sneak-peek) |
| §7 Division of Roles | → Moved into **Research Plan & Current Status** |

---

## 3. Introduction (New Section)

The Introduction must cover four elements per the spec. Map from existing content:

### 3a. Problem Statement
**Keep:** The existing Project Summary opening paragraph (Gorwa, Mendelsohn, Sasse framing) is strong — keep it.

**Add:** One concrete motivating example or statistic to quantify the real-world scale of the problem. The feedback noted: *"Could quantify the scale of covert hate speech prevalence or cite specific incidents/cases where detection failures enabled harm."* Candidates:
- The 2022 Buffalo shooter's manifesto circulated on fringe platforms for hours before removal, containing coded "Great Replacement" language standard classifiers missed.
- Moonshot CVE (2021) estimated 30–40% of online hate speech uses some form of evasion to avoid detection.
- Or, lead with our own preliminary finding: in our 8,720-item corpus, Perspective API (the industry-standard tool) flags only **2.1%** of content while OpenAI flags **9.6%** — a **4.6× gap** that is even larger in polarized communities (3.1% vs 13.3%).

### 3b. Motivation and Significance
**Keep:** Thomas et al. LGBTQ+ (1.86×) and 18–24 age group (3.99×) statistics — they're strong.

**Trim:** The T&S themes section (§3) becomes redundant as a standalone section. Condense the Gorwa, Stephenson, and Thomas discussions into 1–2 sentences each and fold them into the Introduction.

**Add:** Answer the four spec questions explicitly:
- *Who is most affected?* Marginalized communities targeted by coded language (LGBTQ+, racial/ethnic minorities, religious minorities)
- *How does it intersect with privacy/wellbeing?* Detection failures enable sustained harassment while preserving perpetrator deniability
- *Why is it understudied/technically challenging?* The ambiguity of dogwhistle context — same word innocuous vs. hateful depending on community
- *What's at stake?* Platforms operating in good faith may unknowingly host organized hate campaigns that evade all automated gatekeeping

### 3c. Scope and Project
**Add explicit out-of-scope statement.** The spec requires this. Something like:
> "We focus on English-language posts and top-level comments collected between January–April 2026. We are auditing two specific off-the-shelf APIs (Perspective and OpenAI Moderation) and measuring the effect of a preprocessing normalization layer. We are not building or fine-tuning a new hate speech classifier, not conducting temporal analysis of how evasion evolves, and not studying platform policy or moderation decision-making."

**Add:** Specify that data is drawn from 10 subreddits (list them — see §5 below).

### 3d. Project Outcomes (sneak-peek)
**Add a short paragraph** (2–3 sentences) previewing what you'll deliver:
> "By the end of the semester, we will produce: (1) a 125-item human-annotated gold standard with inter-annotator agreement metrics; (2) a precision/recall/F1 comparison of Perspective and OpenAI Moderation on overt vs. covert hate, before and after normalization; and (3) a public-domain normalization pipeline with design recommendations for integrating it into production moderation workflows."

---

## 4. Related Work (New Section — Must Be Added)

The Intention Document has no standalone Related Work section. The midpoint spec requires one. Must include 3–5 sources with analytical discussion and identified gaps.

### Sources to include (all already in the references):
1. **Gorwa et al. (2020)** — algorithmic moderation opacity; positions our audit as filling the empirical gap they call for
2. **Mendelsohn et al. (2023)** — dogwhistle typology and 300+ term glossary; our lexicon is seeded from their work; gap: they demonstrate classifiers miss dogwhistles but don't audit specific production APIs on naturalistic Reddit data
3. **Sasse et al.** — emergent dogwhistles; gap: their discovery methods focus on habitats/communities, not on measuring what existing moderation APIs miss
4. **Thomas et al. (2021)** — hate & harassment SoK taxonomy; provides the harm framework we use; gap: broad survey, doesn't audit evasion-specific failure modes of particular tools
5. **Add one more:** Gröndahl et al. (2018), "All You Need is 'Love': Evading Hate Speech Detection" — directly relevant, shows simple text perturbations defeat state-of-the-art classifiers. This is the most directly related prior work on evasion attacks against hate speech classifiers and should be added to references.

### Gap paragraph:
> "While prior work has demonstrated that dogwhistles evade classifiers in controlled settings (Mendelsohn et al.) and that text perturbations defeat hate speech models (Gröndahl et al.), no study has audited the specific production-grade APIs that platforms actually use — Perspective and OpenAI Moderation — against naturalistic, in-the-wild evasive content collected from Reddit. Our project fills this gap by using a corpus of real Reddit posts (not synthetic perturbations), measuring the detection gap empirically, and evaluating whether a lightweight normalization layer can close it without model retraining."

---

## 5. Research Plan & Current Status (Restructured Section)

### 5a. Research Questions
Keep all three — they are solid. No changes needed.

### 5b. Methodological Approach — Data Collection Update

**Critical update required:** The intention document states data will be collected via PRAW. **In practice, Reddit API approval was pending and data was collected via the Arctic Shift public archive instead.** The midpoint report must document this pivot.

**Add:** Specific subreddit list (the feedback explicitly asked for this — *"I would benefit from seeing candidate subreddit examples or selection criteria beyond 'politically polarized.'"*):

| Subreddit | Type | Items |
|---|---|---|
| r/conspiracy | Polarized | 1,197 |
| r/Firearms | Polarized | 1,151 |
| r/KotakuInAction | Polarized | 1,028 |
| r/PoliticalCompassMemes | Polarized | 934 |
| r/PublicFreakout | Polarized | 817 |
| r/Conservative | Polarized | 461 |
| r/explainlikeimfive | Control | 928 |
| r/science | Control | 806 |
| r/AskHistorians | Control | 740 |
| r/todayilearned | Control | 658 |
| **Total** | | **8,720** |

Add the selection rationale: polarized subreddits were chosen for known history of coded political/identity speech and proximity to hate-adjacent communities without being dedicated hate subreddits; control subreddits are well-moderated, topic-specific, with low expected hate speech prevalence.

### 5c. Methodological Approach — Normalization Pipeline Update

**Update status:** The pipeline is **fully built and applied**. Describe implemented sub-components:
- `homoglyphs.py`: resolves ~150 Unicode codepoints (Cyrillic, Greek, fullwidth Latin) to ASCII
- `leetspeak.py`: three-pass normalizer (interstitial punctuation stripping, leet substitution with context guards, asterisk censoring → `<CENSORED>` placeholder)
- `lexicon.py`: ~50 seed terms from Mendelsohn et al. + ADL, pattern-matched with word boundaries; returns flags, does NOT rewrite text
- `pipeline.py`: chains all three, adds `normalized_text`, `dogwhistle_flags`, `normalization_changed` columns

**Add key pipeline finding:** Of 8,720 items, 72 (0.8%) had text changed by normalization and 166 (1.9%) received dogwhistle flags — though only 31 (0.4%) had flags beyond ambiguous terms like "based."

### 5d. Methodological Approach — Annotation Update

**Major update required.** The intention document says "Two team members will independently annotate a stratified subset of 100-200." This has been substantially revised:

**Annotation Protocol v2:**
- Two annotators (Amory Gao and Mohamed El Atoubi) independently labeled a **50-item stratified pilot sample** (annotation_sample.csv) drawn from the full corpus
- Inter-rater agreement was computed using **Cohen's kappa** (κ = 0.00), revealing a severe calibration failure: one annotator labeled all 50 items "Not Hateful" while the other labeled the majority as hate speech
- **This addresses the professor's feedback gap directly**: The midpoint report should describe the kappa calculation, our kappa threshold (κ ≥ 0.6 as acceptable), and the disagreement resolution process (adjudication on all items where annotators disagree, with a third-party tiebreaker if needed)

**Pilot lesson and revised sampling:** The original sample was found to contain primarily ambiguous/benign content due to an over-broad dogwhistle lexicon matching terms like "based", "echo", "juice" in benign contexts. A revised 50-item sample (annotation_sample_covert_v2.csv) was constructed using multi-signal scoring (OpenAI hate score, detection gap between APIs, dogwhistle specificity) to ensure ~80% of items are genuinely covert hate-speech-bearing content.

**Revised annotation workflow:**
1. Calibration session: annotators jointly review 10 example items with the codebook
2. Independent annotation of 50-item curated sample
3. Cohen's kappa computed; items below agreement threshold adjudicated
4. Expand to full 100-item gold standard if κ ≥ 0.6 on pilot

### 5e. Progress to Date (New Subsection)

**Must be added** per the spec. Concrete accomplishments:

| Component | Status |
|---|---|
| Data collection (8,720 items) | ✅ Complete |
| Perspective API scoring | ✅ Complete (8,720/8,720) |
| OpenAI Moderation API scoring (raw) | ✅ Complete (8,720/8,720) |
| Normalization pipeline built | ✅ Complete |
| Normalization applied to full corpus | ✅ Complete |
| OpenAI Moderation API scoring (normalized) | 🔄 ~80% complete (6,966/8,720) |
| Dogwhistle lexicon (~50 terms) | ✅ Complete |
| Annotation codebook | ✅ Complete |
| Pilot annotation (50 items) | ✅ Complete (calibration failed; revised sample ready) |
| Revised annotation sample (50 items) | ✅ Complete (annotation_sample_covert_v2.csv) |
| Re-annotation with calibrated protocol | ⏳ Pending |
| Detection gap analysis | ⏳ Pending (awaiting gold standard) |

**Preliminary findings (add to report):**
- In the raw corpus, Perspective flags 2.1% of all items vs. OpenAI's 9.6% — a 4.6× gap suggesting a large class of content flagged by one API and not the other
- In polarized subreddits specifically, the gap is larger: Perspective 3.1% vs. OpenAI 13.3%
- 702 items are flagged by OpenAI but missed by Perspective (potential covert hate evasion)
- Only 53 items are flagged by Perspective but missed by OpenAI

### 5f. Next Steps (Required Subsection)

The spec requires this. List what remains:

1. Complete OpenAI Moderation scoring on normalized dataset (~1,754 items remaining; limited by API rate quotas, submitting batches daily)
2. Calibration session with annotators → re-annotation of revised 50-item sample
3. Compute Cohen's kappa; adjudicate disagreements; expand to 100 items if agreement is acceptable
4. Run detection gap analysis: precision/recall/F1 for all four conditions (Perspective raw, OpenAI raw, Perspective+norm, OpenAI+norm) against gold standard
5. Breakdown by evasion strategy type (dogwhistle, leetspeak, homoglyph, etc.)
6. Write final report; create visualizations

**Ethical/logistical barriers:**
- Annotator wellbeing: structured review shifts with mandatory breaks; no more than 2 hours of annotation per session
- API rate limits: OpenAI Moderation quota limits daily batch throughput; using async batch API to maximize throughput within quota
- Annotation quality: second calibration round if κ < 0.6 on revised sample

---

## 6. Challenges/Risks Section Updates

The existing §5 is good but needs one addition:

**Add inter-rater reliability risk** (directly called out in professor feedback):
> "A risk we have already encountered is annotator calibration failure: our initial pilot annotation round produced a Cohen's kappa of κ = 0.00, because one annotator classified all items as 'Not Hateful.' This was caused partly by a sample design flaw — our initial sample over-represented ambiguous lexicon matches (e.g., the word 'based' in benign political contexts) rather than items with clear covert hate signals. We have addressed this by (a) redesigning the sample using multi-signal scoring to ensure ~80% prevalence of covert hate, and (b) adding a mandatory calibration session before independent annotation begins."

---

## 7. Expected Outcomes Section Updates

The professor feedback noted: *"You mention 'technical interventions and policy considerations' but don't preview what these might look like."*

**Add specifics:**
- **Technical:** Recommend platforms integrate a two-stage preprocessing pipeline before moderation API calls: (1) Unicode normalization + leetspeak decoding, (2) dogwhistle flag escalation for human review rather than automated action
- **Human-in-the-loop:** For items with dogwhistle flags, suggest escalating to human moderators rather than relying on automated action, given the high false-positive rate of lexicon matching
- **API improvement:** Propose that Perspective API add an "identity_attack_coded" subscale that scores items for coded rather than explicit derogatory language

---

## 8. References — One Addition Needed

Add:

> GRÖNDAHL, T., PAJOLA, L., JUUTI, M., CONTI, M., AND ASOKAN, N. All you need is "love": Evading hate speech detection. In *Proceedings of the 11th ACM Workshop on Artificial Intelligence and Security* (2018), pp. 2–12.

---

## 9. Minor/Stylistic Fixes

- **Section 3 verbosity**: The two paragraphs about "upcoming course themes" (Red Teaming, Cross-Cultural, Hate & Harassment) are now past dates (those lectures have happened). Reframe as "we engaged with these themes" not "we will connect to upcoming themes."
- **PRAW footnotes**: The PRAW footnotes and rate-limit footnote should be replaced or supplemented with Arctic Shift documentation since that's the actual data source.
- **Page count**: The midpoint should be ~5 pages. The intention document is already 5 pages, so restructuring (not expanding) is the priority.

---

## Summary of Highest-Priority Changes

| Priority | Change |
|---|---|
| 🔴 Required | Restructure sections into Introduction / Related Work / Research Plan |
| 🔴 Required | Add "Progress to Date" subsection with concrete status table |
| 🔴 Required | Add "Next Steps" subsection |
| 🔴 Required | Add Related Work section (currently missing entirely) |
| 🟠 Important | Document Arctic Shift pivot and list actual subreddits |
| 🟠 Important | Document annotation calibration failure + revised protocol |
| 🟠 Important | Add inter-rater reliability methodology (kappa, threshold, adjudication) |
| 🟠 Important | Add preliminary detection gap numbers (2.1% vs 9.6%) |
| 🟡 Feedback gap | Add concrete technical intervention previews |
| 🟡 Feedback gap | Add quantified prevalence/urgency statistic in intro |
| 🟡 Feedback gap | Add Gröndahl et al. reference |
| 🟢 Minor | Update title to professor's suggestion |
| 🟢 Minor | Trim §3 "upcoming themes" framing (those lectures are past) |
| 🟢 Minor | Add explicit out-of-scope statement |
