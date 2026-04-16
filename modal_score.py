"""
Modal cloud runner for OpenAI Moderation scoring of the toxic sample.

Runs two scoring conditions in parallel:
  - pre-norm:  reddit_toxic_sample_raw.csv  → toxic_openai_scored.csv
  - post-norm: reddit_toxic_sample.csv      → toxic_openai_normalized_scored.csv

Results are persisted to a Modal Volume so they survive container restarts.
Final CSVs are downloaded back to data/processed/ automatically.

Setup (one-time):
    modal secret create openai-key OPENAI_API_KEY=sk-proj-...

Run:
    modal run modal_score.py

Monitor:
    modal.com/apps  →  trust-safety-scoring

Download results after completion:
    modal run modal_score.py::download
"""

import io
import csv
import json
import logging
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App + infrastructure
# ---------------------------------------------------------------------------

app = modal.App("trust-safety-scoring-v2")

# Persistent volume — survives container restarts and is readable after the job
volume = modal.Volume.from_name("trust-safety-data", create_if_missing=True)
VOLUME_PATH = Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.0", "requests")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_COLS = [
    "openai_flagged",
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
]
NULL_ROW = {col: "" for col in SCORE_COLS}
RATE = 0.5          # requests per second (30 RPM)
SAVE_EVERY = 100    # checkpoint interval


# ---------------------------------------------------------------------------
# Core scoring logic (self-contained, no local imports)
# ---------------------------------------------------------------------------

def _parse_result(result) -> dict:
    scores = result.category_scores
    return {
        "openai_flagged": result.flagged,
        "openai_hate": scores.hate,
        "openai_hate_threatening": scores.hate_threatening,
        "openai_harassment": scores.harassment,
        "openai_harassment_threatening": scores.harassment_threatening,
        "openai_self_harm": scores.self_harm,
    }


def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(out.getvalue(), encoding="utf-8")


def _score(
    input_csv_bytes: bytes,
    scored_csv_bytes: bytes,
    label: str,
    openai_key: str,
    out_path: Path,
) -> None:
    from openai import OpenAI

    log = logging.getLogger(label)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s")

    client = OpenAI(api_key=openai_key)
    delay = 1.0 / RATE

    # Load texts
    texts = [
        row.get("text", "") or ""
        for row in csv.DictReader(io.StringIO(input_csv_bytes.decode("utf-8")))
    ]

    # Load or initialise scored CSV
    reader = csv.DictReader(io.StringIO(scored_csv_bytes.decode("utf-8")))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

    # Add score columns if not present
    for col in SCORE_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = ""

    # Find null rows
    null_indices = [
        i for i, row in enumerate(rows)
        if not row.get("openai_flagged", "").strip()
        or row.get("openai_flagged", "").strip() == "None"
    ]
    log.info("Found %d rows needing scores", len(null_indices))

    scored = flagged = errors = 0

    for pos, idx in enumerate(null_indices, 1):
        text = texts[idx][:10_000]
        backoff = 5
        while True:
            try:
                response = client.moderations.create(
                    model="omni-moderation-latest",
                    input=text,
                )
                result = _parse_result(response.results[0])
                for col, val in result.items():
                    if col in fieldnames:
                        rows[idx][col] = "" if val is None else val
                scored += 1
                if result["openai_flagged"]:
                    flagged += 1
                break
            except Exception as e:
                if "429" in str(e) or "RateLimitError" in type(e).__name__ or "Too Many" in str(e):
                    log.warning("429 on row %d — waiting %ds", idx, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                else:
                    log.warning("Row %d failed: %s", idx, e)
                    errors += 1
                    break

        if pos % SAVE_EVERY == 0 or pos == len(null_indices):
            _write_csv(out_path, fieldnames, rows)
            volume.commit()
            log.info("Progress: %d/%d scored | flagged: %d | errors: %d",
                     pos, len(null_indices), flagged, errors)

        time.sleep(delay)

    log.info("Done. Scored %d | flagged %d | errors %d", scored, flagged, errors)


# ---------------------------------------------------------------------------
# Modal functions — one per condition, run in parallel
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("openai-key")],
    timeout=86400,      # 24h hard limit
    memory=512,
)
def score_pre_norm(input_csv: bytes, scored_csv: bytes) -> bytes:
    """Score raw (pre-normalization) toxic sample."""
    import os
    out_path = VOLUME_PATH / "toxic_openai_scored.csv"
    _score(input_csv, scored_csv, "pre-norm", os.environ["OPENAI_API_KEY"], out_path)
    return out_path.read_bytes()


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("openai-key")],
    timeout=86400,
    memory=512,
)
def score_post_norm(input_csv: bytes, scored_csv: bytes) -> bytes:
    """Score normalized (post-normalization) toxic sample."""
    import os
    out_path = VOLUME_PATH / "toxic_openai_normalized_scored.csv"
    _score(input_csv, scored_csv, "post-norm", os.environ["OPENAI_API_KEY"], out_path)
    return out_path.read_bytes()


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run():
    """Upload CSVs, launch both scoring jobs in parallel, download results."""
    import sys
    local_data = Path("data/processed")

    raw_csv      = (local_data / "reddit_toxic_sample_raw.csv").read_bytes()
    norm_csv     = (local_data / "reddit_toxic_sample.csv").read_bytes()

    # Bootstrap scored CSVs from input if they don't have score columns yet
    pre_scored   = (local_data / "toxic_openai_scored.csv").read_bytes() \
                   if (local_data / "toxic_openai_scored.csv").exists() else raw_csv
    post_scored  = (local_data / "toxic_openai_normalized_scored.csv").read_bytes() \
                   if (local_data / "toxic_openai_normalized_scored.csv").exists() else norm_csv

    print("Launching pre-norm and post-norm scoring in parallel...")

    pre_handle  = score_pre_norm.spawn(raw_csv, pre_scored)
    post_handle = score_post_norm.spawn(norm_csv, post_scored)

    print(f"Jobs running. Monitor at: https://modal.com/apps/taahamohsin/main")
    print("Waiting for results...")

    pre_result  = pre_handle.get()
    post_result = post_handle.get()

    (local_data / "toxic_openai_scored.csv").write_bytes(pre_result)
    (local_data / "toxic_openai_normalized_scored.csv").write_bytes(post_result)

    print("Done. Results saved to data/processed/")


@app.function(image=image, volumes={str(VOLUME_PATH): volume})
def read_volume_file(filename: str) -> bytes:
    """Read a file from the Modal Volume."""
    p = VOLUME_PATH / filename
    return p.read_bytes() if p.exists() else b""


@app.local_entrypoint()
def download():
    """Re-download results from Modal Volume without re-running scoring."""
    local_data = Path("data/processed")
    for name in ["toxic_openai_scored.csv", "toxic_openai_normalized_scored.csv"]:
        data = read_volume_file.remote(name)
        if data:
            (local_data / name).write_bytes(data)
            print(f"Downloaded {name} ({len(data):,} bytes)")
        else:
            print(f"{name} not found in volume yet")
