"""
Synchronous fallback for OpenAI Moderation API.
Reads null rows from an existing scored CSV, calls the API directly with rate
limiting, and patches results in place. Fully resumable — already-scored rows
are skipped automatically.

Usage:
    python3 src/detection/openai_sync.py \
        --input data/processed/reddit_normalized.csv \
        --scored data/processed/openai_normalized_scored.csv \
        [--rate 20]   # requests per second (default: 20)
"""

import argparse
import csv
import io
import logging
import time
from pathlib import Path

import yaml
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCORE_COLS = [
    "openai_flagged",
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_result(result) -> dict:
    scores = result.category_scores
    return {
        "openai_flagged": result.flagged,
        "openai_hate": scores.hate,
        "openai_hate_threatening": scores.hate_threatening,
        "openai_harassment": scores.harassment,
        "openai_harassment_threatening": scores.harassment_threatening,
        "openai_self_harm": scores.self_harm,
    }


def run(input_csv: str, scored_csv: str, rate: float, config_path: str) -> None:
    cfg = load_config(config_path)
    client = OpenAI(api_key=cfg["apis"]["openai"]["key"])
    delay = 1.0 / rate

    # Load texts from input CSV
    with open(input_csv, newline="", encoding="utf-8") as f:
        texts = [row.get("text", "") or "" for row in csv.DictReader(f)]

    # Load scored CSV
    with open(scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add score columns if bootstrapping from a file that doesn't have them yet
    for col in SCORE_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = ""

    # Find null rows
    null_indices = [
        i
        for i, row in enumerate(rows)
        if not row.get("openai_flagged", "").strip()
        or row.get("openai_flagged", "").strip() == "None"
    ]
    log.info("Found %d rows still needing scores", len(null_indices))

    scored = 0
    flagged = 0
    errors = 0
    save_every = 100

    for pos, idx in enumerate(null_indices, 1):
        text = texts[idx][:10_000]
        backoff = 5
        while True:
            try:
                response = client.moderations.create(
                    model="omni-moderation-latest",
                    input=text,
                )
                result = parse_result(response.results[0])
                for col, val in result.items():
                    if col in fieldnames:
                        rows[idx][col] = "" if val is None else val
                scored += 1
                if result["openai_flagged"]:
                    flagged += 1
                break
            except Exception as e:
                if (
                    "429" in str(e)
                    or "Too Many Requests" in str(e)
                    or "RateLimitError" in type(e).__name__
                ):
                    log.warning(
                        "429 on row %d — waiting %ds before retry", idx, backoff
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                else:
                    log.warning("Row %d failed (non-retryable): %s", idx, e)
                    errors += 1
                    break

        if pos % save_every == 0 or pos == len(null_indices):
            _write_csv(scored_csv, fieldnames, rows)
            log.info(
                "Progress: %d/%d scored | flagged: %d | errors: %d",
                pos,
                len(null_indices),
                flagged,
                errors,
            )

        time.sleep(delay)

    log.info("Done. Scored %d rows, %d flagged, %d errors.", scored, flagged, errors)


def _write_csv(path: str, fieldnames: list, rows: list) -> None:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(out.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--scored", required=True)
    parser.add_argument(
        "--rate", type=float, default=3, help="Requests per second (default: 3)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.input, args.scored, args.rate, args.config)
