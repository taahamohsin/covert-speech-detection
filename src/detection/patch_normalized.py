"""
One-off script: apply all successful results from today's batch output files
into openai_normalized_scored.csv, then report what still needs retrying.

Usage:
    conda run python3 src/detection/patch_normalized.py
"""

import csv
import glob
import io
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
SCORED_CSV = BASE / "data/processed/openai_normalized_scored.csv"
SCORE_COLS = [
    "openai_flagged",
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
    "openai_violence",
    "openai_illicit",
]


def parse_result(response_body: dict) -> dict | None:
    try:
        r = response_body["results"][0]
        scores = r["category_scores"]
        return {
            "openai_flagged": r["flagged"],
            "openai_hate": scores["hate"],
            "openai_hate_threatening": scores["hate/threatening"],
            "openai_harassment": scores["harassment"],
            "openai_harassment_threatening": scores["harassment/threatening"],
            "openai_self_harm": scores["self-harm"],
            "openai_violence": scores.get("violence"),
            "openai_illicit": scores.get("illicit"),
        }
    except Exception:
        return None


def load_all_successes() -> dict[int, dict]:
    scores: dict[int, dict] = {}
    for f in sorted(BASE.glob("batch_*_output.jsonl")):
        count = 0
        for line in open(f):
            d = json.loads(line)
            if d.get("response", {}).get("status_code") != 200:
                continue
            idx = int(d["custom_id"])
            if idx in scores:
                continue  # already have it from an earlier file
            result = parse_result(d["response"]["body"])
            if result:
                scores[idx] = result
                count += 1
        log.info("%-70s  +%d (total so far: %d)", f.name, count, len(scores))
    return scores


def patch_csv(scores: dict[int, dict]) -> None:
    with open(SCORED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    patched = 0
    for i, row in enumerate(rows):
        if i in scores:
            for col, val in scores[i].items():
                if col not in fieldnames:
                    continue
                row[col] = "" if val is None else val
            patched += 1

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    with open(SCORED_CSV, "w", newline="", encoding="utf-8") as f:
        f.write(out.getvalue())

    log.info("Patched %d rows into %s", patched, SCORED_CSV)


def report_remaining(scores: dict[int, dict], total: int) -> list[int]:
    null_ids = [i for i in range(total) if i not in scores]
    log.info("Remaining null rows: %d / %d", len(null_ids), total)
    return null_ids


if __name__ == "__main__":
    log.info("Loading successes from all batch output files...")
    scores = load_all_successes()
    log.info("Total unique successes found: %d", len(scores))

    patch_csv(scores)

    # Report what still needs retrying
    with open(SCORED_CSV, newline="", encoding="utf-8") as f:
        total = sum(1 for _ in csv.DictReader(f))

    null_ids = report_remaining(scores, total)

    if null_ids:
        log.info("Next retry command:")
        log.info(
            "  conda run python3 src/detection/openai_retry.py \\\n"
            "    --error-jsonl <any_error.jsonl> \\\n"
            "    --output-jsonl <any_output.jsonl> \\\n"
            "    --input data/processed/reddit_normalized.csv \\\n"
            "    --scored data/processed/openai_normalized_scored.csv \\\n"
            "    --chunk-size 2000\n"
            "  (openai_retry.py will be updated to accept --null-ids mode)"
        )
        # Write the null IDs to a simple text file for reference
        null_path = BASE / "data/processed/openai_normalized_null_ids.txt"
        null_path.write_text("\n".join(map(str, null_ids)))
        log.info("Null IDs written to %s", null_path)
    else:
        log.info("All rows scored — openai_normalized_scored.csv is complete.")
