"""
Retry failed OpenAI Moderation API batch requests.

Reads the error JSONL from a previous batch run, extracts 429-failed IDs,
resubmits them as a new batch, and patches the existing scored CSV with results.

Usage:
    conda run python3 src/detection/openai_retry.py \
        --error-jsonl batch_<id>_error.jsonl \
        --output-jsonl batch_<id>_output.jsonl \
        --input data/processed/reddit_normalized.csv \
        --scored data/processed/openai_scored.csv \
        [--chunk-size 2000]
"""

import argparse
import json
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
    "openai_violence",
    "openai_illicit",
]
NULL_ROW = {col: None for col in SCORE_COLS}


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_failed_ids(error_jsonl: str) -> list[int]:
    """Return sorted list of custom_ids that failed with 429."""
    failed = []
    with open(error_jsonl) as f:
        for line in f:
            record = json.loads(line)
            status = record.get("response", {}).get("status_code")
            if status == 429:
                failed.append(int(record["custom_id"]))
    log.info("Found %d failed IDs (429) in %s", len(failed), error_jsonl)
    return sorted(failed)


def load_null_ids_from_csv(scored_csv: str) -> list[int]:
    """Return sorted list of row indices where openai_flagged is null/empty."""
    import csv as _csv
    null_ids = []
    with open(scored_csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(_csv.DictReader(f)):
            val = row.get("openai_flagged", "").strip()
            if not val or val == "None":
                null_ids.append(i)
    log.info("Found %d null rows in %s", len(null_ids), scored_csv)
    return null_ids


def load_existing_results(output_jsonl: str) -> dict[int, dict]:
    """Parse existing successful results keyed by custom_id."""
    results = {}
    with open(output_jsonl) as f:
        for line in f:
            record = json.loads(line)
            idx = int(record["custom_id"])
            if record.get("error") or record.get("response", {}).get("status_code") != 200:
                continue
            body = record["response"]["body"]
            try:
                r = body["results"][0]
                scores = r["category_scores"]
                results[idx] = {
                    "openai_flagged": r["flagged"],
                    "openai_hate": scores["hate"],
                    "openai_hate_threatening": scores["hate/threatening"],
                    "openai_harassment": scores["harassment"],
                    "openai_harassment_threatening": scores["harassment/threatening"],
                    "openai_self_harm": scores["self-harm"],
                    "openai_violence": scores.get("violence", None),
                    "openai_illicit": scores.get("illicit", None),
                }
            except Exception:
                results[idx] = NULL_ROW
    log.info("Loaded %d existing results from %s", len(results), output_jsonl)
    return results


def load_texts(input_csv: str) -> list[str]:
    """Load text column from CSV without pandas dependency."""
    import csv
    texts = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row.get("text", "") or "")
    return texts


def build_jsonl(ids_and_texts: list[tuple[int, str]]) -> bytes:
    lines = []
    for idx, text in ids_and_texts:
        lines.append(json.dumps({
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/moderations",
            "body": {"model": "omni-moderation-latest", "input": text[:10_000]},
        }))
    return "\n".join(lines).encode("utf-8")


def parse_result(response_body: dict) -> dict:
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
            "openai_violence": scores.get("violence", None),
            "openai_illicit": scores.get("illicit", None),
        }
    except Exception:
        return NULL_ROW


def submit_and_poll(client: OpenAI, jsonl_bytes: bytes, chunk_label: str) -> dict[int, dict]:
    """Upload, submit, poll a batch and return {custom_id: scores}."""
    log.info("[%s] Uploading %d bytes...", chunk_label, len(jsonl_bytes))
    batch_file = client.files.create(
        file=("batch_input.jsonl", jsonl_bytes, "application/jsonl"),
        purpose="batch",
    )
    log.info("[%s] File uploaded: %s", chunk_label, batch_file.id)

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/moderations",
        completion_window="24h",
    )
    log.info("[%s] Batch created: %s (status: %s)", chunk_label, batch.id, batch.status)

    poll_interval = 10
    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        log.info(
            "[%s] Status: %s | completed: %d, failed: %d, total: %d",
            chunk_label,
            batch.status,
            counts.completed,
            counts.failed,
            counts.total,
        )
        poll_interval = min(poll_interval * 1.5, 60)

    if batch.status != "completed":
        log.error("[%s] Batch ended with status: %s", chunk_label, batch.status)
        return {}

    log.info("[%s] Downloading results...", chunk_label)
    content = client.files.content(batch.output_file_id).content
    results = {}
    new_failures = 0
    for line in content.decode("utf-8").strip().split("\n"):
        record = json.loads(line)
        idx = int(record["custom_id"])
        status = record.get("response", {}).get("status_code")
        if record.get("error") or status != 200:
            new_failures += 1
            results[idx] = NULL_ROW
        else:
            results[idx] = parse_result(record["response"]["body"])

    if new_failures:
        log.warning("[%s] %d items still failed after retry", chunk_label, new_failures)

    # Save raw output for this chunk
    chunk_out = Path(f"batch_retry_{batch.id}_output.jsonl")
    chunk_out.write_bytes(content)
    log.info("[%s] Raw output saved to %s", chunk_label, chunk_out)

    return results


def patch_csv(scored_csv: str, all_scores: dict[int, dict]) -> None:
    """Overwrite scored CSV, filling in scores for previously-null rows."""
    import csv, io

    with open(scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    patched = 0
    for i, row in enumerate(rows):
        if i in all_scores:
            for col, val in all_scores[i].items():
                if col not in fieldnames:
                    continue
                row[col] = "" if val is None else val
            patched += 1

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        f.write(out.getvalue())

    log.info("Patched %d rows in %s", patched, scored_csv)


def run(
    input_csv: str,
    scored_csv: str,
    chunk_size: int,
    error_jsonl: str | None = None,
    output_jsonl: str | None = None,
    config_path: str = "config/config.yaml",
) -> None:
    cfg = load_config(config_path)
    client = OpenAI(api_key=cfg["apis"]["openai"]["key"])

    # Prefer null-ids-from-CSV mode; fall back to error JSONL
    if error_jsonl:
        failed_ids = load_failed_ids(error_jsonl)
    else:
        failed_ids = load_null_ids_from_csv(scored_csv)

    existing_scores = load_existing_results(output_jsonl) if output_jsonl else {}
    texts = load_texts(input_csv)

    log.info("Total texts loaded: %d", len(texts))

    # Collect new results across all chunks
    new_scores: dict[int, dict] = {}

    chunks = [failed_ids[i:i + chunk_size] for i in range(0, len(failed_ids), chunk_size)]
    log.info("Submitting %d failed items in %d chunk(s) of up to %d", len(failed_ids), len(chunks), chunk_size)

    for ci, chunk_ids in enumerate(chunks):
        ids_and_texts = [(idx, texts[idx]) for idx in chunk_ids if idx < len(texts)]
        jsonl_bytes = build_jsonl(ids_and_texts)
        chunk_scores = submit_and_poll(client, jsonl_bytes, f"chunk {ci+1}/{len(chunks)}")
        new_scores.update(chunk_scores)

    # Merge: existing successes + new retry results
    all_scores = {**existing_scores, **new_scores}
    log.info("Total resolved items: %d / %d", len(all_scores), len(texts))

    flagged = sum(1 for v in all_scores.values() if v.get("openai_flagged"))
    log.info("Flagged: %d (%.1f%%)", flagged, 100 * flagged / max(len(all_scores), 1))

    patch_csv(scored_csv, all_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retry failed OpenAI Moderation batch requests.")
    parser.add_argument("--error-jsonl", default=None, help="Path to batch error JSONL (optional; if omitted, null rows in --scored are used)")
    parser.add_argument("--output-jsonl", default=None, help="Path to batch output JSONL file (prior successes, optional)")
    parser.add_argument("--input", required=True, help="Original input CSV (same as used for initial batch)")
    parser.add_argument("--scored", required=True, help="Existing scored CSV to patch with retry results")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Items per retry batch (default: 2000)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    run(
        error_jsonl=args.error_jsonl,
        output_jsonl=args.output_jsonl,
        input_csv=args.input,
        scored_csv=args.scored,
        chunk_size=args.chunk_size,
        config_path=args.config,
    )
