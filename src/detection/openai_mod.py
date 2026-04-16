"""
Component 2a: OpenAI Moderation API baseline detection via Batch API.
Submits all items as a single batch job (async), polls until complete, then saves results.
Output CSV adds columns: openai_flagged, openai_hate, openai_hate_threatening,
openai_harassment, openai_harassment_threatening, openai_self_harm.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCORE_COLS = [
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
]
NULL_ROW = {col: None for col in ["openai_flagged"] + SCORE_COLS}


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_jsonl(texts: list[str]) -> bytes:
    """Build JSONL batch input file content."""
    lines = []
    for i, text in enumerate(texts):
        lines.append(json.dumps({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/moderations",
            "body": {"model": "omni-moderation-latest", "input": text[:10_000]},
        }))
    return "\n".join(lines).encode("utf-8")


def parse_result(response_body: dict) -> dict:
    """Extract scores from a single moderation response body."""
    try:
        result = response_body["results"][0]
        scores = result["category_scores"]
        return {
            "openai_flagged": result["flagged"],
            "openai_hate": scores["hate"],
            "openai_hate_threatening": scores["hate/threatening"],
            "openai_harassment": scores["harassment"],
            "openai_harassment_threatening": scores["harassment/threatening"],
            "openai_self_harm": scores["self-harm"],
        }
    except Exception:
        return NULL_ROW


def run(input_path: str, output_path: str, config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    client = OpenAI(api_key=cfg["apis"]["openai"]["key"])

    df = pd.read_csv(input_path)
    log.info("Loaded %d items from %s", len(df), input_path)
    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].tolist()

    # Step 1: Upload JSONL batch file
    log.info("Uploading batch input file (%d requests)...", len(texts))
    jsonl_bytes = build_jsonl(texts)
    batch_file = client.files.create(
        file=("batch_input.jsonl", jsonl_bytes, "application/jsonl"),
        purpose="batch",
    )
    log.info("File uploaded: %s", batch_file.id)

    # Step 2: Create batch job
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/moderations",
        completion_window="24h",
    )
    log.info("Batch created: %s (status: %s)", batch.id, batch.status)

    # Step 3: Poll until complete
    poll_interval = 10
    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        log.info(
            "Status: %s | completed: %d, failed: %d, total: %d",
            batch.status,
            counts.completed,
            counts.failed,
            counts.total,
        )
        poll_interval = min(poll_interval * 1.5, 60)

    if batch.status != "completed":
        log.error("Batch ended with status: %s", batch.status)
        return

    # Step 4: Download and parse results
    log.info("Batch complete. Downloading results...")
    result_content = client.files.content(batch.output_file_id).content
    result_lines = result_content.decode("utf-8").strip().split("\n")

    # Build index: custom_id (str int) → score dict
    scores_by_idx = {}
    for line in result_lines:
        record = json.loads(line)
        idx = int(record["custom_id"])
        if record.get("error"):
            log.warning("Item %d errored: %s", idx, record["error"])
            scores_by_idx[idx] = NULL_ROW
        else:
            scores_by_idx[idx] = parse_result(record["response"]["body"])

    # Align results with original dataframe order
    results = [scores_by_idx.get(i, NULL_ROW) for i in range(len(texts))]
    scores_df = pd.DataFrame(results)

    out_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    log.info("Saved %d scored items to %s", len(out_df), output_path)
    log.info(
        "Flagged: %d / %d (%.1f%%)",
        out_df["openai_flagged"].sum(),
        len(out_df),
        100 * out_df["openai_flagged"].mean(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Moderation Batch API on collected data.")
    parser.add_argument("--input", required=True, help="Path to raw CSV (data/raw/*.csv)")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.input, args.output, args.config)
