"""
Component 2a: OpenAI Moderation API baseline detection.
Passes each item's text through /v1/moderations and records all category scores.
Output CSV adds columns: openai_flagged, openai_hate, openai_hate_threatening,
openai_harassment, openai_harassment_threatening, openai_self_harm.
"""

import argparse
import csv
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


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_moderation(client: OpenAI, texts: list[str]) -> list[dict]:
    """Call /v1/moderations for a batch of texts. Returns one result per text."""
    response = client.moderations.create(input=texts)
    return response.results


def score_item(result) -> dict:
    scores = result.category_scores
    return {
        "openai_flagged": result.flagged,
        "openai_hate": scores.hate,
        "openai_hate_threatening": scores.hate_threatening,
        "openai_harassment": scores.harassment,
        "openai_harassment_threatening": scores.harassment_threatening,
        "openai_self_harm": scores.self_harm,
    }


def run(input_path: str, output_path: str, config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    client = OpenAI(api_key=cfg["apis"]["openai"]["key"])

    df = pd.read_csv(input_path)
    log.info("Loaded %d items from %s", len(df), input_path)

    # Fill missing text
    df["text"] = df["text"].fillna("").astype(str)

    results = []
    batch_size = 32  # OpenAI accepts up to 32 inputs per call
    total = len(df)

    for start in range(0, total, batch_size):
        batch = df["text"].iloc[start : start + batch_size].tolist()
        # Truncate to 10k chars per item (API limit)
        batch = [t[:10_000] for t in batch]

        try:
            moderation_results = run_moderation(client, batch)
            results.extend(score_item(r) for r in moderation_results)
        except Exception as exc:
            log.error("Batch %d-%d failed: %s", start, start + batch_size, exc)
            # Fill with nulls so row count stays aligned
            results.extend([{col: None for col in ["openai_flagged"] + SCORE_COLS}] * len(batch))
            time.sleep(2)

        if (start // batch_size + 1) % 10 == 0:
            log.info("  Progress: %d/%d items scored", min(start + batch_size, total), total)

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
    parser = argparse.ArgumentParser(description="Run OpenAI Moderation API on collected data.")
    parser.add_argument("--input", required=True, help="Path to raw CSV (data/raw/*.csv)")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.input, args.output, args.config)
