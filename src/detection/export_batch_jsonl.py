"""
Exports the dataset as a JSONL file for manual upload to OpenAI Batch API.
Upload the output file at platform.openai.com/batches.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def run(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df["text"] = df["text"].fillna("").astype(str)

    lines = []
    for i, text in enumerate(df["text"]):
        lines.append(json.dumps({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/moderations",
            "body": {"input": text[:10_000]},
        }))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Written {len(lines)} requests to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/reddit_raw_20260406_011406.csv")
    parser.add_argument("--output", default="data/processed/openai_batch_input.jsonl")
    args = parser.parse_args()
    run(args.input, args.output)
