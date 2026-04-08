"""
Parses OpenAI batch output + error files, merges successes,
and resubmits failed items as a new batch.

Usage:
  # First run: parse results and resubmit failures
  conda run python3 src/detection/parse_and_retry_batch.py \
    --input data/raw/reddit_raw_20260406_011406.csv \
    --output-file batch_<id>_output.jsonl \
    --error-file batch_<id>_error.jsonl \
    --state data/processed/openai_state.json

  # Subsequent runs (after downloading new batch output):
  conda run python3 src/detection/parse_and_retry_batch.py \
    --input data/raw/reddit_raw_20260406_011406.csv \
    --output-file batch_<new_id>_output.jsonl \
    --error-file batch_<new_id>_error.jsonl \
    --state data/processed/openai_state.json \
    --final-output data/processed/openai_scored.csv  # add when all done
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI


NULL_ROW = {
    "openai_flagged": None,
    "openai_hate": None,
    "openai_hate_threatening": None,
    "openai_harassment": None,
    "openai_harassment_threatening": None,
    "openai_self_harm": None,
    "openai_violence": None,
    "openai_illicit": None,
}


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_response(body: dict) -> dict:
    r = body["results"][0]
    s = r["category_scores"]
    return {
        "openai_flagged": r["flagged"],
        "openai_hate": s["hate"],
        "openai_hate_threatening": s["hate/threatening"],
        "openai_harassment": s["harassment"],
        "openai_harassment_threatening": s["harassment/threatening"],
        "openai_self_harm": s["self-harm"],
        "openai_violence": s.get("violence"),
        "openai_illicit": s.get("illicit"),
    }


def load_state(state_path: str) -> dict:
    if Path(state_path).exists():
        with open(state_path) as f:
            return json.load(f)
    return {}  # custom_id (str) -> score dict


def save_state(state: dict, state_path: str) -> None:
    with open(state_path, "w") as f:
        json.dump(state, f)


def build_jsonl(indices: list[int], texts: list[str]) -> bytes:
    lines = []
    for i in indices:
        lines.append(json.dumps({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/moderations",
            "body": {"model": "omni-moderation-latest", "input": texts[i][:10_000]},
        }))
    return "\n".join(lines).encode("utf-8")


def run(args):
    cfg = load_config(args.config)
    client = OpenAI(api_key=cfg["apis"]["openai"]["key"])

    df = pd.read_csv(args.input)
    texts = df["text"].fillna("").astype(str).tolist()
    total = len(texts)

    # Load existing state (scores accumulated across runs)
    state = load_state(args.state)
    print(f"State: {len(state)} items already scored")

    # Parse successes from output file
    new_successes = 0
    with open(args.output_file) as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["custom_id"]
            if cid in state:
                continue
            if rec["response"]["status_code"] == 200:
                state[cid] = parse_response(rec["response"]["body"])
                new_successes += 1

    print(f"New successes from output file: {new_successes}")
    print(f"Total scored so far: {len(state)} / {total}")

    # Find remaining failures
    failed_ids = [i for i in range(total) if str(i) not in state]
    print(f"Still failing: {len(failed_ids)}")

    save_state(state, args.state)

    # If all done, write final CSV
    if not failed_ids or args.final_output:
        if failed_ids:
            print(f"WARNING: {len(failed_ids)} items still missing — filling with nulls")
        results = [state.get(str(i), NULL_ROW) for i in range(total)]
        scores_df = pd.DataFrame(results)
        out_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
        out_path = args.final_output or "data/processed/openai_scored.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        flagged = out_df["openai_flagged"].sum()
        print(f"Saved {len(out_df)} items to {out_path}")
        print(f"Flagged: {flagged} / {len(out_df)} ({100*flagged/len(out_df):.1f}%)")
        return

    # Submit new batch for failures
    print(f"Submitting new batch for {len(failed_ids)} failed items...")
    jsonl_bytes = build_jsonl(failed_ids, texts)
    batch_file = client.files.create(
        file=("retry_input.jsonl", jsonl_bytes, "application/jsonl"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/moderations",
        completion_window="24h",
    )
    print(f"New batch submitted: {batch.id}")
    print(f"Download results from: platform.openai.com/batches/{batch.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/reddit_raw_20260406_011406.csv")
    parser.add_argument("--output-file", required=True, help="Batch output JSONL")
    parser.add_argument("--error-file", required=True, help="Batch error JSONL")
    parser.add_argument("--state", default="data/processed/openai_state.json")
    parser.add_argument("--final-output", help="Set to write final CSV (even if incomplete)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args)
