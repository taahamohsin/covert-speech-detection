"""
Splits unscored rows from toxic_openai_scored.csv and toxic_openai_normalized_scored.csv
into 3 non-overlapping chunks based on throughput ratio:
  - Chunk A (58%): Tier 1 key — largest share
  - Chunk B (21%): Free key 2
  - Chunk C (21%): Free key 3

For each condition (pre-norm, post-norm), outputs:
  - chunk_pre_A.csv / chunk_pre_B.csv / chunk_pre_C.csv
  - chunk_post_A.csv / chunk_post_B.csv / chunk_post_C.csv

Each chunk file contains ONLY the unscored rows (with all original columns),
so each Modal job can treat it as a standalone scored CSV to fill in.

Merge script (merge_chunks.py) recombines results back into the main CSVs.
"""

import csv
import math
from pathlib import Path

DATA = Path("data/processed")

RATIO = [0.58, 0.21, 0.21]  # Tier1, Free2, Free3
LABELS = ["A", "B", "C"]

SCORE_COLS_OPENAI = [
    "openai_flagged",
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
]


def split(scored_csv: Path, input_csv: Path, tag: str):
    with open(scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    with open(input_csv, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))

    # Find unscored indices (scattered, not sequential)
    unscored_indices = [
        i for i, r in enumerate(rows)
        if r.get("openai_flagged", "").strip() in ("", "None")
    ]
    print(f"{tag}: {len(unscored_indices)} unscored rows out of {len(rows)}")

    # Compute chunk sizes
    total = len(unscored_indices)
    sizes = [math.floor(r * total) for r in RATIO]
    sizes[-1] = total - sum(sizes[:-1])  # remainder goes to last chunk

    # Split indices into chunks
    chunks = []
    start = 0
    for size in sizes:
        chunks.append(unscored_indices[start:start + size])
        start += size

    # Write each chunk as a standalone CSV
    # Each chunk CSV contains only the unscored rows with their original text
    for label, chunk_indices in zip(LABELS, chunks):
        chunk_rows = []
        for idx in chunk_indices:
            row = dict(rows[idx])
            # Make sure text comes from input CSV (same row order)
            row["text"] = input_rows[idx].get("text", "") if idx < len(input_rows) else row.get("text", "")
            chunk_rows.append((idx, row))

        out_path = DATA / f"chunk_{tag}_{label}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["_orig_idx"] + fieldnames)
            writer.writeheader()
            for idx, row in chunk_rows:
                writer.writerow({"_orig_idx": idx, **row})

        print(f"  Chunk {label}: {len(chunk_rows)} rows → {out_path.name}")

    # Also write a summary of scored indices for reference
    scored_indices = [
        i for i, r in enumerate(rows)
        if r.get("openai_flagged", "").strip() not in ("", "None")
    ]
    print(f"  Already scored: {len(scored_indices)} rows (will be preserved in merge)")


if __name__ == "__main__":
    print("=== Pre-norm ===")
    split(
        DATA / "toxic_openai_scored.csv",
        DATA / "reddit_toxic_sample_raw.csv",
        "pre",
    )
    print()
    print("=== Post-norm ===")
    split(
        DATA / "toxic_openai_normalized_scored.csv",
        DATA / "reddit_toxic_sample.csv",
        "post",
    )
