"""
Merges scored chunk CSVs back into the main scored CSVs.
Reads chunk_pre_A.csv, chunk_pre_B.csv, chunk_pre_C.csv (and post equivalents),
patches scores into toxic_openai_scored.csv and toxic_openai_normalized_scored.csv
using _orig_idx to map rows back correctly.

Run after all three Modal jobs complete:
    python3 merge_chunks.py
"""

import csv
from pathlib import Path

DATA = Path("data/processed")

SCORE_COLS = [
    "openai_flagged",
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
]

LABELS = ["A", "B", "C"]


def merge(scored_csv: Path, tag: str):
    # Load main CSV
    with open(scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    patched = 0
    for label in LABELS:
        chunk_path = DATA / f"chunk_{tag}_{label}_scored.csv"
        if not chunk_path.exists():
            print(f"  WARNING: {chunk_path.name} not found, skipping")
            continue

        with open(chunk_path, newline="", encoding="utf-8") as f:
            chunk_rows = list(csv.DictReader(f))

        for chunk_row in chunk_rows:
            orig_idx = int(chunk_row["_orig_idx"])
            if chunk_row.get("openai_flagged", "").strip() not in ("", "None"):
                for col in SCORE_COLS:
                    rows[orig_idx][col] = chunk_row.get(col, "")
                patched += 1

        print(f"  Chunk {label}: processed {len(chunk_rows)} rows")

    # Write back
    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scored = sum(1 for r in rows if r.get("openai_flagged", "").strip() not in ("", "None"))
    flagged = sum(1 for r in rows if str(r.get("openai_flagged", "")).strip() == "True")
    print(f"  Patched {patched} rows. Total scored: {scored}/{len(rows)}, flagged: {flagged} ({100*flagged/scored:.1f}%)")


def merge_unscored(scored_csv: Path, unscored_scored_csv: Path):
    """Merge unscored_pre_scored.csv / unscored_post_scored.csv back by item_id."""
    with open(scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Build item_id index
    id_to_idx = {r["item_id"]: i for i, r in enumerate(rows)}

    if not unscored_scored_csv.exists():
        print(f"  WARNING: {unscored_scored_csv.name} not found, skipping")
        return

    with open(unscored_scored_csv, newline="", encoding="utf-8") as f:
        unscored_rows = list(csv.DictReader(f))

    patched = 0
    for row in unscored_rows:
        if row.get("openai_flagged", "").strip() not in ("", "None"):
            idx = id_to_idx.get(row["item_id"])
            if idx is not None:
                for col in SCORE_COLS:
                    rows[idx][col] = row.get(col, "")
                patched += 1

    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scored = sum(1 for r in rows if r.get("openai_flagged", "").strip() not in ("", "None"))
    flagged = sum(1 for r in rows if str(r.get("openai_flagged", "")).strip() == "True")
    print(f"  Patched {patched} rows from {unscored_scored_csv.name}. Total scored: {scored}/{len(rows)}, flagged: {flagged} ({100*flagged/scored:.1f}%)")


if __name__ == "__main__":
    print("=== Merging pre-norm chunks ===")
    merge(DATA / "toxic_openai_scored.csv", "pre")
    print()
    print("=== Merging post-norm chunks ===")
    merge(DATA / "toxic_openai_normalized_scored.csv", "post")
    print()
    print("=== Merging unscored pre-norm ===")
    merge_unscored(DATA / "toxic_openai_scored.csv", DATA / "unscored_pre_scored.csv")
    print()
    print("=== Merging unscored post-norm ===")
    merge_unscored(DATA / "toxic_openai_normalized_scored.csv", DATA / "unscored_post_scored.csv")
