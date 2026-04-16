"""
Modal cloud runner for scoring chunk CSVs in parallel across 3 API keys.

Each chunk app scores its assigned chunk file and writes results back to the
Modal Volume. Run all three in parallel from separate terminals:

    modal run --detach modal_score_chunk.py::run_A
    modal run --detach modal_score_chunk.py::run_B
    modal run --detach modal_score_chunk.py::run_C

Download and merge when all done:
    modal run modal_score_chunk.py::download
    python3 merge_chunks.py

Secrets required (create once):
    modal secret create openai-key    OPENAI_API_KEY=sk-...  # Tier 1 (already exists)
    modal secret create openai-key-2  OPENAI_API_KEY=sk-...  # Free key 2
    modal secret create openai-key-3  OPENAI_API_KEY=sk-...  # Free key 3
"""

import csv
import io
import logging
import os
import time
from pathlib import Path

import modal

app = modal.App("trust-safety-chunks")

volume = modal.Volume.from_name("trust-safety-data", create_if_missing=True)
VOLUME_PATH = Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.0")
)

SCORE_COLS = [
    "openai_flagged",
    "openai_hate",
    "openai_hate_threatening",
    "openai_harassment",
    "openai_harassment_threatening",
    "openai_self_harm",
]

SAVE_EVERY = 50


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


def _score_chunk(chunk_csv_bytes: bytes, label: str, openai_key: str, out_filename: str, rate: float) -> bytes:
    from openai import OpenAI

    log = logging.getLogger(label)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s")

    client = OpenAI(api_key=openai_key)
    delay = 1.0 / rate
    out_path = VOLUME_PATH / out_filename

    reader = csv.DictReader(io.StringIO(chunk_csv_bytes.decode("utf-8")))
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

    # Add score cols if not present
    for col in SCORE_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = ""

    # Resume from volume checkpoint if exists
    if out_path.exists():
        existing = list(csv.DictReader(io.StringIO(out_path.read_text(encoding="utf-8"))))
        for i, row in enumerate(existing):
            if i < len(rows) and row.get("openai_flagged", "").strip() not in ("", "None"):
                rows[i] = row

    null_indices = [
        i for i, row in enumerate(rows)
        if row.get("openai_flagged", "").strip() in ("", "None")
    ]
    log.info("Chunk %s: %d rows to score", label, len(null_indices))

    scored = flagged = errors = 0

    for pos, idx in enumerate(null_indices, 1):
        text = rows[idx].get("text", "")[:10_000]
        backoff = 5
        while True:
            try:
                response = client.moderations.create(
                    model="omni-moderation-latest",
                    input=text,
                )
                result = _parse_result(response.results[0])
                for col, val in result.items():
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
    return out_path.read_bytes()


# ---------------------------------------------------------------------------
# Chunk A — Tier 1 key, rate 0.5 req/s
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key")], timeout=86400, memory=512)
def score_chunk_pre_A(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "pre-A", os.environ["OPENAI_API_KEY"], "chunk_pre_A_scored.csv", rate=0.3)

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key")], timeout=86400, memory=512)
def score_chunk_post_A(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "post-A", os.environ["OPENAI_API_KEY"], "chunk_post_A_scored.csv", rate=0.3)


# ---------------------------------------------------------------------------
# Chunk B — Free key 2, rate 0.05 req/s (3 RPM)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key-2")], timeout=86400, memory=512)
def score_chunk_pre_B(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "pre-B", os.environ["OPENAI_API_KEY"], "chunk_pre_B_scored.csv", rate=0.05)

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key-2")], timeout=86400, memory=512)
def score_chunk_post_B(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "post-B", os.environ["OPENAI_API_KEY"], "chunk_post_B_scored.csv", rate=0.05)


# ---------------------------------------------------------------------------
# Chunk C — Free key 3, rate 0.05 req/s (3 RPM)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key-3")], timeout=86400, memory=512)
def score_chunk_pre_C(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "pre-C", os.environ["OPENAI_API_KEY"], "chunk_pre_C_scored.csv", rate=0.05)

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key-3")], timeout=86400, memory=512)
def score_chunk_post_C(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "post-C", os.environ["OPENAI_API_KEY"], "chunk_post_C_scored.csv", rate=0.05)


# ---------------------------------------------------------------------------
# Local entrypoints — run each from a separate terminal
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_A():
    local_data = Path("data/processed")
    print("Chunk A: scoring pre-norm first, then post-norm (sequential to avoid shared key contention)...")
    pre_result = score_chunk_pre_A.remote((local_data / "chunk_pre_A.csv").read_bytes())
    (local_data / "chunk_pre_A.csv").write_bytes(pre_result)
    print("Chunk A pre-norm done. Starting post-norm...")
    post_result = score_chunk_post_A.remote((local_data / "chunk_post_A.csv").read_bytes())
    (local_data / "chunk_post_A.csv").write_bytes(post_result)
    print("Chunk A done.")


@app.local_entrypoint()
def run_B():
    local_data = Path("data/processed")
    print("Chunk B: scoring pre-norm first, then post-norm (sequential to stay under free tier quota)...")
    pre_result = score_chunk_pre_B.remote((local_data / "chunk_pre_B.csv").read_bytes())
    (local_data / "chunk_pre_B.csv").write_bytes(pre_result)
    print("Chunk B pre-norm done. Starting post-norm...")
    post_result = score_chunk_post_B.remote((local_data / "chunk_post_B.csv").read_bytes())
    (local_data / "chunk_post_B.csv").write_bytes(post_result)
    print("Chunk B done.")


@app.local_entrypoint()
def run_C():
    local_data = Path("data/processed")
    print("Chunk C: scoring pre-norm first, then post-norm (sequential to stay under free tier quota)...")
    pre_result = score_chunk_pre_C.remote((local_data / "chunk_pre_C.csv").read_bytes())
    (local_data / "chunk_pre_C.csv").write_bytes(pre_result)
    print("Chunk C pre-norm done. Starting post-norm...")
    post_result = score_chunk_post_C.remote((local_data / "chunk_post_C.csv").read_bytes())
    (local_data / "chunk_post_C.csv").write_bytes(post_result)
    print("Chunk C done.")


# ---------------------------------------------------------------------------
# Finish — score remaining unscored rows sequentially with tier 1 key
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key")], timeout=86400, memory=512)
def score_unscored_pre(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "unscored-pre", os.environ["OPENAI_API_KEY"], "unscored_pre_scored.csv", rate=0.3)

@app.function(image=image, volumes={str(VOLUME_PATH): volume},
              secrets=[modal.Secret.from_name("openai-key")], timeout=86400, memory=512)
def score_unscored_post(chunk_csv: bytes) -> bytes:
    return _score_chunk(chunk_csv, "unscored-post", os.environ["OPENAI_API_KEY"], "unscored_post_scored.csv", rate=0.3)


@app.local_entrypoint()
def run_unscored():
    local_data = Path("data/processed")
    print("Scoring unscored_pre.csv first...")
    pre_result = score_unscored_pre.remote((local_data / "unscored_pre.csv").read_bytes())
    (local_data / "unscored_pre_scored.csv").write_bytes(pre_result)
    print("Pre done. Scoring unscored_post.csv...")
    post_result = score_unscored_post.remote((local_data / "unscored_post.csv").read_bytes())
    (local_data / "unscored_post_scored.csv").write_bytes(post_result)
    print("Done.")


@app.local_entrypoint()
def run_unscored_post():
    local_data = Path("data/processed")
    print("Scoring unscored_post.csv...")
    post_result = score_unscored_post.remote((local_data / "unscored_post.csv").read_bytes())
    (local_data / "unscored_post_scored.csv").write_bytes(post_result)
    print("Done.")


# ---------------------------------------------------------------------------
# Download all chunk results from Volume
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOLUME_PATH): volume})
def read_volume_file(filename: str) -> bytes:
    p = VOLUME_PATH / filename
    return p.read_bytes() if p.exists() else b""


@app.local_entrypoint()
def download():
    local_data = Path("data/processed")
    for name in [
        "chunk_pre_A_scored.csv", "chunk_pre_B_scored.csv", "chunk_pre_C_scored.csv",
        "chunk_post_A_scored.csv", "chunk_post_B_scored.csv", "chunk_post_C_scored.csv",
        "unscored_pre_scored.csv", "unscored_post_scored.csv",
    ]:
        data = read_volume_file.remote(name)
        if data:
            (local_data / name).write_bytes(data)
            print(f"Downloaded {name} ({len(data):,} bytes)")
        else:
            print(f"{name} not found in volume yet")
