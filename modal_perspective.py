"""
Modal cloud runner for Perspective API scoring of the toxic sample.

Runs two scoring conditions in parallel:
  - pre-norm:  reddit_toxic_sample_raw.csv  → toxic_perspective_scored.csv
  - post-norm: reddit_toxic_sample.csv      → toxic_perspective_normalized_scored.csv

Results persist to a Modal Volume (shared with modal_score.py).

Setup (one-time, already done):
    modal secret create perspective-key PERSPECTIVE_SA_JSON="$(cat config/service-account.json)"

Run (detached — safe to close laptop):
    modal run --detach modal_perspective.py::run

Download results when done:
    modal run modal_perspective.py::download
"""

import csv
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import modal

app = modal.App("trust-safety-perspective")

volume = modal.Volume.from_name("trust-safety-data", create_if_missing=True)
VOLUME_PATH = Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "google-auth>=2.0")
)

PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
ATTRIBUTES = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]
FLAGGED_THRESHOLD = 0.7
RPM = 55
SAVE_EVERY = 100


def _build_credentials(sa_json_str: str):
    from google.oauth2 import service_account
    info = json.loads(sa_json_str)
    return service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/userinfo.email"]
    )


def _get_token(creds) -> str:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    if not creds.valid or creds.expiry is None:
        creds.refresh(GoogleAuthRequest())
    return creds.token


def _analyze(text: str, creds, token: str) -> dict:
    import requests
    body = {
        "comment": {"text": text[:20_000]},
        "requestedAttributes": {attr: {} for attr in ATTRIBUTES},
    }
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(
        PERSPECTIVE_URL,
        json=body,
        headers=headers,
        timeout=30,
    )
    if resp.status_code == 429:
        raise RuntimeError("429")
    resp.raise_for_status()
    data = resp.json()
    return {
        attr.lower(): data["attributeScores"][attr]["summaryScore"]["value"]
        for attr in ATTRIBUTES
        if attr in data.get("attributeScores", {})
    }


def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(out.getvalue(), encoding="utf-8")


SCORE_COLS = [f"perspective_{a.lower()}" for a in ATTRIBUTES] + ["perspective_flagged"]


def _score(
    input_csv_bytes: bytes,
    scored_csv_bytes: bytes,
    label: str,
    sa_json: str,
    out_path: Path,
) -> None:
    log = logging.getLogger(label)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s")

    creds = _build_credentials(sa_json)
    delay = 60.0 / RPM

    texts = [
        row.get("text", "") or ""
        for row in csv.DictReader(io.StringIO(input_csv_bytes.decode("utf-8")))
    ]

    reader = csv.DictReader(io.StringIO(scored_csv_bytes.decode("utf-8")))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

    for col in SCORE_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = ""

    null_indices = [
        i for i, row in enumerate(rows)
        if not row.get("perspective_flagged", "").strip()
        or row.get("perspective_flagged", "").strip() == "None"
    ]
    log.info("Found %d rows needing scores", len(null_indices))

    scored = flagged = errors = 0

    for pos, idx in enumerate(null_indices, 1):
        text = texts[idx]
        if not text.strip():
            for col in SCORE_COLS:
                rows[idx][col] = ""
            scored += 1
        else:
            backoff = 5
            while True:
                try:
                    token = _get_token(creds)
                    scores = _analyze(text, creds, token)
                    tox = scores.get("toxicity", 0.0)
                    ia  = scores.get("identity_attack", 0.0)
                    is_flagged = tox >= FLAGGED_THRESHOLD or ia >= FLAGGED_THRESHOLD
                    for attr in ATTRIBUTES:
                        col = f"perspective_{attr.lower()}"
                        rows[idx][col] = scores.get(attr.lower(), "")
                    rows[idx]["perspective_flagged"] = is_flagged
                    scored += 1
                    if is_flagged:
                        flagged += 1
                    break
                except Exception as e:
                    if "429" in str(e):
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


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("perspective-key")],
    timeout=86400,
    memory=512,
)
def score_perspective_pre_norm(input_csv: bytes, scored_csv: bytes) -> bytes:
    out_path = VOLUME_PATH / "toxic_perspective_scored.csv"
    _score(input_csv, scored_csv, "persp-pre", os.environ["PERSPECTIVE_SA_JSON"], out_path)
    return out_path.read_bytes()


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("perspective-key")],
    timeout=86400,
    memory=512,
)
def score_perspective_post_norm(input_csv: bytes, scored_csv: bytes) -> bytes:
    out_path = VOLUME_PATH / "toxic_perspective_normalized_scored.csv"
    _score(input_csv, scored_csv, "persp-post", os.environ["PERSPECTIVE_SA_JSON"], out_path)
    return out_path.read_bytes()


@app.function(image=image, volumes={str(VOLUME_PATH): volume})
def read_volume_file(filename: str) -> bytes:
    p = VOLUME_PATH / filename
    return p.read_bytes() if p.exists() else b""


@app.local_entrypoint()
def run():
    local_data = Path("data/processed")

    raw_csv  = (local_data / "reddit_toxic_sample_raw.csv").read_bytes()
    norm_csv = (local_data / "reddit_toxic_sample.csv").read_bytes()

    pre_scored  = (local_data / "toxic_perspective_scored.csv").read_bytes() \
                  if (local_data / "toxic_perspective_scored.csv").exists() else raw_csv
    post_scored = (local_data / "toxic_perspective_normalized_scored.csv").read_bytes() \
                  if (local_data / "toxic_perspective_normalized_scored.csv").exists() else norm_csv

    print("Launching Perspective pre-norm scoring (sequential to stay under quota)...")
    pre_result = score_perspective_pre_norm.remote(raw_csv, pre_scored)
    (local_data / "toxic_perspective_scored.csv").write_bytes(pre_result)
    print("Pre-norm done. Starting post-norm...")

    post_scored = (local_data / "toxic_perspective_normalized_scored.csv").read_bytes() \
                  if (local_data / "toxic_perspective_normalized_scored.csv").exists() else norm_csv
    post_result = score_perspective_post_norm.remote(norm_csv, post_scored)
    (local_data / "toxic_perspective_normalized_scored.csv").write_bytes(post_result)

    print("Done. Results saved to data/processed/")


@app.local_entrypoint()
def download():
    local_data = Path("data/processed")
    for name in ["toxic_perspective_scored.csv", "toxic_perspective_normalized_scored.csv"]:
        data = read_volume_file.remote(name)
        if data:
            (local_data / name).write_bytes(data)
            print(f"Downloaded {name} ({len(data):,} bytes)")
        else:
            print(f"{name} not found in volume yet")
