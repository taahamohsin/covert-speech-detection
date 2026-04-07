"""
Component 2c: Google Perspective API baseline detection.
Authenticates via service account JSON (google-auth).
Uses a token bucket rate limiter + ThreadPoolExecutor to saturate quota.
Returns TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, THREAT scores.
Output CSV adds columns: perspective_toxicity, perspective_severe_toxicity,
perspective_identity_attack, perspective_insult, perspective_threat, perspective_flagged.
"""

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yaml
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
SCOPES = ["https://www.googleapis.com/auth/userinfo.email"]
FLAGGED_THRESHOLD = 0.7


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate_per_minute: int):
        self._capacity = rate_per_minute
        self._tokens = rate_per_minute
        self._refill_rate = rate_per_minute / 60.0  # tokens per second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
            time.sleep(0.05)


def build_credentials(service_account_path: str):
    return service_account.Credentials.from_service_account_file(
        service_account_path, scopes=SCOPES
    )


def get_access_token(creds) -> str:
    if not creds.valid or creds.expired:
        creds.refresh(GoogleAuthRequest())
    return creds.token


def analyze_text(
    idx: int,
    text: str,
    attributes: list[str],
    creds,
    bucket: TokenBucket,
) -> tuple[int, dict]:
    """Score one text item. Returns (original_index, score_dict)."""
    bucket.acquire()
    token = get_access_token(creds)
    body = {
        "comment": {"text": text[:20_000]},
        "requestedAttributes": {attr: {} for attr in attributes},
        "languages": ["en"],
    }
    headers = {"Authorization": f"Bearer {token}"}
    delay = 5.0
    while True:
        try:
            resp = requests.post(PERSPECTIVE_URL, json=body, headers=headers, timeout=15)
            if resp.status_code == 429:
                log.warning("Item %d: 429 hit, backing off %.1fs", idx, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            resp.raise_for_status()
            data = resp.json()
            scores = {
                attr.lower(): data["attributeScores"][attr]["summaryScore"]["value"]
                for attr in attributes
            }
            return idx, scores
        except requests.exceptions.HTTPError:
            raise
        except Exception as exc:
            log.error("Item %d failed: %s", idx, exc)
            return idx, {attr.lower(): None for attr in attributes}


def run(input_path: str, output_path: str, config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    pcfg = cfg["apis"]["perspective"]
    attributes = pcfg["attributes"]
    rpm = pcfg.get("requests_per_minute", 60)
    max_workers = pcfg.get("max_workers", 8)

    creds = build_credentials(pcfg["service_account_json"])
    bucket = TokenBucket(rpm)

    df = pd.read_csv(input_path)
    log.info("Loaded %d items from %s", len(df), input_path)
    log.info("Rate limit: %d RPM, %d workers — est. %.0f min", rpm, max_workers, len(df) / rpm)

    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].tolist()
    total = len(texts)

    results = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_text, i, text, attributes, creds, bucket): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(futures):
            idx, scores = future.result()
            results[idx] = scores
            completed += 1
            if completed % 100 == 0:
                log.info("  Progress: %d/%d items scored", completed, total)

    scores_df = pd.DataFrame(results)
    scores_df.columns = [f"perspective_{c}" for c in scores_df.columns]

    scores_df["perspective_flagged"] = (
        scores_df.get("perspective_toxicity", pd.Series(0.0)) >= FLAGGED_THRESHOLD
    ) | (
        scores_df.get("perspective_identity_attack", pd.Series(0.0)) >= FLAGGED_THRESHOLD
    )

    out_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    log.info("Saved %d scored items to %s", len(out_df), output_path)
    log.info(
        "Flagged: %d / %d (%.1f%%)",
        out_df["perspective_flagged"].sum(),
        len(out_df),
        100 * out_df["perspective_flagged"].mean(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Perspective API on collected data.")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.input, args.output, args.config)
