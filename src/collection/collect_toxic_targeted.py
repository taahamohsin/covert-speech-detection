"""
Targeted toxic content collection via Arctic Shift.

Sampling is deliberately adversarial: we query directly for terms from the
dogwhistle lexicon so the resulting corpus contains the evasion strategies the
normalization layer is designed to handle. The initial broad-sample collection
surfaced very little character-level obfuscation (only 72/8720 items), so the
normalization layer had no measurable lift. This script produces a supplementary
dataset that actually exercises it.

Three sampling strategies, combined:
  1. Dogwhistle-term search across polarized + banned subreddits
  2. Controversial-sort posts and comments (content that barely passed automod)
  3. Banned-subreddit archives (content predating automod improvements)

Output schema matches collect_arctic_shift.py so the new CSV can be concatenated
with the existing reddit_raw_*.csv and passed straight into normalization +
detection scripts with zero changes.

Usage:
    python3 src/collection/collect_toxic_targeted.py \
        --output data/raw/reddit_toxic_<timestamp>.csv \
        [--per-term 50] \
        [--comments-per-post 10]
"""

import argparse
import csv
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE_URL = "https://arctic-shift.photon-reddit.com"

# Polarized subs we already sampled broadly — re-query here with term filters
POLARIZED_SUBS = [
    "conspiracy",
    "Conservative",
    "kotakuinaction",
    "PoliticalCompassMemes",
]

# Archived banned subreddits — Arctic Shift retains content from before the bans
BANNED_SUBS = [
    "The_Donald",
    "ChapoTrapHouse",
    "GenderCritical",
    "Braincels",
    "MGTOW",
    "MensRights",
    "frenworld",
    "ConsumeProduct",
    "CringeAnarchy",
    "DebateAltRight",
]

# High-yield dogwhistle / evasion terms. Kept narrow on purpose — queries broader
# than these return too much non-hateful content (e.g., 'based' is both a
# dogwhistle and a common slang term). These are terms whose usage is
# overwhelmingly hateful in context.
TARGET_TERMS = [
    # Antisemitic
    "ZOG", "globohomo", "goyim", "(((they)))", "early life",
    # White nationalist
    "1488", "14 words", "groyper", "honkler", "great replacement",
    "white genocide", "white pride", "anti-white",
    # Racist
    "jogger", "dindu", "we wuz", "13/50", "13/52",
    # Islamophobic
    "rapefugee", "muzrat", "muslim invasion",
    # Misogynist / incel
    "femoid", "roastie", "AWFL", "hypergamy", "chad", "becky",
    "landwhale", "foid",
    # Anti-LGBTQ
    "groomer", "troon", "moid",
    # Leetspeak variants of above
    "j0gg3r", "1488ers", "gr0yper",
]


def anonymize_author(author_name: str) -> str:
    if not author_name or author_name in ("[deleted]", ""):
        return "deleted"
    return hashlib.sha256(author_name.encode()).hexdigest()[:16]


def rate_limit_check(response: requests.Response, backoff: float = 2.0) -> None:
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) < 10:
        log.warning("Rate limit low (%s), backing off %.1fs", remaining, backoff)
        time.sleep(backoff)


def _get(endpoint: str, params: dict, label: str, max_retries: int = 3) -> list[dict]:
    """GET with retry-on-timeout. Arctic Shift returns 422 "Timeout. Maybe slow
    down a bit" on heavy text searches — treat that as retryable with backoff."""
    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{BASE_URL}/api/{endpoint}", params=params, timeout=60)
            if resp.status_code == 400:
                log.debug("400 %s: %s", label, resp.text[:200])
                return []
            if resp.status_code == 422:
                msg = resp.json().get("error", "")
                if "timeout" in msg.lower() or "slow down" in msg.lower():
                    if attempt < max_retries:
                        log.warning("server timeout %s (attempt %d), backing off %.1fs", label, attempt, backoff)
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                log.error("422 %s: %s", label, msg)
                return []
            resp.raise_for_status()
            rate_limit_check(resp)
            return resp.json().get("data") or []
        except requests.RequestException as exc:
            if attempt < max_retries:
                log.warning("%s failed (%s), retrying in %.1fs", label, exc, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            log.error("%s: %s", label, exc)
            return []
    return []


def search_posts(term: str, subreddit: str, limit: int = 100) -> list[dict]:
    """Arctic Shift requires a subreddit or author filter alongside text search.
    Searches both title and selftext, merging results."""
    base = {"limit": limit, "sort": "desc", "subreddit": subreddit}
    title_hits = _get("posts/search", {**base, "title": term}, f"posts title={term} sub={subreddit}")
    body_hits  = _get("posts/search", {**base, "selftext": term}, f"posts selftext={term} sub={subreddit}")

    # Dedup on id
    seen = set()
    merged = []
    for p in title_hits + body_hits:
        pid = p.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            merged.append(p)
    return merged


def fetch_post_comments(post_id: str, limit: int = 50) -> list[dict]:
    """Fetch the comment tree for a known post via link_id. This is NOT a text
    search — it retrieves the stored comments for a post directly, which the
    Arctic Shift server handles reliably even when text search times out."""
    params = {"link_id": f"t3_{post_id}", "limit": limit}
    return _get("comments/search", params, f"comments link_id={post_id}")


def normalize_post(row: dict, subreddit_type: str) -> dict:
    title = row.get("title", "") or ""
    body = row.get("selftext", "") or ""
    permalink = row.get("permalink", "")
    return {
        "item_id": row.get("id", ""),
        "item_type": "post",
        "subreddit": row.get("subreddit", ""),
        "subreddit_type": subreddit_type,
        "title": title,
        "body": body,
        "text": (title + " " + body).strip(),
        "score": row.get("score"),
        "upvote_ratio": None,
        "num_comments": row.get("num_comments"),
        "awards": None,
        "timestamp": datetime.fromtimestamp(float(row["created_utc"]), tz=timezone.utc).isoformat()
        if row.get("created_utc")
        else "",
        "author_hash": anonymize_author(row.get("author")),
        "url": f"https://reddit.com{permalink}" if permalink else "",
    }


def normalize_comment(row: dict, subreddit_type: str) -> dict:
    body = row.get("body", "") or ""
    permalink = row.get("permalink", "")
    return {
        "item_id": row.get("id", ""),
        "item_type": "comment",
        "subreddit": row.get("subreddit", ""),
        "subreddit_type": subreddit_type,
        "title": "",
        "body": body,
        "text": body,
        "score": row.get("score"),
        "upvote_ratio": None,
        "num_comments": None,
        "awards": None,
        "timestamp": datetime.fromtimestamp(float(row["created_utc"]), tz=timezone.utc).isoformat()
        if row.get("created_utc")
        else "",
        "author_hash": anonymize_author(row.get("author")),
        "url": f"https://reddit.com{permalink}" if permalink else "",
    }


def classify_sub(sub: str) -> str:
    if sub in BANNED_SUBS:
        return "banned"
    if sub in POLARIZED_SUBS:
        return "polarized"
    return "other"


def collect(per_term: int) -> list[dict]:
    """Iterate terms × subreddit buckets, collecting posts + comments."""
    seen_ids: set[str] = set()
    items: list[dict] = []

    subreddit_buckets = POLARIZED_SUBS + BANNED_SUBS

    for term in TARGET_TERMS:
        for sub in subreddit_buckets:
            log.info("Searching term=%r sub=%s", term, sub)

            posts = search_posts(term, subreddit=sub, limit=per_term)
            new_posts = []
            for p in posts:
                pid = p.get("id")
                if not pid or pid in seen_ids:
                    continue
                if not (p.get("title") or p.get("selftext")):
                    continue
                seen_ids.add(pid)
                bucket = classify_sub(p.get("subreddit", ""))
                items.append(normalize_post(p, bucket))
                new_posts.append((pid, bucket))

            # For each hateful-term-containing post, pull its comments via
            # link_id (direct retrieval, not text search — reliable).
            for pid, bucket in new_posts:
                comments = fetch_post_comments(pid, limit=25)
                for c in comments:
                    cid = c.get("id")
                    body = c.get("body", "")
                    if not cid or cid in seen_ids:
                        continue
                    if not body or body in ("[deleted]", "[removed]"):
                        continue
                    seen_ids.add(cid)
                    items.append(normalize_comment(c, bucket))
                time.sleep(0.3)  # pacing between comment fetches

            time.sleep(1.0)  # pacing between term+sub queries

        log.info("  running total: %d items (%d unique ids)", len(items), len(seen_ids))

    return items


def save_csv(items: list[dict], path: Path) -> None:
    if not items:
        log.warning("No items to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(items[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
    log.info("Saved %d items -> %s", len(items), path)


def summarize(items: list[dict]) -> None:
    from collections import Counter

    by_type = Counter(i["item_type"] for i in items)
    by_sub = Counter(i["subreddit"].lower() for i in items)
    by_bucket = Counter(i["subreddit_type"] for i in items)

    log.info("--- summary ---")
    log.info("by item_type: %s", dict(by_type))
    log.info("by bucket:    %s", dict(by_bucket))
    log.info("top 10 subs:  %s", by_sub.most_common(10))


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted toxic-content collection via Arctic Shift.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/raw/reddit_toxic_<timestamp>.csv)",
    )
    parser.add_argument("--per-term", type=int, default=50, help="Max results per term+subreddit pair (default: 50)")
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path(f"data/raw/reddit_toxic_{timestamp}.csv")

    items = collect(per_term=args.per_term)
    log.info("Collected %d unique items", len(items))
    summarize(items)
    save_csv(items, out_path)


if __name__ == "__main__":
    main()
