"""
Component 1: Reddit Data Collection via Arctic Shift
Interim replacement for collect_reddit.py while PRAW credentials are pending.
Produces identical output schema for downstream compatibility.
"""

import csv
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE_URL = "https://arctic-shift.photon-reddit.com"


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def anonymize_author(author_name: str) -> str:
    """SHA-256 hash the username so we store no PII."""
    if not author_name or author_name in ("[deleted]", ""):
        return "deleted"
    return hashlib.sha256(author_name.encode()).hexdigest()[:16]


def rate_limit_check(response: requests.Response, backoff: float) -> None:
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) < 10:
        log.warning("Rate limit low (%s remaining), backing off %.1fs", remaining, backoff)
        time.sleep(backoff)


def _to_epoch(date_str: str) -> int:
    """Convert a YYYY-MM-DD string to a UTC epoch timestamp."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def fetch_posts(subreddit: str, cfg: dict) -> list[dict]:
    """Paginate /api/posts/search for a subreddit, return raw API rows."""
    as_cfg = cfg["arctic_shift"]
    limit = 100  # API max per request
    target = as_cfg["posts_per_subreddit"]
    backoff = as_cfg["rate_limit_backoff_seconds"]

    results = []
    after = _to_epoch(as_cfg["date_after"])
    before = _to_epoch(as_cfg["date_before"])

    while len(results) < target:
        params = {
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "limit": limit,
            "sort": "asc",  # ascending by date so 'after' pagination works cleanly
        }
        try:
            resp = requests.get(f"{BASE_URL}/api/posts/search", params=params, timeout=30)
            resp.raise_for_status()
            rate_limit_check(resp, backoff)
        except requests.RequestException as exc:
            log.error("Error fetching posts for r/%s: %s", subreddit, exc)
            break

        batch = resp.json().get("data", [])
        if not batch:
            break

        results.extend(batch)
        log.debug("  r/%s posts: %d so far", subreddit, len(results))

        if len(batch) < limit:
            break  # end of available data

        # Advance window: use created_utc of last item as new 'after' (already an int)
        after = int(batch[-1]["created_utc"])

    return results[:target]


def fetch_comments(post_id: str, subreddit: str, cfg: dict) -> list[dict]:
    """Fetch top-level comments for a post via /api/comments/search."""
    as_cfg = cfg["arctic_shift"]
    backoff = as_cfg["rate_limit_backoff_seconds"]
    limit = as_cfg["comments_per_post"]

    params = {
        "link_id": f"t3_{post_id}" if not post_id.startswith("t3_") else post_id,
        "limit": limit,
    }
    try:
        resp = requests.get(f"{BASE_URL}/api/comments/search", params=params, timeout=30)
        resp.raise_for_status()
        rate_limit_check(resp, backoff)
        data = resp.json().get("data", [])
        # Keep only true top-level comments (parent_id starts with t3_)
        return [c for c in data if c.get("parent_id", "").startswith("t3_")]
    except requests.RequestException as exc:
        log.error("Error fetching comments for post %s: %s", post_id, exc)
        return []


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
        "upvote_ratio": None,   # not available in Arctic Shift
        "num_comments": row.get("num_comments"),
        "awards": None,         # not available in Arctic Shift
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


def save_csv(items: list[dict], path: Path) -> None:
    if not items:
        log.warning("No items to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        writer.writeheader()
        writer.writerows(items)
    log.info("Saved %d items to %s", len(items), path)


def save_json(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    log.info("Saved %d items to %s", len(items), path)


def collect_subreddit(subreddit: str, subreddit_type: str, cfg: dict) -> list[dict]:
    log.info("Collecting r/%s (%s) ...", subreddit, subreddit_type)
    items = []

    posts = fetch_posts(subreddit, cfg)
    log.info("  r/%s: %d posts fetched", subreddit, len(posts))

    for i, post in enumerate(posts, 1):
        items.append(normalize_post(post, subreddit_type))

        if post.get("num_comments", 0):
            raw_comments = fetch_comments(post["id"], subreddit, cfg)
            for c in raw_comments:
                if c.get("body") in ("[deleted]", "[removed]", None):
                    continue
                items.append(normalize_comment(c, subreddit_type))

        if i % 50 == 0:
            log.info("  r/%s: processed %d/%d posts ...", subreddit, i, len(posts))

    log.info("  r/%s: %d total items (posts + comments)", subreddit, len(items))
    return items


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    all_items: list[dict] = []

    for group_type, subreddit_list in cfg["subreddits"].items():
        log.info("=== %s subreddits ===", group_type.upper())
        for name in subreddit_list:
            items = collect_subreddit(name, group_type, cfg)
            all_items.extend(items)

    log.info("Total items collected: %d", len(all_items))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(cfg["output"]["raw_dir"])

    save_csv(all_items, raw_dir / f"reddit_raw_{timestamp}.csv")
    save_json(all_items, raw_dir / f"reddit_raw_{timestamp}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Reddit data via Arctic Shift archive.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
