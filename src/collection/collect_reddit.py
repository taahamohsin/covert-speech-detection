"""
Component 1: Reddit Data Collection via PRAW
Collects posts and top-level comments from polarized and control subreddits.
Stores results as CSV with anonymized author IDs.
"""

import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import praw
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_reddit_client(cfg: dict) -> praw.Reddit:
    r = cfg["reddit"]
    return praw.Reddit(
        client_id=r["client_id"],
        client_secret=r["client_secret"],
        user_agent=r["user_agent"],
    )


def anonymize_author(author_name: str) -> str:
    """SHA-256 hash the username so we store no PII."""
    if author_name is None:
        return "deleted"
    return hashlib.sha256(author_name.encode()).hexdigest()[:16]


def extract_post(submission, subreddit_type: str) -> dict:
    return {
        "item_id": submission.id,
        "item_type": "post",
        "subreddit": submission.subreddit.display_name,
        "subreddit_type": subreddit_type,   # polarized | control
        "title": submission.title,
        "body": submission.selftext,
        "text": (submission.title + " " + submission.selftext).strip(),
        "score": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "awards": submission.total_awards_received,
        "timestamp": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
        "author_hash": anonymize_author(
            submission.author.name if submission.author else None
        ),
        "url": f"https://reddit.com{submission.permalink}",
    }


def extract_comment(comment, subreddit_type: str) -> dict:
    return {
        "item_id": comment.id,
        "item_type": "comment",
        "subreddit": comment.subreddit.display_name,
        "subreddit_type": subreddit_type,
        "title": "",
        "body": comment.body,
        "text": comment.body,
        "score": comment.score,
        "upvote_ratio": None,
        "num_comments": None,
        "awards": comment.total_awards_received,
        "timestamp": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
        "author_hash": anonymize_author(
            comment.author.name if comment.author else None
        ),
        "url": f"https://reddit.com{comment.permalink}",
    }


def collect_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    subreddit_type: str,
    cfg: dict,
    total_so_far: int,
) -> list[dict]:
    """Collect posts + top-level comments from one subreddit."""
    col_cfg = cfg["collection"]
    target = cfg["collection"]["target_total"]
    items = []

    subreddit = reddit.subreddit(subreddit_name)
    seen_ids = set()

    sort_configs = []
    for method in col_cfg["sort_methods"]:
        if method in ("hot", "new", "rising"):
            sort_configs.append((method, None))
        else:
            for tf in col_cfg.get("time_filters", ["month"]):
                sort_configs.append((method, tf))

    for method, time_filter in sort_configs:
        if total_so_far + len(items) >= target:
            break

        log.info(
            "  r/%s — %s%s",
            subreddit_name,
            method,
            f"/{time_filter}" if time_filter else "",
        )

        try:
            if time_filter:
                listings = getattr(subreddit, method)(
                    limit=col_cfg["limit_per_sort"], time_filter=time_filter
                )
            else:
                listings = getattr(subreddit, method)(limit=col_cfg["limit_per_sort"])

            for submission in listings:
                if submission.id in seen_ids:
                    continue
                seen_ids.add(submission.id)

                post_row = extract_post(submission, subreddit_type)
                items.append(post_row)

                # Fetch top-level comments
                try:
                    submission.comments.replace_more(limit=0)
                    top_comments = submission.comments.list()
                    # Only keep direct top-level comments (depth 0)
                    top_level = [c for c in top_comments if c.parent_id.startswith("t3_")]
                    for comment in top_level[: col_cfg["comment_limit"]]:
                        if comment.body in ("[deleted]", "[removed]"):
                            continue
                        items.append(extract_comment(comment, subreddit_type))
                except Exception as exc:
                    log.warning("Failed to fetch comments for %s: %s", submission.id, exc)

                # Respect rate limit
                time.sleep(60 / cfg["reddit"]["rate_limit_per_minute"])

        except Exception as exc:
            log.error("Error collecting r/%s (%s): %s", subreddit_name, method, exc)

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
    log.info("Saved %d items to %s", len(items), path)


def save_json(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    log.info("Saved %d items to %s", len(items), path)


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    reddit = build_reddit_client(cfg)

    all_items: list[dict] = []

    subreddit_groups = {
        "polarized": cfg["subreddits"]["polarized"],
        "control": cfg["subreddits"]["control"],
    }

    for group_type, subreddit_list in subreddit_groups.items():
        log.info("=== Collecting from %s subreddits ===", group_type)
        for name in subreddit_list:
            if len(all_items) >= cfg["collection"]["target_total"]:
                log.info("Reached target total of %d items, stopping.", cfg["collection"]["target_total"])
                break
            log.info("Collecting r/%s ...", name)
            items = collect_subreddit(reddit, name, group_type, cfg, len(all_items))
            log.info("  -> %d items collected from r/%s", len(items), name)
            all_items.extend(items)

    log.info("Total items collected: %d", len(all_items))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(cfg["output"]["raw_dir"])
    fmt = cfg["output"]["format"]

    if fmt == "csv":
        save_csv(all_items, raw_dir / f"reddit_raw_{timestamp}.csv")
    else:
        save_json(all_items, raw_dir / f"reddit_raw_{timestamp}.json")

    # Always save a JSON copy for programmatic downstream use
    save_json(all_items, raw_dir / f"reddit_raw_{timestamp}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Reddit data for hate speech audit.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
