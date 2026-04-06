"""
Component 2b: Dynabench RoBERTa hate speech classifier (local, no API key needed).
Model: facebook/roberta-hate-speech-dynabench-r4-target (Vidgen et al., 2021)
Labels: 'hate' / 'nothate' with confidence scores.
Output CSV adds columns: dynabench_label, dynabench_hate_score, dynabench_flagged.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_classifier(cfg: dict):
    dcfg = cfg["apis"]["dynabench"]
    log.info("Loading model %s on %s ...", dcfg["model_id"], dcfg["device"])
    clf = pipeline(
        "text-classification",
        model=dcfg["model_id"],
        device=-1 if dcfg["device"] == "cpu" else 0,
        truncation=True,
        max_length=512,
    )
    log.info("Model loaded.")
    return clf, dcfg["batch_size"]


def run(input_path: str, output_path: str, config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    clf, batch_size = load_classifier(cfg)

    df = pd.read_csv(input_path)
    log.info("Loaded %d items from %s", len(df), input_path)

    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].tolist()

    labels, hate_scores = [], []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        try:
            preds = clf(batch)
            for pred in preds:
                label = pred["label"].lower()   # 'hate' or 'nothate'
                score = pred["score"]
                labels.append(label)
                # Normalise: if label is 'hate', score IS hate score; else invert
                hate_scores.append(score if label == "hate" else 1 - score)
        except Exception as exc:
            log.error("Batch %d-%d failed: %s", start, start + batch_size, exc)
            labels.extend([None] * len(batch))
            hate_scores.extend([None] * len(batch))

        if (start // batch_size + 1) % 10 == 0:
            log.info("  Progress: %d/%d items scored", min(start + batch_size, total), total)

    df["dynabench_label"] = labels
    df["dynabench_hate_score"] = hate_scores
    df["dynabench_flagged"] = df["dynabench_label"] == "hate"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d scored items to %s", len(df), output_path)
    log.info(
        "Flagged: %d / %d (%.1f%%)",
        df["dynabench_flagged"].sum(),
        len(df),
        100 * df["dynabench_flagged"].mean(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynabench RoBERTa classifier on collected data.")
    parser.add_argument("--input", required=True, help="Path to raw CSV (data/raw/*.csv)")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.input, args.output, args.config)
