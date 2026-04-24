"""
Evaluation: precision, recall, F1 for all four detection conditions
against the human-annotated gold standard.

Usage:
  python3 src/evaluation/metrics.py

Outputs:
  data/figures/fig5_precision_recall_f1.png
  data/figures/fig6_f1_by_hate_type.png
  data/figures/fig7_f1_by_evasion_strategy.png
  Prints full results table to stdout
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

GOLD = "data/annotations/Gold Standard Annotations Final - Mohamed.csv"
DETECTION = "data/processed/detection_results_final.csv"
FIGURES = Path("data/figures")


def compute_metrics(y_true, y_pred):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1}


def load_data():
    gold = pd.read_csv(GOLD).rename(columns={"Item Id": "item_id"})
    gold["label"] = gold["Hate Class"].str.strip().str.lower()
    gold["evasion"] = gold["Evasion Strategy"].str.strip().str.lower().fillna("none")
    gold = gold[gold["label"] != "borderline/ambiguous"].copy()
    gold["is_hate"] = gold["label"].isin(["overt hate", "covert hate"])

    det = pd.read_csv(DETECTION)
    merged = gold.merge(det[[
        "item_id",
        "openai_flagged", "openai_norm_flagged",
        "perspective_flagged", "perspective_norm_flagged",
    ]], on="item_id")
    return merged


def run_evasion_analysis(merged):
    hate = merged[merged["is_hate"]].copy()
    strategies = sorted(hate["evasion"].unique())

    print("\n--- By evasion strategy (OpenAI pre-norm, hate items only) ---")
    by_strategy = {}
    for strategy in strategies:
        sub = merged[merged["evasion"] == strategy].copy() if strategy != "none" \
              else merged[(merged["evasion"] == "none") & merged["is_hate"]].copy()
        if len(sub) == 0:
            continue
        sub_true = sub["is_hate"]
        sub_pred = sub["openai_flagged"].astype(bool)
        m = compute_metrics(sub_true, sub_pred)
        by_strategy[strategy] = m
        print(f"  {strategy:<28} Recall={m['Recall']:.3f}  F1={m['F1']:.3f}  (n={int(sub_true.sum())})")

    return by_strategy


def run_evaluation(merged):
    y_true = merged["is_hate"]

    conditions = {
        "OpenAI\npre-norm":       merged["openai_flagged"].astype(bool),
        "OpenAI\npost-norm":      merged["openai_norm_flagged"].astype(bool),
        "Perspective\npre-norm":  merged["perspective_flagged"].astype(bool),
        "Perspective\npost-norm": merged["perspective_norm_flagged"].astype(bool),
    }

    print(f"Evaluation set: {len(merged)} items "
          f"({y_true.sum()} hate, {(~y_true).sum()} not hateful)\n")
    print(f"{'Condition':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-" * 80)

    results = {}
    for name, y_pred in conditions.items():
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        label = name.replace("\n", " ")
        print(f"{label:<25} {m['Precision']:>10.3f} {m['Recall']:>10.3f} "
              f"{m['F1']:>10.3f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}")

    print("\n--- By hate type (pre-norm only) ---")
    by_type = {}
    for api, col in [("OpenAI", "openai_flagged"), ("Perspective", "perspective_flagged")]:
        by_type[api] = {}
        for hate_type in ["overt hate", "covert hate"]:
            sub = merged[merged["label"].isin([hate_type, "not hateful"])].copy()
            sub_true = sub["label"] == hate_type
            sub_pred = sub[col].astype(bool)
            m = compute_metrics(sub_true, sub_pred)
            by_type[api][hate_type] = m
            print(f"  {api} / {hate_type:<12}  "
                  f"Precision={m['Precision']:.3f}  "
                  f"Recall={m['Recall']:.3f}  "
                  f"F1={m['F1']:.3f}  "
                  f"(n={int(sub_true.sum())})")

    return results, by_type



def plot_overall(results):
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [k.replace("\n", " ") for k in results]
    precision = [results[k]["Precision"] for k in results]
    recall    = [results[k]["Recall"]    for k in results]
    f1        = [results[k]["F1"]        for k in results]

    x = np.arange(len(labels))
    width = 0.25
    colors = ["#4C72B0", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, precision, width, label="Precision", color=colors[0], edgecolor="white")
    b2 = ax.bar(x,         recall,    width, label="Recall",    color=colors[1], edgecolor="white")
    b3 = ax.bar(x + width, f1,        width, label="F1",        color=colors[2], edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision, Recall, F1 by Detection Condition\n(n=134, Borderline/Ambiguous excluded)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig5_precision_recall_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


def plot_by_hate_type(by_type):
    FIGURES.mkdir(parents=True, exist_ok=True)
    hate_types = ["overt hate", "covert hate"]
    apis = ["OpenAI", "Perspective"]
    metrics_list = ["Precision", "Recall", "F1"]
    colors = {"Precision": "#4C72B0", "Recall": "#C44E52", "F1": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, api in zip(axes, apis):
        x = np.arange(len(hate_types))
        width = 0.25
        for i, metric in enumerate(metrics_list):
            vals = [by_type[api][ht][metric] for ht in hate_types]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=metric, color=colors[metric], edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["Overt Hate\n(n=12)", "Covert Hate\n(n=24)"], fontsize=10)
        ax.set_title(f"{api} — by Hate Type", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Score", fontsize=11)
            ax.legend(fontsize=10)

    plt.suptitle("Detection Performance by Hate Type (pre-norm)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIGURES / "fig6_f1_by_hate_type.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_by_evasion(by_strategy):
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = list(by_strategy.keys())
    recalls = [by_strategy[s]["Recall"] for s in labels]
    f1s     = [by_strategy[s]["F1"]     for s in labels]
    ns      = [by_strategy[s]["TP"] + by_strategy[s]["FN"] for s in labels]

    x = np.arange(len(labels))
    width = 0.35
    colors = ["#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, recalls, width, label="Recall", color=colors[0], edgecolor="white")
    b2 = ax.bar(x + width / 2, f1s,     width, label="F1",     color=colors[1], edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(labels, ns)], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("OpenAI Detection Performance by Evasion Strategy (pre-norm)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig7_f1_by_evasion_strategy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    merged = load_data()
    results, by_type = run_evaluation(merged)
    by_strategy = run_evasion_analysis(merged)
    plot_overall(results)
    plot_by_hate_type(by_type)
    plot_by_evasion(by_strategy)


if __name__ == "__main__":
    main()
