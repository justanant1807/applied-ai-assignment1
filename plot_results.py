import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("out")
PLOT_DIR = OUT_DIR / "plots"

V1_PATH = OUT_DIR / "summary_v1_strict.csv"
V2_PATH = OUT_DIR / "summary_v2_relaxed.csv"


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["expected"] = df["expected"].astype(int)
    df["visible"] = df["visible"].astype(int)
    return df


def compute_presence_metrics(df: pd.DataFrame):
    gt_present = df["expected"] > 0
    pred_present = df["visible"] > 0

    TP = ((gt_present) & (pred_present)).sum()
    TN = ((~gt_present) & (~pred_present)).sum()
    FP = ((~gt_present) & (pred_present)).sum()
    FN = ((gt_present) & (~pred_present)).sum()

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def status_distribution(df: pd.DataFrame):
    counts = Counter(df["status"])
    total = sum(counts.values())
    return {
        s: {
            "count": c,
            "percent": (c / total) if total > 0 else 0.0,
        }
        for s, c in counts.items()
    }


def plot_status_distribution(df_v1: pd.DataFrame, df_v2: pd.DataFrame):
    stats_v1 = status_distribution(df_v1)
    stats_v2 = status_distribution(df_v2)

    # Union of statuses
    all_status = sorted(set(stats_v1.keys()) | set(stats_v2.keys()))

    v1_counts = [stats_v1.get(s, {"count": 0})["count"] for s in all_status]
    v2_counts = [stats_v2.get(s, {"count": 0})["count"] for s in all_status]

    x = range(len(all_status))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar([xi - width / 2 for xi in x], v1_counts, width=width, label="STRICT")
    plt.bar([xi + width / 2 for xi in x], v2_counts, width=width, label="RELAXED")

    plt.xticks(list(x), all_status, rotation=20, ha="right")
    plt.ylabel("Number of SKUs")
    plt.title("Status distribution: STRICT vs RELAXED")
    plt.legend()
    plt.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "status_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_presence_metrics(df_v1: pd.DataFrame, df_v2: pd.DataFrame):
    m1 = compute_presence_metrics(df_v1)
    m2 = compute_presence_metrics(df_v2)

    metrics = ["accuracy", "precision", "recall", "f1"]
    v1_vals = [m1[k] for k in metrics]
    v2_vals = [m2[k] for k in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar([xi - width / 2 for xi in x], v1_vals, width=width, label="STRICT")
    plt.bar([xi + width / 2 for xi in x], v2_vals, width=width, label="RELAXED")

    plt.xticks(list(x), [m.upper() for m in metrics])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Presence metrics: STRICT vs RELAXED")
    plt.legend()
    plt.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "presence_metrics.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_abs_error_hist_v2(df_v2: pd.DataFrame):
    # Only SKUs that should be present
    df_present = df_v2[df_v2["expected"] > 0].copy()
    df_present["abs_err"] = (df_present["visible"] - df_present["expected"]).abs()

    plt.figure(figsize=(8, 4))
    max_err = int(df_present["abs_err"].max())
    bins = range(0, max(5, max_err) + 2)  # small integer bins

    plt.hist(df_present["abs_err"], bins=bins, align="left", rwidth=0.8)
    plt.xlabel("|visible - expected|")
    plt.ylabel("Number of SKUs")
    plt.title("Absolute count error histogram (RELAXED)")
    plt.xticks(bins)
    plt.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "abs_error_hist_v2.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    if not V1_PATH.exists() or not V2_PATH.exists():
        print("ERROR: summary_v1_strict.csv or summary_v2_relaxed.csv not found in out/")
        return

    df_v1 = load_summary(V1_PATH)
    df_v2 = load_summary(V2_PATH)

    print(f"Loaded {len(df_v1)} rows from {V1_PATH}")
    print(f"Loaded {len(df_v2)} rows from {V2_PATH}")

    plot_status_distribution(df_v1, df_v2)
    plot_presence_metrics(df_v1, df_v2)
    plot_abs_error_hist_v2(df_v2)

    print("All plots saved in:", PLOT_DIR)


if __name__ == "__main__":
    main()
