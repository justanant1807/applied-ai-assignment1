# mlops/plot_results_cli.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(df: pd.DataFrame):
    # presence
    present_gt = df["expected"] > 0
    present_pred = df["visible"] > 0

    tp = int(((present_gt == 1) & (present_pred == 1)).sum())
    fp = int(((present_gt == 0) & (present_pred == 1)).sum())
    fn = int(((present_gt == 1) & (present_pred == 0)).sum())
    tn = int(((present_gt == 0) & (present_pred == 0)).sum())

    total = len(df)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # count
    df_pos = df[df["expected"] > 0].copy()
    if len(df_pos) == 0:
        mae = rmse = np.nan
    else:
        err = (df_pos["visible"] - df_pos["expected"]).astype(float)
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mae": mae,
        "rmse": rmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple summary CSVs and plot metrics."
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="List of summary CSVs to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=False,
        help="Optional labels for each summary (e.g. v2 v4 v5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [Path(p) for p in args.summaries]

    if args.labels and len(args.labels) != len(paths):
        raise ValueError("Number of labels must match number of summaries.")

    labels = args.labels or [p.stem for p in paths]

    metrics_list = []
    for p, label in zip(paths, labels):
        df = pd.read_csv(p)
        m = compute_metrics(df)
        m["label"] = label
        metrics_list.append(m)
        print(f"{label}: F1={m['f1']:.3f}, MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}")

    df_m = pd.DataFrame(metrics_list)

    # Plot F1 vs versions
    plt.figure()
    plt.bar(df_m["label"], df_m["f1"])
    plt.title("Presence F1 by version")
    plt.xlabel("Version")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig("out/plot_f1_by_version.png")

    # Plot MAE vs versions
    plt.figure()
    plt.bar(df_m["label"], df_m["mae"])
    plt.title("SKU-level MAE by version")
    plt.xlabel("Version")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig("out/plot_mae_by_version.png")

    print("Saved plots to out/plot_f1_by_version.png and out/plot_mae_by_version.png")


if __name__ == "__main__":
    main()
