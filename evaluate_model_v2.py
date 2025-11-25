import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["expected"] = df["expected"].astype(int)
    df["visible"] = df["visible"].astype(int)

    # extra columns for v2
    df["abs_err"] = (df["visible"] - df["expected"]).abs()
    df["over"] = (df["visible"] - df["expected"]).clip(lower=0)
    df["under"] = (df["expected"] - df["visible"]).clip(lower=0)
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


def compute_count_metrics(df: pd.DataFrame):
    abs_err = df["abs_err"].to_numpy()
    mse = float(np.mean(abs_err ** 2)) if len(abs_err) > 0 else 0.0
    rmse = float(np.sqrt(mse))

    mae = float(np.mean(abs_err)) if len(abs_err) > 0 else 0.0

    # “mass” interpretation: how many extra / missed items total
    total_over = int(df["over"].sum())
    total_under = int(df["under"].sum())

    avg_over_per_sku = float(df["over"].mean()) if len(df) > 0 else 0.0
    avg_under_per_sku = float(df["under"].mean()) if len(df) > 0 else 0.0

    # per-bin exact accuracy (all SKUs in bin have exact counts)
    if "image_id" in df.columns:
        per_bin = df.groupby("image_id").apply(
            lambda g: bool((g["visible"] == g["expected"]).all())
        )
        bin_exact_acc = float(per_bin.mean()) if len(per_bin) > 0 else 0.0
        num_bins = len(per_bin)
    else:
        bin_exact_acc = 0.0
        num_bins = 0

    return {
        "mae": mae,
        "rmse": rmse,
        "total_over": total_over,
        "total_under": total_under,
        "avg_over_per_sku": avg_over_per_sku,
        "avg_under_per_sku": avg_under_per_sku,
        "bin_exact_acc": bin_exact_acc,
        "num_bins": num_bins,
    }


def status_distribution(df: pd.DataFrame):
    counts = Counter(df["status"])
    total = sum(counts.values())
    dist = {}
    for s, c in counts.items():
        dist[s] = {
            "count": c,
            "percent": (c / total) if total > 0 else 0.0,
        }
    return dist


def pretty_print_single(path: Path, df: pd.DataFrame):
    print("=" * 70)
    print(f"📄 Summary file: {path}")
    print(f"   Rows (SKUs): {len(df)}")
    if "image_id" in df.columns:
        num_bins = df["image_id"].nunique()
        print(f"   Bins (images): {num_bins}")

    # Presence metrics
    pres = compute_presence_metrics(df)
    print("\n🟦 Presence metrics (SKU-level, expected>0 only):")
    print(f"   TP={pres['TP']}  FN={pres['FN']}  FP={pres['FP']}  TN={pres['TN']}")
    print(f"   Accuracy : {pres['accuracy']:.3f}")
    print(f"   Precision: {pres['precision']:.3f}")
    print(f"   Recall   : {pres['recall']:.3f}")
    print(f"   F1       : {pres['f1']:.3f}")

    # Count metrics
    cnt = compute_count_metrics(df)
    print("\n🟩 Count metrics (SKU-level):")
    print(f"   MAE (|visible-expected|) : {cnt['mae']:.3f}")
    print(f"   RMSE                     : {cnt['rmse']:.3f}")
    print(f"   Total overcount mass     : {cnt['total_over']} items")
    print(f"   Total undercount mass    : {cnt['total_under']} items")
    print(f"   Avg over per SKU         : {cnt['avg_over_per_sku']:.3f}")
    print(f"   Avg under per SKU        : {cnt['avg_under_per_sku']:.3f}")
    print(f"   Bin exact accuracy       : {cnt['bin_exact_acc']:.3f}")
    print(f"   #Bins                    : {cnt['num_bins']}")

    # Status distribution
    dist = status_distribution(df)
    print("\n🟨 Status distribution:")
    for s, v in dist.items():
        print(f"   {s:17s}: {v['count']:5d}  ({100.0*v['percent']:.1f}%)")

    print()


def compare_summaries(paths):
    records = []

    for path_str in paths:
        path = Path(path_str)
        df = load_summary(path)

        pres = compute_presence_metrics(df)
        cnt = compute_count_metrics(df)

        records.append(
            {
                "file": path.name,
                "rows": len(df),
                "bins": df["image_id"].nunique() if "image_id" in df.columns else 0,
                "presence_acc": pres["accuracy"],
                "presence_recall": pres["recall"],
                "presence_f1": pres["f1"],
                "mae": cnt["mae"],
                "rmse": cnt["rmse"],
                "total_over": cnt["total_over"],
                "total_under": cnt["total_under"],
                "bin_exact_acc": cnt["bin_exact_acc"],
            }
        )

    df_cmp = pd.DataFrame(records)

    print("=" * 70)
    print("📊 COMPARISON (v2 metrics)")
    print(df_cmp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SKU detection summaries (v2 with FP/FN mass)."
    )
    p.add_argument(
        "--summary",
        type=str,
        help="Path to a single summary CSV (e.g. out/summary_v2_relaxed.csv)",
    )
    p.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple summary CSVs (2 or more paths)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.summary is None and not args.compare:
        print("Usage examples:")
        print("  python evaluate_model_v2.py --summary out/summary_v2_relaxed.csv")
        print(
            "  python evaluate_model_v2.py --compare out/summary_v1_strict.csv out/summary_v2_relaxed.csv"
        )
        return

    if args.summary:
        path = Path(args.summary)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            return
        df = load_summary(path)
        pretty_print_single(path, df)

    if args.compare:
        if len(args.compare) < 2:
            print("ERROR: --compare needs at least 2 files.")
            return
        compare_summaries(args.compare)


if __name__ == "__main__":
    main()
