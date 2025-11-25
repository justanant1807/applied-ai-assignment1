# mlops/evaluate_model_v3_counts.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SKU verification summaries (presence + count metrics)."
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to summary CSV (e.g. out/summary_v5_clip020_box017.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary)

    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)

    required_cols = {
        "image_id",
        "asin",
        "sku_name",
        "expected",
        "visible",
        "status",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns in CSV: {missing}")
        return

    print(f"Summary file: {summary_path}")
    print(f"   Rows (SKUs): {len(df)}")
    print(f"   Bins (images): {df['image_id'].nunique()}")

    # ---------- Presence metrics ----------
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

    print("\n🟦 Presence metrics (SKU present vs not):")
    print(f"   TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"   Accuracy : {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall   : {rec:.3f}")
    print(f"   F1       : {f1:.3f}")
    print("   Note: FP/TN are 0 here because metadata only lists SKUs that are truly present.")

    # ---------- Count metrics ----------
    df_pos = df[df["expected"] > 0].copy()
    n_pos = len(df_pos)

    print("\n🟩 Count metrics (SKU-level):")
    print(f"   Num SKUs with expected > 0: {n_pos}")
    if n_pos == 0:
        print("   (No positive SKUs; skipping count metrics.)")
        return

    err = (df_pos["visible"] - df_pos["expected"]).astype(float)
    abs_err = err.abs()
    sq_err = err ** 2

    mae = abs_err.mean()
    rmse = np.sqrt(sq_err.mean())

    over_mass = (df_pos["visible"] - df_pos["expected"]).clip(lower=0).sum()
    under_mass = (df_pos["expected"] - df_pos["visible"]).clip(lower=0).sum()

    print(f"   MAE (|visible-expected|) : {mae:.3f}")
    print(f"   RMSE                     : {rmse:.3f}")
    print(f"   Total overcount mass     : {over_mass:.0f} items")
    print(f"   Total undercount mass    : {under_mass:.0f} items")
    print(f"   Avg over per SKU         : {over_mass / n_pos:.3f}")
    print(f"   Avg under per SKU        : {under_mass / n_pos:.3f}")

    # Bin-level exact accuracy
    def bin_perfect(g):
        return int((g["visible"].values == g["expected"].values).all())

    per_bin = df_pos.groupby("image_id").apply(bin_perfect)
    num_bins = len(per_bin)
    num_perfect = int(per_bin.sum())
    bin_acc = num_perfect / num_bins if num_bins > 0 else 0.0

    print("\n   Bin-level accuracy:")
    print(f"   #Bins              : {num_bins}")
    print(f"   Perfect bins       : {num_perfect}")
    print(f"   Bin exact accuracy : {bin_acc:.3f}")

    # Count-style confusion
    tp_exact = int((df_pos["visible"] == df_pos["expected"]).sum())
    fp_over = int((df_pos["visible"] > df_pos["expected"]).sum())
    fn_under = int((df_pos["visible"] < df_pos["expected"]).sum())

    print("\n🟥 Count-style confusion (per SKU):")
    print(f"   Exact matches (TP_exact)               : {tp_exact}")
    print(f"   Over-count SKUs (FP_over / count FPs)  : {fp_over}")
    print(f"   Under-count SKUs (FN_under / count FNs): {fn_under}")
    print("   Here:")
    print("     • Count false positives (FP_over) = SKUs that ARE in the bin but")
    print("       our model predicted MORE units than the ground-truth quantity.")
    print("     • Count false negatives (FN_under) = SKUs that ARE in the bin but")
    print("       our model predicted FEWER units than the ground-truth quantity.")

    # Status distribution
    print("\n🟨 Status distribution:")
    status_counts = df["status"].value_counts()
    total_rows = len(df)
    for status, cnt in status_counts.items():
        pct = cnt / total_rows if total_rows > 0 else 0.0
        print(f"   {status:15s}: {cnt:5d}  ({pct*100:4.1f}%)")


if __name__ == "__main__":
    main()

