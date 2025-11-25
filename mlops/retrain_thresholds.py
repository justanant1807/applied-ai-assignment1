# mlops/retrain_thresholds.py

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def compute_score(df: pd.DataFrame):
    """Combine presence F1 and count MAE into a single score (higher is better)."""
    present_gt = df["expected"] > 0
    present_pred = df["visible"] > 0

    tp = int(((present_gt == 1) & (present_pred == 1)).sum())
    fp = int(((present_gt == 0) & (present_pred == 1)).sum())
    fn = int(((present_gt == 1) & (present_pred == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    df_pos = df[df["expected"] > 0].copy()
    if len(df_pos) == 0:
        mae = np.inf
    else:
        err = (df_pos["visible"] - df_pos["expected"]).astype(float)
        mae = err.abs().mean()

    # we want high F1, low MAE → simple score
    score = f1 - 0.1 * mae
    return score, f1, mae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick best summary CSV and write config as 'retrained' thresholds."
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="List of summary CSVs (each corresponding to a (BOX, CLIP) setting).",
    )
    parser.add_argument(
        "--box",
        nargs="+",
        type=float,
        required=True,
        help="BOX thresholds corresponding to summaries.",
    )
    parser.add_argument(
        "--clip",
        nargs="+",
        type=float,
        required=True,
        help="CLIP thresholds corresponding to summaries.",
    )
    parser.add_argument(
        "--out-config",
        type=str,
        default="weights/v5_config.json",
        help="Where to write the chosen config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (len(args.summaries) == len(args.box) == len(args.clip)):
        raise ValueError("summaries, box, and clip must have same length.")

    best_idx = None
    best_score = -np.inf
    results = []

    for i, (path, b, c) in enumerate(zip(args.summaries, args.box, args.clip)):
        df = pd.read_csv(path)
        score, f1, mae = compute_score(df)
        results.append((path, b, c, score, f1, mae))
        print(f"{path}: BOX={b:.2f}, CLIP={c:.2f}, F1={f1:.3f}, MAE={mae:.3f}, score={score:.3f}")
        if score > best_score:
            best_score = score
            best_idx = i

    best_path, best_box, best_clip, best_score, best_f1, best_mae = results[best_idx]
    print("\nBest config:")
    print(f"   summary: {best_path}")
    print(f"   BOX_THRESHOLD = {best_box}")
    print(f"   CLIP_THRESHOLD = {best_clip}")
    print(f"   F1={best_f1:.3f}, MAE={best_mae:.3f}")

    out_cfg = {
        "BOX_THRESHOLD": best_box,
        "CLIP_THRESHOLD": best_clip,
        "source_summary": str(best_path),
    }

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_cfg, f, indent=2)

    print(f"\nWrote retrained config to {out_path}")


if __name__ == "__main__":
    main()
