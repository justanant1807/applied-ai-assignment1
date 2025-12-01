"""
EDA.py

Exploratory Data Analysis for Assignment 1: Bin-level SKU verification.

This script inspects:
  1. METADATA  (metadata/*.json, BIN_FCSKU_DATA)
  2. IMAGES    (bin-images/*.{jpg,jpeg,png})
  3. SUMMARIES (out/summary_*.csv from the detection pipeline)

It prints key statistics to the console and writes plots / small CSVs to:
    out/eda/

Usage (from project root):
    python EDA.py

Optional arguments:
    python EDA.py --metadata-dir metadata --image-dir bin-images --summary-dir out --output-dir out/eda

Dependencies:
    pandas, numpy, matplotlib, Pillow, tqdm
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Make tqdm optional: if not installed, fall back to identity
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# -------------------
# Argument parsing
# -------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EDA for Assignment 1 dataset")
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="metadata",
        help="Directory containing per-bin JSON metadata files",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="bin-images",
        help="Directory containing bin images",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="out",
        help="Directory containing summary_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/eda",
        help="Directory where plots / EDA artifacts will be saved",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=0,
        help="If >0, randomly sample this many images for image stats (0 = use all)",
    )
    return parser.parse_args()


# -------------------
# Utility
# -------------------

def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_int_from_stem(path: Path):
    """Best-effort int conversion from filename stem '00062' -> 62."""
    try:
        return int(path.stem)
    except ValueError:
        return path.stem  # fall back to string if weird name


# -------------------
# METADATA EDA
# -------------------

def load_all_metadata(metadata_dir: Path) -> pd.DataFrame:
    """
    Load all JSON metadata into a flat DataFrame with columns:
        image_id, asin, sku_name, expected
    """
    rows = []

    json_paths = sorted(metadata_dir.glob("*.json"))
    if not json_paths:
        print(f"[META] No JSON files found in {metadata_dir} - skipping metadata EDA.")
        return pd.DataFrame()

    for path in tqdm(json_paths, desc="Loading metadata JSONs"):
        image_id = safe_int_from_stem(path)
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        sku_dict = meta.get("BIN_FCSKU_DATA", {})
        for asin, data in sku_dict.items():
            rows.append(
                {
                    "image_id": image_id,
                    "asin": asin,
                    "sku_name": data.get("name", ""),
                    "expected": int(data.get("quantity", 0)),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["expected"] = df["expected"].astype(int)
    return df


def analyze_metadata(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return

    print("\n================ METADATA STATS ================")
    num_bins = df["image_id"].nunique()
    num_rows = len(df)
    print(f"Number of bins (JSON files): {num_bins}")
    print(f"Total SKU rows across all bins: {num_rows}")

    # SKUs per bin
    skus_per_bin = df.groupby("image_id")["asin"].nunique()
    print("\nSKUs per bin:")
    print(f"  mean: {skus_per_bin.mean():.2f}")
    print(f"  min : {skus_per_bin.min()}  max: {skus_per_bin.max()}")
    print(f"  25% : {skus_per_bin.quantile(0.25):.2f}")
    print(f"  50% : {skus_per_bin.median():.2f}")
    print(f"  75% : {skus_per_bin.quantile(0.75):.2f}")

    # Quantities
    print("\nExpected quantity per SKU row:")
    print(df["expected"].describe())

    # Top SKUs by frequency
    sku_counts = df["asin"].value_counts().head(10)
    print("\nTop 10 SKUs by number of bins they appear in:")
    print(sku_counts)

    ensure_out_dir(out_dir)

    # Hist: SKUs per bin
    plt.figure()
    plt.hist(skus_per_bin, bins=20)
    plt.xlabel("# SKUs per bin")
    plt.ylabel("# bins")
    plt.title("Distribution of SKUs per bin")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_skus_per_bin.png")
    plt.close()

    # Hist: expected quantity
    plt.figure()
    plt.hist(df["expected"], bins=20)
    plt.xlabel("Expected quantity per SKU row")
    plt.ylabel("# rows")
    plt.title("Distribution of expected quantities")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_expected_qty_per_sku_row.png")
    plt.close()

    # Save flattened metadata for reference
    df.to_csv(out_dir / "flattened_metadata.csv", index=False)
    print(f"[META] Saved flattened metadata to {out_dir / 'flattened_metadata.csv'}")


# -------------------
# IMAGE EDA
# -------------------

def collect_image_stats(image_dir: Path, sample_images: int = 0) -> pd.DataFrame:
    """
    Read basic statistics (width, height, aspect) for images.
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    img_paths = []
    for e in exts:
        img_paths.extend(image_dir.glob(e))

    img_paths = sorted(set(p.resolve() for p in img_paths))
    if not img_paths:
        print(f"[IMG] No images found in {image_dir} - skipping image EDA.")
        return pd.DataFrame()

    if sample_images and sample_images < len(img_paths):
        print(f"[IMG] Sampling {sample_images} / {len(img_paths)} images for stats.")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(img_paths), size=sample_images, replace=False)
        img_paths = [img_paths[i] for i in idx]

    rows = []
    for path in tqdm(img_paths, desc="Loading images for stats"):
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception as e:  # pragma: no cover
            print(f"[IMG] Failed to open {path}: {e}")
            continue

        if h == 0:
            aspect = np.nan
        else:
            aspect = w / h

        rows.append(
            {
                "image_id": safe_int_from_stem(path),
                "path": str(path),
                "width": w,
                "height": h,
                "aspect": aspect,
                "orientation": (
                    "landscape" if w > h else "portrait" if h > w else "square"
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_images(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return

    print("\n================ IMAGE STATS ================")
    print(f"Number of images analysed: {len(df)}")
    print("\nWidth (px):")
    print(df["width"].describe())
    print("\nHeight (px):")
    print(df["height"].describe())
    print("\nAspect ratio (w/h):")
    print(df["aspect"].describe())

    orientation_counts = df["orientation"].value_counts()
    print("\nImage orientation counts:")
    print(orientation_counts)

    ensure_out_dir(out_dir)

    # Hist: width
    plt.figure()
    plt.hist(df["width"], bins=20)
    plt.xlabel("Width (px)")
    plt.ylabel("# images")
    plt.title("Image width distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_image_width.png")
    plt.close()

    # Hist: height
    plt.figure()
    plt.hist(df["height"], bins=20)
    plt.xlabel("Height (px)")
    plt.ylabel("# images")
    plt.title("Image height distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_image_height.png")
    plt.close()

    # Hist: aspect ratio
    plt.figure()
    plt.hist(df["aspect"].dropna(), bins=20)
    plt.xlabel("Aspect ratio (w/h)")
    plt.ylabel("# images")
    plt.title("Image aspect ratio distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_image_aspect.png")
    plt.close()

    # Bar: orientation
    plt.figure()
    orientation_counts.plot(kind="bar")
    plt.xlabel("Orientation")
    plt.ylabel("# images")
    plt.title("Image orientation counts")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_image_orientation.png")
    plt.close()

    # Save for reference
    df.to_csv(out_dir / "image_stats.csv", index=False)
    print(f"[IMG] Saved image stats to {out_dir / 'image_stats.csv'}")


# -------------------
# SUMMARY / MODEL EDA
# -------------------

def load_summary_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "expected" in df.columns:
        df["expected"] = df["expected"].astype(int)
    if "visible" in df.columns:
        df["visible"] = df["visible"].astype(int)
    return df


def compute_presence_metrics(df: pd.DataFrame):
    """
    Treat 'expected > 0' as GT present, 'visible > 0' as prediction present.
    """
    gt_present = df["expected"] > 0
    pred_present = df["visible"] > 0

    TP = int(((gt_present) & (pred_present)).sum())
    TN = int(((~gt_present) & (~pred_present)).sum())
    FP = int(((~gt_present) & (pred_present)).sum())
    FN = int(((gt_present) & (~pred_present)).sum())

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


def analyze_single_summary(df: pd.DataFrame, label: str, out_dir: Path):
    print(f"\n================ SUMMARY: {label} ================")
    print(f"# rows (SKU-bin pairs): {len(df)}")
    print(f"# unique bins (image_id): {df['image_id'].nunique()}")

    status_counts = Counter(df["status"])
    print("\nStatus distribution:")
    for k, v in status_counts.items():
        print(f"  {k:15s} : {v}")

    if "pretty_status" in df.columns:
        print("\nPretty status distribution:")
        for k, v in Counter(df["pretty_status"]).items():
            print(f"  {k:25s} : {v}")

    metrics = compute_presence_metrics(df)
    print("\nPresence/absence metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:10s} : {v:.4f}")
        else:
            print(f"  {k:10s} : {v}")

    # Plots
    ensure_out_dir(out_dir)

    # Bar of status counts
    plt.figure()
    keys = list(status_counts.keys())
    vals = [status_counts[k] for k in keys]
    plt.bar(keys, vals)
    plt.xlabel("Status")
    plt.ylabel("# rows")
    plt.title(f"Status distribution - {label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"bar_status_{label}.png")
    plt.close()

    # Scatter: expected vs visible
    plt.figure()
    plt.scatter(df["expected"], df["visible"], alpha=0.3)
    plt.xlabel("Expected quantity")
    plt.ylabel("Detected (visible) quantity")
    plt.title(f"Expected vs visible counts - {label}")
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_expected_vs_visible_{label}.png")
    plt.close()

    return metrics


def analyze_all_summaries(summary_dir: Path, out_dir: Path):
    summary_paths = sorted(summary_dir.glob("summary_*.csv"))
    if not summary_paths:
        print(f"[SUM] No summary_*.csv files found in {summary_dir} - skipping summary EDA.")
        return

    all_metrics = []
    for path in summary_paths:
        label = path.stem  # e.g., summary_v5_clip020_box017
        df = load_summary_df(path)
        metrics = analyze_single_summary(df, label, out_dir)
        metrics["summary_file"] = path.name
        all_metrics.append(metrics)

    # Save metrics table + F1 trend plot
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "summary_presence_metrics.csv", index=False)
    print(f"[SUM] Saved presence metrics to {out_dir / 'summary_presence_metrics.csv'}")

    if len(metrics_df) > 1:
        plt.figure()
        plt.plot(metrics_df["summary_file"], metrics_df["f1"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1 score")
        plt.xlabel("Summary file")
        plt.title("F1 trend across summary versions")
        plt.tight_layout()
        plt.savefig(out_dir / "line_f1_across_summaries.png")
        plt.close()


# -------------------
# MAIN
# -------------------

def main():
    args = parse_args()

    metadata_dir = Path(args.metadata_dir)
    image_dir = Path(args.image_dir)
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.output_dir)

    ensure_out_dir(out_dir)

    # 1) METADATA
    meta_df = load_all_metadata(metadata_dir)
    analyze_metadata(meta_df, out_dir)

    # 2) IMAGES
    img_df = collect_image_stats(image_dir, sample_images=args.sample_images)
    analyze_images(img_df, out_dir)

    # 3) SUMMARIES / MODEL OUTPUT
    analyze_all_summaries(summary_dir, out_dir)

    print("\nEDA complete. Check outputs in:", out_dir)


if __name__ == "__main__":
    main()
