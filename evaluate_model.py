import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


OUT_DIR = Path("out")


def load_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["expected"] = int(r["expected"])
            r["visible"] = int(r["visible"])
            rows.append(r)
    return rows


def compute_presence_metrics(rows):
    TP = TN = FP = FN = 0

    for r in rows:
        gt_present = r["expected"] > 0
        pred_present = r["visible"] > 0

        if gt_present and pred_present:
            TP += 1
        elif (not gt_present) and (not pred_present):
            TN += 1
        elif (not gt_present) and pred_present:
            FP += 1
        elif gt_present and (not pred_present):
            FN += 1

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


def compute_count_metrics(rows):
    present_rows = [r for r in rows if r["expected"] > 0]
    if not present_rows:
        return {
            "num_present_rows": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "exact_count_accuracy": 0.0,
        }

    abs_errors = []
    sq_errors = []
    exact = 0

    for r in present_rows:
        e = r["expected"]
        v = r["visible"]
        diff = v - e
        abs_errors.append(abs(diff))
        sq_errors.append(diff * diff)
        if v == e:
            exact += 1

    n = len(present_rows)
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    exact_acc = exact / n

    return {
        "num_present_rows": n,
        "mae": mae,
        "rmse": rmse,
        "exact_count_accuracy": exact_acc,
    }


def compute_bin_level_metrics(rows):
    bins = defaultdict(list)
    for r in rows:
        bins[r["image_id"]].append(r)

    total_bins = len(bins)
    perfect_bins = 0

    for image_id, rs in bins.items():
        if all(r["visible"] == r["expected"] for r in rs):
            perfect_bins += 1

    bin_accuracy = perfect_bins / total_bins if total_bins > 0 else 0.0

    return {
        "num_bins": total_bins,
        "perfect_bins": perfect_bins,
        "bin_exact_accuracy": bin_accuracy,
    }


def status_distribution(rows):
    counts = Counter(r["status"] for r in rows)
    total = sum(counts.values())
    return {
        status: {
            "count": c,
            "percent": (c / total) if total > 0 else 0.0,
        }
        for status, c in counts.items()
    }


def print_single_report(summary_path: Path, name: str = None):
    rows = load_summary(summary_path)
    if not rows:
        print(f"[ERROR] {summary_path} is empty.")
        return

    label = name if name is not None else summary_path.name
    print(f"\n===== EVALUATION: {label} =====")
    print(f"Rows: {len(rows)}")

    presence = compute_presence_metrics(rows)
    count_metrics = compute_count_metrics(rows)
    bin_metrics = compute_bin_level_metrics(rows)
    status_stats = status_distribution(rows)

    print("\n-- Presence (SKU present vs not) --")
    print(
        f"TP={presence['TP']}, FP={presence['FP']}, "
        f"FN={presence['FN']}, TN={presence['TN']}"
    )
    print(f"Accuracy : {presence['accuracy']:.3f}")
    print(f"Precision: {presence['precision']:.3f}")
    print(f"Recall   : {presence['recall']:.3f}")
    print(f"F1-score : {presence['f1']:.3f}")

    print("\n-- Count quality (expected > 0) --")
    print(f"Num SKUs with expected > 0: {count_metrics['num_present_rows']}")
    print(f"Exact count accuracy      : {count_metrics['exact_count_accuracy']:.3f}")
    print(f"Mean Absolute Error (MAE) : {count_metrics['mae']:.3f}")
    print(f"Root MSE (RMSE)           : {count_metrics['rmse']:.3f}")

    print("\n-- Bin-level accuracy --")
    print(f"Total bins         : {bin_metrics['num_bins']}")
    print(f"Perfect bins       : {bin_metrics['perfect_bins']}")
    print(f"Bin exact accuracy : {bin_metrics['bin_exact_accuracy']:.3f}")

    print("\n-- Status distribution --")
    for status, info in status_stats.items():
        print(
            f"{status:17s}: {info['count']:4d} "
            f"({info['percent'] * 100:5.1f}%)"
        )

    # return metrics for comparison
    return {
        "presence": presence,
        "count": count_metrics,
        "bin": bin_metrics,
        "status": status_stats,
    }


def compare_reports(path1: Path, path2: Path, name1: str, name2: str):
    m1 = print_single_report(path1, name1)
    m2 = print_single_report(path2, name2)

    print("\n===== COMPARISON (", name1, "vs", name2, ") =====")

    def fmt(x):
        return f"{x:.3f}"

    print("\n-- Presence metrics --")
    print(
        f"{'Metric':20s}{name1:>15s}{name2:>15s}"
    )
    print(
        f"{'Accuracy':20s}{fmt(m1['presence']['accuracy']):>15s}{fmt(m2['presence']['accuracy']):>15s}"
    )
    print(
        f"{'Precision':20s}{fmt(m1['presence']['precision']):>15s}{fmt(m2['presence']['precision']):>15s}"
    )
    print(
        f"{'Recall':20s}{fmt(m1['presence']['recall']):>15s}{fmt(m2['presence']['recall']):>15s}"
    )
    print(
        f"{'F1-score':20s}{fmt(m1['presence']['f1']):>15s}{fmt(m2['presence']['f1']):>15s}"
    )

    print("\n-- Count metrics --")
    print(
        f"{'Metric':20s}{name1:>15s}{name2:>15s}"
    )
    print(
        f"{'Exact count acc.':20s}"
        f"{fmt(m1['count']['exact_count_accuracy']):>15s}"
        f"{fmt(m2['count']['exact_count_accuracy']):>15s}"
    )
    print(
        f"{'MAE':20s}{fmt(m1['count']['mae']):>15s}{fmt(m2['count']['mae']):>15s}"
    )
    print(
        f"{'RMSE':20s}{fmt(m1['count']['rmse']):>15s}{fmt(m2['count']['rmse']):>15s}"
    )

    print("\n-- Bin-level metrics --")
    print(
        f"{'Metric':20s}{name1:>15s}{name2:>15s}"
    )
    print(
        f"{'Bin exact acc.':20s}"
        f"{fmt(m1['bin']['bin_exact_accuracy']):>15s}"
        f"{fmt(m2['bin']['bin_exact_accuracy']):>15s}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SKU verification summaries (strict/relaxed)."
    )
    p.add_argument(
        "--summary",
        type=str,
        help="Path to a single summary CSV to evaluate "
             "(default: out/summary_v1_strict.csv)",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("STRICT", "RELAXED"),
        help="Compare two summary CSVs: e.g. "
             "--compare out/summary_v1_strict.csv out/summary_v2_relaxed.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        path1 = Path(args.compare[0])
        path2 = Path(args.compare[1])
        if not path1.exists() or not path2.exists():
            print("[ERROR] One or both comparison files do not exist.")
            print(" Given:", path1, path2)
            return
        compare_reports(path1, path2, "STRICT", "RELAXED")
    else:
        if args.summary:
            summary_path = Path(args.summary)
        else:
            summary_path = OUT_DIR / "summary_v1_strict.csv"

        if not summary_path.exists():
            print(f"[ERROR] Summary file not found: {summary_path}")
            return
        print_single_report(summary_path)


if __name__ == "__main__":
    main()
