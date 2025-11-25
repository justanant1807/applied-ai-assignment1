# mlops/inference_logger.py

from pathlib import Path
from datetime import datetime

import pandas as pd


LOG_PATH = Path("out/inference_logs.csv")


def log_inference_run(
    version_label: str,
    image_id: str,
    df_bin,
    df_order=None,
):
    """
    Append a single inference record to out/inference_logs.csv.

    df_bin: DataFrame with per-SKU rows for this bin.
    df_order: optional DataFrame with order validation rows.
    """
    timestamp = datetime.utcnow().isoformat()

    num_skus = len(df_bin["asin"].unique())
    num_full_match = (df_bin["status"] == "FULL_MATCH").sum()
    num_not_found = (df_bin["status"] == "NOT_FOUND").sum()

    if df_order is not None:
        num_order_lines = len(df_order)
        num_pass = (df_order["order_status"] == "PASS").sum()
    else:
        num_order_lines = 0
        num_pass = 0

    record = {
        "timestamp_utc": timestamp,
        "version": version_label,
        "image_id": image_id,
        "num_skus": num_skus,
        "num_full_match": int(num_full_match),
        "num_not_found": int(num_not_found),
        "num_order_lines": int(num_order_lines),
        "num_order_pass": int(num_pass),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        df_log = pd.read_csv(LOG_PATH)
        df_log = pd.concat([df_log, pd.DataFrame([record])], ignore_index=True)
    else:
        df_log = pd.DataFrame([record])

    df_log.to_csv(LOG_PATH, index=False)
