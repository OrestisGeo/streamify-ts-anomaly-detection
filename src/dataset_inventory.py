import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loading import read_out_file


def build_inventory(raw_data_dir, file_extension="*.out"):
    """
    Scans all time-series files and creates an inventory table.
    """

    data_files = list(raw_data_dir.rglob(file_extension))

    records = []

    for path in tqdm(data_files):
        try:
            df = read_out_file(path)
        except Exception as e:
            print("Failed:", path, "|", e)
            continue

        relative_path = path.relative_to(raw_data_dir)
        parts = relative_path.parts

        domain = parts[0] if len(parts) > 1 else "unknown"

        length = len(df)
        anomaly_count = int((df["label"] == 1).sum())
        anomaly_ratio = anomaly_count / length if length > 0 else np.nan

        records.append({
            "path": str(path),
            "relative_path": str(relative_path),
            "domain": domain,
            "file_name": path.name,
            "length": length,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_ratio,
            "missing_values": int(df["value"].isna().sum()),
            "value_mean": float(df["value"].mean()),
            "value_std": float(df["value"].std()),
        })

    return pd.DataFrame(records)


def summarize_domains(inventory):
    """
    Creates a summary table per domain.
    """

    domain_summary = (
        inventory
        .groupby("domain")
        .agg(
            n_series=("file_name", "count"),
            median_length=("length", "median"),
            min_length=("length", "min"),
            max_length=("length", "max"),
            median_anomaly_ratio=("anomaly_ratio", "median"),
        )
        .sort_values("n_series", ascending=False)
    )

    return domain_summary


def filter_candidate_series(
    inventory,
    min_length=2000,
    max_length=100000,
    min_anomaly_ratio=0.001,
    max_anomaly_ratio=0.10,
):
    """
    Keeps time series that are useful candidates for the project.
    """

    candidates = inventory[
        (inventory["length"] >= min_length) &
        (inventory["length"] <= max_length) &
        (inventory["anomaly_count"] > 0) &
        (inventory["anomaly_ratio"] >= min_anomaly_ratio) &
        (inventory["anomaly_ratio"] <= max_anomaly_ratio) &
        (inventory["missing_values"] == 0)
    ].copy()

    candidates = candidates.sort_values(
        by=["domain", "length", "anomaly_ratio"],
        ascending=[True, True, True]
    )

    return candidates