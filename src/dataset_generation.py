from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import read_out_file


def z_normalize(values):
    """
    Applies z-normalization to a time series.
    If the standard deviation is zero, the original centered values are returned.
    """
    values = np.asarray(values, dtype=float)

    mean = np.nanmean(values)
    std = np.nanstd(values)

    if std == 0 or np.isnan(std):
        return values - mean

    return (values - mean) / std


def create_normality_datasets(
    selected_series,
    raw_data_dir,
    generated_data_dir,
    normalize=True,
):
    """
    Creates Normality 1, Normality 2, ..., Normality N datasets
    by concatenating the selected time series in their given order.

    Parameters
    ----------
    selected_series : pandas.DataFrame
        DataFrame containing the selected time series.
        It must contain: order, domain, file_name, relative_path.

    raw_data_dir : pathlib.Path
        Path to the raw dataset directory.

    generated_data_dir : pathlib.Path
        Path where the generated datasets will be saved.

    normalize : bool
        If True, each individual time series is z-normalized before concatenation.

    Returns
    -------
    summary : pandas.DataFrame
        Summary table for the generated normality datasets.
    """

    generated_data_dir = Path(generated_data_dir)
    generated_data_dir.mkdir(parents=True, exist_ok=True)

    selected_series = selected_series.sort_values("order").reset_index(drop=True)

    loaded_segments = []

    for _, row in selected_series.iterrows():
        path = Path(raw_data_dir) / row["relative_path"]

        df = read_out_file(path)

        values = df["value"].to_numpy(dtype=float)
        labels = df["label"].to_numpy(dtype=int)

        if normalize:
            values = z_normalize(values)

        segment_df = pd.DataFrame({
            "value": values,
            "label": labels,
            "source_domain": row["domain"],
            "source_file": row["file_name"],
            "source_order": row["order"],
            "original_index": np.arange(len(df)),
        })

        loaded_segments.append(segment_df)

    summary_records = []

    for normality_level in range(1, len(loaded_segments) + 1):
        used_segments = loaded_segments[:normality_level]

        generated_df = pd.concat(
            used_segments,
            ignore_index=True
        )

        generated_df["time_index"] = np.arange(len(generated_df))

        output_name = f"normality_{normality_level}.csv"
        output_path = generated_data_dir / output_name

        generated_df.to_csv(output_path, index=False)

        boundary_indices = []
        current_length = 0

        for segment in used_segments[:-1]:
            current_length += len(segment)
            boundary_indices.append(current_length)

        summary_records.append({
            "dataset_name": output_name,
            "normality_level": normality_level,
            "n_segments": normality_level,
            "length": len(generated_df),
            "anomaly_count": int((generated_df["label"] == 1).sum()),
            "anomaly_ratio": float((generated_df["label"] == 1).mean()),
            "domains": " + ".join([seg["source_domain"].iloc[0] for seg in used_segments]),
            "distribution_shifts": normality_level - 1,
            "boundary_indices": boundary_indices,
            "normalized_per_segment": normalize,
        })

    summary = pd.DataFrame(summary_records)

    summary.to_csv(generated_data_dir / "generated_datasets_summary.csv", index=False)

    return summary