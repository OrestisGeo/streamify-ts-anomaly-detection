import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


def top_k_threshold(scores, labels):
    """
    Selects the top-k highest anomaly scores as anomalies,
    where k is the number of true anomalies.

    This is used only for evaluation.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    k = int(labels.sum())

    if k == 0:
        return np.zeros_like(labels)

    threshold = np.partition(scores, -k)[-k]
    predictions = (scores >= threshold).astype(int)

    return predictions


def evaluate_scores(
    scores,
    labels,
    method_name,
    dataset_name,
    runtime_seconds,
    setting="offline",
):
    """
    Evaluates anomaly scores against binary labels.
    Higher score means more anomalous.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    valid_mask = np.isfinite(scores)

    scores = scores[valid_mask]
    labels = labels[valid_mask]

    if len(np.unique(labels)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(labels, scores)
        pr_auc = average_precision_score(labels, scores)

    predictions = top_k_threshold(scores, labels)

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        "dataset": dataset_name,
        "method": method_name,
        "setting": setting,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "runtime_seconds": runtime_seconds,
        "n_points": len(labels),
        "n_anomalies": int(labels.sum()),
        "anomaly_ratio": float(labels.mean()),
    }