from pathlib import Path
import sys
import time

import numpy as np


def load_sand_class(sand_code_dir="external/TSB-UAD/TSB_UAD/models"):
    """
    Loads the official SAND class from the cloned TSB-UAD repository.

    The TSB-UAD repository is expected to be cloned under:
    external/TSB-UAD/
    """

    # Compatibility fix for newer NumPy versions.
    # The official SAND code uses np.Inf.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    sand_code_dir = Path(sand_code_dir).resolve()

    if not sand_code_dir.exists():
        raise FileNotFoundError(
            f"SAND code directory not found: {sand_code_dir}\n"
            "Clone the official TSB-UAD repository first:\n"
            "!git clone https://github.com/TheDatumOrg/TSB-UAD.git external/TSB-UAD"
        )

    if str(sand_code_dir) not in sys.path:
        sys.path.insert(0, str(sand_code_dir))

    from sand import SAND

    return SAND


def align_scores_to_labels(scores, labels):
    """
    Aligns SAND scores with the original point-level labels.

    SAND can return fewer scores than the original time-series length because
    it works with subsequences. If needed, we pad the beginning with the
    minimum score so that scores and labels have the same length.
    """

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    n_labels = len(labels)

    if len(scores) == n_labels:
        return scores

    if len(scores) < n_labels:
        pad_length = n_labels - len(scores)
        pad_value = np.nanmin(scores)

        return np.concatenate([
            np.full(pad_length, pad_value),
            scores,
        ])

    return scores[-n_labels:]


def run_sand_online(
    values,
    labels,
    sand_code_dir="external/TSB-UAD/TSB_UAD/models",
    pattern_length=500,
    subsequence_length=100,
    k=6,
    alpha=0.5,
    init_length=3000,
    batch_size=1000,
    overlaping_rate=10,
):
    """
    Runs SAND in online/streaming mode.

    Parameters
    ----------
    values : array-like
        Time-series values.

    labels : array-like
        Point-level anomaly labels. Used only for score alignment.

    pattern_length : int
        Pattern length used by SAND. Must be greater than subsequence_length.

    subsequence_length : int
        Subsequence length used by SAND.

    k : int
        Number of clusters/subsequences in SAND.

    alpha : float
        Update rate in online mode.

    init_length : int
        Initial part of the stream used for initialization.

    batch_size : int
        Number of new points arriving in each batch.

    overlaping_rate : int
        SAND parameter name is intentionally kept as in the official code.
    """

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if pattern_length <= subsequence_length:
        raise ValueError("pattern_length must be greater than subsequence_length.")

    if init_length <= subsequence_length:
        raise ValueError("init_length must be greater than subsequence_length.")

    if batch_size <= subsequence_length:
        raise ValueError("batch_size must be greater than subsequence_length.")

    if len(values) <= init_length:
        raise ValueError("Time series length must be greater than init_length.")

    SAND = load_sand_class(sand_code_dir=sand_code_dir)

    start_time = time.time()

    model = SAND(
        pattern_length=pattern_length,
        subsequence_length=subsequence_length,
        k=k,
    )

    model.fit(
        values,
        online=True,
        alpha=alpha,
        init_length=init_length,
        batch_size=batch_size,
        overlaping_rate=overlaping_rate,
        verbose=False,
    )

    scores = np.asarray(model.decision_scores_, dtype=float)
    scores = align_scores_to_labels(scores, labels)

    runtime_seconds = time.time() - start_time

    return scores, runtime_seconds