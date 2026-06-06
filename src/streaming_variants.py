import numpy as np

from src.offline_methods import (
    run_isolation_forest_offline,
    run_autoencoder_offline,
)


def run_method_on_batches(
    values,
    method_function,
    batch_size,
    window_size,
    method_kwargs,
):
    """
    Runs an offline method independently on each incoming batch.

    This is the naive streaming variant:
    - the method sees only the current batch
    - there is no memory from previous batches
    - the method itself is not modified
    """

    values = np.asarray(values, dtype=float)

    all_scores = np.full(len(values), np.nan)
    total_runtime = 0.0

    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        batch_values = values[start:end]

        if len(batch_values) < window_size:
            all_scores[start:end] = 0.0
            continue

        batch_scores, batch_runtime = method_function(
            values=batch_values,
            window_size=window_size,
            **method_kwargs,
        )

        all_scores[start:end] = batch_scores
        total_runtime += batch_runtime

    # Replace possible NaN values with the minimum finite score
    finite_mask = np.isfinite(all_scores)

    if finite_mask.any():
        min_score = np.nanmin(all_scores)
        all_scores[~finite_mask] = min_score
    else:
        all_scores[:] = 0.0

    return all_scores, total_runtime


def run_batch_isolation_forest_streaming(
    values,
    batch_size=5000,
    window_size=100,
    n_estimators=100,
    contamination="auto",
    random_state=42,
):
    """
    Variant 1 for Isolation Forest:
    independently fit and score each incoming batch.
    """

    return run_method_on_batches(
        values=values,
        method_function=run_isolation_forest_offline,
        batch_size=batch_size,
        window_size=window_size,
        method_kwargs={
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
        },
    )


def run_batch_autoencoder_streaming(
    values,
    batch_size=5000,
    window_size=100,
    epochs=5,
    batch_size_training=256,
    random_state=42,
):
    """
    Variant 1 for Dense Autoencoder:
    independently fit and score each incoming batch.
    """

    return run_method_on_batches(
        values=values,
        method_function=run_autoencoder_offline,
        batch_size=batch_size,
        window_size=window_size,
        method_kwargs={
            "epochs": epochs,
            "batch_size": batch_size_training,
            "random_state": random_state,
        },
    )