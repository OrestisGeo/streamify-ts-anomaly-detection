import time
import numpy as np
import tensorflow as tf

from src.offline_methods import (
    run_isolation_forest_offline,
    run_autoencoder_offline,
    create_sliding_windows,
    window_scores_to_point_scores,
    build_dense_autoencoder,
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


def run_method_on_rolling_batches(
    values,
    method_function,
    batch_size,
    window_size,
    history_batches,
    method_kwargs,
):
    """
    Runs an offline method in a rolling-window streaming setting.

    For each incoming batch:
    - use the current batch plus a short recent history
    - fit/score the method on that rolling context
    - keep only the scores that correspond to the current batch

    This is Variant 2:
    adaptive rolling-window streaming.
    """

    values = np.asarray(values, dtype=float)

    all_scores = np.full(len(values), np.nan)
    total_runtime = 0.0

    history_size = history_batches * batch_size

    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))

        context_start = max(0, start - history_size)
        context_values = values[context_start:end]

        current_batch_length = end - start

        if len(context_values) < window_size:
            all_scores[start:end] = 0.0
            continue

        context_scores, context_runtime = method_function(
            values=context_values,
            window_size=window_size,
            **method_kwargs,
        )

        # Keep only the scores that correspond to the current batch
        batch_scores = context_scores[-current_batch_length:]

        all_scores[start:end] = batch_scores
        total_runtime += context_runtime

    finite_mask = np.isfinite(all_scores)

    if finite_mask.any():
        min_score = np.nanmin(all_scores)
        all_scores[~finite_mask] = min_score
    else:
        all_scores[:] = 0.0

    return all_scores, total_runtime


def run_rolling_isolation_forest_streaming(
    values,
    batch_size=5000,
    window_size=100,
    history_batches=2,
    n_estimators=100,
    contamination="auto",
    random_state=42,
):
    """
    Variant 2 for Isolation Forest:
    rolling-window streaming with recent batch history.
    """

    return run_method_on_rolling_batches(
        values=values,
        method_function=run_isolation_forest_offline,
        batch_size=batch_size,
        window_size=window_size,
        history_batches=history_batches,
        method_kwargs={
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
        },
    )


def run_rolling_autoencoder_streaming(
    values,
    batch_size=5000,
    window_size=100,
    history_batches=2,
    epochs=5,
    batch_size_training=256,
    random_state=42,
):
    """
    Variant 2 for Dense Autoencoder:
    rolling-window streaming with recent batch history.
    """

    return run_method_on_rolling_batches(
        values=values,
        method_function=run_autoencoder_offline,
        batch_size=batch_size,
        window_size=window_size,
        history_batches=history_batches,
        method_kwargs={
            "epochs": epochs,
            "batch_size": batch_size_training,
            "random_state": random_state,
        },
    )


def run_online_finetune_autoencoder_streaming(
    values,
    batch_size=5000,
    window_size=100,
    init_batches=1,
    initial_epochs=10,
    finetune_epochs=1,
    batch_size_training=256,
    random_state=42,
):
    """
    Variant 2 alternative for Dense Autoencoder:
    online fine-tuning streaming adaptation.

    The model is first trained on an initial part of the stream.
    Then, for each new batch:
    - anomaly scores are computed using the current model
    - the model is fine-tuned on the current batch for a few epochs

    This avoids retraining the autoencoder from scratch for every batch.
    """

    start_time = time.time()

    values = np.asarray(values, dtype=float)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    all_scores = np.full(len(values), np.nan)

    init_size = init_batches * batch_size
    init_size = min(init_size, len(values))

    if init_size < window_size:
        raise ValueError("Initial training segment is shorter than window_size.")

    # Initial training segment
    init_values = values[:init_size]
    init_windows = create_sliding_windows(init_values, window_size=window_size)

    model = build_dense_autoencoder(input_dim=window_size)

    model.fit(
        init_windows,
        init_windows,
        epochs=initial_epochs,
        batch_size=batch_size_training,
        validation_split=0.1,
        verbose=0,
    )

    # Score the initial segment
    init_reconstructed = model.predict(
        init_windows,
        batch_size=batch_size_training,
        verbose=0,
    )

    init_window_scores = np.mean(
        (init_windows - init_reconstructed) ** 2,
        axis=1,
    )

    init_point_scores = window_scores_to_point_scores(
        window_scores=init_window_scores,
        series_length=len(init_values),
        window_size=window_size,
    )

    all_scores[:init_size] = init_point_scores

    # Process the rest of the stream batch by batch
    for start in range(init_size, len(values), batch_size):
        end = min(start + batch_size, len(values))
        batch_values = values[start:end]

        if len(batch_values) < window_size:
            all_scores[start:end] = np.nanmin(all_scores)
            continue

        batch_windows = create_sliding_windows(
            batch_values,
            window_size=window_size,
        )

        # Score current batch before updating the model
        reconstructed = model.predict(
            batch_windows,
            batch_size=batch_size_training,
            verbose=0,
        )

        batch_window_scores = np.mean(
            (batch_windows - reconstructed) ** 2,
            axis=1,
        )

        batch_point_scores = window_scores_to_point_scores(
            window_scores=batch_window_scores,
            series_length=len(batch_values),
            window_size=window_size,
        )

        all_scores[start:end] = batch_point_scores

        # Fine-tune model using the current batch
        model.fit(
            batch_windows,
            batch_windows,
            epochs=finetune_epochs,
            batch_size=batch_size_training,
            verbose=0,
        )

    finite_mask = np.isfinite(all_scores)

    if finite_mask.any():
        min_score = np.nanmin(all_scores)
        all_scores[~finite_mask] = min_score
    else:
        all_scores[:] = 0.0

    runtime = time.time() - start_time

    return all_scores, runtime


def run_selective_finetune_autoencoder_streaming(
    values,
    batch_size=5000,
    window_size=100,
    init_batches=1,
    initial_epochs=10,
    finetune_epochs=1,
    batch_size_training=256,
    normal_fraction=0.8,
    random_state=42,
):
    """
    Variant 2 alternative for Dense Autoencoder:
    selective online fine-tuning streaming adaptation.

    The model is first trained on an initial part of the stream.
    Then, for each new batch:
    - anomaly scores are computed using the current model
    - windows with the lowest reconstruction errors are selected
    - the model is fine-tuned only on these likely-normal windows

    This reduces the risk of adapting the model to anomalous patterns.
    """

    start_time = time.time()

    values = np.asarray(values, dtype=float)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    all_scores = np.full(len(values), np.nan)

    init_size = init_batches * batch_size
    init_size = min(init_size, len(values))

    if init_size < window_size:
        raise ValueError("Initial training segment is shorter than window_size.")

    if not 0 < normal_fraction <= 1:
        raise ValueError("normal_fraction must be in the interval (0, 1].")

    # Initial training segment
    init_values = values[:init_size]
    init_windows = create_sliding_windows(init_values, window_size=window_size)

    model = build_dense_autoencoder(input_dim=window_size)

    model.fit(
        init_windows,
        init_windows,
        epochs=initial_epochs,
        batch_size=batch_size_training,
        validation_split=0.1,
        verbose=0,
    )

    # Score the initial segment
    init_reconstructed = model.predict(
        init_windows,
        batch_size=batch_size_training,
        verbose=0,
    )

    init_window_scores = np.mean(
        (init_windows - init_reconstructed) ** 2,
        axis=1,
    )

    init_point_scores = window_scores_to_point_scores(
        window_scores=init_window_scores,
        series_length=len(init_values),
        window_size=window_size,
    )

    all_scores[:init_size] = init_point_scores

    # Process the rest of the stream batch by batch
    for start in range(init_size, len(values), batch_size):
        end = min(start + batch_size, len(values))
        batch_values = values[start:end]

        if len(batch_values) < window_size:
            all_scores[start:end] = np.nanmin(all_scores)
            continue

        batch_windows = create_sliding_windows(
            batch_values,
            window_size=window_size,
        )

        # Score current batch before updating the model
        reconstructed = model.predict(
            batch_windows,
            batch_size=batch_size_training,
            verbose=0,
        )

        batch_window_scores = np.mean(
            (batch_windows - reconstructed) ** 2,
            axis=1,
        )

        batch_point_scores = window_scores_to_point_scores(
            window_scores=batch_window_scores,
            series_length=len(batch_values),
            window_size=window_size,
        )

        all_scores[start:end] = batch_point_scores

        # Select likely-normal windows for fine-tuning.
        # We keep the windows with the lowest reconstruction errors.
        threshold = np.quantile(batch_window_scores, normal_fraction)

        likely_normal_mask = batch_window_scores <= threshold
        selected_windows = batch_windows[likely_normal_mask]

        # Safety fallback
        if len(selected_windows) == 0:
            selected_windows = batch_windows

        # Fine-tune only on likely-normal windows
        model.fit(
            selected_windows,
            selected_windows,
            epochs=finetune_epochs,
            batch_size=batch_size_training,
            verbose=0,
        )

    finite_mask = np.isfinite(all_scores)

    if finite_mask.any():
        min_score = np.nanmin(all_scores)
        all_scores[~finite_mask] = min_score
    else:
        all_scores[:] = 0.0

    runtime = time.time() - start_time

    return all_scores, runtime