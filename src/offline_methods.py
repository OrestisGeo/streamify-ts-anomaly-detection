import time
import numpy as np

from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow.keras import layers, models


def create_sliding_windows(values, window_size=100):
    """
    Creates overlapping sliding windows from a 1D time series.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < window_size:
        raise ValueError("Time series is shorter than window_size.")

    windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=window_size
    )

    return windows


def window_scores_to_point_scores(window_scores, series_length, window_size):
    """
    Converts window-level anomaly scores to point-level scores.

    Each window score is assigned to the last point of the window.
    The first window_size - 1 points receive the minimum score.
    """
    point_scores = np.full(series_length, np.nan)

    point_scores[window_size - 1:] = window_scores

    # Fill early points with the minimum available score
    min_score = np.nanmin(window_scores)
    point_scores[:window_size - 1] = min_score

    return point_scores


def run_isolation_forest_offline(
    values,
    window_size=100,
    n_estimators=100,
    contamination="auto",
    random_state=42,
):
    """
    Runs Isolation Forest in offline mode using the entire time series.
    """
    start_time = time.time()

    windows = create_sliding_windows(values, window_size=window_size)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(windows)

    # decision_function: higher = more normal
    # We multiply by -1 so higher score = more anomalous
    window_scores = -model.decision_function(windows)

    point_scores = window_scores_to_point_scores(
        window_scores=window_scores,
        series_length=len(values),
        window_size=window_size,
    )

    runtime = time.time() - start_time

    return point_scores, runtime


def build_dense_autoencoder(input_dim):
    """
    Builds a simple dense autoencoder.
    """
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def run_autoencoder_offline(
    values,
    window_size=100,
    epochs=10,
    batch_size=256,
    random_state=42,
):
    """
    Runs a simple Dense Autoencoder in offline mode.

    The model is trained on all windows and anomaly scores are
    reconstruction errors.
    """
    start_time = time.time()

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    windows = create_sliding_windows(values, window_size=window_size)

    model = build_dense_autoencoder(input_dim=window_size)

    model.fit(
        windows,
        windows,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
    )

    reconstructed = model.predict(
        windows,
        batch_size=batch_size,
        verbose=0,
    )

    window_scores = np.mean((windows - reconstructed) ** 2, axis=1)

    point_scores = window_scores_to_point_scores(
        window_scores=window_scores,
        series_length=len(values),
        window_size=window_size,
    )

    runtime = time.time() - start_time

    return point_scores, runtime