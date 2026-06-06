from pathlib import Path


PROJECT_ROOT = Path.cwd()

# Main folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GENERATED_DATA_DIR = DATA_DIR / "generated"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

POSTER_DIR = PROJECT_ROOT / "poster"


# Dataset preparation tables
DATASET_INVENTORY_PATH = TABLES_DIR / "dataset_inventory.csv"
DOMAIN_SUMMARY_PATH = TABLES_DIR / "domain_summary.csv"
DATASET_CANDIDATES_PATH = TABLES_DIR / "dataset_candidates.csv"
CANDIDATE_DOMAIN_SUMMARY_PATH = TABLES_DIR / "candidate_domain_summary.csv"
SELECTED_SERIES_PATH = TABLES_DIR / "selected_series.csv"


# Generated Normality datasets
GENERATED_DATASETS_SUMMARY_PATH = GENERATED_DATA_DIR / "generated_datasets_summary.csv"

NORMALITY_1_PATH = GENERATED_DATA_DIR / "normality_1.csv"
NORMALITY_2_PATH = GENERATED_DATA_DIR / "normality_2.csv"
NORMALITY_3_PATH = GENERATED_DATA_DIR / "normality_3.csv"


# Step 4 - Offline baseline results
OFFLINE_BASELINE_RESULTS_PATH = TABLES_DIR / "offline_baseline_results.csv"


# Step 5 - SAND streaming baseline results
SAND_STREAMING_RESULTS_PATH = TABLES_DIR / "sand_streaming_results.csv"


# Step 6 - Streaming Variant 1 results
# Naive batch streaming: each batch is processed independently.
VARIANT1_BATCH_RESULTS_PATH = TABLES_DIR / "variant1_batch_results.csv"


# Step 6 - Streaming Variant 2 results
# Rolling-window/adaptive streaming: recent batches are used as short-term memory.
VARIANT2_ROLLING_RESULTS_PATH = TABLES_DIR / "variant2_rolling_results.csv"


# Combined result tables
ALL_BASELINE_RESULTS_PATH = TABLES_DIR / "all_baseline_results.csv"
ALL_STREAMING_RESULTS_PATH = TABLES_DIR / "all_streaming_results.csv"
ALL_RESULTS_PATH = TABLES_DIR / "all_results.csv"


# Optional plot output paths
OFFLINE_BASELINE_PLOT_PATH = PLOTS_DIR / "offline_baseline_results.png"
SAND_RESULTS_PLOT_PATH = PLOTS_DIR / "sand_streaming_results.png"
VARIANT1_RESULTS_PLOT_PATH = PLOTS_DIR / "variant1_batch_results.png"
VARIANT2_RESULTS_PLOT_PATH = PLOTS_DIR / "variant2_rolling_results.png"
ALL_RESULTS_METRICS_PLOT_PATH = PLOTS_DIR / "all_results_metrics.png"
ALL_RESULTS_RUNTIME_PLOT_PATH = PLOTS_DIR / "all_results_runtime.png"


def create_project_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    POSTER_DIR.mkdir(parents=True, exist_ok=True)