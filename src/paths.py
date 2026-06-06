from pathlib import Path


PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GENERATED_DATA_DIR = DATA_DIR / "generated"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

# Result table paths
DATASET_INVENTORY_PATH = TABLES_DIR / "dataset_inventory.csv"
DOMAIN_SUMMARY_PATH = TABLES_DIR / "domain_summary.csv"
DATASET_CANDIDATES_PATH = TABLES_DIR / "dataset_candidates.csv"
SELECTED_SERIES_PATH = TABLES_DIR / "selected_series.csv"

OFFLINE_BASELINE_RESULTS_PATH = TABLES_DIR / "offline_baseline_results.csv"
SAND_STREAMING_RESULTS_PATH = TABLES_DIR / "sand_streaming_results.csv"
ALL_BASELINE_RESULTS_PATH = TABLES_DIR / "all_baseline_results.csv"


def create_project_dirs():
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)