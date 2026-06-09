# Streamify Time-Series Anomaly Detection

Project #2 for the course **Mining of Massive Datasets**.

This project studies time-series anomaly detection in both **offline** and **streaming-like** settings. The experiments use the public **TSB-UAD** benchmark and construct three generated datasets by concatenating time series from different domains. The main goal is to examine how anomaly detection methods behave when the normal behaviour of the stream changes over time because of **distribution shifts**.

---

## Main idea

The project compares:

1. **Offline baselines** that have access to the full time series.
2. **SAND**, used as an online/streaming baseline.
3. **Variant 1: Batch streaming**, where each incoming stream batch is processed independently.
4. **Variant 2: Rolling-window streaming**, where each incoming stream batch is processed together with recent historical batches.

The final comparison focuses on **PR-AUC**, **F1-score**, **Accuracy**, and **Runtime**. Accuracy is reported for completeness, but the most useful metrics for anomaly detection are PR-AUC and F1 because the datasets are highly imbalanced.

---

## Repository structure

```text
streamify-ts-anomaly-detection/
├── data/
│   ├── raw/                    # raw TSB-UAD dataset, not committed
│   └── generated/              # generated Normality datasets
├── external/                   # external repositories cloned during execution, not committed
├── notebooks/
│   └── project2_main.ipynb     # main notebook
├── results/
│   ├── tables/                 # CSV result tables
│   └── plots/                  # generated plots
├── src/
│   ├── data_loading.py
│   ├── dataset_generation.py
│   ├── dataset_inventory.py
│   ├── evaluation.py
│   ├── offline_methods.py
│   ├── paths.py
│   ├── sand_baseline.py
│   └── streaming_variants.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to run the notebook

The main notebook is:

```text
notebooks/project2_main.ipynb
```

The notebook is designed so that expensive experiments are controlled by execution flags. This allows the evaluator to either:

- load the already generated datasets and result CSV files, or
- recompute selected parts of the project from the raw dataset.

---

## Recommended run for evaluation

For a normal review of the project, the recommended option is to open the notebook and run it with all expensive computation flags set to `False`.

In the notebook, use:

```python
USE_GOOGLE_DRIVE_DATA = False

RUN_BUILD_INVENTORY = False
RUN_GENERATE_NORMALITIES = False
RUN_OFFLINE_BASELINES = False
RUN_SAND_BASELINE = False
RUN_VARIANT1_BATCH = False
RUN_VARIANT2_ROLLING = False
RUN_VARIANT2_FINETUNE_AE = False
RUN_VARIANT2_SELECTIVE_FINETUNE_AE = False
```

With these values, the notebook loads existing files from:

```text
data/generated/
results/tables/
results/plots/
```

This is the fastest way to inspect the workflow, the tables, and the final plots without rerunning the expensive experiments.

---

## Running in Google Colab

A typical Colab workflow is:

1. Clone the repository.
2. Install the dependencies.
3. Open and run `notebooks/project2_main.ipynb`.

Example setup:

```python
from pathlib import Path

username = "OrestisGeo"
repo = "streamify-ts-anomaly-detection"
repo_dir = Path(repo)

if not repo_dir.exists():
    !git clone https://github.com/{username}/{repo}.git
else:
    print(f"Repository already exists: {repo}")

%cd {repo}
```

Install the required packages:

```python
%pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
tensorflow
stumpy
tslearn
```

The packages `stumpy` and `tslearn` are needed for the SAND implementation.

---

## Dataset setup

The raw TSB-UAD dataset is **not included in the repository**, because it is too large for normal Git tracking.

To recompute the project from scratch, download and unzip the public TSB-UAD dataset and place the domain folders under:

```text
data/raw/
```

Expected structure:

```text
data/raw/
├── KDD21/
├── NASA-MSL/
├── SMD/
├── OPPORTUNITY/
├── Daphnet/
└── ...
```

The `.out` files are read as univariate time series with point-level anomaly labels.

### Option A: local or Colab runtime dataset

Place the unzipped dataset directly under:

```text
data/raw/
```

and keep:

```python
USE_GOOGLE_DRIVE_DATA = False
```

### Option B: Google Drive dataset

If the dataset is stored in Google Drive, set:

```python
USE_GOOGLE_DRIVE_DATA = True
```

and update:

```python
GOOGLE_DRIVE_DATASET_DIR = "your_drive_path_to_tsb_uad_dataset"
```

The notebook mounts Google Drive and creates a symbolic link:

```text
data/raw/ -> GOOGLE_DRIVE_DATASET_DIR
```

This avoids copying the full raw dataset into the Colab runtime.

---

## Generated Normality datasets

The notebook generates three streaming-like datasets:

| Dataset | Domains | Length | Anomalies | Anomaly ratio | Distribution shifts |
|---|---|---:|---:|---:|---:|
| Normality 1 | SMD | 23,688 | 980 | 4.14% | 0 |
| Normality 2 | SMD + OPPORTUNITY | 56,960 | 2,274 | 3.99% | 1 |
| Normality 3 | SMD + OPPORTUNITY + Daphnet | 66,559 | 2,597 | 3.90% | 2 |

A distribution shift occurs when the stream moves from one domain to another. In this project, this means that the normal behaviour of the time series changes over time.

Generated datasets are saved in:

```text
data/generated/
├── normality_1.csv
├── normality_2.csv
├── normality_3.csv
└── generated_datasets_summary.csv
```

Each generated dataset contains:

```text
value
label
source_domain
source_file
source_order
original_index
time_index
```

---

## Methods

### Offline baselines

The offline baselines are:

- **Isolation Forest**
- **Dense Autoencoder**

Both methods use sliding windows of size 100. They are considered offline methods because the full dataset is available before scoring.

### SAND baseline

SAND is used as the streaming/online baseline through the official TSB-UAD implementation:

```text
https://github.com/TheDatumOrg/TSB-UAD
```

During execution, the repository is cloned under:

```text
external/TSB-UAD/
```

The local wrapper is:

```text
src/sand_baseline.py
```

### Variant 1: Batch streaming

Variant 1 adapts the offline methods to a simple batch-wise streaming setting.

Each stream batch is processed independently:

```text
current batch -> fit/score method -> next batch
```

No information from previous batches is used.

Implemented methods:

- Batch Isolation Forest
- Batch Dense Autoencoder

### Variant 2: Rolling-window streaming

Variant 2 uses a short recent history when processing each incoming batch:

```text
previous batches + current batch -> fit/score method
```

Only the scores that correspond to the current batch are kept for evaluation.

Implemented methods:

- Rolling Isolation Forest
- Rolling Dense Autoencoder

In the notebook, the stream batch size is defined inside the Variant 1 and Variant 2 sections as `STREAM_BATCH_SIZE`, and the input sliding-window size is defined as `WINDOW_SIZE = 100`.

### Optional Autoencoder experiments

The notebook also contains optional Autoencoder adaptation cells:

- Online Fine-tuning Dense Autoencoder
- Selective Online Fine-tuning Dense Autoencoder

These are kept as additional experiments. They are not required for reproducing the main Variant 1 vs Variant 2 comparison and can remain disabled:

```python
RUN_VARIANT2_FINETUNE_AE = False
RUN_VARIANT2_SELECTIVE_FINETUNE_AE = False
```

---

## Evaluation

Each method produces an anomaly score for every time point. Higher scores indicate points that are more likely to be anomalous.

The notebook computes:

| Metric | Meaning |
|---|---|
| ROC-AUC | Overall ranking quality between normal and anomalous points |
| PR-AUC | Ranking quality focused on the anomaly class |
| F1-score | Binary detection quality after thresholding |
| Accuracy | Secondary metric, included for completeness |
| Runtime | Execution time in seconds |

For F1-score and Accuracy, the notebook uses **top-k thresholding**:

```text
1. Rank all anomaly scores from highest to lowest.
2. Label the top k points as anomalies.
3. Set k equal to the true number of anomalies in the dataset.
```

This makes all methods predict the same number of anomalous points and allows a fair comparison of their ranked anomaly scores.

---

## Reproducing the main results from scratch

To recompute the main results from the raw TSB-UAD dataset:

1. Place the raw TSB-UAD dataset under `data/raw/`, or configure Google Drive access.
2. Set the following flags:

```python
RUN_BUILD_INVENTORY = True
RUN_GENERATE_NORMALITIES = True
RUN_OFFLINE_BASELINES = True
RUN_SAND_BASELINE = True
RUN_VARIANT1_BATCH = True
RUN_VARIANT2_ROLLING = True
RUN_VARIANT2_FINETUNE_AE = False
RUN_VARIANT2_SELECTIVE_FINETUNE_AE = False
```

3. Run the notebook from top to bottom.

This recomputes the dataset selection, the generated Normality datasets, the offline baselines, SAND, and the two main streaming variants.

---

## Rerunning only Variant 1 and Variant 2

If the generated datasets and baseline results already exist, and only the streaming variants need to be recomputed, use:

```python
RUN_BUILD_INVENTORY = False
RUN_GENERATE_NORMALITIES = False
RUN_OFFLINE_BASELINES = False
RUN_SAND_BASELINE = False
RUN_VARIANT1_BATCH = True
RUN_VARIANT2_ROLLING = True
RUN_VARIANT2_FINETUNE_AE = False
RUN_VARIANT2_SELECTIVE_FINETUNE_AE = False
```

Then run only these notebook sections:

```text
Step 6 - Variant 1: Naive batch streaming
Step 7 - Variant 2: Rolling-window streaming
Final evaluation table
Poster result tables
Poster plots
```

Do not run the optional fine-tuning or selective fine-tuning cells unless those additional experiments are needed.

---

## Result files

The main tables are saved under:

```text
results/tables/
```

Important result files:

```text
offline_baseline_results.csv
sand_streaming_results.csv
variant1_batch_results.csv
variant2_rolling_results.csv
all_results.csv
poster_main_results.csv
poster_variant_comparison.csv
poster_best_methods.csv
poster_compact_results.csv
```

Optional Autoencoder experiment files may also be created if their flags are enabled:

```text
variant2_finetune_autoencoder_results.csv
variant2_selective_finetune_autoencoder_results.csv
poster_ae_variants.csv
```

The poster plots are saved under:

```text
results/plots/
```

Important plot files:

```text
poster_pr_auc_comparison.png
poster_f1_comparison.png
poster_accuracy_comparison.png
poster_runtime_comparison.png
```

Depending on the final poster layout, the accuracy plot may be omitted and Accuracy may instead be shown inside the result tables.

---

## Exporting the notebook to HTML

If an HTML version of the notebook is required, run:

```bash
jupyter nbconvert --to html notebooks/project2_main.ipynb
```

This creates:

```text
notebooks/project2_main.html
```

The HTML file is a static version of the notebook that includes the code, markdown text, output tables, and plots.

In Google Colab, the same command can be executed in a code cell:

```python
!jupyter nbconvert --to html notebooks/project2_main.ipynb
```

Before exporting, it is recommended to run the notebook once so that the output cells are saved.

---

## Git and version control notes

The following folders and files should not be committed:

```text
data/raw/
external/
__pycache__/
.ipynb_checkpoints/
*.pyc
*.zip
```

These are excluded through `.gitignore`.

The raw dataset should remain outside GitHub. Generated datasets, result tables, and final plots can be committed if they are needed for reproducibility and their size is reasonable.

Do not store GitHub tokens, passwords, or personal credentials inside notebooks or source files.
