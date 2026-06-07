# Streamify Time-Series Anomaly Detection Methods

Project #2 for the course **Mining of Massive Datasets**.

This project studies anomaly detection methods for time series in both **offline** and **streaming** settings. The experiments use the **TSB-UAD** benchmark dataset and create new streaming-like datasets by concatenating time series from different domains. The goal is to examine how anomaly detection methods behave when the definition of "normal" changes over time due to distribution shifts.

---

## Project Summary

The project follows these main steps:

1. Use the public TSB-UAD dataset.
2. Select time series from different domains.
3. Generate new datasets with different normality levels:
   - **Normality 1**: one time series.
   - **Normality 2**: two concatenated time series from different domains.
   - **Normality 3**: three concatenated time series from different domains.
4. Run two offline anomaly detection baselines:
   - Isolation Forest.
   - Dense Autoencoder.
5. Run **SAND** as the streaming baseline.
6. Adapt the offline methods to streaming settings:
   - Variant 1: naive batch streaming.
   - Variant 2: rolling-window / adaptive streaming.
   - Additional Autoencoder experiments: online fine-tuning and selective online fine-tuning.
7. Compare all methods using anomaly detection metrics and runtime.


---

## Dataset Setup

The raw TSB-UAD dataset is **not included in the GitHub repository**, because it is too large for normal Git tracking.

To reproduce the full workflow, download and unzip the TSB-UAD public dataset, then place it under:

```text
data/raw/
```

The expected structure is that domain folders are directly under `data/raw/`, for example:

```text
data/raw/
??? KDD21/
??? NASA-MSL/
??? SMD/
??? OPPORTUNITY/
??? Daphnet/
??? ...
```

The time-series files are `.out` files. Each file is read as a univariate time series with point-level anomaly labels.

### Option A — Local / Colab runtime dataset

Place the unzipped dataset directly under:

```text
data/raw/
```

Then keep this notebook flag as:

```python
USE_GOOGLE_DRIVE_DATA = False
```

This is the default option.

### Option B — Google Drive dataset

If the unzipped dataset is stored in Google Drive, set:

```python
USE_GOOGLE_DRIVE_DATA = True
```

and update:

```python
GOOGLE_DRIVE_DATASET_DIR = "your_drive_path_to_tsb_uad_dataset"
```

The notebook will mount Google Drive and create a symbolic link:

```text
data/raw/ -> GOOGLE_DRIVE_DATASET_DIR
```

This avoids copying the full raw dataset into the Colab runtime.

---

## Installation

In Google Colab or a local environment, install the dependencies with:

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

The `stumpy` and `tslearn` packages are needed for the official SAND implementation.

---

## Running in Google Colab

Open the notebook:

```text
notebooks/project2_main.ipynb
```

If the repository is private, the first setup cell uses a GitHub token through `getpass`, so the token is not stored inside the notebook.

The notebook is organized so that expensive computations are controlled by execution flags.

Default recommended flags:

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

With these flags set to `False`, the notebook loads existing generated datasets and result CSV files instead of recomputing everything.

To recompute a specific part, set only the relevant flag to `True`.

---

## Reproducing from Scratch

To fully reproduce all results from the raw dataset:

1. Place the full unzipped TSB-UAD dataset under `data/raw/`.
2. Set the required flags to `True` in the notebook:

```python
RUN_BUILD_INVENTORY = True
RUN_GENERATE_NORMALITIES = True
RUN_OFFLINE_BASELINES = True
RUN_SAND_BASELINE = True
RUN_VARIANT1_BATCH = True
RUN_VARIANT2_ROLLING = True
RUN_VARIANT2_FINETUNE_AE = True
RUN_VARIANT2_SELECTIVE_FINETUNE_AE = True
```

3. Run the notebook from top to bottom.

For normal use, it is better to keep the flags `False` and load the already-created CSV files.

---

## Generated Datasets

The selected time series are saved in:

```text
results/tables/selected_series.csv
```

The generated Normality datasets are saved in:

```text
data/generated/
??? normality_1.csv
??? normality_2.csv
??? normality_3.csv
??? generated_datasets_summary.csv
```

The generated datasets follow this structure:

| Dataset | Description | Distribution shifts |
|---|---|---:|
| `normality_1.csv` | one selected time series | 0 |
| `normality_2.csv` | first + second selected time series | 1 |
| `normality_3.csv` | first + second + third selected time series | 2 |

Each generated dataset includes:

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

1. **Isolation Forest**
2. **Dense Autoencoder**

Both methods are applied to sliding windows generated from the complete time series. This is still an offline setting, because the full dataset is available before scoring.

### SAND streaming baseline

SAND is used through the official implementation from the TSB-UAD repository:

```text
https://github.com/TheDatumOrg/TSB-UAD
```

During notebook execution, the repository is cloned under:

```text
external/TSB-UAD/
```

The `external/` folder is ignored by Git and should not be committed.

The local wrapper is:

```text
src/sand_baseline.py
```

It loads the official SAND class, runs SAND in online mode, aligns the anomaly scores with point-level labels, and evaluates the results.

### Streaming Variant 1 — Naive batch streaming

In Variant 1, the offline methods are applied independently to each incoming batch:

```text
current batch -> fit/score method -> next batch
```

There is no memory from previous batches.

Implemented methods:

- Batch Isolation Forest.
- Batch Dense Autoencoder.

### Streaming Variant 2 — Rolling-window streaming

In Variant 2, each batch is processed together with a short recent history:

```text
recent previous batches + current batch -> fit/score method
```

Only the scores for the current batch are kept.

Implemented methods:

- Rolling Isolation Forest.
- Rolling Dense Autoencoder.

### Additional Autoencoder variants

Two additional Dense Autoencoder streaming adaptations were also tested:

1. **Online Fine-tuning Dense Autoencoder**
   - The model is initially trained on the first part of the stream.
   - For each new batch, it scores the batch and then fine-tunes on it.

2. **Selective Fine-tuning Dense Autoencoder**
   - Similar to online fine-tuning.
   - However, it fine-tunes only on windows with low reconstruction error, which are treated as more likely normal.

These variants are included as additional experiments and saved in separate result files.

---

## Evaluation

The methods are evaluated using:

| Metric | Meaning |
|---|---|
| ROC-AUC | General ranking quality between normal and anomaly points |
| PR-AUC | More informative under class imbalance |
| F1-score | Binary anomaly detection quality after thresholding |
| Runtime | Execution time in seconds |

For F1 evaluation, the project uses a **top-k thresholding strategy**, where `k` is the number of true anomalies in the dataset. The `k` highest anomaly scores are classified as anomalies. Because the number of predicted anomalies equals the number of true anomalies, precision, recall, and F1 can become numerically equal.

Accuracy is not used as the main metric because anomaly detection datasets are highly imbalanced. A model that predicts every point as normal could have high accuracy while detecting no anomalies.

---

## Result Files

The main result files are saved under:

```text
results/tables/
```

Important files:

```text
dataset_inventory.csv
domain_summary.csv
dataset_candidates.csv
candidate_domain_summary.csv
selected_series.csv

offline_baseline_results.csv
sand_streaming_results.csv
variant1_batch_results.csv
variant2_rolling_results.csv
variant2_finetune_autoencoder_results.csv
variant2_selective_finetune_autoencoder_results.csv

all_baseline_results.csv
all_streaming_results.csv
all_results.csv
```

The exact set of generated files may depend on which experiments have been run.

---

## Git / Version Control Notes

The following folders/files should not be committed:

```text
data/raw/
external/
__pycache__/
.ipynb_checkpoints/
*.pyc
*.zip
```

These are excluded through `.gitignore`.

Raw datasets should stay outside GitHub. Generated datasets and result tables can be committed if their size is reasonable and they are needed for reproducibility.

---

## Typical Workflow

### Normal notebook use

Use the existing generated files and result tables:

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

Then run all cells.

### Recompute one experiment

For example, to rerun only SAND:

```python
RUN_SAND_BASELINE = True
```

Run the SAND section, then set it back to `False`.

### Save new results

After generating new result CSV files in Colab:

```python
!git status
!git add results/tables/<new_result_file>.csv
!git commit -m "Add new experiment results"
!git push
```

If the notebook was saved directly to GitHub through Colab, run before committing:

```python
!git pull --rebase origin main
```

---

## Main Notebook

The main notebook is:

```text
notebooks/project2_main.ipynb
```

It contains:

1. Setup.
2. Dataset loading configuration.
3. Dataset inventory and time-series selection.
4. Normality dataset generation.
5. Offline baselines.
6. SAND streaming baseline.
7. Streaming Variant 1.
8. Streaming Variant 2.
9. Additional Autoencoder experiments.
10. Result comparison.

---