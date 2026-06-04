import pandas as pd

def read_out_file(path):
    """
    Reads a TSB-UAD .out file.

    Expected output:
    - first column: time-series value
    - second column: anomaly label
    """

    # First try normal CSV reading
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", header=None)

    # If there is no proper header, assign names
    if df.shape[1] >= 2:
        # Keep only first two columns
        df = df.iloc[:, :2].copy()

        # If columns are not already value/label, rename them
        df.columns = ["value", "label"]
    else:
        raise ValueError(f"Could not read file correctly: {path}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    return df