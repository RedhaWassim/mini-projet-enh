"""
TON-IoT Dataset Preprocessing Pipeline.

Handles loading, cleaning, feature engineering, and splitting for the
TON-IoT Network dataset (train_test_network.csv).

The TON-IoT dataset has 44 columns including network flow features,
protocol-specific fields (DNS, SSL, HTTP), and two label columns:
  - 'label': binary (0=normal, 1=attack)
  - 'type': multi-class attack category (10 classes)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Columns that are too sparse or high-cardinality to encode usefully
DROP_COLUMNS = [
    "dns_query",         # free-text domain names
    "ssl_subject",       # certificate subjects (mostly '-')
    "ssl_issuer",        # certificate issuers (mostly '-')
    "http_uri",          # request URIs (unique per request)
    "http_user_agent",   # user agent strings (high cardinality)
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",        # Zeek weird event names
    "weird_addl",        # Zeek weird additional info
    "weird_notice",      # Zeek weird notice flag
]

# Categorical columns to label-encode
CATEGORICAL_COLUMNS = [
    "proto",            # tcp, udp, icmp, etc.
    "service",          # http, dns, ssl, etc.
    "conn_state",       # S0, S1, SF, REJ, etc.
    "dns_AA",           # authoritative answer flag
    "dns_RD",           # recursion desired
    "dns_RA",           # recursion available
    "dns_rejected",     # DNS rejection flag
    "ssl_version",      # TLSv12, SSLv3, etc.
    "ssl_cipher",       # cipher suite
    "ssl_resumed",      # session resumed flag
    "ssl_established",  # connection established flag
    "http_trans_depth", # HTTP transaction depth
    "http_method",      # GET, POST, etc.
    "http_version",     # 1.0, 1.1, etc.
]

# Numeric columns to keep directly
NUMERIC_COLUMNS = [
    "src_port", "dst_port",
    "duration",
    "src_bytes", "dst_bytes",
    "missed_bytes",
    "src_pkts", "src_ip_bytes",
    "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]


def load_toniot(csv_path: str = "ton_iot_dataset/train_test_network.csv"):
    """
    Load the TON-IoT network dataset from CSV.

    Returns:
        df: pandas DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"TON-IoT dataset not found at '{csv_path}'. "
            f"Please download it and place it in the ton_iot_dataset/ folder."
        )

    print(f"[INFO] Loading TON-IoT dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} samples with {len(df.columns)} columns")
    return df


def analyze_toniot(df: pd.DataFrame):
    """Print a summary analysis of the TON-IoT dataset."""
    print(f"\n{'='*60}")
    print(f"  TON-IoT DATASET ANALYSIS")
    print(f"{'='*60}")

    print(f"\n  Shape: {df.shape[0]:,} samples x {df.shape[1]} features")

    # Class distribution
    print(f"\n  Attack Type Distribution:")
    type_counts = df["type"].value_counts()
    for attack_type, count in type_counts.items():
        pct = 100.0 * count / len(df)
        print(f"    {attack_type:20s}: {count:>8,} ({pct:5.1f}%)")

    # Binary label distribution
    print(f"\n  Binary Label Distribution:")
    label_counts = df["label"].value_counts()
    for lbl, count in label_counts.items():
        tag = "attack" if lbl == 1 else "normal"
        pct = 100.0 * count / len(df)
        print(f"    {tag:20s}: {count:>8,} ({pct:5.1f}%)")

    # Dash placeholders
    dash_counts = (df == "-").sum()
    sparse_cols = dash_counts[dash_counts > 0].sort_values(ascending=False)
    print(f"\n  Columns with '-' placeholders ({len(sparse_cols)} columns):")
    for col, cnt in sparse_cols.head(10).items():
        pct = 100.0 * cnt / len(df)
        print(f"    {col:30s}: {cnt:>8,} ({pct:5.1f}%)")
    if len(sparse_cols) > 10:
        print(f"    ... and {len(sparse_cols) - 10} more")

    print(f"{'='*60}\n")


def preprocess_toniot(df: pd.DataFrame, label_col: str = "type"):
    """
    Full preprocessing pipeline for the TON-IoT dataset.

    Steps:
      1. Replace '-' with NaN
      2. Drop high-cardinality/sparse text columns
      3. Label-encode categorical columns (NaN -> 'missing')
      4. Keep numeric columns (fill NaN with 0)
      5. StandardScale all features
      6. Encode labels

    Returns:
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,) integer-encoded labels
        label_encoder: fitted LabelEncoder
        feature_names: list of feature names
    """
    print(f"\n[PREPROCESS] Starting TON-IoT feature engineering...")
    df = df.copy()

    # Replace '-' with NaN globally
    df.replace("-", np.nan, inplace=True)

    # 1. Drop columns we don't need
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    # Also drop the IP columns (used for graph construction, not features)
    # and the label columns
    cols_to_drop += ["src_ip", "dst_ip", "label", label_col]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    print(f"[PREPROCESS] Dropping {len(cols_to_drop)} columns: {cols_to_drop[:5]}...")

    # 2. Encode categoricals
    cat_encoders = {}
    cat_features = []
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns or col in cols_to_drop:
            continue
        df[col] = df[col].fillna("missing").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        cat_encoders[col] = le
        cat_features.append(col)

    print(f"[PREPROCESS] Encoded {len(cat_features)} categorical features: {cat_features[:5]}...")

    # 3. Numeric columns
    num_features = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns or col in cols_to_drop:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        num_features.append(col)

    print(f"[PREPROCESS] Using {len(num_features)} numeric features: {num_features[:5]}...")

    # 4. Build feature matrix
    feature_cols = cat_features + num_features
    X = df[feature_cols].values.astype(np.float32)

    # 5. Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"[PREPROCESS] Feature matrix shape: {X.shape} (after scaling)")

    # 6. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col])
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    print(f"[PREPROCESS] Classes ({num_classes}): {class_names}")
    print(f"[PREPROCESS] Feature engineering complete: {X.shape[1]} features\n")

    return X, y, label_encoder, feature_cols


def create_splits(n_samples: int, y: np.ndarray,
                  train_ratio: float = 0.6, val_ratio: float = 0.2,
                  seed: int = 42):
    """
    Create stratified train/val/test splits.

    Returns:
        train_idx, val_idx, test_idx: np.ndarray
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    indices = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_ratio),
        stratify=y, random_state=seed
    )

    val_relative = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_relative),
        stratify=y[temp_idx], random_state=seed
    )

    print(f"[SPLITS] Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    return train_idx, val_idx, test_idx


def augment_training_data(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                          seed: int = 42):
    """
    Apply SMOTE to the training set for class balance.

    Returns:
        X_train_aug, y_train_aug
    """
    X_train = X[train_idx]
    y_train = y[train_idx]

    print(f"\n[AUGMENTATION] Training set class distribution before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt:,} samples")

    try:
        from imblearn.over_sampling import SMOTE

        # Set k_neighbors to min(5, smallest_class_count - 1)
        min_count = counts.min()
        k = min(5, min_count - 1) if min_count > 1 else 1

        smote = SMOTE(random_state=seed, k_neighbors=k)
        X_aug, y_aug = smote.fit_resample(X_train, y_train)

        print(f"\n[AUGMENTATION] Training set class distribution after SMOTE:")
        unique2, counts2 = np.unique(y_aug, return_counts=True)
        for cls, cnt in zip(unique2, counts2):
            print(f"  Class {cls}: {cnt:,} samples")
        print(f"  Total: {len(y_aug):,} (was {len(y_train):,})")

        return X_aug, y_aug

    except ImportError:
        print("[AUGMENTATION] imblearn not available, skipping SMOTE")
        return X_train, y_train
