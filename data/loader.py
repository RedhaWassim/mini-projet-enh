"""
Data Loading, Analysis, and Preprocessing for Cybersecurity Dataset.

This module handles:
1. Loading the raw CSV dataset (excel_labeling_complet.csv or similar).
2. Exploratory data analysis (EDA) with class distribution, feature types, statistics.
3. Feature encoding: one-hot for categorical, standardization for numeric.
4. Synthetic data generation for testing when the real dataset is unavailable.

Design rationale:
- Cybersecurity datasets typically contain network flow features (IPs, ports, protocol,
  duration, byte counts, packet counts, flags) with a multi-class label column.
- We auto-detect feature types and the label column to remain dataset-agnostic.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter


# ---------------------------------------------------------------------------
# Synthetic data generator (for development/testing when real CSV is absent)
# ---------------------------------------------------------------------------

def generate_synthetic_cybersecurity_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic cybersecurity network-flow dataset mimicking CICIDS/UNSW-NB15.

    Features include:
      - src_ip, dst_ip: source/destination IP addresses
      - src_port, dst_port: source/destination ports
      - protocol: TCP/UDP/ICMP
      - duration, total_fwd_packets, total_bwd_packets, flow_bytes_per_s, etc.
      - label: multi-class attack label

    Class distribution is intentionally imbalanced (realistic for IDS datasets).
    """
    rng = np.random.RandomState(seed)

    # Attack types with imbalanced distribution
    attack_types = ["Normal", "DDoS", "PortScan", "Bot", "Infiltration",
                    "Web Attack", "Brute Force", "Heartbleed"]
    attack_probs = [0.50, 0.15, 0.12, 0.08, 0.05, 0.05, 0.04, 0.01]

    labels = rng.choice(attack_types, size=n_samples, p=attack_probs)

    # Generate IP addresses (as categorical features)
    src_ips = [f"192.168.{rng.randint(0, 10)}.{rng.randint(1, 255)}" for _ in range(n_samples)]
    dst_ips = [f"10.0.{rng.randint(0, 5)}.{rng.randint(1, 255)}" for _ in range(n_samples)]

    # Attack-correlated feature generation
    label_idx = {name: i for i, name in enumerate(attack_types)}
    l_ids = np.array([label_idx[l] for l in labels])

    data = pd.DataFrame({
        "src_ip": src_ips,
        "dst_ip": dst_ips,
        "src_port": rng.randint(1024, 65535, n_samples),
        "dst_port": rng.choice([80, 443, 22, 53, 8080, 3389, 445, 21], n_samples),
        "protocol": rng.choice(["TCP", "UDP", "ICMP"], n_samples, p=[0.7, 0.2, 0.1]),
        "duration": np.abs(rng.exponential(5.0, n_samples) + l_ids * 0.5),
        "total_fwd_packets": np.abs(rng.poisson(10 + l_ids * 3, n_samples)).astype(float),
        "total_bwd_packets": np.abs(rng.poisson(8 + l_ids * 2, n_samples)).astype(float),
        "flow_bytes_per_s": np.abs(rng.exponential(1000, n_samples) + l_ids * 500),
        "flow_packets_per_s": np.abs(rng.exponential(50, n_samples) + l_ids * 20),
        "fwd_packet_length_mean": np.abs(rng.normal(500, 200, n_samples) + l_ids * 50),
        "bwd_packet_length_mean": np.abs(rng.normal(400, 150, n_samples) + l_ids * 40),
        "flow_iat_mean": np.abs(rng.exponential(100, n_samples) + l_ids * 30),
        "fwd_iat_mean": np.abs(rng.exponential(80, n_samples) + l_ids * 25),
        "bwd_iat_mean": np.abs(rng.exponential(90, n_samples) + l_ids * 20),
        "fin_flag_count": rng.poisson(1 + l_ids * 0.3, n_samples).astype(float),
        "syn_flag_count": rng.poisson(1 + l_ids * 0.5, n_samples).astype(float),
        "rst_flag_count": rng.poisson(0.5 + l_ids * 0.2, n_samples).astype(float),
        "psh_flag_count": rng.poisson(2 + l_ids * 0.4, n_samples).astype(float),
        "ack_flag_count": rng.poisson(5 + l_ids * 1.0, n_samples).astype(float),
        "label": labels,
    })

    return data


# ---------------------------------------------------------------------------
# Data loading and analysis
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load the cybersecurity dataset from CSV. Falls back to synthetic data
    if the file is not found.
    """
    if csv_path and os.path.exists(csv_path):
        # Try common encodings and separators
        for sep in [",", ";", "\t"]:
            for encoding in ["utf-8", "latin-1", "utf-16"]:
                try:
                    df = pd.read_csv(csv_path, sep=sep, encoding=encoding, low_memory=False)
                    if len(df.columns) > 1:
                        print(f"[INFO] Loaded dataset from {csv_path}")
                        print(f"       Shape: {df.shape}, Separator: '{sep}', Encoding: {encoding}")
                        return df
                except Exception:
                    continue
        print(f"[WARN] Could not parse {csv_path}. Falling back to synthetic data.")

    print("[INFO] Using synthetic cybersecurity dataset for development.")
    return generate_synthetic_cybersecurity_data()


def detect_label_column(df: pd.DataFrame) -> str:
    """Auto-detect the label/target column by name heuristics."""
    candidates = ["label", "Label", "LABEL", "attack_cat", "Attack", "attack",
                   "class", "Class", "category", "Category", "type", "Type",
                   "classification", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fall back to last column if it has few unique values
    last_col = df.columns[-1]
    if df[last_col].nunique() < 50:
        return last_col
    raise ValueError("Cannot auto-detect label column. Please specify it manually.")


def analyze_dataset(df: pd.DataFrame, label_col: str) -> dict:
    """
    Perform exploratory data analysis on the cybersecurity dataset.

    Returns a dict with analysis results (also prints a summary).
    """
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    # Basic info
    print(f"\nShape: {df.shape[0]} samples, {df.shape[1]} features")

    # Feature type classification
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if label_col in categorical_cols:
        categorical_cols.remove(label_col)

    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols) > 10 else ''}")

    # Class distribution
    class_dist = df[label_col].value_counts()
    print(f"\nTarget variable: '{label_col}' with {len(class_dist)} classes")
    print("\nClass distribution:")
    for cls, count in class_dist.items():
        pct = 100.0 * count / len(df)
        print(f"  {cls:25s}: {count:7d} ({pct:5.1f}%)")

    # Imbalance ratio
    imbalance_ratio = class_dist.max() / class_dist.min()
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.1f}x")

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values: {total_missing} total ({100 * total_missing / df.size:.2f}%)")
    if total_missing > 0:
        print("  Columns with missing values:")
        for col in missing[missing > 0].index:
            print(f"    {col}: {missing[col]} ({100 * missing[col] / len(df):.1f}%)")

    # Numeric feature statistics
    if numeric_cols:
        print(f"\nNumeric feature statistics (first 8):")
        print(df[numeric_cols[:8]].describe().round(2).to_string())

    # Inf values
    if numeric_cols:
        inf_counts = np.isinf(df[numeric_cols]).sum()
        total_inf = inf_counts.sum()
        if total_inf > 0:
            print(f"\nInfinite values found: {total_inf}")

    print("\n" + "=" * 70)
    print("CYBERSECURITY-SPECIFIC CHALLENGES")
    print("=" * 70)
    print("""
    1. Class Imbalance: Rare attacks (e.g., Heartbleed, Infiltration) are
       heavily underrepresented -> use weighted loss or oversampling.
    2. Concept Drift: Attack patterns evolve over time; temporal splits
       are more realistic than random splits for deployment.
    3. Feature Heterogeneity: Mix of continuous (byte counts), discrete
       (packet counts), categorical (protocol, flags), and network-specific
       (IP addresses, ports) features.
    4. High Dimensionality: Many correlated flow-level features; feature
       selection or PCA may help baselines but GNNs can learn implicitly.
    5. Adversarial Nature: Attackers may craft traffic to evade detection,
       motivating robust models that leverage structural (graph) patterns.
    """)

    return {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "n_classes": len(class_dist),
        "class_distribution": class_dist.to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "imbalance_ratio": imbalance_ratio,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_features(df: pd.DataFrame, label_col: str):
    """
    Encode features and labels for modeling.

    Returns:
        X: np.ndarray of shape (n_samples, n_features) — encoded feature matrix
        y: np.ndarray of shape (n_samples,) — integer-encoded labels
        label_encoder: fitted LabelEncoder for inverse transforms
        feature_names: list of feature names after encoding
    """
    df = df.copy()

    # Separate features and labels
    y_raw = df[label_col].astype(str).values
    df = df.drop(columns=[label_col])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Handle IP addresses: extract octets as numeric features instead of one-hot
    ip_cols = [c for c in categorical_cols if "ip" in c.lower()]
    for col in ip_cols:
        try:
            parts = df[col].astype(str).str.split(".", expand=True).astype(float)
            for i in range(parts.shape[1]):
                df[f"{col}_oct{i}"] = parts[i]
                numeric_cols.append(f"{col}_oct{i}")
            categorical_cols.remove(col)
            df = df.drop(columns=[col])
        except Exception:
            pass  # keep as categorical if parsing fails

    # Handle missing / infinite values in numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    # One-hot encode remaining categorical features (limit cardinality)
    encoded_parts = []
    feature_names = []

    # Numeric features
    valid_numeric = [c for c in numeric_cols if c in df.columns]
    if valid_numeric:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[valid_numeric].values.astype(float))
        encoded_parts.append(X_num)
        feature_names.extend(valid_numeric)

    # Categorical features with cardinality cap
    for col in categorical_cols:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique()
        if n_unique > 50:
            # High cardinality: use frequency encoding
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0).values
            encoded_parts.append(df[[f"{col}_freq"]].values)
            feature_names.append(f"{col}_freq")
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).values.astype(float)
            encoded_parts.append(dummies)
            feature_names.extend([f"{col}_{v}" for v in pd.get_dummies(df[col], prefix=col, drop_first=False).columns])

    if not encoded_parts:
        raise ValueError("No features were successfully encoded.")

    X = np.hstack(encoded_parts)
    print(f"[INFO] Encoded feature matrix: {X.shape} (samples x features)")
    print(f"[INFO] Number of classes: {len(le.classes_)} -> {list(le.classes_)}")

    return X, y, le, feature_names


def create_splits(n_samples: int, y: np.ndarray,
                  train_ratio: float = 0.6, val_ratio: float = 0.2,
                  seed: int = 42):
    """
    Create stratified train/val/test index splits.

    Returns:
        train_idx, val_idx, test_idx: np.ndarray of indices
    """
    indices = np.arange(n_samples)
    test_ratio = 1.0 - train_ratio - val_ratio

    # Check if all classes have enough samples for stratified splitting.
    # We need at least 3 per class for a 3-way split; otherwise fall back to
    # non-stratified splitting to avoid errors with very rare classes.
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    use_stratify = min_count >= 3

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio),
        stratify=y if use_stratify else None, random_state=seed
    )
    relative_val = val_ratio / (val_ratio + test_ratio)

    # For the second split, re-check stratification feasibility on temp subset
    temp_counts = Counter(y[temp_idx])
    use_stratify_2 = min(temp_counts.values()) >= 2 if temp_counts else False

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1.0 - relative_val),
        stratify=y[temp_idx] if use_stratify_2 else None, random_state=seed
    )

    print(f"[INFO] Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx
