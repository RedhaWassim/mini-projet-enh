"""
Data Loading, Analysis, and Preprocessing for Cybersecurity Dataset.

This module handles:
1. Loading the raw CSV dataset (excel_labeling_complet.csv).
2. Exploratory data analysis (EDA) with class distribution, feature types, statistics.
3. Feature engineering: extract rich features from sparse packet-level data.
4. Feature encoding: one-hot for categorical, standardization for numeric.
5. Synthetic data generation for testing when the real dataset is unavailable.

Dataset characteristics (excel_labeling_complet.csv):
  - 13,514 packet-level records from a network capture session (~485 seconds)
  - 9 columns: packet_id, timestamp, label, src_ip, dst_ip, protocol, src_port, dst_port, length
  - 5 classes: BENIGNE, Copy 54ndc47 (SMB), Service Creation, Discover local hosts, View remote shares
  - Imbalanced: SMB copy attack dominates (68.7%), View remote shares is rare (0.1%)
  - 245 rows with missing IP/protocol (local/broadcast packets)

Feature engineering rationale:
  Since the raw dataset has very few columns (only port, protocol, length, timestamp, IPs),
  we must derive meaningful features to give models enough signal:
  - Temporal features: inter-arrival times, burst detection, time windows
  - Communication features: IP pair encoding, port categories, flow direction
  - Statistical aggregates: per-IP and per-pair traffic statistics (windowed)
  - Structural features: IP frequency, port frequency, protocol distribution
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
    Generate a synthetic cybersecurity packet-level dataset mimicking
    the structure of excel_labeling_complet.csv.
    """
    rng = np.random.RandomState(seed)

    attack_types = ["BENIGNE", "Copy 54ndc47 (SMB)", "Service Creation",
                    "Discover local hosts", "View remote shares"]
    attack_probs = [0.24, 0.69, 0.04, 0.03, 0.001]
    # Normalize
    attack_probs = [p / sum(attack_probs) for p in attack_probs]

    labels = rng.choice(attack_types, size=n_samples, p=attack_probs)

    src_ips_pool = ["192.168.10.100", "192.168.30.100", "192.168.10.103",
                    "192.168.60.101", "192.168.10.102"]
    dst_ips_pool = ["192.168.10.100", "192.168.30.100", "192.168.10.103",
                    "192.168.60.101", "10.0.0.2", "8.8.8.8"]

    base_time = 1771502223.0
    timestamps = base_time + np.sort(rng.uniform(0, 500, n_samples))

    data = pd.DataFrame({
        "packet_id": np.arange(n_samples),
        "timestamp": timestamps,
        "label": labels,
        "src_ip": rng.choice(src_ips_pool, n_samples),
        "dst_ip": rng.choice(dst_ips_pool, n_samples),
        "protocol": rng.choice([6.0, 17.0, 1.0, np.nan], n_samples, p=[0.75, 0.1, 0.01, 0.14]),
        "src_port": rng.randint(0, 65535, n_samples),
        "dst_port": rng.choice([80, 443, 445, 135, 5044, 0, 21], n_samples),
        "length": rng.choice([54, 66, 74, 1514], n_samples, p=[0.1, 0.3, 0.2, 0.4]),
    })

    return data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load the cybersecurity dataset from CSV. Falls back to synthetic data
    if the file is not found.
    """
    if csv_path and os.path.exists(csv_path):
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
    last_col = df.columns[-1]
    if df[last_col].nunique() < 50:
        return last_col
    raise ValueError("Cannot auto-detect label column. Please specify it manually.")


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(df: pd.DataFrame, label_col: str) -> dict:
    """
    Perform exploratory data analysis on the cybersecurity dataset.
    Returns a dict with analysis results (also prints a summary).
    """
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]} samples, {df.shape[1]} columns")

    # Feature type classification
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if label_col in categorical_cols:
        categorical_cols.remove(label_col)

    print(f"\nRaw numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Raw categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # Class distribution
    class_dist = df[label_col].value_counts()
    print(f"\nTarget variable: '{label_col}' with {len(class_dist)} classes")
    print("\nClass distribution:")
    for cls, count in class_dist.items():
        pct = 100.0 * count / len(df)
        print(f"  {cls:30s}: {count:7d} ({pct:5.1f}%)")

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
        print(f"\nNumeric feature statistics:")
        print(df[numeric_cols].describe().round(2).to_string())

    # Unique IPs
    if "src_ip" in df.columns:
        print(f"\nUnique source IPs: {df['src_ip'].nunique()}")
        print(f"Unique destination IPs: {df['dst_ip'].nunique()}")
        print(f"Unique (src_ip, dst_ip) pairs: {df.groupby(['src_ip', 'dst_ip']).ngroups}")

    if "timestamp" in df.columns:
        ts = df["timestamp"]
        print(f"\nCapture duration: {ts.max() - ts.min():.1f} seconds")
        print(f"Average packet rate: {len(df) / (ts.max() - ts.min()):.1f} packets/sec")

    print("\n" + "=" * 70)
    print("CYBERSECURITY-SPECIFIC OBSERVATIONS")
    print("=" * 70)
    print("""
    1. Class Imbalance: 'Copy 54ndc47 (SMB)' dominates at ~68.7% while
       'View remote shares' has only 13 samples (0.1%) -> inverse-frequency
       class weights are essential.
    2. Sparse Raw Features: Only 9 columns (packet_id, timestamp, label,
       src_ip, dst_ip, protocol, src_port, dst_port, length) -> heavy
       feature engineering is required (temporal, communication, statistical).
    3. Attack Semantics: The attacks represent lateral movement stages in a
       cyber kill chain (discovery -> SMB copy -> service creation -> shares).
       This sequential nature motivates temporal and graph-based modeling.
    4. Network Structure: Few unique IPs suggest a small network topology.
       Graph structure from IP communication patterns will be highly informative.
    5. Missing Data: 245 packets lack IP/protocol (likely broadcast/local) ->
       impute with dedicated category.
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
# Feature engineering (critical for this sparse packet-level dataset)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Derive rich features from the sparse packet-level dataset.

    Feature groups:
    1. Packet-level: length, protocol one-hot, port categories
    2. IP encoding: IP octets as numeric, IP pair identity
    3. Temporal: inter-arrival time, relative timestamp, time-based burst features
    4. Communication context: per-source and per-destination aggregated statistics
    5. Flow context: per-(src_ip, dst_ip) pair statistics

    This transforms the 9-column raw data into a ~50+ dimensional feature space,
    giving models enough signal for multi-class attack classification.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Handle missing values ----
    df["src_ip"] = df["src_ip"].fillna("MISSING")
    df["dst_ip"] = df["dst_ip"].fillna("MISSING")
    df["protocol"] = df["protocol"].fillna(-1)

    features = pd.DataFrame(index=df.index)

    # ---- 1. Basic packet features ----
    features["length"] = df["length"].astype(float)
    features["length_log"] = np.log1p(df["length"].astype(float))
    features["src_port"] = df["src_port"].astype(float)
    features["dst_port"] = df["dst_port"].astype(float)

    # Port categories (well-known, registered, dynamic)
    features["src_port_wellknown"] = (df["src_port"] < 1024).astype(float)
    features["src_port_registered"] = ((df["src_port"] >= 1024) & (df["src_port"] < 49152)).astype(float)
    features["src_port_dynamic"] = (df["src_port"] >= 49152).astype(float)
    features["dst_port_wellknown"] = (df["dst_port"] < 1024).astype(float)
    features["dst_port_registered"] = ((df["dst_port"] >= 1024) & (df["dst_port"] < 49152)).astype(float)
    features["dst_port_dynamic"] = (df["dst_port"] >= 49152).astype(float)

    # Key service ports (important for attack type discrimination)
    for port in [80, 443, 445, 135, 21, 53]:
        features[f"dst_port_is_{port}"] = (df["dst_port"] == port).astype(float)
    features["same_port"] = (df["src_port"] == df["dst_port"]).astype(float)

    # ---- 2. Protocol one-hot ----
    features["proto_tcp"] = (df["protocol"] == 6).astype(float)
    features["proto_udp"] = (df["protocol"] == 17).astype(float)
    features["proto_icmp"] = (df["protocol"] == 1).astype(float)
    features["proto_missing"] = (df["protocol"] == -1).astype(float)

    # ---- 3. IP encoding ----
    # Extract IP octets as numeric features
    for col in ["src_ip", "dst_ip"]:
        parts = df[col].str.split(".", expand=True)
        for i in range(4):
            features[f"{col}_oct{i}"] = pd.to_numeric(parts[i], errors="coerce").fillna(0)

    # Whether IPs are in the same subnet
    features["same_subnet_16"] = (
        (features["src_ip_oct0"] == features["dst_ip_oct0"]) &
        (features["src_ip_oct1"] == features["dst_ip_oct1"])
    ).astype(float)
    features["same_subnet_24"] = (
        features["same_subnet_16"].astype(bool) &
        (features["src_ip_oct2"] == features["dst_ip_oct2"])
    ).astype(float)

    # Is IP missing/broadcast
    features["src_ip_missing"] = (df["src_ip"] == "MISSING").astype(float)
    features["dst_ip_missing"] = (df["dst_ip"] == "MISSING").astype(float)

    # Is IP internal (192.168.x.x or 10.x.x.x)
    features["src_ip_internal"] = (
        (features["src_ip_oct0"] == 192) | (features["src_ip_oct0"] == 10)
    ).astype(float)
    features["dst_ip_internal"] = (
        (features["dst_ip_oct0"] == 192) | (features["dst_ip_oct0"] == 10)
    ).astype(float)
    features["both_internal"] = (features["src_ip_internal"] * features["dst_ip_internal"])

    # ---- 4. Temporal features ----
    ts = df["timestamp"].values
    t_min = ts.min()
    features["relative_time"] = ts - t_min
    features["relative_time_norm"] = (ts - t_min) / max(ts.max() - t_min, 1e-9)

    # Inter-arrival time
    iat = np.diff(ts, prepend=ts[0])
    features["inter_arrival_time"] = iat
    features["iat_log"] = np.log1p(np.abs(iat))

    # Burst detection: very small IAT suggests burst traffic
    features["is_burst"] = (iat < 0.001).astype(float)

    # Rolling window features (last N packets)
    for window in [10, 50]:
        features[f"iat_mean_{window}"] = pd.Series(iat).rolling(window, min_periods=1).mean().values
        features[f"iat_std_{window}"] = pd.Series(iat).rolling(window, min_periods=1).std().fillna(0).values
        features[f"length_mean_{window}"] = pd.Series(df["length"].values.astype(float)).rolling(window, min_periods=1).mean().values
        features[f"length_std_{window}"] = pd.Series(df["length"].values.astype(float)).rolling(window, min_periods=1).std().fillna(0).values

    # ---- 5. Communication context (per-IP statistics) ----
    # How many packets this src_ip has sent (frequency encoding)
    src_freq = df["src_ip"].value_counts(normalize=True)
    features["src_ip_freq"] = df["src_ip"].map(src_freq).fillna(0).values

    dst_freq = df["dst_ip"].value_counts(normalize=True)
    features["dst_ip_freq"] = df["dst_ip"].map(dst_freq).fillna(0).values

    # Per-source average length
    src_avg_len = df.groupby("src_ip")["length"].transform("mean")
    features["src_avg_length"] = src_avg_len.values.astype(float)

    # Per-destination average length
    dst_avg_len = df.groupby("dst_ip")["length"].transform("mean")
    features["dst_avg_length"] = dst_avg_len.values.astype(float)

    # Number of unique destinations per source
    src_dst_count = df.groupby("src_ip")["dst_ip"].transform("nunique")
    features["src_unique_dsts"] = src_dst_count.values.astype(float)

    # Number of unique sources per destination
    dst_src_count = df.groupby("dst_ip")["src_ip"].transform("nunique")
    features["dst_unique_srcs"] = dst_src_count.values.astype(float)

    # Number of unique ports used by source
    src_port_count = df.groupby("src_ip")["dst_port"].transform("nunique")
    features["src_unique_ports"] = src_port_count.values.astype(float)

    # ---- 6. Flow-pair context ----
    pair_key = df["src_ip"].astype(str) + "->" + df["dst_ip"].astype(str)
    pair_count = pair_key.value_counts(normalize=True)
    features["pair_freq"] = pair_key.map(pair_count).fillna(0).values

    pair_total = pair_key.value_counts()
    features["pair_packet_count"] = pair_key.map(pair_total).fillna(0).values.astype(float)

    # Per-pair average length
    df["_pair"] = pair_key
    pair_avg_len = df.groupby("_pair")["length"].transform("mean")
    features["pair_avg_length"] = pair_avg_len.values.astype(float)

    # Dst port frequency
    port_freq = df["dst_port"].value_counts(normalize=True)
    features["dst_port_freq"] = df["dst_port"].map(port_freq).fillna(0).values

    # ---- 7. Advanced features (scanning, bidirectional, rate) ----
    # Packet count per src_ip in sliding windows
    src_series = df["src_ip"].values
    for window in [10, 50, 100]:
        counts = []
        for i in range(len(src_series)):
            start = max(0, i - window)
            counts.append(np.sum(src_series[start:i+1] == src_series[i]))
        features[f"src_pkt_count_{window}"] = np.array(counts, dtype=float)

    # Entropy of dst_port per src_ip (scanning detection)
    def port_entropy(group):
        counts = group.value_counts(normalize=True)
        return -(counts * np.log2(counts + 1e-10)).sum()

    src_port_entropy = df.groupby("src_ip")["dst_port"].transform(
        lambda x: port_entropy(x)
    )
    features["src_port_entropy"] = src_port_entropy.values.astype(float)

    # Bidirectional flag: does the reverse pair exist?
    pair_set = set(pair_key.unique())
    reverse_key = df["dst_ip"].astype(str) + "->" + df["src_ip"].astype(str)
    features["has_reverse_flow"] = reverse_key.isin(pair_set).astype(float).values

    # Time since first packet from same src_ip
    src_first_time = df.groupby("src_ip")["timestamp"].transform("min")
    features["time_since_src_first"] = (df["timestamp"] - src_first_time).values.astype(float)

    # Packet rate per src_ip (total packets / time span)
    src_time_span = df.groupby("src_ip")["timestamp"].transform(lambda x: x.max() - x.min() + 1e-6)
    src_total_pkts = df.groupby("src_ip")["packet_id"].transform("count")
    features["src_packet_rate"] = (src_total_pkts / src_time_span).values.astype(float)

    # Bytes rate per src_ip
    src_total_bytes = df.groupby("src_ip")["length"].transform("sum")
    features["src_bytes_rate"] = (src_total_bytes / src_time_span).values.astype(float)

    print(f"[INFO] Feature engineering complete: {features.shape[1]} features derived from {df.shape[1]} raw columns")

    # Store label for later use
    features["_label"] = df[label_col].values

    return features


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_features(df: pd.DataFrame, label_col: str):
    """
    Full preprocessing pipeline: feature engineering + encoding.

    Returns:
        X: np.ndarray of shape (n_samples, n_features) — encoded feature matrix
        y: np.ndarray of shape (n_samples,) — integer-encoded labels
        label_encoder: fitted LabelEncoder for inverse transforms
        feature_names: list of feature names after encoding
    """
    # Feature engineering
    features_df = engineer_features(df, label_col)

    # Separate labels
    y_raw = features_df["_label"].astype(str).values
    features_df = features_df.drop(columns=["_label"])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Handle inf/nan
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    for col in features_df.columns:
        if features_df[col].isna().any():
            features_df[col] = features_df[col].fillna(features_df[col].median() if not features_df[col].isna().all() else 0)

    feature_names = list(features_df.columns)

    # Standardize all features
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values.astype(float))

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


# ---------------------------------------------------------------------------
# Data augmentation (SMOTE for class imbalance)
# ---------------------------------------------------------------------------

def augment_training_data(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                          seed: int = 42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to the training set
    to handle severe class imbalance.

    SMOTE generates synthetic samples for minority classes by interpolating
    between existing minority samples and their k-nearest neighbors in feature
    space. This is preferable to simple duplication because it creates novel
    but realistic training examples.

    For classes with very few samples (< k_neighbors+1), we fall back to
    RandomOverSampler (simple duplication with noise).

    Only the training set is augmented; val/test remain untouched.

    Args:
        X: Full feature matrix (n_samples, n_features)
        y: Full label vector (n_samples,)
        train_idx: Indices of training samples

    Returns:
        X_train_aug: Augmented training features
        y_train_aug: Augmented training labels
    """
    X_train = X[train_idx]
    y_train = y[train_idx]

    class_counts = Counter(y_train)
    print(f"\n[AUGMENTATION] Training set class distribution before SMOTE:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} samples")

    try:
        from imblearn.over_sampling import SMOTENC, SMOTE, RandomOverSampler
        from imblearn.combine import SMOTEENN

        # Check if any class has fewer than 6 samples (SMOTE needs k_neighbors=5)
        min_count = min(class_counts.values())

        if min_count < 6:
            # Use a pipeline: RandomOverSampler first to get min 6 per class,
            # then SMOTE for further balancing
            print(f"  [INFO] Min class count={min_count} < 6, using RandomOverSampler + SMOTE")

            # Step 1: Random oversample tiny classes to at least 6
            target_min = max(6, min_count)
            ros_strategy = {}
            for cls, count in class_counts.items():
                if count < target_min:
                    ros_strategy[cls] = target_min
            if ros_strategy:
                ros = RandomOverSampler(sampling_strategy=ros_strategy, random_state=seed)
                X_train, y_train = ros.fit_resample(X_train, y_train)

            # Step 2: SMOTE to balance all classes
            # Target: minority classes get up to 30% of the majority class count
            majority_count = max(Counter(y_train).values())
            smote_target = {}
            for cls, count in Counter(y_train).items():
                target = min(majority_count, max(count, int(majority_count * 0.3)))
                if target > count:
                    smote_target[cls] = target
            if smote_target:
                k = min(5, min(Counter(y_train).values()) - 1)
                smote = SMOTE(sampling_strategy=smote_target, k_neighbors=max(1, k),
                              random_state=seed)
                X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            # Standard SMOTE
            majority_count = max(class_counts.values())
            smote_target = {}
            for cls, count in class_counts.items():
                target = min(majority_count, max(count, int(majority_count * 0.3)))
                if target > count:
                    smote_target[cls] = target
            if smote_target:
                smote = SMOTE(sampling_strategy=smote_target, k_neighbors=5,
                              random_state=seed)
                X_train, y_train = smote.fit_resample(X_train, y_train)

    except ImportError:
        print("  [WARN] imbalanced-learn not installed. Using manual oversampling.")
        # Manual oversampling fallback: duplicate minority samples with small noise
        majority_count = max(class_counts.values())
        target_count = int(majority_count * 0.3)
        rng = np.random.RandomState(seed)
        extra_X, extra_y = [], []
        for cls, count in class_counts.items():
            if count < target_count:
                n_needed = target_count - count
                cls_mask = y_train == cls
                cls_X = X_train[cls_mask]
                # Duplicate with small Gaussian noise
                indices = rng.choice(len(cls_X), n_needed, replace=True)
                noise = rng.normal(0, 0.01, (n_needed, cls_X.shape[1]))
                extra_X.append(cls_X[indices] + noise)
                extra_y.append(np.full(n_needed, cls))
        if extra_X:
            X_train = np.vstack([X_train] + extra_X)
            y_train = np.concatenate([y_train] + extra_y)

    aug_counts = Counter(y_train)
    print(f"\n[AUGMENTATION] Training set class distribution after SMOTE:")
    for cls, count in sorted(aug_counts.items()):
        print(f"  Class {cls}: {count} samples")
    print(f"  Total: {len(y_train)} (was {len(train_idx)})")

    return X_train, y_train
