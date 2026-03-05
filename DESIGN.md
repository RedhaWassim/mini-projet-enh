# GNN-Based Cybersecurity Intrusion Detection: Design Document

## 1. Problem Description and Assumptions

### Problem Statement
Multi-class classification of network traffic flows into normal traffic and
various attack types (DDoS, PortScan, Bot, Infiltration, Web Attack, Brute Force,
Heartbleed, etc.) using Graph Neural Networks.

### Why GNNs Are Appropriate
1. **Relational structure**: Network traffic has natural graph structure — hosts
   communicate with each other, creating interaction patterns that flat classifiers
   cannot exploit.
2. **Campaign detection**: Cyber attacks are rarely isolated events. A GNN can
   propagate suspicion signals through connected flows, enabling detection of
   coordinated attack campaigns.
3. **Context enrichment**: A flow's neighbors provide contextual information
   (what other traffic is the source/destination involved in?) that augments
   per-flow features.
4. **Heterophilic robustness**: Normal flows adjacent to attack flows on the
   same host create heterophilic edges — motivating architectures like LSGNN
   that handle label disagreement across edges.

### Dataset Assumptions
- CSV-format cybersecurity dataset with network flow features
- Features: IP addresses, ports, protocol, duration, packet/byte statistics, flags
- Target: multi-class label column (attack type or normal)
- Expected challenges: class imbalance, high dimensionality, mixed feature types

---

## 2. Graph Construction Choices and Rationale

### Node Definition
Each network flow (row) is a node. This is the finest granularity, preserving
all per-flow information for classification.

### Edge Definition (Dual Strategy)

**Primary: k-NN Similarity Graph (k=10, cosine distance)**
- Connects flows with similar network characteristics
- Inductive bias: similar traffic patterns → similar traffic types
- Robust even without explicit network topology information

**Secondary: Communication-Based Edges**
- Connect flows sharing the same (src_ip, dst_ip) pair → captures host
  interaction patterns
- Connect flows targeting the same dst_port → captures service-level patterns
- Inductive bias: flows on the same communication channel may be part of the
  same activity (normal or attack)

### Prediction Target
Node-level multi-class classification: classify each flow as normal or a
specific attack type.

### Feature Encoding
- **Numeric features**: StandardScaler normalization
- **IP addresses**: Split into octets as numeric features
- **Categorical features**: One-hot encoding (cardinality ≤ 50) or frequency
  encoding (cardinality > 50)
- **Train/Val/Test**: 60/20/20 stratified split

---

## 3. Baseline SOTA GNN: LSGNN

### Key Ideas
The Local Similarity Graph Neural Network (LSGNN) extends standard message
passing with **local similarity encoding** — for each edge (i,j), an MLP
scores how similar the two endpoint representations are. This score gates
the message, allowing the model to:
- Amplify messages from similar neighbors (homophilic edges)
- Suppress or transform messages from dissimilar neighbors (heterophilic edges)

### Mathematical Formulation

**Layer ℓ computation:**

1. **Local Similarity Score:**
   ```
   s_ij^(ℓ) = σ(MLP_sim([h_i^(ℓ) ∥ h_j^(ℓ) ∥ |h_i^(ℓ) - h_j^(ℓ)|]))
   ```

2. **Similarity-Weighted Aggregation:**
   ```
   m_i^(ℓ) = Σ_{j ∈ N(i)} s_ij^(ℓ) · W_msg^(ℓ) · h_j^(ℓ)
   ```

3. **Node Update with Residual:**
   ```
   h_i^(ℓ+1) = LayerNorm(ReLU(W_upd^(ℓ) [h_i^(ℓ) ∥ m_i^(ℓ)]) + W_res · h_i^(ℓ))
   ```

4. **Classification:**
   ```
   ŷ_i = softmax(MLP_cls(h_i^(L)))
   ```

5. **Loss:**
   ```
   L = CrossEntropy(ŷ, y) with inverse-frequency class weights
   ```

### Why LSGNN for Cybersecurity
- Handles heterophilic patterns (normal ↔ attack edges on same host)
- Residual connections prevent oversmoothing of attack signals
- Similarity scoring naturally identifies "suspicious neighborhood" patterns

---

## 4. Novel Modification: Dual-Task Loss (Edge Consistency Regularization)

### Motivation
Standard GNN training only optimizes node labels. But in cybersecurity:
- Attack campaigns create clusters of same-label flows
- The boundary between normal/attack traffic is a critical signal
- Edge structure carries semantic meaning beyond just connectivity

### Formulation

**Edge Consistency Prediction:**
```
p_ij = σ(MLP_edge(h_i ⊙ h_j))    (⊙ = element-wise product)
t_ij = 𝟙[y_i = y_j]               (binary: same class?)
L_edge = BCE(p_ij, t_ij)
```

**Combined Loss:**
```
L = L_node + λ · L_edge
```

where λ ∈ [0, 1] is tunable (default: 0.3).

### Why This Helps
1. **Cluster tightening**: Explicitly trains embeddings to be similar for
   same-type flows connected by edges → better attack campaign detection.
2. **Boundary sharpening**: Forces the model to distinguish normal↔attack
   boundaries → improved detection of rare attacks.
3. **Structural regularization**: Prevents overfitting to node features by
   incorporating edge-level supervision.
4. **Clean extension**: λ=0 recovers baseline exactly → clean ablation.

---

## 5. Experimental Design

### Models Compared
| Model | Type | Key Feature |
|-------|------|-------------|
| RandomForest | Tabular baseline | No neural network, no graph |
| MLP | Neural baseline | Neural network, no graph |
| LSGNN | SOTA GNN | Local similarity message passing |
| LSGNN-DualTask | Modified GNN | + Edge consistency regularization |

### Hyperparameters
- Hidden dim: 128, Layers: 3, Dropout: 0.3
- Learning rate: 1e-3, Weight decay: 5e-4
- k-NN k: 10, λ: 0.3
- Early stopping patience: 30 epochs
- Class weights: inverse-frequency

### Metrics
- Accuracy, Macro-F1, Weighted-F1
- Per-class F1, Recall, Precision
- Confusion matrix
- ROC-AUC (macro/weighted)

### Ablation Study
- Sweep λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
- λ=0 is equivalent to baseline LSGNN

### Hypotheses
1. GNN models (LSGNN, DualTask) outperform flat baselines (RF, MLP) by
   exploiting graph structure.
2. LSGNN-DualTask improves over LSGNN baseline, especially on rare attack
   classes, due to edge consistency regularization.
3. Optimal λ is in [0.1, 0.5] — too high overwhelms node classification.

---

## 6. Code Structure

```
mini-projet-enh/
├── main.py                    # Experiment runner (entry point)
├── requirements.txt           # Dependencies
├── DESIGN.md                  # This document
├── data/
│   ├── __init__.py
│   ├── loader.py              # Data loading, EDA, preprocessing
│   └── graph_construction.py  # Graph building (k-NN + communication edges)
├── models/
│   ├── __init__.py
│   ├── baselines.py           # MLP, RandomForest
│   ├── lsgnn.py               # LSGNN baseline (SOTA GNN)
│   └── lsgnn_dual.py          # LSGNN-DualTask (novel modification)
└── utils/
    ├── __init__.py
    ├── training.py            # Training loops for GNN models
    └── evaluation.py          # Metrics, comparison tables, analysis
```

### Usage
```bash
# Run with synthetic data (for testing)
python main.py

# Run with real dataset
python main.py --data_path excel_labeling_complet.csv

# Run with ablation study
python main.py --data_path excel_labeling_complet.csv --run_ablation

# Run with multi-seed evaluation
python main.py --data_path excel_labeling_complet.csv --multi_seed

# Full run
python main.py --data_path excel_labeling_complet.csv --run_ablation --multi_seed
```
