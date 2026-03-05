"""
Graph Construction Pipeline for Cybersecurity Data.

Design choices and rationale:
─────────────────────────────
Node definition:
  Each network flow (row in the dataset) is a node. This is the finest granularity
  and preserves all per-flow feature information for classification.

Edge definition (k-NN similarity graph):
  We connect flows that are "similar" in feature space using k-nearest-neighbors.
  This encodes the inductive bias that flows with similar network characteristics
  (packet sizes, durations, byte rates) likely share the same traffic type.

  Additional communication-based edges:
  We also connect flows sharing the same (src_ip, dst_ip) pair or the same dst_port,
  capturing the structural pattern that attacks often target the same hosts/services.

Why GNN is appropriate here:
  1. Network traffic has natural relational structure (host-to-host communication).
  2. Attack campaigns involve correlated flows — a GNN can propagate suspicion signals.
  3. Heterophilic patterns exist: a normal flow may connect to an attack flow on the
     same host, requiring models robust to label disagreement across edges (→ LSGNN).
  4. The graph structure provides regularization beyond what flat classifiers capture.

Prediction target: Node-level multi-class classification (classify each flow).
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def build_knn_edges(X: np.ndarray, k: int = 10, metric: str = "cosine") -> np.ndarray:
    """
    Build a k-NN graph from the feature matrix.

    For each node, we find its k nearest neighbors and create directed edges.
    We then symmetrize to get an undirected graph.

    Args:
        X: Feature matrix (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric for neighbor search

    Returns:
        edge_index: np.ndarray of shape (2, n_edges) — COO format
    """
    n = X.shape[0]
    k_actual = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Build edge list (exclude self-loops from kNN: column 0 is the node itself)
    src = np.repeat(np.arange(n), k_actual)
    dst = indices[:, 1:].flatten()

    # Symmetrize
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ], axis=0)

    # Remove duplicates
    edge_set = set()
    unique_edges = [[], []]
    for i in range(edge_index.shape[1]):
        e = (edge_index[0, i], edge_index[1, i])
        if e not in edge_set:
            edge_set.add(e)
            unique_edges[0].append(e[0])
            unique_edges[1].append(e[1])

    edge_index = np.array(unique_edges, dtype=np.int64)
    return edge_index


def build_communication_edges(df, ip_src_col: str = None, ip_dst_col: str = None,
                               port_col: str = None) -> Optional[np.ndarray]:
    """
    Build edges based on communication patterns:
    - Flows sharing the same (src_ip, dst_ip) pair are connected.
    - Flows targeting the same dst_port are connected (with sampling to limit density).

    This captures the structural bias that attack campaigns target specific hosts/services.
    """
    edges_src, edges_dst = [], []

    # IP-pair based edges
    if ip_src_col and ip_dst_col and ip_src_col in df.columns and ip_dst_col in df.columns:
        pair_key = df[ip_src_col].astype(str) + "->" + df[ip_dst_col].astype(str)
        groups = pair_key.groupby(pair_key).groups

        for _, indices in groups.items():
            idx_list = list(indices)
            if len(idx_list) > 1 and len(idx_list) <= 100:
                # Connect all pairs within small groups (chain for large groups)
                for i in range(len(idx_list)):
                    for j in range(i + 1, min(i + 5, len(idx_list))):
                        edges_src.append(idx_list[i])
                        edges_dst.append(idx_list[j])
            elif len(idx_list) > 100:
                # For large groups, connect sequential flows (chain topology)
                for i in range(len(idx_list) - 1):
                    edges_src.append(idx_list[i])
                    edges_dst.append(idx_list[i + 1])

    # Port-based edges (sample to avoid huge cliques)
    if port_col and port_col in df.columns:
        port_groups = df.groupby(port_col).groups
        for _, indices in port_groups.items():
            idx_list = list(indices)
            if 2 <= len(idx_list) <= 50:
                for i in range(len(idx_list)):
                    for j in range(i + 1, min(i + 3, len(idx_list))):
                        edges_src.append(idx_list[i])
                        edges_dst.append(idx_list[j])

    if not edges_src:
        return None

    # Symmetrize
    all_src = edges_src + edges_dst
    all_dst = edges_dst + edges_src
    return np.array([all_src, all_dst], dtype=np.int64)


def build_graph_data(X: np.ndarray, y: np.ndarray,
                     train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                     df=None, k: int = 10,
                     ip_src_col: str = None, ip_dst_col: str = None,
                     port_col: str = None):
    """
    Build a PyTorch Geometric Data object from features, labels, and edges.

    Combines k-NN similarity edges with optional communication-based edges.

    Args:
        X: Feature matrix (n, d)
        y: Label vector (n,)
        train_idx, val_idx, test_idx: split indices
        df: Original DataFrame for communication edges
        k: k for k-NN graph
        ip_src_col, ip_dst_col, port_col: column names for communication edges

    Returns:
        torch_geometric.data.Data object
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        # Lightweight fallback when PyG is not installed
        class Data:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            def to(self, device):
                for k, v in self.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        setattr(self, k, v.to(device))
                return self
            def __repr__(self):
                info = ", ".join(f"{k}={list(v.shape) if isinstance(v, torch.Tensor) else v}"
                                for k, v in self.__dict__.items()
                                if not k.startswith("_"))
                return f"Data({info})"

    n = X.shape[0]
    print(f"\n[Graph Construction] Building graph for {n} nodes...")

    # 1. k-NN edges (primary structure)
    print(f"  Building k-NN graph (k={k})...")
    knn_edges = build_knn_edges(X, k=k)
    print(f"  k-NN edges: {knn_edges.shape[1]}")

    # 2. Communication-based edges (auxiliary structure)
    comm_edges = None
    if df is not None:
        print("  Building communication-based edges...")
        comm_edges = build_communication_edges(df, ip_src_col, ip_dst_col, port_col)
        if comm_edges is not None:
            print(f"  Communication edges: {comm_edges.shape[1]}")

    # 3. Merge edge sets
    if comm_edges is not None:
        edge_index = np.concatenate([knn_edges, comm_edges], axis=1)
    else:
        edge_index = knn_edges

    # Remove duplicates from merged edges
    edge_tuples = set(zip(edge_index[0], edge_index[1]))
    edge_index = np.array(list(edge_tuples), dtype=np.int64).T
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)

    print(f"  Total unique edges: {edge_index.shape[1]}")
    print(f"  Average degree: {edge_index.shape[1] / n:.1f}")

    # 4. Compute edge homophily (fraction of edges connecting same-class nodes)
    same_label = y[edge_index[0]] == y[edge_index[1]]
    homophily = same_label.mean()
    print(f"  Edge homophily: {homophily:.3f} ({'homophilic' if homophily > 0.5 else 'heterophilic'})")

    # 5. Build masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # 6. Assemble Data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Store metadata
    data.num_classes = len(np.unique(y))
    data.homophily = homophily

    print(f"  Data object: {data}")
    return data
