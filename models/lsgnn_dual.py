"""
LSGNN with Dual-Task Loss: Node Classification + Edge Consistency Regularization.

══════════════════════════════════════════════════════════════════════════
NOVEL MODIFICATION: DUAL-TASK LOSS
══════════════════════════════════════════════════════════════════════════

Motivation:
  Standard GNN training only optimizes node-level classification loss. However,
  in cybersecurity graphs, the edge structure carries important semantic
  information: edges between nodes of the same attack type indicate campaign
  membership, while edges between normal and attack nodes indicate host
  compromise boundaries. Training the model to also predict edge-level
  relationships can:

  1. Encourage representations that respect structural patterns, not just
     node features.
  2. Act as a regularizer preventing overfitting to node features alone.
  3. Help the model learn to distinguish "same-type" from "cross-type"
     connections — directly useful for detecting attack campaigns.

Formulation:

  Given node embeddings h_i from the LSGNN encoder:

  Node classification loss (standard):
    L_node = CrossEntropy(MLP_cls(h_i), y_i)

  Edge consistency loss (novel):
    For each edge (i, j) ∈ E, we predict whether the endpoints share the
    same label:
      p_{ij} = σ(MLP_edge(h_i ⊙ h_j))   (element-wise product as edge repr.)
      t_{ij} = 𝟙[y_i = y_j]              (binary target: same class?)
      L_edge = BCE(p_{ij}, t_{ij})

  Combined loss:
    L = L_node + λ · L_edge

  where λ ∈ [0, 1] is a tunable coefficient (default: 0.3).

Why this helps for cybersecurity:
  - Attack campaigns create clusters of same-label flows. The edge loss
    explicitly trains the model to embed these clusters tightly.
  - The boundary between normal and attack traffic is sharpened, improving
    detection of rare attacks that might otherwise be drowned out.
  - It provides a soft structural prior without changing the architecture,
    making it a clean and principled extension.

Ablation:
  Setting λ=0 recovers the baseline LSGNN exactly. We can sweep λ to
  measure the contribution of the edge consistency term.

══════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lsgnn import LSGNN


class EdgeConsistencyHead(nn.Module):
    """
    Auxiliary head that predicts whether two connected nodes share the same label.

    Uses element-wise product of node embeddings as edge representation,
    which captures similarity in the learned feature space.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h_src, h_dst):
        """
        Predict edge consistency score.

        Args:
            h_src: Embeddings of source nodes (E, hidden_dim)
            h_dst: Embeddings of destination nodes (E, hidden_dim)

        Returns:
            logits: (E, 1) — probability that endpoints share the same label
        """
        # Element-wise product captures pairwise similarity
        edge_repr = h_src * h_dst
        return self.edge_mlp(edge_repr)


class LSGNNDualTask(nn.Module):
    """
    LSGNN with Dual-Task Loss for joint node classification and edge
    consistency prediction.

    This model extends the baseline LSGNN with:
      1. An edge consistency prediction head
      2. A combined loss function L = L_node + λ * L_edge
      3. Edge sampling for efficient training on large graphs

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimensionality
        num_classes: Number of classes for node classification
        num_layers: Number of LSGNN layers
        dropout: Dropout probability
        lambda_edge: Weight of edge consistency loss (λ)
        edge_sample_ratio: Fraction of edges to sample per batch (efficiency)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.3,
                 lambda_edge: float = 0.3, edge_sample_ratio: float = 0.5):
        super().__init__()

        # Shared LSGNN backbone
        self.backbone = LSGNN(input_dim, hidden_dim, num_classes,
                              num_layers, dropout)

        # Edge consistency head
        self.edge_head = EdgeConsistencyHead(hidden_dim, dropout)

        self.lambda_edge = lambda_edge
        self.edge_sample_ratio = edge_sample_ratio

    def forward(self, x, edge_index):
        """Standard forward pass — returns node classification logits."""
        return self.backbone(x, edge_index)

    def compute_dual_loss(self, x, edge_index, y, node_mask=None,
                          class_weights=None, label_smoothing=0.1):
        """
        Compute the combined dual-task loss.

        Args:
            x: Node features (N, d)
            edge_index: (2, E)
            y: Node labels (N,)
            node_mask: Boolean mask for supervised nodes (train mask)
            class_weights: Optional class weight tensor for CE loss
            label_smoothing: Label smoothing for cross-entropy

        Returns:
            total_loss, loss_node, loss_edge: scalar tensors
        """
        # Get embeddings from backbone (before classifier)
        h = self.backbone.get_embeddings(x, edge_index)

        # Node classification loss
        logits = self.backbone.classifier(h)

        if node_mask is not None:
            if class_weights is not None:
                ce = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=label_smoothing)
            else:
                ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            loss_node = ce(logits[node_mask], y[node_mask])
        else:
            loss_node = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        # Edge consistency loss
        # Sample edges for efficiency
        n_edges = edge_index.size(1)
        if self.edge_sample_ratio < 1.0 and n_edges > 1000:
            n_sample = max(int(n_edges * self.edge_sample_ratio), 100)
            perm = torch.randperm(n_edges, device=edge_index.device)[:n_sample]
            sampled_edge_index = edge_index[:, perm]
        else:
            sampled_edge_index = edge_index

        src, dst = sampled_edge_index[0], sampled_edge_index[1]
        h_src, h_dst = h[src], h[dst]

        edge_logits = self.edge_head(h_src, h_dst).squeeze(-1)

        # Binary target: 1 if same class, 0 otherwise
        edge_targets = (y[src] == y[dst]).float()

        loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_targets)

        # Combined loss
        total_loss = loss_node + self.lambda_edge * loss_edge

        return total_loss, loss_node, loss_edge

    def get_node_predictions(self, x, edge_index):
        """Get node classification predictions."""
        logits = self.forward(x, edge_index)
        return logits.argmax(dim=1)
