"""
Local Similarity Graph Neural Network (LSGNN) — Baseline SOTA GNN.

══════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULATION
══════════════════════════════════════════════════════════════════════════

LSGNN is designed for robust node classification on both homophilic and
heterophilic graphs. The key idea is to augment standard message passing
with a local similarity encoding that captures how similar each node is
to its neighbors, allowing the model to adaptively weight messages.

Given:
  - Graph G = (V, E) with adjacency A
  - Node features X ∈ ℝ^{N×d}
  - Labels Y ∈ {1, ..., C}^N

Layer-wise computation (for layer ℓ):

1. Local Similarity Encoding:
   For each edge (i, j) ∈ E, compute a similarity score:
     s_{ij}^{(ℓ)} = σ(MLP_sim(h_i^{(ℓ)} ∥ h_j^{(ℓ)} ∥ |h_i^{(ℓ)} - h_j^{(ℓ)}|))

   This captures whether neighbors have similar or dissimilar representations,
   which is critical in heterophilic graphs where connected nodes often have
   different labels.

2. Similarity-Weighted Message Passing:
     m_i^{(ℓ)} = Σ_{j ∈ N(i)} s_{ij}^{(ℓ)} · W_msg^{(ℓ)} h_j^{(ℓ)}

   Messages are weighted by learned similarity, so the model can attend to
   or suppress neighbors based on local feature patterns.

3. Node Update with Residual:
     h_i^{(ℓ+1)} = σ(W_upd^{(ℓ)} [h_i^{(ℓ)} ∥ m_i^{(ℓ)}]) + h_i^{(ℓ)}

   Residual connections preserve input signal across layers.

4. Classification:
     ŷ_i = softmax(W_out h_i^{(L)})

Loss: L = CrossEntropy(ŷ, y) with optional class weights for imbalance.

══════════════════════════════════════════════════════════════════════════
WHY LSGNN FOR CYBERSECURITY
══════════════════════════════════════════════════════════════════════════

1. Cybersecurity graphs are often heterophilic: a normal flow may be
   adjacent to attack flows on the same host. Standard GCN/GAT fail here
   because they assume similar neighbors -> similar labels.

2. The local similarity encoding allows LSGNN to distinguish "this neighbor
   is similar to me (same traffic type)" from "this neighbor is different
   (potential anomaly)" — directly useful for intrusion detection.

3. Residual connections prevent oversmoothing, important when attack signals
   are localized and should not be diluted by many normal neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class LocalSimilarityConv(MessagePassing if HAS_PYG else nn.Module):
    """
    Single LSGNN convolution layer with local similarity-weighted message passing.

    Implements:
      1. Edge-level similarity scoring via MLP on concatenated node features
      2. Similarity-weighted aggregation
      3. Node update with residual connection
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        if HAS_PYG:
            super().__init__(aggr="add")
        else:
            super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Similarity scoring MLP: takes [h_i ∥ h_j ∥ |h_i - h_j|] -> scalar
        self.sim_mlp = nn.Sequential(
            nn.Linear(3 * in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

        # Message transform
        self.W_msg = nn.Linear(in_channels, out_channels, bias=False)

        # Node update: takes [h_i ∥ m_i] -> h_i'
        self.W_upd = nn.Linear(in_channels + out_channels, out_channels)

        # Residual projection if dimensions change
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        if HAS_PYG:
            # Add self-loops for self-message
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            out = self.propagate(edge_index, x=x)
        else:
            out = self._manual_propagate(x, edge_index)

        # Node update with residual
        combined = torch.cat([x, out], dim=-1)
        updated = self.W_upd(combined)
        updated = updated + self.residual(x)  # residual connection
        updated = self.norm(updated)
        updated = F.relu(updated)
        updated = self.dropout(updated)
        return updated

    def message(self, x_i, x_j):
        """PyG message function: compute similarity-weighted messages."""
        # Local similarity encoding
        diff = torch.abs(x_i - x_j)
        sim_input = torch.cat([x_i, x_j, diff], dim=-1)
        sim_score = self.sim_mlp(sim_input)  # (E, 1)

        # Similarity-weighted message
        msg = self.W_msg(x_j)
        return sim_score * msg

    def _manual_propagate(self, x, edge_index):
        """Fallback for when PyG is not available."""
        # Add self-loops
        n = x.size(0)
        self_loops = torch.arange(n, device=edge_index.device).unsqueeze(0).expand(2, -1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        src, dst = edge_index[0], edge_index[1]
        x_i = x[dst]
        x_j = x[src]

        diff = torch.abs(x_i - x_j)
        sim_input = torch.cat([x_i, x_j, diff], dim=-1)
        sim_score = self.sim_mlp(sim_input)
        msg = self.W_msg(x_j)
        weighted_msg = sim_score * msg

        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)
        return out


class LSGNN(nn.Module):
    """
    Local Similarity Graph Neural Network for node classification.

    Architecture:
      Input projection -> [LocalSimilarityConv] x L -> Classifier head

    Args:
        input_dim: Number of input features per node
        hidden_dim: Hidden layer dimensionality
        num_classes: Number of output classes
        num_layers: Number of LSGNN convolution layers
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(LocalSimilarityConv(hidden_dim, hidden_dim, dropout))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x: Node features (N, input_dim)
            edge_index: Edge indices (2, E)

        Returns:
            logits: (N, num_classes)
        """
        # Project input features to hidden space
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Message passing layers
        for conv in self.convs:
            h = conv(h, edge_index)

        # Classify
        logits = self.classifier(h)
        return logits

    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings (before the classifier head).
        Useful for visualization and the dual-task extension.
        """
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(h, edge_index)

        return h
