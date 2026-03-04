"""
Training Utilities for GNN Models.

Provides standardized training loops for:
  1. LSGNN baseline (standard cross-entropy on nodes)
  2. LSGNN-DualTask (combined node + edge loss)

Both use early stopping on validation macro-F1 and cosine LR scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


def compute_class_weights(y, num_classes):
    """
    Compute inverse-frequency class weights for handling class imbalance.

    This is critical for cybersecurity datasets where rare attacks (e.g.,
    Heartbleed) may have 100x fewer samples than normal traffic.
    """
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(y) / (num_classes * counts)
    # Clip extreme weights to prevent instability
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_gnn(model, data, epochs: int = 300, lr: float = 1e-3,
              weight_decay: float = 5e-4, patience: int = 30,
              device: str = "cpu", use_class_weights: bool = True):
    """
    Train a standard GNN model (e.g., LSGNN baseline) with cross-entropy loss.

    Args:
        model: GNN model with forward(x, edge_index) -> logits
        data: PyG Data object with x, edge_index, y, train_mask, val_mask
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        use_class_weights: Whether to use inverse-frequency weights

    Returns:
        model: Trained model (with best weights loaded)
        history: dict with training curves
    """
    model = model.to(device)
    data = data.to(device)

    # Class weights for imbalanced data
    if use_class_weights:
        weights = compute_class_weights(
            data.y[data.train_mask].cpu().numpy(), data.num_classes
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_pred = val_logits[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            val_acc = (val_pred == val_true).mean()

        history["train_loss"].append(loss.item())
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}, best val F1: {best_val_f1:.4f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: loss={loss.item():.4f}, "
                  f"val_F1={val_f1:.4f}, val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


def train_gnn_dual(model, data, epochs: int = 300, lr: float = 1e-3,
                   weight_decay: float = 5e-4, patience: int = 30,
                   device: str = "cpu", use_class_weights: bool = True):
    """
    Train the LSGNN-DualTask model with combined node + edge loss.

    Same training loop structure as train_gnn, but uses the dual-task
    loss from LSGNNDualTask.compute_dual_loss().
    """
    model = model.to(device)
    data = data.to(device)

    # Class weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(
            data.y[data.train_mask].cpu().numpy(), data.num_classes
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "node_loss": [], "edge_loss": [],
               "val_f1": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        total_loss, loss_node, loss_edge = model.compute_dual_loss(
            data.x, data.edge_index, data.y,
            node_mask=data.train_mask,
            class_weights=class_weights
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_pred = val_logits[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            val_acc = (val_pred == val_true).mean()

        history["train_loss"].append(total_loss.item())
        history["node_loss"].append(loss_node.item())
        history["edge_loss"].append(loss_edge.item())
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}, best val F1: {best_val_f1:.4f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: total_loss={total_loss.item():.4f} "
                  f"(node={loss_node.item():.4f}, edge={loss_edge.item():.4f}), "
                  f"val_F1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history
