"""
Training Utilities for GNN Models.

Provides standardized training loops for:
  1. LSGNN baseline (standard cross-entropy on nodes)
  2. LSGNN-DualTask (combined node + edge loss)

Both use early stopping on validation macro-F1, cosine LR scheduling with
linear warmup, and label smoothing for improved generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


def compute_class_weights(y, num_classes):
    """
    Compute inverse-frequency class weights for handling class imbalance.

    This is critical for cybersecurity datasets where rare attacks
    may have 100x fewer samples than normal traffic.
    """
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (num_classes * counts)
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Cosine annealing with linear warmup.

    Warmup helps GNNs avoid early divergence when the graph message
    passing hasn't stabilized yet.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_gnn(model, data, epochs: int = 300, lr: float = 1e-3,
              weight_decay: float = 5e-4, patience: int = 30,
              device: str = None, use_class_weights: bool = True,
              label_smoothing: float = 0.1, warmup_epochs: int = 10):
    """
    Train a standard GNN model (e.g., LSGNN baseline) with cross-entropy loss.

    Improvements over basic training:
      - Linear warmup (first `warmup_epochs`) for stable initial convergence
      - Label smoothing (default 0.1) for better generalization
      - Gradient clipping at max_norm=1.0

    Returns:
        model: Trained model (with best weights loaded)
        history: dict with training curves
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  [TRAIN_GNN] Starting training on device: {device.upper()}")
    print(f"  [TRAIN_GNN] Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"  [TRAIN_GNN] Patience: {patience} | Warmup: {warmup_epochs} epochs")
    print(f"  [TRAIN_GNN] Label Smoothing: {label_smoothing}")
    print(f"{'='*60}")
    model = model.to(device)
    data = data.to(device)
    print(f"  [TRAIN_GNN] Model and data moved to {device.upper()}")

    # Class weights for imbalanced data
    if use_class_weights:
        weights = compute_class_weights(
            data.y[data.train_mask].cpu().numpy(), data.num_classes
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_f1": [], "val_acc": []}

    print(f"  [TRAIN_GNN] Training started...\n")
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        # Compute train accuracy
        with torch.no_grad():
            train_pred = logits[data.train_mask].argmax(dim=1)
            train_acc = (train_pred == data.y[data.train_mask]).float().mean().item()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

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

        # Status marker
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = "★ BEST"
        else:
            wait += 1
            marker = f"  wait={wait}/{patience}"

        print(f"  [GNN] Epoch {epoch+1:>4d}/{epochs} | "
              f"loss={loss.item():.4f} | train_acc={train_acc:.4f} | "
              f"val_F1={val_f1:.4f} | val_acc={val_acc:.4f} | "
              f"lr={current_lr:.6f} | grad={grad_norm:.4f} | {marker}")

        if wait >= patience:
            print(f"\n  [GNN] ⏹ Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            print(f"  [GNN] Best val F1: {best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [TRAIN_GNN] ✓ Loaded best model weights (val_F1={best_val_f1:.4f})")
    model = model.to(device)
    print(f"  [TRAIN_GNN] Training complete. Final best val_F1: {best_val_f1:.4f}")
    print(f"{'='*60}\n")
    return model, history


def train_gnn_dual(model, data, epochs: int = 300, lr: float = 1e-3,
                   weight_decay: float = 5e-4, patience: int = 30,
                   device: str = None, use_class_weights: bool = True,
                   label_smoothing: float = 0.1, warmup_epochs: int = 10):
    """
    Train the LSGNN-DualTask model with combined node + edge loss.

    Same training loop structure as train_gnn, but uses the dual-task
    loss from LSGNNDualTask.compute_dual_loss().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  [TRAIN_DUAL] Starting DualTask training on device: {device.upper()}")
    print(f"  [TRAIN_DUAL] Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"  [TRAIN_DUAL] Patience: {patience} | Warmup: {warmup_epochs} epochs")
    print(f"  [TRAIN_DUAL] Label Smoothing: {label_smoothing}")
    print(f"{'='*60}")
    model = model.to(device)
    data = data.to(device)
    print(f"  [TRAIN_DUAL] Model and data moved to {device.upper()}")

    # Class weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(
            data.y[data.train_mask].cpu().numpy(), data.num_classes
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "node_loss": [], "edge_loss": [],
               "val_f1": [], "val_acc": []}

    print(f"  [TRAIN_DUAL] Training started...\n")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        total_loss, loss_node, loss_edge = model.compute_dual_loss(
            data.x, data.edge_index, data.y,
            node_mask=data.train_mask,
            class_weights=class_weights,
            label_smoothing=label_smoothing
        )

        # Compute train accuracy
        with torch.no_grad():
            train_logits = model(data.x, data.edge_index)
            train_pred = train_logits[data.train_mask].argmax(dim=1)
            train_acc = (train_pred == data.y[data.train_mask]).float().mean().item()

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

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

        # Status marker
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = "★ BEST"
        else:
            wait += 1
            marker = f"  wait={wait}/{patience}"

        print(f"  [DUAL] Epoch {epoch+1:>4d}/{epochs} | "
              f"total={total_loss.item():.4f} (node={loss_node.item():.4f}, edge={loss_edge.item():.4f}) | "
              f"train_acc={train_acc:.4f} | val_F1={val_f1:.4f} | val_acc={val_acc:.4f} | "
              f"lr={current_lr:.6f} | grad={grad_norm:.4f} | {marker}")

        if wait >= patience:
            print(f"\n  [DUAL] ⏹ Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            print(f"  [DUAL] Best val F1: {best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [TRAIN_DUAL] ✓ Loaded best model weights (val_F1={best_val_f1:.4f})")
    model = model.to(device)
    print(f"  [TRAIN_DUAL] Training complete. Final best val_F1: {best_val_f1:.4f}")
    print(f"{'='*60}\n")
    return model, history
