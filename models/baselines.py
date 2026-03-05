"""
Baseline Non-Graph Models for Cybersecurity Classification.

These serve as comparison points to demonstrate the value of GNN-based approaches.
We implement:
  1. MLP (Multi-Layer Perceptron) — a neural baseline on flat features.
  2. RandomForest wrapper — a strong tabular baseline.

Both operate on the raw feature matrix X without graph structure.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# MLP Baseline (PyTorch)
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Simple feedforward MLP for multi-class classification on tabular features.

    Architecture: Input -> [Linear -> BN -> ReLU -> Dropout] x L -> Linear -> Output
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(model, X_train, y_train, X_val, y_val,
              epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
              patience: int = 20, device: str = None, class_weights=None):
    """
    Train the MLP classifier with early stopping on validation F1.

    Returns:
        best_state_dict, training_history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  [TRAIN_MLP] Starting MLP training on device: {device.upper()}")
    print(f"  [TRAIN_MLP] Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"  [TRAIN_MLP] Patience: {patience}")
    print(f"{'='*60}")
    model = model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    print(f"  [TRAIN_MLP] Model and data moved to {device.upper()}")

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_f1": []}

    print(f"  [TRAIN_MLP] Training started...\n")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)

        # Train accuracy
        with torch.no_grad():
            train_pred = out.argmax(dim=1).cpu().numpy()
            train_acc = (train_pred == y_train).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_pred = val_out.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

        history["train_loss"].append(loss.item())
        history["val_f1"].append(val_f1)

        # Status marker
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = "★ BEST"
        else:
            wait += 1
            marker = f"  wait={wait}/{patience}"

        print(f"  [MLP] Epoch {epoch+1:>4d}/{epochs} | "
              f"loss={loss.item():.4f} | train_acc={train_acc:.4f} | "
              f"val_F1={val_f1:.4f} | lr={current_lr:.6f} | {marker}")

        if wait >= patience:
            print(f"\n  [MLP] ⏹ Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            print(f"  [MLP] Best val F1: {best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [TRAIN_MLP] ✓ Loaded best model weights (val_F1={best_val_f1:.4f})")
    print(f"  [TRAIN_MLP] Training complete. Final best val_F1: {best_val_f1:.4f}")
    print(f"{'='*60}\n")
    return model, history


# ---------------------------------------------------------------------------
# Random Forest Baseline
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train, n_estimators: int = 200, seed: int = 42):
    """
    Train a RandomForest classifier as a strong tabular baseline.

    RandomForest is particularly effective for cybersecurity data due to:
    - Natural handling of mixed feature types
    - Robustness to outliers and noisy features
    - Built-in feature importance ranking
    """
    print(f"\n  [TRAIN_RF] Training RandomForest with {n_estimators} estimators...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",  # Handle imbalanced classes
        random_state=seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print(f"  [TRAIN_RF] ✓ RandomForest training complete ({n_estimators} trees)")
    return clf
