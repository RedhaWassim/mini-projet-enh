"""
Evaluation Utilities for Model Comparison.

Provides standardized metrics computation, formatted output, and plotting
utilities for comparing baseline, GNN, and modified GNN models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from collections import defaultdict


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (n,)
        y_pred: Predicted labels (n,)
        y_prob: Predicted probabilities (n, C) — optional, for AUC
        class_names: List of class name strings

    Returns:
        dict with all metrics
    """
    # Use explicit label list if class_names provided, so arrays have consistent size
    labels = list(range(len(class_names))) if class_names is not None else None

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        "per_class_precision": precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }

    # AUC (multi-class, one-vs-rest)
    if y_prob is not None:
        try:
            results["macro_auc"] = roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            )
            results["weighted_auc"] = roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovr"
            )
        except ValueError:
            results["macro_auc"] = None
            results["weighted_auc"] = None

    if class_names is not None:
        results["class_names"] = class_names
        labels = list(range(len(class_names)))
        results["classification_report"] = classification_report(
            y_true, y_pred, labels=labels,
            target_names=class_names, zero_division=0
        )

    return results


def print_metrics(results: dict, model_name: str = "Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {model_name}")
    print(f"{'=' * 70}")

    print(f"\n  Accuracy:         {results['accuracy']:.4f}")
    print(f"  Macro F1:         {results['macro_f1']:.4f}")
    print(f"  Weighted F1:      {results['weighted_f1']:.4f}")
    print(f"  Macro Precision:  {results['macro_precision']:.4f}")
    print(f"  Macro Recall:     {results['macro_recall']:.4f}")

    if "macro_auc" in results and results["macro_auc"] is not None:
        print(f"  Macro AUC:        {results['macro_auc']:.4f}")
        print(f"  Weighted AUC:     {results['weighted_auc']:.4f}")

    # Per-class metrics
    class_names = results.get("class_names", None)
    n_classes = len(results["per_class_f1"])

    print(f"\n  {'Class':25s} {'F1':>8s} {'Recall':>8s} {'Precision':>8s}")
    print(f"  {'-' * 49}")
    for i in range(n_classes):
        name = class_names[i] if class_names else f"Class {i}"
        print(f"  {name:25s} {results['per_class_f1'][i]:8.4f} "
              f"{results['per_class_recall'][i]:8.4f} "
              f"{results['per_class_precision'][i]:8.4f}")

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    cm = results["confusion_matrix"]
    if class_names:
        header = "  " + " " * 15 + "".join(f"{n[:6]:>7s}" for n in class_names)
        print(header)
    for i in range(cm.shape[0]):
        name = class_names[i][:14] if class_names else f"Class {i}"
        row = "".join(f"{cm[i, j]:7d}" for j in range(cm.shape[1]))
        print(f"  {name:15s}{row}")

    print(f"\n{'=' * 70}")


def compare_models(all_results: dict):
    """
    Print a comparison table across all models.

    Args:
        all_results: dict mapping model_name -> metrics dict
    """
    print(f"\n{'#' * 70}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'#' * 70}")

    metrics_to_show = ["accuracy", "macro_f1", "weighted_f1", "macro_recall"]
    auc_available = any("macro_auc" in r and r["macro_auc"] is not None
                        for r in all_results.values())
    if auc_available:
        metrics_to_show.append("macro_auc")

    # Header
    header = f"  {'Model':30s}" + "".join(f"{m:>15s}" for m in metrics_to_show)
    print(f"\n{header}")
    print(f"  {'-' * (30 + 15 * len(metrics_to_show))}")

    # Rows
    for model_name, results in all_results.items():
        row = f"  {model_name:30s}"
        for m in metrics_to_show:
            val = results.get(m, None)
            if val is not None:
                row += f"{val:15.4f}"
            else:
                row += f"{'N/A':>15s}"
        print(row)

    # Best model per metric
    print(f"\n  Best model per metric:")
    for m in metrics_to_show:
        vals = {name: r.get(m, -1) for name, r in all_results.items()
                if r.get(m) is not None}
        if vals:
            best = max(vals, key=vals.get)
            print(f"    {m:20s}: {best} ({vals[best]:.4f})")

    print(f"\n{'#' * 70}")


def analyze_attack_improvements(baseline_results: dict, gnn_results: dict,
                                 modified_results: dict, class_names: list):
    """
    Analyze which attack types benefit most from GNN and the modification.

    This is particularly relevant for cybersecurity where we care about
    per-attack-type detection improvements.
    """
    print(f"\n{'=' * 70}")
    print(f"  PER-CLASS IMPROVEMENT ANALYSIS")
    print(f"{'=' * 70}")

    base_f1 = baseline_results["per_class_f1"]
    gnn_f1 = gnn_results["per_class_f1"]
    mod_f1 = modified_results["per_class_f1"]

    print(f"\n  {'Class':25s} {'Baseline':>10s} {'GNN':>10s} {'Modified':>10s} "
          f"{'GNN Gain':>10s} {'Mod Gain':>10s}")
    print(f"  {'-' * 75}")

    for i, name in enumerate(class_names):
        gnn_gain = gnn_f1[i] - base_f1[i]
        mod_gain = mod_f1[i] - gnn_f1[i]
        print(f"  {name:25s} {base_f1[i]:10.4f} {gnn_f1[i]:10.4f} {mod_f1[i]:10.4f} "
              f"{gnn_gain:+10.4f} {mod_gain:+10.4f}")

    print(f"\n  Average GNN improvement over baseline: "
          f"{np.mean(gnn_f1 - base_f1):+.4f} F1")
    print(f"  Average modification improvement over GNN: "
          f"{np.mean(mod_f1 - gnn_f1):+.4f} F1")
    print(f"\n{'=' * 70}")
