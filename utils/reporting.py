"""
Report Generation Utilities.

Saves comprehensive experiment results to disk:
  - JSON summary with all metrics
  - CSV tables (overall comparison, per-class breakdown)
  - Training curve plots
  - Confusion matrix heatmaps
  - Full text report
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_float(val):
    """Convert numpy types to Python float for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.floating)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ──────────────────────────────────────────────────────────────────────
# 1. JSON Summary
# ──────────────────────────────────────────────────────────────────────

def save_json_report(all_results: dict, output_dir: str, config: dict = None):
    """Save all metrics as a structured JSON file."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": config or {},
        "models": {},
    }

    for model_name, metrics in all_results.items():
        entry = {}
        for key, val in metrics.items():
            if key == "confusion_matrix":
                entry[key] = val.tolist() if isinstance(val, np.ndarray) else val
            elif key == "classification_report":
                entry[key] = val
            else:
                entry[key] = _safe_float(val)
        report["models"][model_name] = entry

    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  [SAVED] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# 2. CSV Tables
# ──────────────────────────────────────────────────────────────────────

def save_overall_csv(all_results: dict, output_dir: str):
    """Save overall metrics comparison as CSV."""
    rows = []
    for model_name, metrics in all_results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy"),
            "Macro_F1": metrics.get("macro_f1"),
            "Weighted_F1": metrics.get("weighted_f1"),
            "Macro_Precision": metrics.get("macro_precision"),
            "Macro_Recall": metrics.get("macro_recall"),
            "Macro_AUC": metrics.get("macro_auc"),
            "Weighted_AUC": metrics.get("weighted_auc"),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "overall_comparison.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  [SAVED] {path}")
    return path


def save_perclass_csv(all_results: dict, output_dir: str):
    """Save per-class F1, Precision, Recall for every model."""
    rows = []
    for model_name, metrics in all_results.items():
        class_names = metrics.get("class_names", [])
        f1s = metrics.get("per_class_f1", [])
        precs = metrics.get("per_class_precision", [])
        recs = metrics.get("per_class_recall", [])

        for i, cname in enumerate(class_names):
            rows.append({
                "Model": model_name,
                "Class": cname,
                "F1": f1s[i] if i < len(f1s) else None,
                "Precision": precs[i] if i < len(precs) else None,
                "Recall": recs[i] if i < len(recs) else None,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "per_class_metrics.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  [SAVED] {path}")
    return path


def save_ablation_csv(ablation_results: dict, output_dir: str):
    """Save ablation study results as CSV."""
    rows = []
    for name, metrics in ablation_results.items():
        lam_val = name.split("=")[1] if "=" in name else name
        rows.append({
            "Lambda": float(lam_val),
            "Accuracy": metrics.get("accuracy"),
            "Macro_F1": metrics.get("macro_f1"),
            "Weighted_F1": metrics.get("weighted_f1"),
            "Macro_Precision": metrics.get("macro_precision"),
            "Macro_Recall": metrics.get("macro_recall"),
        })

    df = pd.DataFrame(rows).sort_values("Lambda")
    path = os.path.join(output_dir, "ablation_lambda.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  [SAVED] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# 3. Training Curve Plots
# ──────────────────────────────────────────────────────────────────────

def save_training_curves(histories: dict, output_dir: str):
    """Plot and save training loss and validation F1 curves per model."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available, skipping training curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    for name, hist in histories.items():
        if "train_loss" in hist:
            ax.plot(hist["train_loss"], label=name, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation F1 curves
    ax = axes[1]
    for name, hist in histories.items():
        if "val_f1" in hist:
            ax.plot(hist["val_f1"], label=name, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Validation F1 Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def save_dual_loss_curves(dual_history: dict, output_dir: str):
    """Plot node loss vs edge loss for LSGNN-DualTask."""
    if not HAS_MPL:
        return
    if "node_loss" not in dual_history or "edge_loss" not in dual_history:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(dual_history["node_loss"]) + 1)
    ax.plot(epochs, dual_history["node_loss"], label="Node Loss", alpha=0.8)
    ax.plot(epochs, dual_history["edge_loss"], label="Edge Loss", alpha=0.8)
    ax.plot(epochs, dual_history["train_loss"], label="Total Loss", alpha=0.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("LSGNN-DualTask: Node vs Edge Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "dual_loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────
# 4. Confusion Matrix Heatmaps
# ──────────────────────────────────────────────────────────────────────

def save_confusion_matrices(all_results: dict, output_dir: str):
    """Save confusion matrix heatmaps for each model."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available, skipping confusion matrices")
        return

    models_with_cm = {k: v for k, v in all_results.items()
                      if "confusion_matrix" in v and v["confusion_matrix"] is not None}

    if not models_with_cm:
        return

    n = len(models_with_cm)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, models_with_cm.items()):
        cm = metrics["confusion_matrix"]
        class_names = metrics.get("class_names", [f"C{i}" for i in range(cm.shape[0])])
        short_names = [c[:12] for c in class_names]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name, fontsize=11)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────
# 5. Text Report
# ──────────────────────────────────────────────────────────────────────

def save_text_report(all_results: dict, output_dir: str, config: dict = None):
    """Save a human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  EXPERIMENT REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    if config:
        lines.append("\n  Configuration:")
        for k, v in config.items():
            lines.append(f"    {k}: {v}")

    # Overall comparison
    lines.append(f"\n{'─' * 70}")
    lines.append("  OVERALL PERFORMANCE COMPARISON")
    lines.append(f"{'─' * 70}")
    header = f"  {'Model':25s} {'Acc':>8s} {'M-F1':>8s} {'W-F1':>8s} {'M-Prec':>8s} {'M-Rec':>8s} {'AUC':>8s}"
    lines.append(header)
    lines.append(f"  {'-' * 65}")

    for name, m in all_results.items():
        auc_str = f"{m.get('macro_auc', 0):.4f}" if m.get("macro_auc") else "N/A"
        lines.append(
            f"  {name:25s} {m['accuracy']:8.4f} {m['macro_f1']:8.4f} "
            f"{m['weighted_f1']:8.4f} {m['macro_precision']:8.4f} "
            f"{m['macro_recall']:8.4f} {auc_str:>8s}"
        )

    # Per-class metrics for each model
    for name, m in all_results.items():
        class_names = m.get("class_names", [])
        if not class_names:
            continue

        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {name} — Per-Class Metrics")
        lines.append(f"{'─' * 70}")
        lines.append(f"  {'Class':30s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s}")
        lines.append(f"  {'-' * 56}")

        for i, cname in enumerate(class_names):
            f1 = m["per_class_f1"][i]
            p = m["per_class_precision"][i]
            r = m["per_class_recall"][i]
            lines.append(f"  {cname:30s} {f1:8.4f} {p:10.4f} {r:8.4f}")

        # Confusion matrix
        cm = m.get("confusion_matrix")
        if cm is not None:
            lines.append(f"\n  Confusion Matrix:")
            header_row = "  " + " " * 15 + "".join(f"{c[:7]:>8s}" for c in class_names)
            lines.append(header_row)
            for i in range(cm.shape[0]):
                row_name = class_names[i][:14]
                row_vals = "".join(f"{cm[i, j]:8d}" for j in range(cm.shape[1]))
                lines.append(f"  {row_name:15s}{row_vals}")

        # Classification report (sklearn)
        cr = m.get("classification_report")
        if cr:
            lines.append(f"\n  Sklearn Classification Report:")
            for line in cr.split("\n"):
                lines.append(f"  {line}")

    lines.append(f"\n{'=' * 70}")
    lines.append("  END OF REPORT")
    lines.append(f"{'=' * 70}")

    text = "\n".join(lines)
    path = os.path.join(output_dir, "report.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"  [SAVED] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# 6. Master Function
# ──────────────────────────────────────────────────────────────────────

def generate_full_report(all_results: dict, histories: dict = None,
                         ablation_results: dict = None,
                         multi_seed_results: dict = None,
                         config: dict = None,
                         output_dir: str = "results"):
    """
    Generate all reports and save to output_dir/.

    Args:
        all_results: dict mapping model_name -> metrics dict
        histories: dict mapping model_name -> training history dict
        ablation_results: dict mapping "lambda=X" -> metrics (optional)
        multi_seed_results: dict mapping model_name -> list of F1 scores (optional)
        config: dict of experiment configuration (optional)
        output_dir: directory to save all reports
    """
    _ensure_dir(output_dir)

    print(f"\n{'=' * 60}")
    print(f"  GENERATING REPORTS -> {output_dir}/")
    print(f"{'=' * 60}")

    # Always save these
    save_json_report(all_results, output_dir, config)
    save_overall_csv(all_results, output_dir)
    save_perclass_csv(all_results, output_dir)
    save_text_report(all_results, output_dir, config)
    save_confusion_matrices(all_results, output_dir)

    # Training curves if histories provided
    if histories:
        save_training_curves(histories, output_dir)
        # Dual-task specific curves
        if "LSGNN-DualTask" in histories:
            save_dual_loss_curves(histories["LSGNN-DualTask"], output_dir)

    # Ablation study
    if ablation_results:
        save_ablation_csv(ablation_results, output_dir)

    # Multi-seed results
    if multi_seed_results:
        rows = []
        for model_name, scores in multi_seed_results.items():
            rows.append({
                "Model": model_name,
                "Mean_Macro_F1": np.mean(scores),
                "Std_Macro_F1": np.std(scores),
                "Min": np.min(scores),
                "Max": np.max(scores),
                "Scores": str(scores),
            })
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "multi_seed_results.csv")
        df.to_csv(path, index=False, float_format="%.4f")
        print(f"  [SAVED] {path}")

    print(f"\n  All reports saved to: {output_dir}/")
    print(f"{'=' * 60}")
