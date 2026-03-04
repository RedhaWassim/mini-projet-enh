"""
Main Experiment Runner: GNN-Based Cybersecurity Intrusion Detection Pipeline.

This script orchestrates the complete experimental pipeline:
  1. Load and analyze the cybersecurity dataset
  2. Preprocess features and construct the graph
  3. Train baseline models (RandomForest, MLP)
  4. Train LSGNN baseline GNN
  5. Train LSGNN-DualTask (novel modification)
  6. Compare all models with comprehensive metrics
  7. Run ablation study on λ (edge loss weight)

Usage:
    python main.py                                    # Use synthetic data
    python main.py --data_path path/to/dataset.csv    # Use real dataset
    python main.py --data_path path/to/dataset.csv --label_col Label

All results are printed to stdout with formatted tables.
"""

import argparse
import os
import sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Project imports
from data.loader import load_dataset, detect_label_column, analyze_dataset, preprocess_features, create_splits
from data.graph_construction import build_graph_data
from models.baselines import MLPClassifier, train_mlp, train_random_forest
from models.lsgnn import LSGNN
from models.lsgnn_dual import LSGNNDualTask
from utils.training import train_gnn, train_gnn_dual, compute_class_weights
from utils.evaluation import compute_metrics, print_metrics, compare_models, analyze_attack_improvements


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_ip_port_columns(df):
    """Auto-detect IP and port column names from the DataFrame."""
    cols = [c.lower() for c in df.columns]
    col_map = dict(zip(cols, df.columns))

    ip_src = None
    ip_dst = None
    port_col = None

    for key in ["src_ip", "source ip", "srcip", "src ip", "ip.src", "ip_src"]:
        if key in cols:
            ip_src = col_map[key]
            break

    for key in ["dst_ip", "destination ip", "dstip", "dst ip", "ip.dst", "ip_dst"]:
        if key in cols:
            ip_dst = col_map[key]
            break

    for key in ["dst_port", "destination port", "dstport", "dst port", "port", "dport"]:
        if key in cols:
            port_col = col_map[key]
            break

    return ip_src, ip_dst, port_col


def run_experiment(args):
    """Run the complete experiment pipeline."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[CONFIG] Device: {device}")
    print(f"[CONFIG] Seed: {args.seed}")
    print(f"[CONFIG] Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"[CONFIG] k-NN k: {args.k}, Lambda edge: {args.lambda_edge}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load and Analyze Dataset
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 1: DATA LOADING AND ANALYSIS")
    print("█" * 70)

    set_seed(args.seed)
    df = load_dataset(args.data_path)

    # Detect label column
    if args.label_col:
        label_col = args.label_col
    else:
        label_col = detect_label_column(df)
    print(f"[INFO] Using label column: '{label_col}'")

    # Analyze
    analysis = analyze_dataset(df, label_col)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Feature Preprocessing
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 2: FEATURE PREPROCESSING")
    print("█" * 70)

    X, y, label_encoder, feature_names = preprocess_features(df, label_col)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # Create train/val/test splits
    train_idx, val_idx, test_idx = create_splits(len(y), y, seed=args.seed)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Graph Construction
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 3: GRAPH CONSTRUCTION")
    print("█" * 70)

    ip_src, ip_dst, port_col = detect_ip_port_columns(df)
    print(f"[INFO] Detected columns -> src_ip: {ip_src}, dst_ip: {ip_dst}, port: {port_col}")

    data = build_graph_data(
        X, y, train_idx, val_idx, test_idx,
        df=df, k=args.k,
        ip_src_col=ip_src, ip_dst_col=ip_dst, port_col=port_col
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Baseline Models (Non-Graph)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 4: BASELINE MODELS (NON-GRAPH)")
    print("█" * 70)

    all_results = {}

    # --- Random Forest ---
    print("\n--- Training Random Forest ---")
    rf = train_random_forest(X[train_idx], y[train_idx], seed=args.seed)
    rf_pred = rf.predict(X[test_idx])
    rf_prob = rf.predict_proba(X[test_idx])
    rf_results = compute_metrics(y[test_idx], rf_pred, rf_prob, class_names)
    print_metrics(rf_results, "Random Forest (Tabular Baseline)")
    all_results["RandomForest"] = rf_results

    # --- MLP ---
    print("\n--- Training MLP ---")
    class_weights = compute_class_weights(y[train_idx], num_classes)
    mlp = MLPClassifier(X.shape[1], args.hidden_dim, num_classes,
                        num_layers=3, dropout=0.3)
    mlp, mlp_hist = train_mlp(
        mlp, X[train_idx], y[train_idx], X[val_idx], y[val_idx],
        epochs=200, lr=1e-3, device=device,
        class_weights=class_weights.numpy()
    )
    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(torch.tensor(X[test_idx], dtype=torch.float32).to(device))
        mlp_pred = mlp_logits.argmax(dim=1).cpu().numpy()
        mlp_prob = torch.softmax(mlp_logits, dim=1).cpu().numpy()
    mlp_results = compute_metrics(y[test_idx], mlp_pred, mlp_prob, class_names)
    print_metrics(mlp_results, "MLP (Neural Baseline)")
    all_results["MLP"] = mlp_results

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: LSGNN Baseline
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 5: LSGNN BASELINE (SOTA GNN)")
    print("█" * 70)

    set_seed(args.seed)
    lsgnn = LSGNN(
        input_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    print(f"\n  LSGNN parameters: {sum(p.numel() for p in lsgnn.parameters()):,}")

    lsgnn, lsgnn_hist = train_gnn(
        lsgnn, data, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        device=device
    )

    # Evaluate LSGNN
    lsgnn.eval()
    data_eval = data.to(device)
    with torch.no_grad():
        lsgnn_logits = lsgnn(data_eval.x, data_eval.edge_index)
        lsgnn_pred = lsgnn_logits[data_eval.test_mask].argmax(dim=1).cpu().numpy()
        lsgnn_prob = torch.softmax(lsgnn_logits[data_eval.test_mask], dim=1).cpu().numpy()
    lsgnn_results = compute_metrics(y[test_idx], lsgnn_pred, lsgnn_prob, class_names)
    print_metrics(lsgnn_results, "LSGNN (Baseline GNN)")
    all_results["LSGNN"] = lsgnn_results

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: LSGNN-DualTask (Novel Modification)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 6: LSGNN-DUALTASK (NOVEL MODIFICATION)")
    print("█" * 70)

    set_seed(args.seed)
    lsgnn_dual = LSGNNDualTask(
        input_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lambda_edge=args.lambda_edge,
        edge_sample_ratio=0.5
    )
    print(f"\n  LSGNN-DualTask parameters: {sum(p.numel() for p in lsgnn_dual.parameters()):,}")
    print(f"  Lambda (edge loss weight): {args.lambda_edge}")

    lsgnn_dual, dual_hist = train_gnn_dual(
        lsgnn_dual, data, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        device=device
    )

    # Evaluate LSGNN-DualTask
    lsgnn_dual.eval()
    with torch.no_grad():
        dual_logits = lsgnn_dual(data_eval.x, data_eval.edge_index)
        dual_pred = dual_logits[data_eval.test_mask].argmax(dim=1).cpu().numpy()
        dual_prob = torch.softmax(dual_logits[data_eval.test_mask], dim=1).cpu().numpy()
    dual_results = compute_metrics(y[test_idx], dual_pred, dual_prob, class_names)
    print_metrics(dual_results, "LSGNN-DualTask (Modified GNN)")
    all_results["LSGNN-DualTask"] = dual_results

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Comparison and Analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  STEP 7: MODEL COMPARISON AND ANALYSIS")
    print("█" * 70)

    compare_models(all_results)

    # Per-class improvement analysis
    analyze_attack_improvements(
        rf_results, lsgnn_results, dual_results, class_names
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Ablation Study on λ (Edge Loss Weight)
    # ══════════════════════════════════════════════════════════════════
    if args.run_ablation:
        print("\n" + "█" * 70)
        print("  STEP 8: ABLATION STUDY (λ SENSITIVITY)")
        print("█" * 70)

        lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        ablation_results = {}

        for lam in lambda_values:
            print(f"\n  --- λ = {lam} ---")
            set_seed(args.seed)

            model_abl = LSGNNDualTask(
                input_dim=X.shape[1],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                lambda_edge=lam,
                edge_sample_ratio=0.5
            )

            model_abl, _ = train_gnn_dual(
                model_abl, data, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience,
                device=device
            )

            model_abl.eval()
            with torch.no_grad():
                abl_logits = model_abl(data_eval.x, data_eval.edge_index)
                abl_pred = abl_logits[data_eval.test_mask].argmax(dim=1).cpu().numpy()
                abl_prob = torch.softmax(abl_logits[data_eval.test_mask], dim=1).cpu().numpy()

            abl_metrics = compute_metrics(y[test_idx], abl_pred, abl_prob, class_names)
            ablation_results[f"λ={lam}"] = abl_metrics

        # Print ablation summary
        print(f"\n{'=' * 60}")
        print(f"  ABLATION: Effect of λ on Test Performance")
        print(f"{'=' * 60}")
        print(f"\n  {'λ':>6s} {'Accuracy':>12s} {'Macro-F1':>12s} {'Weighted-F1':>12s}")
        print(f"  {'-' * 42}")
        for name, res in ablation_results.items():
            lam_val = name.split("=")[1]
            print(f"  {lam_val:>6s} {res['accuracy']:12.4f} "
                  f"{res['macro_f1']:12.4f} {res['weighted_f1']:12.4f}")

        # Note: λ=0 is equivalent to baseline LSGNN (no edge loss)
        print(f"\n  Note: λ=0 is equivalent to LSGNN baseline (edge loss disabled).")
        print(f"  The optimal λ balances node classification with edge consistency.")
        print(f"{'=' * 60}")

    # ══════════════════════════════════════════════════════════════════
    # MULTI-SEED EVALUATION (optional)
    # ══════════════════════════════════════════════════════════════════
    if args.multi_seed:
        print("\n" + "█" * 70)
        print("  MULTI-SEED EVALUATION (Robustness Check)")
        print("█" * 70)

        seeds = [42, 123, 456, 789, 1024]
        seed_results = {"LSGNN": [], "LSGNN-DualTask": []}

        for s in seeds:
            print(f"\n  --- Seed {s} ---")

            # LSGNN
            set_seed(s)
            m1 = LSGNN(X.shape[1], args.hidden_dim, num_classes,
                       args.num_layers, args.dropout)
            m1, _ = train_gnn(m1, data, epochs=args.epochs, lr=args.lr,
                              weight_decay=args.weight_decay,
                              patience=args.patience, device=device)
            m1.eval()
            with torch.no_grad():
                l1 = m1(data_eval.x, data_eval.edge_index)
                p1 = l1[data_eval.test_mask].argmax(dim=1).cpu().numpy()
            r1 = compute_metrics(y[test_idx], p1, class_names=class_names)
            seed_results["LSGNN"].append(r1["macro_f1"])

            # LSGNN-DualTask
            set_seed(s)
            m2 = LSGNNDualTask(X.shape[1], args.hidden_dim, num_classes,
                               args.num_layers, args.dropout,
                               lambda_edge=args.lambda_edge)
            m2, _ = train_gnn_dual(m2, data, epochs=args.epochs, lr=args.lr,
                                   weight_decay=args.weight_decay,
                                   patience=args.patience, device=device)
            m2.eval()
            with torch.no_grad():
                l2 = m2(data_eval.x, data_eval.edge_index)
                p2 = l2[data_eval.test_mask].argmax(dim=1).cpu().numpy()
            r2 = compute_metrics(y[test_idx], p2, class_names=class_names)
            seed_results["LSGNN-DualTask"].append(r2["macro_f1"])

        print(f"\n{'=' * 50}")
        print(f"  Multi-Seed Results (Macro-F1)")
        print(f"{'=' * 50}")
        for model_name, scores in seed_results.items():
            mean_f1 = np.mean(scores)
            std_f1 = np.std(scores)
            print(f"  {model_name:25s}: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"{'=' * 50}")

    print("\n[DONE] Experiment complete.")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Based Cybersecurity Intrusion Detection Pipeline"
    )

    # Data arguments
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV dataset (default: synthetic data)")
    parser.add_argument("--label_col", type=str, default=None,
                        help="Name of the label column (auto-detected if not given)")

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimensionality for all models")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")

    # Graph construction
    parser.add_argument("--k", type=int, default=10,
                        help="k for k-NN graph construction")

    # Training
    parser.add_argument("--epochs", type=int, default=300,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")

    # Dual-task modification
    parser.add_argument("--lambda_edge", type=float, default=0.3,
                        help="Weight of edge consistency loss (λ)")

    # Experiment options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run ablation study on λ")
    parser.add_argument("--multi_seed", action="store_true",
                        help="Run multi-seed evaluation for robustness")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
