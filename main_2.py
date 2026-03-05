"""
Main Experiment Runner for TON-IoT Dataset.

Runs the same pipeline as main.py but on the TON-IoT network dataset:
  1. Load and analyze the TON-IoT dataset
  2. Preprocess features (encode categoricals, scale numerics, SMOTE)
  3. Construct the k-NN graph
  4. Train baseline models (RandomForest, MLP)
  5. Train LSGNN baseline
  6. Train LSGNN-DualTask
  7. Compare all models
  8. Generate reports to results_2/

Usage:
    python main_2.py
    python main_2.py --epochs 20 --patience 10
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from ton_iot_models.preprocessing import (
    load_toniot, analyze_toniot, preprocess_toniot,
    create_splits, augment_training_data
)
from data.graph_construction import build_graph_data
from models.baselines import MLPClassifier, train_mlp, train_random_forest
from models.lsgnn import LSGNN
from models.lsgnn_dual import LSGNNDualTask
from utils.training import train_gnn, train_gnn_dual, compute_class_weights
from utils.evaluation import (
    compute_metrics, print_metrics, compare_models, analyze_attack_improvements
)
from utils.reporting import generate_full_report


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_gnn(model, data, device, class_names):
    """Evaluate a GNN model on the test set."""
    model.eval()
    data_d = data.to(device)
    with torch.no_grad():
        logits = model(data_d.x, data_d.edge_index)
        test_logits = logits[data_d.test_mask]
        pred = test_logits.argmax(dim=1).cpu().numpy()
        prob = torch.softmax(test_logits, dim=1).cpu().numpy()
    y_true = data_d.y[data_d.test_mask].cpu().numpy()
    return compute_metrics(y_true, pred, prob, class_names)


def run_experiment(args):
    """Run the complete TON-IoT experiment pipeline."""

    if args.gpu:
        if not torch.cuda.is_available():
            print("[ERROR] --gpu flag set but CUDA is not available!")
            print("[ERROR] Install PyTorch with CUDA support or remove --gpu flag.")
            return
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'#'*70}")
    print(f"  TON-IoT INTRUSION DETECTION EXPERIMENT")
    print(f"{'#'*70}")
    print(f"\n[CONFIG] Device: {device}")
    if device == "cuda":
        print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CONFIG] GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"[CONFIG] Seed: {args.seed}")
    print(f"[CONFIG] Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"[CONFIG] k-NN k: {args.k}, Lambda edge: {args.lambda_edge}")
    print(f"[CONFIG] Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"[CONFIG] Max samples: {args.max_samples}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load and Analyze Dataset
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 1: DATA LOADING AND ANALYSIS")
    print("=" * 70)

    set_seed(args.seed)
    df = load_toniot(args.data_path)

    # Subsample if dataset is too large for k-NN graph construction
    if args.max_samples and len(df) > args.max_samples:
        print(f"\n[INFO] Subsampling {args.max_samples:,} from {len(df):,} samples (stratified by 'type')...")
        from sklearn.model_selection import train_test_split
        keep_ratio = args.max_samples / len(df)
        _, df = train_test_split(
            df, test_size=keep_ratio, stratify=df["type"], random_state=args.seed
        )
        df = df.reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(df):,} samples")

    analyze_toniot(df)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Feature Preprocessing & Augmentation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE PREPROCESSING & AUGMENTATION")
    print("=" * 70)

    X, y, label_encoder, feature_names = preprocess_toniot(df, label_col="type")
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # Create train/val/test splits
    train_idx, val_idx, test_idx = create_splits(len(y), y, seed=args.seed)

    # SMOTE augmentation on training data (for baselines)
    X_train_aug, y_train_aug = augment_training_data(X, y, train_idx, seed=args.seed)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Graph Construction
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 3: GRAPH CONSTRUCTION")
    print("=" * 70)

    # TON-IoT has src_ip, dst_ip, dst_port columns
    ip_src = "src_ip" if "src_ip" in df.columns else None
    ip_dst = "dst_ip" if "dst_ip" in df.columns else None
    port_col = "dst_port" if "dst_port" in df.columns else None
    print(f"[INFO] Using columns -> src_ip: {ip_src}, dst_ip: {ip_dst}, port: {port_col}")

    data = build_graph_data(
        X, y, train_idx, val_idx, test_idx,
        df=df, k=args.k,
        ip_src_col=ip_src, ip_dst_col=ip_dst, port_col=port_col
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Baseline Models (Non-Graph)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 4: BASELINE MODELS (NON-GRAPH)")
    print("=" * 70)

    all_results = {}
    all_histories = {}

    # --- Random Forest ---
    print("\n--- Training Random Forest (with SMOTE augmentation) ---")
    rf = train_random_forest(X_train_aug, y_train_aug, seed=args.seed)
    rf_pred = rf.predict(X[test_idx])
    rf_prob = rf.predict_proba(X[test_idx])
    rf_results = compute_metrics(y[test_idx], rf_pred, rf_prob, class_names)
    print_metrics(rf_results, "Random Forest (Tabular Baseline + SMOTE)")
    all_results["RandomForest"] = rf_results

    # --- MLP ---
    print("\n--- Training MLP (with SMOTE augmentation) ---")
    class_weights_mlp = compute_class_weights(y_train_aug, num_classes)
    mlp = MLPClassifier(X.shape[1], args.hidden_dim, num_classes,
                        num_layers=3, dropout=0.3)
    mlp, mlp_hist = train_mlp(
        mlp, X_train_aug, y_train_aug, X[val_idx], y[val_idx],
        epochs=200, lr=1e-3, device=device,
        class_weights=class_weights_mlp.numpy()
    )
    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(torch.tensor(X[test_idx], dtype=torch.float32).to(device))
        mlp_pred = mlp_logits.argmax(dim=1).cpu().numpy()
        mlp_prob = torch.softmax(mlp_logits, dim=1).cpu().numpy()
    mlp_results = compute_metrics(y[test_idx], mlp_pred, mlp_prob, class_names)
    print_metrics(mlp_results, "MLP (Neural Baseline + SMOTE)")
    all_results["MLP"] = mlp_results
    all_histories["MLP"] = mlp_hist

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: LSGNN Baseline
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 5: LSGNN BASELINE (SOTA GNN)")
    print("=" * 70)

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

    lsgnn_results = evaluate_gnn(lsgnn, data, device, class_names)
    print_metrics(lsgnn_results, "LSGNN (Baseline GNN)")
    all_results["LSGNN"] = lsgnn_results
    all_histories["LSGNN"] = lsgnn_hist

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: LSGNN-DualTask (Novel Modification)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 6: LSGNN-DUALTASK (NOVEL MODIFICATION)")
    print("=" * 70)

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

    dual_results = evaluate_gnn(lsgnn_dual, data, device, class_names)
    print_metrics(dual_results, "LSGNN-DualTask (Modified GNN)")
    all_results["LSGNN-DualTask"] = dual_results
    all_histories["LSGNN-DualTask"] = dual_hist

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Comparison and Analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 7: MODEL COMPARISON AND ANALYSIS")
    print("=" * 70)

    compare_models(all_results)

    analyze_attack_improvements(
        rf_results, lsgnn_results, dual_results, class_names
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Generate Reports
    # ══════════════════════════════════════════════════════════════════
    config = {
        "dataset": "TON-IoT Network",
        "data_path": args.data_path,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "k": args.k,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "lambda_edge": args.lambda_edge,
    }

    generate_full_report(
        all_results=all_results,
        histories=all_histories,
        config=config,
        output_dir=args.output_dir,
    )

    print("\n[DONE] TON-IoT experiment complete.")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Based Intrusion Detection on TON-IoT Dataset"
    )

    # Data arguments
    parser.add_argument("--data_path", type=str,
                        default="ton_iot_dataset/train_test_network.csv",
                        help="Path to TON-IoT CSV dataset")

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimensionality for all models")
    parser.add_argument("--num_layers", type=int, default=2,
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
                        help="Weight of edge consistency loss (lambda)")

    # Experiment options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results_2",
                        help="Directory to save all reports and plots")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Max samples to use (stratified subsample). Set to 0 to use all.")
    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU usage (exits with error if CUDA is not available)")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
