# How to Use — GNN Cybersecurity Intrusion Detection Pipeline

## Prerequisites

```bash
pip install torch torch-geometric numpy pandas scikit-learn imbalanced-learn matplotlib python-pcapng
```

## Project Structure

```
mini-projet-enh/
├── main.py                  # Main experiment runner
├── process_pcap.py          # PCAP processing pipeline
├── data/
│   ├── excel_labeling_complet.csv   # Original reference dataset (13,514 packets)
│   ├── enriched_dataset.csv         # Merged dataset (76,500 packets)
│   └── pcap_processed.csv          # Packets extracted from raw pcap files
├── raw_data/
│   └── 1/ 2/ 3/ 4/ 5/ 6/          # Raw pcapng capture files
├── models/
│   ├── baselines.py         # MLP and Random Forest baselines
│   ├── lsgnn.py             # LSGNN baseline model
│   └── lsgnn_dual.py        # LSGNN-DualTask (our contribution)
├── utils/
│   ├── training.py          # GNN training loops
│   ├── evaluation.py        # Metrics computation and printing
│   └── reporting.py         # Report generation (JSON, CSV, plots, text)
├── results/                 # Generated reports (created automatically)
├── report.tex               # Full LaTeX report (learning-focused)
├── paper_2.tex              # Paper version (no sample counts, with P/R tables)
└── presentation.tex         # 18-slide Beamer presentation
```

## Quick Start

### 1. Run the Full Experiment (Default Settings)

```bash
python main.py
```

This runs the complete pipeline:
- Loads `data/excel_labeling_complet.csv`
- Engineers 68 features from 9 raw columns
- Applies SMOTE for class balancing
- Builds k-NN + communication graph
- Trains Random Forest, MLP, LSGNN, and LSGNN-DualTask
- Evaluates all models on the test set
- **Saves reports to `results/`**

### 2. Run with the Enriched Dataset

```bash
python main.py --data_path data/enriched_dataset.csv
```

Uses the larger 76,500-packet dataset (original + processed pcap files).

### 3. Run Everything (Ablation + Multi-Seed + Reports)

```bash
python main.py --run_ablation --multi_seed --output_dir results
```

This additionally runs:
- **Ablation study**: tests lambda values [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
- **Multi-seed evaluation**: runs 5 seeds [42, 123, 456, 789, 1024] for robustness

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `data/excel_labeling_complet.csv` | Path to input CSV dataset |
| `--label_col` | auto-detected | Name of the label column |
| `--hidden_dim` | `128` | Hidden layer dimensionality |
| `--num_layers` | `2` | Number of GNN message-passing layers |
| `--dropout` | `0.3` | Dropout probability |
| `--k` | `10` | k for k-NN graph construction |
| `--epochs` | `300` | Maximum training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `5e-4` | AdamW weight decay |
| `--patience` | `30` | Early stopping patience (epochs) |
| `--lambda_edge` | `0.3` | Edge consistency loss weight |
| `--seed` | `42` | Random seed |
| `--run_ablation` | off | Run lambda ablation study |
| `--multi_seed` | off | Run multi-seed robustness evaluation |
| `--output_dir` | `results` | Directory for all generated reports |

## Generated Reports

After running, the `results/` folder contains:

| File | Description |
|------|-------------|
| `results.json` | Complete metrics for all models (machine-readable) |
| `overall_comparison.csv` | Accuracy, F1, Precision, Recall, AUC per model |
| `per_class_metrics.csv` | Per-class F1, Precision, Recall for every model |
| `report.txt` | Human-readable text report with all metrics and confusion matrices |
| `confusion_matrices.png` | Side-by-side confusion matrix heatmaps |
| `training_curves.png` | Training loss and validation F1 over epochs |
| `dual_loss_curves.png` | Node loss vs edge loss for LSGNN-DualTask |
| `ablation_lambda.csv` | Ablation study results (if `--run_ablation`) |
| `multi_seed_results.csv` | Mean/std F1 across seeds (if `--multi_seed`) |

## Processing Raw PCAP Files

If you have new pcapng captures to add:

```bash
# Place pcapng files in raw_data/<folder_number>/
# File names are auto-mapped to labels based on keywords:
#   "copy", "SMB"          -> Copy 54ndc47 (SMB)
#   "service creation"     -> Service Creation
#   "discover local hosts" -> Discover local hosts
#   "view remote shares"   -> View remote shares
#   "local fqdn", "host discovery" -> BENIGNE

python process_pcap.py
```

This produces:
- `data/pcap_processed.csv` — packets from pcap files only
- `data/enriched_dataset.csv` — merged with reference dataset

## Example Workflows

### Experiment with different lambda values

```bash
python main.py --lambda_edge 0.5 --output_dir results_lambda05
```

### Use more GNN layers

```bash
python main.py --num_layers 3 --output_dir results_3layers
```

### Quick test with fewer epochs

```bash
python main.py --epochs 50 --patience 10 --output_dir results_quick
```

### Full reproducibility run

```bash
python main.py --run_ablation --multi_seed --data_path data/enriched_dataset.csv --output_dir results_full
```

## Dataset Format

The input CSV must have these columns (order doesn't matter):

| Column | Type | Example |
|--------|------|---------|
| `packet_id` | int | 0 |
| `timestamp` | float | 1771502223.035 |
| `label` | string | BENIGNE |
| `src_ip` | string | 192.168.10.103 |
| `dst_ip` | string | 10.0.0.2 |
| `protocol` | float | 6.0 (TCP=6, UDP=17, ICMP=1) |
| `src_port` | int | 40004 |
| `dst_port` | int | 80 |
| `length` | int | 74 |

Missing values are allowed (`N/A` for IPs, `NaN` for protocol).

## Labels

| Label | Description |
|-------|-------------|
| BENIGNE | Normal, legitimate network traffic |
| Copy 54ndc47 (SMB) | SMB-based file copy attack |
| Service Creation | Remote service creation for persistence |
| Discover local hosts | Network scanning / host discovery |
| View remote shares | SMB share enumeration |

## Compiling LaTeX Documents

```bash
# Report (learning-focused, with all math)
pdflatex report.tex

# Paper version (no sample counts, with precision/recall tables)
pdflatex paper_2.tex

# Presentation (18 slides)
pdflatex presentation.tex
```
