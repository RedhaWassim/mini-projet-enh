"""
Generate the pipeline architecture figure for the IEEE paper.
Produces figures/pipeline.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-1.0, 8.0)
ax.axis("off")

# ── Color palette ──
C_RAW    = "#E8F0FE"   # light blue
C_FEAT   = "#D2E3FC"   # medium blue
C_GRAPH  = "#C6DAFC"   # darker blue
C_MODEL  = "#FEEFC3"   # light gold
C_DUAL   = "#FCE8B2"   # gold
C_LOSS   = "#F4C7C3"   # soft red
C_OUT    = "#CEEAD6"   # soft green
C_SMOTE  = "#E6F4EA"   # light green
C_BORDER = "#5F6368"
C_ARROW  = "#3C4043"

def draw_box(x, y, w, h, text, color, fontsize=9, bold=False, text2=None):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=C_BORDER,
                         linewidth=1.3, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ty = y + h / 2
    if text2:
        ty = y + h * 0.62
        ax.text(x + w / 2, y + h * 0.32, text2, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#5F6368", style="italic", zorder=3)
    ax.text(x + w / 2, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#202124", zorder=3,
            wrap=True)

def draw_arrow(x1, y1, x2, y2, style="-|>", color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"),
                zorder=1)

def draw_curved_arrow(x1, y1, x2, y2, rad=0.3, color=C_ARROW, lw=1.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=1)

# ════════════════════════════════════════════════════════════════════
# ROW 1 (top, y~6.2): Raw Data → Feature Engineering → 68-D Features
# ════════════════════════════════════════════════════════════════════

# Raw Data
draw_box(0.2, 6.0, 2.4, 1.2, "Raw Packet Data", C_RAW, fontsize=10, bold=True,
         text2="9 cols × 13,514 rows")

# Arrow
draw_arrow(2.6, 6.6, 3.3, 6.6)

# Feature Engineering
draw_box(3.3, 5.8, 3.2, 1.6, "Feature Engineering", C_FEAT, fontsize=10, bold=True,
         text2="7 feature groups")

# Sub-items for feature engineering (small labels below)
feat_items = ["Temporal", "IP Encoding", "Protocol", "Port Cat.",
              "Per-Src Stats", "Per-Dst Stats", "Flow-Pair"]
for i, item in enumerate(feat_items):
    col = i % 4
    row = i // 4
    ax.text(3.5 + col * 0.78, 5.55 - row * 0.3, item,
            fontsize=5.5, color="#5F6368", ha="left", va="center", zorder=3)

# Arrow
draw_arrow(6.5, 6.6, 7.2, 6.6)

# 68-D Feature Matrix
draw_box(7.2, 6.0, 2.4, 1.2, "Feature Matrix", C_FEAT, fontsize=10, bold=True,
         text2="68 dimensions")

# ════════════════════════════════════════════════════════════════════
# ROW 1 right: SMOTE
# ════════════════════════════════════════════════════════════════════

draw_arrow(9.6, 6.6, 10.3, 6.6)
draw_box(10.3, 6.0, 2.4, 1.2, "SMOTE", C_SMOTE, fontsize=10, bold=True,
         text2="8,108 → 12,483 train")

# ════════════════════════════════════════════════════════════════════
# ROW 2 (middle, y~3.5): Graph Construction + Models
# ════════════════════════════════════════════════════════════════════

# k-NN Graph
draw_box(0.2, 3.5, 2.4, 1.2, "k-NN Graph", C_GRAPH, fontsize=10, bold=True,
         text2="k=10, cosine dist.")

# Comm Edges
draw_box(0.2, 1.8, 2.4, 1.2, "Comm. Edges", C_GRAPH, fontsize=10, bold=True,
         text2="IP-pair + port")

# Arrow from Feature Matrix down to k-NN
draw_arrow(8.4, 6.0, 8.4, 5.3)
draw_curved_arrow(8.4, 5.3, 2.6, 4.3, rad=-0.15)

# Arrow from Raw Data down to Comm Edges
draw_curved_arrow(1.4, 6.0, 1.4, 3.0, rad=0.0)

# Merge symbol
draw_box(3.4, 2.8, 1.8, 1.2, "Graph\nMerge", C_GRAPH, fontsize=9, bold=True,
         text2="195,608 edges")

draw_arrow(2.6, 4.1, 3.4, 3.8)
draw_arrow(2.6, 2.4, 3.4, 3.0)

# Arrow to GNN
draw_arrow(5.2, 3.4, 5.9, 3.4)

# ════════════════════════════════════════════════════════════════════
# LSGNN Backbone
# ════════════════════════════════════════════════════════════════════

draw_box(5.9, 2.6, 2.8, 1.6, "LSGNN Backbone", C_MODEL, fontsize=10, bold=True,
         text2="2 layers, 128-dim")

# Arrow from SMOTE down to baselines
draw_arrow(11.5, 6.0, 11.5, 5.3)

# ════════════════════════════════════════════════════════════════════
# Tabular Baselines (right side)
# ════════════════════════════════════════════════════════════════════

draw_box(10.3, 4.0, 2.4, 1.2, "Random Forest", C_OUT, fontsize=9, bold=True,
         text2="Acc: 0.999, F1: 0.955")
draw_box(10.3, 2.4, 2.4, 1.2, "MLP Baseline", C_OUT, fontsize=9, bold=True,
         text2="Acc: 0.965, F1: 0.886")

draw_arrow(11.5, 5.3, 11.5, 5.2)
draw_curved_arrow(11.5, 5.2, 11.5, 4.0, rad=0.0)
draw_curved_arrow(11.5, 5.2, 11.5, 3.6, rad=0.0)

# ════════════════════════════════════════════════════════════════════
# ROW 3 (bottom, y~0.2): Dual-Task outputs
# ════════════════════════════════════════════════════════════════════

# Node classification head
draw_box(5.0, 0.2, 2.2, 1.2, "Node Class.\nHead", C_MODEL, fontsize=9, bold=True,
         text2="MLP → 5 classes")

# Edge consistency head
draw_box(7.8, 0.2, 2.2, 1.2, "Edge Consist.\nHead", C_DUAL, fontsize=9, bold=True,
         text2="h_i ⊙ h_j → σ")

# Arrows from backbone
draw_arrow(6.7, 2.6, 6.1, 1.4)
draw_arrow(7.8, 2.6, 8.5, 1.4)

# Loss combination
draw_box(4.2, -0.9, 6.6, 0.8, "L = L_node + λ · L_edge    (λ = 0.3)", C_LOSS,
         fontsize=10, bold=True)

draw_arrow(6.1, 0.2, 6.5, -0.1)
draw_arrow(8.9, 0.2, 8.5, -0.1)

# Final results boxes for GNN models
draw_box(11.4, 0.9, 2.6, 0.7, "LSGNN Baseline", C_OUT, fontsize=8, bold=True,
         text2="F1: 0.795")
draw_box(11.4, -0.2, 2.6, 0.7, "LSGNN-DualTask", C_OUT, fontsize=8, bold=True,
         text2="F1: 0.833  (+3.8%)")

draw_arrow(10.8, -0.5, 11.4, -0.0)
draw_curved_arrow(7.3, 2.8, 11.4, 1.2, rad=-0.2)

# ════════════════════════════════════════════════════════════════════
# Section labels
# ════════════════════════════════════════════════════════════════════

ax.text(7.0, 7.6, "LSGNN-DualTask: End-to-End Pipeline for Network Intrusion Detection",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1A237E")

# Subtle region labels
ax.text(0.1, 7.4, "① Data & Features", fontsize=8, color="#1565C0",
        fontweight="bold", fontstyle="italic")
ax.text(0.1, 4.9, "② Graph Construction", fontsize=8, color="#1565C0",
        fontweight="bold", fontstyle="italic")
ax.text(4.8, 1.65, "③ Dual-Task Training", fontsize=8, color="#1565C0",
        fontweight="bold", fontstyle="italic")
ax.text(10.2, 5.45, "④ Baselines", fontsize=8, color="#1565C0",
        fontweight="bold", fontstyle="italic")

plt.tight_layout(pad=0.5)
plt.savefig("figures/pipeline.pdf", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig("figures/pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("[OK] Saved figures/pipeline.pdf and figures/pipeline.png")
