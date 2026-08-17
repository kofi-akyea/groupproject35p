#!/usr/bin/env python3
"""
Generate figures for the Project Risk DSS documentation.
Creates model comparison charts and confusion matrices based on pipeline results.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Configure default Matplotlib and Seaborn style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Recorded benchmark performance metrics from test evaluations
models = ['Logistic Regression', 'Random Forest', 'Ordinal Logistic Regression', 
          'XGBoost', 'SVM (RBF)', 'K-Nearest Neighbors']
accuracy = [0.4950, 0.5217, 0.5067, 0.4900, 0.4933, 0.4117]
macro_f2 = [0.5241, 0.5081, 0.5082, 0.4904, 0.4779, 0.4071]
qwk = [0.6604, 0.6481, 0.6829, 0.6239, 0.6317, 0.4948]
within_one = [0.9067, 0.9483, 0.9617, 0.9283, 0.9483, 0.8833]

# Create output directories for saved figure images and PDF exports
figures_dir = Path("figures")
script_figures_dir = Path("script_and_doc/figures")
figures_dir.mkdir(exist_ok=True)
script_figures_dir.mkdir(exist_ok=True)

# Figure 6.1: Subplot grid comparing Accuracy, Macro F2, QWK, and Within-One Accuracy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparative Performance of All Six Models on Test Set (600 samples)', 
             fontsize=16, fontweight='bold', y=0.995)

# Render Exact-Match Accuracy subplot
bars1 = axes[0, 0].barh(models, accuracy, color='#5A4FCF', alpha=0.8)
axes[0, 0].set_xlabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Exact-Match Accuracy', fontweight='bold')
axes[0, 0].set_xlim(0, 0.6)
axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
axes[0, 0].legend()
for i, v in enumerate(accuracy):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# Render Macro F2 Score subplot
bars2 = axes[0, 1].barh(models, macro_f2, color='#6495ED', alpha=0.8)
axes[0, 1].set_xlabel('Macro F2 Score', fontweight='bold')
axes[0, 1].set_title('Macro F2 (Recall-Weighted)', fontweight='bold')
axes[0, 1].set_xlim(0, 0.6)
axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
axes[0, 1].legend()
for i, v in enumerate(macro_f2):
    axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# Render Quadratic Weighted Kappa subplot
bars3 = axes[1, 0].barh(models, qwk, color='#00008B', alpha=0.8)
axes[1, 0].set_xlabel('Quadratic Weighted Kappa', fontweight='bold')
axes[1, 0].set_title('QWK (Ordinal Agreement)', fontweight='bold')
axes[1, 0].set_xlim(0, 0.8)
axes[1, 0].axvline(x=0.6, color='green', linestyle='--', alpha=0.5, label='Good agreement')
axes[1, 0].axvline(x=0.7, color='blue', linestyle='--', alpha=0.5, label='Excellent agreement')
axes[1, 0].legend()
for i, v in enumerate(qwk):
    axes[1, 0].text(v + 0.02, i, f'{v:.4f}', va='center', fontweight='bold')

# Render Within-One Class Accuracy subplot
bars4 = axes[1, 1].barh(models, within_one, color='#98FF98', alpha=0.8)
axes[1, 1].set_xlabel('Within-One Accuracy', fontweight='bold')
axes[1, 1].set_title('Within-One Accuracy (±1 Level)', fontweight='bold')
axes[1, 1].set_xlim(0.8, 1.0)
axes[1, 1].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
axes[1, 1].legend()
for i, v in enumerate(within_one):
    axes[1, 1].text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / "figure_6_1_model_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(script_figures_dir / "figure_6_1_model_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(script_figures_dir / "figure_6_1_model_comparison.pdf", bbox_inches='tight')
plt.close()

print(f"✓ Figure 6.1 saved to {figures_dir}/figure_6_1_model_comparison.png")
print(f"✓ Figure 6.1 saved to {script_figures_dir}/figure_6_1_model_comparison.png")
print(f"✓ Figure 6.1 saved to {script_figures_dir}/figure_6_1_model_comparison.pdf")

# Define confusion matrix arrays for all six models
confusion_matrices = {
    'Logistic Regression': np.array([[91, 21, 9, 0], [58, 76, 52, 24], [12, 42, 53, 48], [0, 11, 26, 77]]),
    'Random Forest': np.array([[54, 65, 2, 0], [16, 158, 32, 4], [1, 80, 45, 29], [0, 24, 34, 56]]),
    'Ordinal Logistic Regression': np.array([[63, 55, 3, 0], [33, 124, 47, 6], [0, 68, 56, 31], [0, 14, 39, 61]]),
    'XGBoost': np.array([[64, 50, 5, 2], [39, 119, 40, 12], [4, 65, 51, 35], [1, 19, 34, 60]]),
    'SVM (RBF)': np.array([[58, 58, 5, 0], [24, 145, 37, 4], [1, 73, 48, 33], [1, 20, 48, 45]]),
    'K-Nearest Neighbors': np.array([[51, 52, 17, 1], [35, 103, 58, 14], [8, 56, 49, 42], [4, 26, 40, 44]])
}

class_names = ['Low', 'Medium', 'High', 'Critical']

# Render 2x3 subplot grid of normalized confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices for All Six Models (Normalised)', 
             fontsize=16, fontweight='bold', y=0.995)

axes_flat = axes.flatten()

for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
    ax = axes_flat[idx]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax, 
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(model_name, fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(figures_dir / "figure_5_1_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.savefig(script_figures_dir / "figure_5_1_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.savefig(script_figures_dir / "figure_5_1_confusion_matrices.pdf", bbox_inches='tight')
plt.close()

print(f"✓ Figure 5.1 saved to {figures_dir}/figure_5_1_confusion_matrices.png")
print(f"✓ Figure 5.1 saved to {script_figures_dir}/figure_5_1_confusion_matrices.png")
print(f"✓ Figure 5.1 saved to {script_figures_dir}/figure_5_1_confusion_matrices.pdf")

# Render grouped bar comparison chart across all metrics
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#5A4FCF', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, macro_f2, width, label='Macro F2', color='#6495ED', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, qwk, width, label='QWK', color='#00008B', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, within_one, width, label='Within-One', color='#98FF98', alpha=0.8)

ax.set_xlabel('Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Comprehensive Model Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(figures_dir / "figure_6_2_grouped_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(script_figures_dir / "figure_6_2_grouped_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(script_figures_dir / "figure_6_2_grouped_comparison.pdf", bbox_inches='tight')
plt.close()

print(f"✓ Figure 6.2 saved to {figures_dir}/figure_6_2_grouped_comparison.png")
print(f"✓ Figure 6.2 saved to {script_figures_dir}/figure_6_2_grouped_comparison.png")
print(f"✓ Figure 6.2 saved to {script_figures_dir}/figure_6_2_grouped_comparison.pdf")

print("\n✅ All figures generated successfully!")
