#!/usr/bin/env python3
"""
Standalone script to generate confusion matrix heatmap for Random Forest model.
This script does not modify the codebase - it only generates visualization.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def generate_confusion_matrix():
    """Generate and save confusion matrix heatmap for Random Forest."""
    
    print("Loading Random Forest model and test data...")
    
    # Load the model
    rf_model = joblib.load(ROOT / "models" / "random_forest.joblib")
    
    # Load test data
    X_test = pd.read_parquet(ROOT / "data" / "processed" / "X_test.parquet")
    y_test = pd.read_parquet(ROOT / "data" / "processed" / "y_test.parquet")
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Make predictions
    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)
    
    # Get class names
    from code.utils.risk_levels import RISK_LEVELS
    class_names = RISK_LEVELS
    
    # Create confusion matrix
    cm = pd.crosstab(
        y_test['Risk_Level'].map({i: name for i, name in enumerate(class_names)}),
        pd.Series(y_pred).map({i: name for i, name in enumerate(class_names)}),
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    
    # Reorder to match expected order
    cm = cm.reindex(index=class_names, columns=class_names, fill_value=0)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a colormap that highlights correct classifications (diagonal)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title('Figure 6.6: Confusion Matrix – Random Forest (Test Set)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Risk Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Risk Level', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = ROOT / "figures" / "figure_6_6_confusion_matrix_rf.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    
    # Also save as PDF for high-quality publication
    pdf_path = ROOT / "figures" / "figure_6_6_confusion_matrix_rf.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()
    
    # Calculate and print additional metrics
    from sklearn.metrics import classification_report, accuracy_score
    print("\nClassification Report:")
    print(classification_report(y_test['Risk_Level'], y_pred, target_names=class_names))
    print(f"Overall Accuracy: {accuracy_score(y_test['Risk_Level'], y_pred):.4f}")
    
    return cm

if __name__ == "__main__":
    try:
        cm = generate_confusion_matrix()
        print("\n✓ Confusion matrix generated successfully!")
    except Exception as e:
        print(f"\n✗ Error generating confusion matrix: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
