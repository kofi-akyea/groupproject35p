#!/usr/bin/env python3
"""
Standalone script to generate SHAP summary plot for Random Forest model.
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
import shap
import joblib

# Set style for better visualization
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 12

def generate_shap_summary():
    """Generate and save SHAP summary plot for Random Forest."""
    
    print("Loading Random Forest model and test data...")
    
    # Load the model
    rf_model = joblib.load(ROOT / "models" / "random_forest.joblib")
    
    # Load test data
    X_test = pd.read_parquet(ROOT / "data" / "processed" / "X_test.parquet")
    
    print(f"Test data shape: {X_test.shape}")
    
    # Get the preprocessor from the pipeline
    preprocessor = rf_model.named_steps['pre']
    classifier = rf_model.named_steps['clf']
    
    # Transform test data
    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Transformed data shape: {X_test_transformed.shape}")
    
    # Get feature names after transformation
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    
    print(f"Number of features after transformation: {len(feature_names)}")
    
    # Create SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(classifier)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a moment)...")
    shap_values = explainer.shap_values(X_test_transformed)
    
    # For multi-class, shap_values is a list of arrays (one per class)
    # We'll use the values for the predicted class or average across classes
    if isinstance(shap_values, list):
        # Average SHAP values across all classes for overall importance
        shap_values_to_plot = np.mean(shap_values, axis=0)
    else:
        shap_values_to_plot = shap_values
    
    print(f"SHAP values shape: {shap_values_to_plot.shape}")
    
    # Create summary plot
    print("Generating SHAP summary plot...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    shap.summary_plot(
        shap_values_to_plot, 
        X_test_transformed,
        feature_names=feature_names,
        plot_type="dot",  # Shows both feature importance and directional impact
        show=False,
        max_display=15  # Show top 15 features
    )
    
    ax.set_title('Figure 6.7: SHAP Summary Plot – Random Forest (Test Set)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = ROOT / "figures" / "figure_6_7_shap_summary_rf.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSHAP summary plot saved to: {output_path}")
    
    # Also save as PDF for high-quality publication
    pdf_path = ROOT / "figures" / "figure_6_7_shap_summary_rf.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()
    
    # Calculate and print feature importance
    mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
    
    # Ensure mean_abs_shap is 1D
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Features by SHAP Importance:")
    print(feature_importance.head(15).to_string(index=False))
    
    return feature_importance

if __name__ == "__main__":
    try:
        importance = generate_shap_summary()
        print("\n✓ SHAP summary plot generated successfully!")
    except Exception as e:
        print(f"\n✗ Error generating SHAP summary plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
