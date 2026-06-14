import logging
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set up logging
logger = logging.getLogger(__name__)

def evaluate_predictions(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list], 
    target_name: str
) -> Dict[str, Any]:
    """Computes evaluation metrics (Accuracy, Precision, Recall, F1-Score) and prints a summary."""
    logger.info(f"Evaluating predictions for target '{target_name}'...")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Precision, Recall, and F1 (Weighted average is most robust for imbalanced classes)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Calculate Macro average metrics
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "confusion_matrix": cm
    }
    
    # Formatted terminal printing
    print("\n" + "=" * 50)
    print(f" Evaluation Metrics for Target: {target_name.upper()}")
    print("=" * 50)
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Weighted Precision: {precision_w:.4f}")
    print(f"  Weighted Recall:    {recall_w:.4f}")
    print(f"  Weighted F1-Score:  {f1_w:.4f}")
    print("-" * 50)
    print(f"  Macro Precision:    {precision_m:.4f}")
    print(f"  Macro Recall:       {recall_m:.4f}")
    print(f"  Macro F1-Score:     {f1_m:.4f}")
    print("=" * 50)
    
    # Display confusion matrix shape
    logger.info(f"Confusion Matrix generated with shape: {cm.shape}")
    
    return metrics
