import logging
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, root_mean_squared_error

# Configure logging
logger = logging.getLogger(__name__)

def evaluate_classification(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], target_name: str) -> float:
    """
    Evaluates a classification model using Accuracy and Classification Report.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        target_name: Name of the target variable.
        
    Returns:
        The accuracy score.
    """
    logger.info(f"Evaluating classification for target: {target_name}")
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    print("\n" + "=" * 65)
    print(f" Classification Evaluation for Target: {target_name.upper()}")
    print("=" * 65)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("=" * 65 + "\n")
    
    return accuracy

def evaluate_regression(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], target_name: str) -> Dict[str, float]:
    """
    Evaluates a regression model using Mean Absolute Error and Root Mean Squared Error.
    If the target is 'year', also computes classification metrics after rounding predictions.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        target_name: Name of the target variable.
        
    Returns:
        A dictionary containing regression metrics (and optionally accuracy).
    """
    logger.info(f"Evaluating regression for target: {target_name}")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    print("\n" + "=" * 65)
    print(f" Regression Evaluation for Target: {target_name.upper()}")
    print("=" * 65)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    metrics = {"mae": mae, "rmse": rmse}
    
    if target_name.lower() in ["year"]:
        # Round predicted floats to the nearest integer year for classification reporting
        y_pred_rounded = np.round(y_pred).astype(int)
        y_true_int = np.round(y_true).astype(int)
        accuracy = accuracy_score(y_true_int, y_pred_rounded)
        print(f"Rounded Year Accuracy: {accuracy:.4f}")
        try:
            report = classification_report(y_true_int, y_pred_rounded, zero_division=0)
            print("\nClassification Report (Rounded Year):")
            print(report)
        except Exception as e:
            logger.warning(f"Could not print classification report for rounded year: {e}")
        metrics["accuracy"] = accuracy
        
    print("=" * 65 + "\n")
    return metrics
