import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score

def calculate_top_k_accuracy(y_true, y_pred_probs, k=3):
    """Calculates top-k accuracy for multi-class classification predictions."""
    if len(y_true) == 0:
        return 0.0
    # y_pred_probs shape: (num_samples, num_classes)
    top_k_preds = np.argsort(y_pred_probs, axis=-1)[:, -k:]
    hits = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(hits))

def calculate_classification_metrics(y_true, y_pred_probs):
    """Calculates standard classification metrics (accuracy, precision, recall, F1, and top-3 accuracy)."""
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    num_classes = y_pred_probs.shape[-1]
    if num_classes >= 3:
        metrics["top_3_accuracy"] = float(calculate_top_k_accuracy(y_true, y_pred_probs, k=3))
    else:
        metrics["top_3_accuracy"] = float(acc)
        
    return metrics

def calculate_mape(y_true, y_pred, epsilon=1e-5):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = np.abs(y_true) > epsilon
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def calculate_regression_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, MAPE, and R2 score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2)
    }
