import os
import logging
import joblib
import numpy as np
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

class RobustLabelEncoder:
    """
    A robust label encoder that maps unseen categories during transform or inverse_transform
    to a specified fallback value (like "UNKNOWN").
    """
    def __init__(self, fallback_value='UNKNOWN'):
        self.fallback_value = fallback_value
        self.classes_ = None
        self.class_to_idx = {}
        self.idx_to_class = {}

    def fit(self, y):
        # Flatten target list/series and find unique classes
        y_str = [str(val).upper().strip() if (val is not None and not (isinstance(val, float) and np.isnan(val))) else self.fallback_value for val in y]
        unique_labels = sorted(list(set(y_str)))
        
        # Ensure fallback_value is represented
        if self.fallback_value not in unique_labels:
            unique_labels.append(self.fallback_value)
            
        self.classes_ = np.array(unique_labels)
        self.class_to_idx = {val: idx for idx, val in enumerate(self.classes_)}
        self.idx_to_class = {idx: val for idx, val in enumerate(self.classes_)}
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("RobustLabelEncoder must be fitted before transforming.")
        y_str = [str(val).upper().strip() if (val is not None and not (isinstance(val, float) and np.isnan(val))) else self.fallback_value for val in y]
        fallback_idx = self.class_to_idx[self.fallback_value]
        return np.array([self.class_to_idx.get(val, fallback_idx) for val in y_str], dtype=int)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_idx):
        if self.classes_ is None:
            raise ValueError("RobustLabelEncoder must be fitted before inverse transforming.")
        if np.isscalar(y_idx) or (isinstance(y_idx, np.ndarray) and y_idx.ndim == 0):
            return self.idx_to_class.get(int(y_idx), self.fallback_value)
        y_idx_arr = np.array(y_idx).ravel()
        decoded = [self.idx_to_class.get(int(idx), self.fallback_value) for idx in y_idx_arr]
        return np.array(decoded)

def save_model(model: Any, filepath: str) -> None:
    """
    Saves a trained model or encoder to disk using joblib.
    """
    try:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        logger.info(f"Saving using joblib to {filepath}...")
        joblib.dump(model, filepath)
        logger.info("Saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save to {filepath}: {e}")
        raise

def load_model(filepath: str) -> Any:
    """
    Loads a saved model or encoder from disk using joblib.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found at {filepath}")
        raise FileNotFoundError(f"File not found at: {filepath}")
        
    try:
        logger.info(f"Loading from {filepath}...")
        model = joblib.load(filepath)
        logger.info("Loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load from {filepath}: {e}")
        raise
