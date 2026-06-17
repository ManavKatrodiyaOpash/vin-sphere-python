import os
import logging
import joblib
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

def save_model(model: Any, filepath: str) -> None:
    """
    Saves a trained model to disk using joblib.
    
    Args:
        model: The trained model object (CatBoost or otherwise).
        filepath: Target filesystem path to save the model.
    """
    try:
        # Create directory path if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logger.info(f"Saving model to {filepath}...")
        joblib.dump(model, filepath)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")
        raise

def load_model(filepath: str) -> Any:
    """
    Loads a saved model from disk using joblib.
    
    Args:
        filepath: Filesystem path to the saved model file.
        
    Returns:
        The loaded model object.
    """
    if not os.path.exists(filepath):
        logger.error(f"Model file not found at {filepath}")
        raise FileNotFoundError(f"Model file not found at: {filepath}")
        
    try:
        logger.info(f"Loading model from {filepath}...")
        model = joblib.load(filepath)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        raise
