import os
import logging
import pickle

# Set up logging
logger = logging.getLogger(__name__)

def save_fallback_model(model_dict: dict, model_dir: str) -> str:
    """Saves the fallback frequency/backoff model dictionary to a serialized file."""
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, "fallback_model.pkl")
    logger.info(f"Saving fallback model to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump(model_dict, f)
    return filepath

def load_fallback_model(model_dir: str) -> dict:
    """Loads the fallback frequency/backoff model dictionary from the serialized file."""
    filepath = os.path.join(model_dir, "fallback_model.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fallback model file not found at '{filepath}'.")
    logger.info(f"Loading fallback model from {filepath}...")
    with open(filepath, "rb") as f:
        return pickle.load(f)
