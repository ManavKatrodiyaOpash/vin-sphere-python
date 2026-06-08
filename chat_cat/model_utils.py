import os
import logging
from typing import Any, Dict
from catboost import CatBoostClassifier

# Set up logging
logger = logging.getLogger(__name__)

# Base parameters with strict memory limits for 16GB RAM machines
BASE_CATBOOST_PARAMS = {
    "iterations": 200,
    "learning_rate": 0.05,
    "depth": 6,
    "early_stopping_rounds": 30,
    "eval_metric": "Accuracy",
    "verbose": 50,
    "random_seed": 42,
    # --- Memory Optimization Settings ---
    "max_ctr_complexity": 1,        # Prevents exponential growth of categorical feature combinations
    "border_count": 32,             # Reduces splits for numeric/engineered features
    "used_ram_limit": "6gb",        # Limits memory during Target Statistics (CTR) computation
    "model_size_reg": 5.0           # Penalizes features that would lead to large model sizes
}

def get_catboost_params(target_name: str, num_classes: int) -> Dict[str, Any]:
    """Generates customized memory-optimized parameters for a given target class size."""
    params = BASE_CATBOOST_PARAMS.copy()
    
    target_name = target_name.upper()
    logger.info(f"Configuring CatBoost params for '{target_name}' with {num_classes} classes...")
    
    # Class balancing
    if target_name in ["REGIONALSPEC", "BODY_TYPE", "ENGINE"]:
        params["auto_class_weights"] = "Balanced"
        
    # Newton leaf estimation method creates an O(N_classes^2) Hessian matrix per object.
    # For targets with hundreds/thousands of classes (like MAKE, MODEL, TRIM), Newton leaf
    # estimation will cause Terabytes of RAM allocations. We MUST use 'Gradient' method instead.
    if num_classes > 50:
        logger.info(f"Target '{target_name}' has high cardinality ({num_classes} classes). Enforcing 'Gradient' leaf estimation to prevent Hessian OOM.")
        params["leaf_estimation_method"] = "Gradient"
        
    # If classes are extremely high, reduce depth to keep model memory usage low
    if num_classes > 500:
        logger.info(f"Target '{target_name}' has extreme cardinality ({num_classes} classes). Reducing tree depth to 5 for memory safety.")
        params["depth"] = 5
        
    return params

def create_model(target_name: str, num_classes: int) -> CatBoostClassifier:
    """Instantiates a CatBoostClassifier with memory-optimized parameters."""
    params = get_catboost_params(target_name, num_classes)
    return CatBoostClassifier(**params)

def save_catboost_model(model: CatBoostClassifier, model_dir: str, target_name: str) -> str:
    """Saves a trained CatBoost model in the model registry."""
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{target_name.lower()}_model.cbm"
    filepath = os.path.join(model_dir, filename)
    logger.info(f"Saving model for '{target_name}' to {filepath}...")
    model.save_model(filepath)
    return filepath

def load_catboost_model(model_dir: str, target_name: str) -> CatBoostClassifier:
    """Loads a CatBoost model from the model registry."""
    filename = f"{target_name.lower()}_model.cbm"
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file for '{target_name}' not found at '{filepath}'.")
    
    logger.info(f"Loading model for '{target_name}' from {filepath}...")
    model = CatBoostClassifier()
    model.load_model(filepath)
    return model
