import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

from feature_engineering import normalize_chassis, extract_features
from model_utils import load_model

# Configure logging
logger = logging.getLogger(__name__)

# Global cache to store loaded models to speed up batch predictions
_MODEL_CACHE: Dict[str, Any] = {}

# Target JSON keys mapped to expected safe target names used in filenames
TARGETS = ["make", "model", "trim", "body_type", "year", "color", "weight", "regional_specs", "origin"]

def get_model(target: str, model_dir: str = "chat_cat_short_vin/models") -> Any:
    """
    Loads and caches the model for a given target.
    
    Args:
        target: The target attribute name.
        model_dir: Directory containing model files.
        
    Returns:
        The loaded model object.
    """
    global _MODEL_CACHE
    safe_name = target.replace(" ", "_")
    model_key = f"{safe_name}_model"
    
    if model_key not in _MODEL_CACHE:
        model_filename = f"{safe_name}_model.pkl"
        model_path = os.path.join(model_dir, model_filename)
        try:
            _MODEL_CACHE[model_key] = load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for {target} from {model_path}: {e}")
            raise FileNotFoundError(f"Model file for {target} is missing or corrupt.")
            
    return _MODEL_CACHE[model_key]
 
def predict_vehicle(chassis_number: str, model_dir: str = "chat_cat_short_vin/models") -> Dict[str, Any]:
    """
    Normalizes the input chassis number, extracts features, and uses the trained
    CatBoost models to predict vehicle attributes.
    
    Args:
        chassis_number: 10-character raw short chassis number.
        model_dir: Directory where the models are saved.
        
    Returns:
        A dictionary containing the predictions in the specified format.
    """
    # 1. Normalize chassis number
    normalized = normalize_chassis(chassis_number)
    if not normalized:
        raise ValueError("Invalid chassis number provided.")
        
    # 2. Extract features (returns a 1-row DataFrame)
    df_input = pd.DataFrame({"chassisNumber": [normalized]})
    X_features = extract_features(df_input["chassisNumber"])
    
    # Ensure categorical features are string type (excluding numeric features)
    numeric_features = ["serial_number", "first_digit_idx", "last_letter_idx", "num_letters", "num_digits"]
    cat_features = [col for col in X_features.columns if col not in numeric_features]
    for col in cat_features:
        X_features[col] = X_features[col].astype(str)
        
    predictions: Dict[str, Any] = {}
    
    # 3. Perform prediction for each target
    for target in TARGETS:
        try:
            model = get_model(target, model_dir)
            pred = model.predict(X_features)
            
            # Extract scalar from numpy array returned by CatBoost
            if isinstance(pred, np.ndarray):
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    val = pred[0][0]
                else:
                    val = pred[0]
            else:
                val = pred
                
            # Process regression outputs (cast to standard Python types)
            if target == "year":
                predictions["year"] = int(np.round(float(val)))
            elif target == "weight":
                predictions["weight"] = float(np.round(float(val), 2))
            else:
                # Replace underscores/hyphens if needed, or leave as string
                predictions[target] = str(val)
                
        except Exception as e:
            logger.warning(f"Error predicting target '{target}': {e}. Setting to default.")
            if target == "year":
                predictions["year"] = 0
            elif target == "weight":
                predictions["weight"] = 0.0
            else:
                predictions[target] = "UNKNOWN"
                
    # Map 'regional_specs' internal target to the user's requested key 'regional specs'
    output = {
        "make": predictions.get("make", "UNKNOWN"),
        "model": predictions.get("model", "UNKNOWN"),
        "trim": predictions.get("trim", "UNKNOWN"),
        "body_type": predictions.get("body_type", "UNKNOWN"),
        "year": predictions.get("year", 0),
        "color": predictions.get("color", "UNKNOWN"),
        "weight": predictions.get("weight", 0.0),
        "regional specs": predictions.get("regional_specs", "UNKNOWN"),
        "origin": predictions.get("origin", "UNKNOWN")
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Japanese Import 10-Character Short Chassis Model Inference Tool")
    parser.add_argument(
        "--chassis",
        type=str,
        required=True,
        help="10-character short chassis number string to decode."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="chat_cat_short_vin/models",
        help="Directory where models are saved."
    )
    
    args = parser.parse_args()
    
    try:
        result = predict_vehicle(args.chassis, args.model_dir)
        print(json.dumps(result, indent=4))
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(json.dumps({"error": str(e)}, indent=4))

if __name__ == "__main__":
    # Ensure warnings or info level prints to console when running as CLI
    logging.basicConfig(level=logging.WARNING)
    main()
