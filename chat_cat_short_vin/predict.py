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
    Decodes a 10-character short chassis number using the unified VINDecoder pipeline.
    """
    pipeline_path = os.path.join(model_dir, "vin_decoder_pipeline.pkl")
    if not os.path.exists(pipeline_path):
        pipeline_path = "vin_decoder_pipeline.pkl"
        
    if os.path.exists(pipeline_path):
        try:
            decoder = load_model(pipeline_path)
            res = decoder.predict(chassis_number)
        except Exception as e:
            logger.warning(f"Failed to load pipeline: {e}. Falling back to manual load.")
            from vin_decoder import VINDecoder
            decoder = VINDecoder(model_dir=model_dir)
            res = decoder.predict(chassis_number)
    else:
        from vin_decoder import VINDecoder
        decoder = VINDecoder(model_dir=model_dir)
        res = decoder.predict(chassis_number)
        
    # Map predictions to output format for backward compatibility
    output = {
        "make": res.get("make", "UNKNOWN"),
        "model": res.get("model", "UNKNOWN"),
        "trim": res.get("trim", "UNKNOWN"),
        "body_type": res.get("body_type", "UNKNOWN"),
        "year": res.get("year", 0),
        "color": res.get("color", "UNKNOWN"),
        "weight": res.get("weight", 0.0),
        "regional specs": res.get("regional_specs", "UNKNOWN"),
        "origin": res.get("origin", "UNKNOWN"),
        "attribute_confidences": {
            "make": res.get("confidence", {}).get("make", 0.0),
            "model": res.get("confidence", {}).get("model", 0.0),
            "year": res.get("confidence", {}).get("year", 0.0),
            "trim": 1.0,
            "body_type": 1.0,
            "color": 1.0,
            "weight": 1.0,
            "regional specs": 1.0,
            "origin": 1.0
        }
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
