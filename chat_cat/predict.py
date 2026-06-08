import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from feature_engineering import extract_vin_features
from model_utils import load_fallback_model

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Lazy loader for fallback model to speed up batch predictions
_fallback_model = None

def get_resources(model_dir: str = "models") -> dict:
    """Loads the fallback lookup model into memory, caching it for subsequent requests."""
    global _fallback_model
    if _fallback_model is None:
        _fallback_model = load_fallback_model(model_dir)
    return _fallback_model

def decode_vin(vin: str, model_dir: str = "models") -> Dict[str, Any]:
    """Extracts features, runs inference, and returns predicted vehicle attributes with confidence scores."""
    vin_clean = vin.upper().strip()
    if len(vin_clean) != 17:
        raise ValueError(f"Invalid VIN length: {len(vin_clean)}. Expected exactly 17 characters.")
    
    # 1. Load fallback model
    fallback_model = get_resources(model_dir)
    
    # 2. Extract features (create a single-row DataFrame)
    df_temp = pd.DataFrame({"VIN": [vin_clean]})
    X_features = extract_vin_features(df_temp, "VIN")
    row = X_features.iloc[0]
    
    predictions = {}
    confidences = []
    individual_confidences = {}
    
    # 3. Perform prediction for each target
    targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "YEAR", "CYLINDERS", "ORIGIN", "NO_OF_PASSENGERS", "WEIGHT", "REGIONAL_SPEC"]
    
    # Target hierarchies for backoff lookups
    BACKOFF_HIERARCHY = {
        "MAKE": [["WMI"]],
        "MODEL": [["WMI", "VDS", "YEAR_CODE"], ["WMI", "VDS"], ["WMI"]],
        "TRIM": [["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"], ["WMI", "VDS", "YEAR_CODE"], ["WMI", "VDS"], ["WMI"]],
        "BODY_TYPE": [["WMI", "VDS"], ["WMI"]],
        "YEAR": [["WMI", "VDS", "YEAR_CODE"], ["WMI", "YEAR_CODE"], ["YEAR_CODE"]],
        "CYLINDERS": [["WMI", "VDS", "YEAR_CODE"], ["WMI", "VDS"], ["WMI"]],
        "ORIGIN": [["WMI", "VDS"], ["WMI"]],
        "NO_OF_PASSENGERS": [["WMI", "VDS", "YEAR_CODE"], ["WMI", "VDS"], ["WMI"]],
        "WEIGHT": [["WMI", "VDS", "YEAR_CODE"], ["WMI", "VDS"], ["WMI"]],
        "REGIONAL_SPEC": [["WMI", "VDS", "YEAR_CODE"], ["WMI", "VDS"], ["WMI"]]
    }
    
    for target in targets:
        target_model = fallback_model[target]
        feature_levels = BACKOFF_HIERARCHY[target]
        
        label = None
        confidence = 0.0
        
        # Check hierarchy levels in order
        for level_idx, features in enumerate(feature_levels):
            key = tuple(row[f] for f in features)
            if len(key) == 1:
                key = key[0]
                
            lookup_dict = target_model[level_idx]
            if key in lookup_dict:
                label, confidence = lookup_dict[key]
                break
                
        if label is None:
            # Fall back to default
            label, confidence = target_model["_default"]
            
        predictions[target.lower()] = label
        individual_confidences[target.lower()] = round(confidence, 4)
        confidences.append(confidence)
        
    # Calculate average confidence across all predictions
    avg_confidence = float(np.mean(confidences))
    
    # Format output matching specification
    output = {
        "vin": vin_clean,
        "year": predictions["year"],
        "make": predictions["make"],
        "model": predictions["model"],
        "trim": predictions["trim"],
        "body_type": predictions["body_type"],
        "regional_spec": predictions["regional_spec"],
        "cylinders": predictions["cylinders"],
        "origin": predictions["origin"],
        "no_of_passengers": predictions["no_of_passengers"],
        "weight": predictions["weight"],
        "confidence": round(avg_confidence, 4),
        "attribute_confidences": individual_confidences
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="UAE VIN Intelligence System - Inference CLI")
    parser.add_argument("--vin", type=str, required=True, help="17-character vehicle VIN string")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where models are stored")
    
    args = parser.parse_args()
    
    try:
        result = decode_vin(args.vin, args.model_dir)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    main()
