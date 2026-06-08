import os
import json
import logging
import pickle
import argparse
import pandas as pd
from typing import Dict, Any, Tuple

from feature_engineering import extract_vin_features
from model_utils import load_catboost_model

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Map target name to its specified feature subset
TARGET_FEATURE_MAPPING = {
    "MAKE": ["WMI"],
    "MODEL": ["WMI", "VDS", "YEAR_CODE"],
    "TRIM": ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],
    "BODY_TYPE": ["WMI", "VDS"],
    "ENGINE": ["WMI", "VDS", "YEAR_CODE"]
}

# Lazy loader for models and encoders to speed up batch predictions
_loaded_models = {}
_loaded_encoders = {}

def get_resources(model_dir: str = "models") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads models and encoders into memory, caching them for subsequent requests."""
    targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "ENGINE"]
    
    for target in targets:
        if target not in _loaded_encoders:
            # Load LabelEncoder first to check class counts
            encoder_path = os.path.join(model_dir, f"{target.lower()}_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder file for target '{target}' not found at: {encoder_path}")
            with open(encoder_path, "rb") as f:
                _loaded_encoders[target] = pickle.load(f)

        if target not in _loaded_models:
            le = _loaded_encoders[target]
            if len(le.classes_) <= 1:
                # If target has only 1 class, it's a constant. No CatBoost model is trained or needed.
                _loaded_models[target] = None
            else:
                # Load CatBoost model
                _loaded_models[target] = load_catboost_model(model_dir, target)
                
    return _loaded_models, _loaded_encoders

def decode_vin(vin: str, model_dir: str = "models") -> Dict[str, Any]:
    """Extracts features, runs inference, and returns predicted vehicle attributes with confidence scores."""
    vin_clean = vin.upper().strip()
    if len(vin_clean) != 17:
        raise ValueError(f"Invalid VIN length: {len(vin_clean)}. Expected exactly 17 characters.")
    
    # 1. Load models and encoders
    models, encoders = get_resources(model_dir)
    
    # 2. Extract features (create a single-row DataFrame)
    df_temp = pd.DataFrame({"VIN": [vin_clean]})
    X_features = extract_vin_features(df_temp, "VIN")
    
    predictions = {}
    confidences = []
    individual_confidences = {}
    
    # 3. Perform prediction for each model
    targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "ENGINE"]
    for target in targets:
        features = TARGET_FEATURE_MAPPING[target]
        model = models[target]
        le = encoders[target]
        
        if model is None:
            # Constant prediction
            label = le.classes_[0]
            confidence = 1.0
        else:
            # Prepare inputs
            X_target = X_features[features]
            
            # Get probability distribution
            proba = model.predict_proba(X_target)[0]
            
            # Extract highest probability index and class label
            class_idx = proba.argmax()
            label = le.inverse_transform([class_idx])[0]
            confidence = float(proba[class_idx])
        
        predictions[target.lower()] = label
        individual_confidences[target.lower()] = round(confidence, 4)
        confidences.append(confidence)
        
    # Calculate average confidence across all predictions
    avg_confidence = float(np.mean(confidences))
    
    # Format output matching specification
    output = {
        "vin": vin_clean,
        "make": predictions["make"],
        "model": predictions["model"],
        "trim": predictions["trim"],
        "body_type": predictions["body_type"],
        "engine": predictions["engine"],
        "confidence": round(avg_confidence, 4),
        "attribute_confidences": individual_confidences  # Pro-grade metadata
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="UAE VIN Intelligence System - Inference CLI")
    parser.add_argument("--vin", type=str, required=True, help="17-character vehicle VIN string")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where models and encoders are stored")
    
    args = parser.parse_args()
    
    try:
        result = decode_vin(args.vin, args.model_dir)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    import numpy as np  # Needed for np.mean
    main()
