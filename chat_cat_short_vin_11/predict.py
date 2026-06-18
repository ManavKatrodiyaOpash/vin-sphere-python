import sys
from pathlib import Path
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

import os
import json
import logging
import argparse
from typing import Dict, Any, List
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import shap

from chat_cat_short_vin_11.feature_engineering import normalize_chassis, extract_features
from chat_cat_short_vin_11.model_utils import load_model

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for models and encoders
_CACHE: Dict[str, Any] = {}

# All targets in prediction sequence
TARGETS = ["make", "model", "year", "trim", "body_type", "origin", "regional_specs", "color", "weight"]

def load_cached_artifact(filename: str, model_dir: str = "chat_cat_short_vin_11/models") -> Any:
    """
    Loads and caches pickle files to optimize batch prediction speed.
    """
    global _CACHE
    filepath = os.path.join(model_dir, filename)
    if filepath not in _CACHE:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found: {filepath}")
        _CACHE[filepath] = load_model(filepath)
    return _CACHE[filepath]

def get_closest_prefixes(chassis_number: str, train_prefixes: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Finds the closest training prefixes to the input chassis using SequenceMatcher.
    """
    input_prefix = chassis_number[:5].upper()
    matches = []
    for train_pref in train_prefixes:
        score = SequenceMatcher(None, input_prefix, train_pref).ratio()
        matches.append({"prefix": train_pref, "similarity": round(score, 4)})
    # Sort by similarity score descending
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
    return matches[:top_n]

def encode_features_helper(X_raw: pd.DataFrame, fe_encoder: Any) -> pd.DataFrame:
    """
    Helper function to transform only categorical columns using OrdinalEncoder,
    matching the logic used during training.
    """
    X_encoded = X_raw.copy()
    cat_cols = [col for col in X_raw.columns if X_raw[col].dtype == object or isinstance(X_raw[col].iloc[0], str)]
    for col in cat_cols:
        X_encoded[col] = X_encoded[col].astype(str)
    if len(cat_cols) > 0:
        X_encoded[cat_cols] = fe_encoder.transform(X_encoded[cat_cols])
    return X_encoded

def predict_vehicle(chassis_number: str, model_dir: str = "chat_cat_short_vin_11/models") -> Dict[str, Any]:
    """
    Normalizes the input chassis number, extracts features, and uses the trained
    comparative classifiers sequentially to predict vehicle attributes with confidence scores.
    """
    normalized = normalize_chassis(chassis_number)
    if not normalized:
        raise ValueError("Invalid chassis number provided.")
        
    # Extract baseline chassis features
    df_input = pd.DataFrame({"chassisNumber": [normalized]})
    X_chassis = extract_features(df_input["chassisNumber"])
    
    # Dict to hold final decoded values and their confidences
    predictions = {}
    confidences = {}
    
    # 1. Predict Make
    try:
        make_fe = load_cached_artifact("make_fe_encoder.pkl", model_dir)
        make_model = load_cached_artifact("make_model.pkl", model_dir)
        make_lbl = load_cached_artifact("make_label_encoder.pkl", model_dir)
        
        X_make = encode_features_helper(X_chassis, make_fe)
        pred_idx = make_model.predict(X_make)[0]
        # In case the model returns a 1-element array
        if isinstance(pred_idx, (np.ndarray, list)):
            pred_idx = pred_idx[0]
            
        predictions["make"] = make_lbl.inverse_transform(pred_idx)
        
        # Compute confidence score
        if hasattr(make_model, "predict_proba"):
            probs = make_model.predict_proba(X_make)[0]
            confidences["make"] = float(np.max(probs))
        else:
            confidences["make"] = 1.0
    except Exception as e:
        logger.warning(f"Error predicting Make: {e}")
        predictions["make"] = "UNKNOWN"
        confidences["make"] = 0.0
        
    # 2. Predict Model (uses chassis features + predicted make)
    try:
        model_fe = load_cached_artifact("model_fe_encoder.pkl", model_dir)
        model_model = load_cached_artifact("model_model.pkl", model_dir)
        model_lbl = load_cached_artifact("model_label_encoder.pkl", model_dir)
        
        X_model_raw = X_chassis.copy()
        X_model_raw["make"] = str(predictions["make"])
        X_model_encoded = encode_features_helper(X_model_raw, model_fe)
        
        pred_idx = model_model.predict(X_model_encoded)[0]
        if isinstance(pred_idx, (np.ndarray, list)):
            pred_idx = pred_idx[0]
            
        predictions["model"] = model_lbl.inverse_transform(pred_idx)
        
        if hasattr(model_model, "predict_proba"):
            probs = model_model.predict_proba(X_model_encoded)[0]
            confidences["model"] = float(np.max(probs))
        else:
            confidences["model"] = 1.0
    except Exception as e:
        logger.warning(f"Error predicting Model: {e}")
        predictions["model"] = "UNKNOWN"
        confidences["model"] = 0.0
        
    # 3. Predict Year (uses chassis features + predicted make + predicted model)
    try:
        year_fe = load_cached_artifact("year_fe_encoder.pkl", model_dir)
        year_model = load_cached_artifact("year_model.pkl", model_dir)
        year_lbl = load_cached_artifact("year_label_encoder.pkl", model_dir)
        
        X_year_raw = X_chassis.copy()
        X_year_raw["make"] = str(predictions["make"])
        X_year_raw["model"] = str(predictions["model"])
        X_year_encoded = encode_features_helper(X_year_raw, year_fe)
        
        pred_idx = year_model.predict(X_year_encoded)[0]
        if isinstance(pred_idx, (np.ndarray, list)):
            pred_idx = pred_idx[0]
            
        raw_year = year_lbl.inverse_transform(pred_idx)
        try:
            predictions["year"] = int(float(raw_year))
        except ValueError:
            predictions["year"] = str(raw_year)
            
        if hasattr(year_model, "predict_proba"):
            probs = year_model.predict_proba(X_year_encoded)[0]
            confidences["year"] = float(np.max(probs))
        else:
            confidences["year"] = 1.0
    except Exception as e:
        logger.warning(f"Error predicting Year: {e}")
        predictions["year"] = 0
        confidences["year"] = 0.0
        
    # 4. Predict Trim (uses chassis features + predicted make + model + year)
    try:
        trim_fe = load_cached_artifact("trim_fe_encoder.pkl", model_dir)
        trim_model = load_cached_artifact("trim_model.pkl", model_dir)
        trim_lbl = load_cached_artifact("trim_label_encoder.pkl", model_dir)
        
        X_trim_raw = X_chassis.copy()
        X_trim_raw["make"] = str(predictions["make"])
        X_trim_raw["model"] = str(predictions["model"])
        X_trim_raw["year"] = str(predictions["year"])
        X_trim_encoded = encode_features_helper(X_trim_raw, trim_fe)
        
        pred_idx = trim_model.predict(X_trim_encoded)[0]
        if isinstance(pred_idx, (np.ndarray, list)):
            pred_idx = pred_idx[0]
            
        predictions["trim"] = trim_lbl.inverse_transform(pred_idx)
        
        if hasattr(trim_model, "predict_proba"):
            probs = trim_model.predict_proba(X_trim_encoded)[0]
            confidences["trim"] = float(np.max(probs))
        else:
            confidences["trim"] = 1.0
    except Exception as e:
        logger.warning(f"Error predicting Trim: {e}")
        predictions["trim"] = "UNKNOWN"
        confidences["trim"] = 0.0
        
    # 5. Predict remaining attributes (body_type, origin, regional_specs, color, weight)
    for target in ["body_type", "origin", "regional_specs", "color", "weight"]:
        try:
            fe_encoder = load_cached_artifact(f"{target}_fe_encoder.pkl", model_dir)
            model = load_cached_artifact(f"{target}_model.pkl", model_dir)
            
            X_encoded = encode_features_helper(X_chassis, fe_encoder)
            pred_val = model.predict(X_encoded)[0]
            if isinstance(pred_val, (np.ndarray, list)):
                pred_val = pred_val[0]
                
            if target == "weight":
                predictions["weight"] = float(np.round(pred_val, 2))
                confidences["weight"] = 1.0
            else:
                lbl_encoder = load_cached_artifact(f"{target}_label_encoder.pkl", model_dir)
                predictions[target] = lbl_encoder.inverse_transform(pred_val)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_encoded)[0]
                    confidences[target] = float(np.max(probs))
                else:
                    confidences[target] = 1.0
        except Exception as e:
            logger.warning(f"Error predicting {target}: {e}")
            if target == "weight":
                predictions["weight"] = 0.0
                confidences["weight"] = 0.0
            else:
                predictions[target] = "UNKNOWN"
                confidences[target] = 0.0
                
    # Build complete final output
    output = {
        "make": predictions["make"],
        "make_confidence": round(confidences["make"], 4),
        "model": predictions["model"],
        "model_confidence": round(confidences["model"], 4),
        "year": predictions["year"],
        "year_confidence": round(confidences["year"], 4),
        "trim": predictions["trim"],
        "trim_confidence": round(confidences["trim"], 4),
        "body_type": predictions["body_type"],
        "body_type_confidence": round(confidences["body_type"], 4),
        "origin": predictions["origin"],
        "origin_confidence": round(confidences["origin"], 4),
        "regional_specs": predictions["regional_specs"],
        "regional_specs_confidence": round(confidences["regional_specs"], 4),
        "color": predictions["color"],
        "color_confidence": round(confidences["color"], 4),
        "weight": predictions["weight"],
        "weight_confidence": round(confidences["weight"], 4)
    }
    return output

def explain_prediction(chassis_number: str, target: str = "make", model_dir: str = "chat_cat_short_vin_11/models") -> Dict[str, Any]:
    """
    Computes explainability insights for a given prediction target:
    1. Finds closest matching prefixes in the training set.
    2. Calculates local feature attributions using SHAP (with fallbacks).
    """
    normalized = normalize_chassis(chassis_number)
    if not normalized:
        raise ValueError("Invalid chassis number provided.")
        
    # Extract baseline chassis features
    df_input = pd.DataFrame({"chassisNumber": [normalized]})
    X_chassis = extract_features(df_input["chassisNumber"])
    
    # Load metadata for training prefix lookup
    metadata = load_cached_artifact("metadata.pkl", model_dir)
    train_prefixes = metadata.get("train_prefixes", [])
    
    # 1. Closest prefixes
    closest_prefixes = get_closest_prefixes(normalized, train_prefixes, top_n=5)
    
    # 2. Local feature attributions
    fe_encoder = load_cached_artifact(f"{target}_fe_encoder.pkl", model_dir)
    model = load_cached_artifact(f"{target}_model.pkl", model_dir)
    
    # Construct proper hierarchical feature inputs for the SHAP target
    X_raw = X_chassis.copy()
    if target == "model":
        # Predict make first
        make_res = predict_vehicle(normalized, model_dir)
        X_raw["make"] = make_res["make"]
    elif target == "year":
        make_res = predict_vehicle(normalized, model_dir)
        X_raw["make"] = make_res["make"]
        X_raw["model"] = make_res["model"]
    elif target == "trim":
        make_res = predict_vehicle(normalized, model_dir)
        X_raw["make"] = make_res["make"]
        X_raw["model"] = make_res["model"]
        X_raw["year"] = str(make_res["year"])
        
    X_encoded = encode_features_helper(X_raw, fe_encoder)
        
    feature_names = list(X_encoded.columns)
    local_explanations = []
    
    # Get predicted class index
    pred_idx = model.predict(X_encoded)[0]
    if isinstance(pred_idx, (np.ndarray, list)):
        pred_idx = int(pred_idx[0])
    else:
        pred_idx = int(pred_idx)
        
    try:
        # Construct tree explainer
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_encoded)
        
        # Resolve class-specific SHAP values
        class_shap = None
        if isinstance(shap_vals, list):
            # TreeExplainer on RandomForest/LGBM returns list of arrays
            class_shap = shap_vals[pred_idx]
            if len(class_shap.shape) > 1:
                class_shap = class_shap[0]
        elif isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) == 3: # (samples, features, classes) e.g., CatBoost
                class_shap = shap_vals[0, :, pred_idx]
            elif len(shap_vals.shape) == 2: # (samples, features)
                class_shap = shap_vals[0]
            else:
                class_shap = shap_vals
                
        if class_shap is not None:
            for name, val in zip(feature_names, class_shap):
                local_explanations.append({"feature": name, "shap_value": float(val)})
            # Sort by absolute impact
            local_explanations = sorted(local_explanations, key=lambda x: abs(x["shap_value"]), reverse=True)
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}. Falling back to model feature importances.")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for name, val in zip(feature_names, importances):
                local_explanations.append({"feature": name, "importance": float(val)})
            # Sort by importance
            local_explanations = sorted(local_explanations, key=lambda x: x["importance"], reverse=True)
            
    # Include features' actual raw values for readability
    for item in local_explanations:
        feat_name = item["feature"]
        item["raw_value"] = str(X_raw[feat_name].iloc[0])
        
    return {
        "closest_prefixes": closest_prefixes,
        "feature_attributions": local_explanations[:10] # Top 10 features
    }

def main():
    parser = argparse.ArgumentParser(description="Japanese Import 11-Character Short Chassis Model Inference Tool")
    parser.add_argument(
        "--chassis",
        type=str,
        required=True,
        help="11-character short chassis number string to decode."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="chat_cat_short_vin_11/models",
        help="Directory where models are saved."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Whether to generate explainability insights."
    )
    
    args = parser.parse_args()
    
    try:
        pred_res = predict_vehicle(args.chassis, args.model_dir)
        output = {"prediction": pred_res}
        
        if args.explain:
            exp_res = explain_prediction(args.chassis, "make", args.model_dir)
            output["explanation"] = exp_res
            
        print(json.dumps(output, indent=4))
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(json.dumps({"error": str(e)}, indent=4))

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
