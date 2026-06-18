import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

from chat_cat_short_vin.feature_engineering import normalize_chassis, extract_features
from chat_cat_short_vin.model_utils import load_model

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
            from chat_cat_short_vin.vin_decoder import VINDecoder
            decoder = VINDecoder(model_dir=model_dir)
            res = decoder.predict(chassis_number)
    else:
        from chat_cat_short_vin.vin_decoder import VINDecoder
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

def explain_prediction(chassis_number: str, target: str = "make", model_dir: str = "chat_cat_short_vin/models") -> Dict[str, Any]:
    """
    Computes explainability insights for a given prediction target:
    1. Finds closest matching prefixes in the training set.
    2. Calculates local feature attributions using SHAP or feature importances.
    """
    from difflib import SequenceMatcher
    import shap
    
    normalized = normalize_chassis(chassis_number)
    if not normalized:
        raise ValueError("Invalid chassis number provided.")
        
    # Load the decoder (which contains all models and similarity engines)
    from chat_cat_short_vin.vin_decoder import VINDecoder
    decoder = VINDecoder(model_dir=model_dir)
    
    # 1. Closest prefixes
    sim_engine = decoder.similarity_engines.get("make")
    closest_prefixes = []
    if sim_engine and hasattr(sim_engine, "train_chassis") and sim_engine.train_chassis:
        train_prefixes = sorted(list(set(c[:5] for c in sim_engine.train_chassis)))
        input_prefix = normalized[:5].upper()
        matches = []
        for train_pref in train_prefixes:
            score = SequenceMatcher(None, input_prefix, train_pref).ratio()
            matches.append({"prefix": train_pref, "similarity": round(score, 4)})
        matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        closest_prefixes = matches[:5]
        
    # 2. Local feature attributions
    X_in = extract_features(pd.Series([normalized]))
    
    predictions = {}
    res = decoder.predict(normalized)
    predictions["make"] = res["make"]
    predictions["model"] = res["model"]
    predictions["year"] = res["year"]
    
    # Construct features for the target:
    X_in_target = X_in.copy()
    if target == "model":
        X_in_target["make"] = predictions["make"]
    elif target == "year":
        X_in_target["make"] = predictions["make"]
        X_in_target["model"] = predictions["model"]
    elif target == "trim":
        X_in_target["make"] = predictions["make"]
        X_in_target["model"] = predictions["model"]
        X_in_target["year"] = str(predictions["year"])
    elif target in ["body_type", "origin", "regional_specs"]:
        X_in_target["make"] = predictions["make"]
        X_in_target["model"] = predictions["model"]
        
    sim_eng = decoder.similarity_engines[target]
    X_sim = sim_eng.transform(pd.Series([normalized]))
    X_all = pd.concat([X_in_target.reset_index(drop=True), X_sim.reset_index(drop=True)], axis=1)
    X_enc = X_all.copy()
    cat_cols = [col for col in X_all.columns if X_all[col].dtype == object or isinstance(X_all[col].iloc[0], str)]
    for col in cat_cols:
        X_enc[col] = X_enc[col].astype(str)
    X_enc[cat_cols] = decoder.ordinal_encoders[target].transform(X_enc[cat_cols])
    
    model = decoder.models[target]
    feature_names = list(X_enc.columns)
    local_explanations = []
    
    # Get predicted class index
    pred_idx = model.predict(X_enc)[0]
    if isinstance(pred_idx, (np.ndarray, list)):
        pred_idx = int(pred_idx[0])
    else:
        pred_idx = int(pred_idx)
        
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_enc)
        
        class_shap = None
        if isinstance(shap_vals, list):
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
            local_explanations = sorted(local_explanations, key=lambda x: abs(x["shap_value"]), reverse=True)
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}. Falling back to model feature importances.")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for name, val in zip(feature_names, importances):
                local_explanations.append({"feature": name, "importance": float(val)})
            local_explanations = sorted(local_explanations, key=lambda x: x["importance"], reverse=True)
            
    # Include features' actual raw values for readability
    for item in local_explanations:
        feat_name = item["feature"]
        item["raw_value"] = str(X_all[feat_name].iloc[0])
        
    return {
        "closest_prefixes": closest_prefixes,
        "feature_attributions": local_explanations[:10] # Top 10 features
    }

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
