import sys
from pathlib import Path
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

from chat_cat_short_vin_09.feature_engineering import normalize_chassis, extract_features
from chat_cat_short_vin_09.model_utils import load_model

# Configure logging
logger = logging.getLogger(__name__)

# Global cache to store loaded models to speed up batch predictions
_MODEL_CACHE: Dict[str, Any] = {}

# Target JSON keys mapped to expected safe target names used in filenames
TARGETS = ["make", "model", "trim", "body_type", "year", "color", "weight", "regional_specs", "origin"]

def get_model(target: str, model_dir: str = "chat_cat_short_vin_09/models") -> Any:
    """
    Loads and caches the model for a given target.
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

def predict_vehicle(chassis_number: str, model_dir: str = "chat_cat_short_vin_09/models") -> Dict[str, Any]:
    """
    Decodes a 9-character short chassis number using the unified VINDecoder pipeline.
    """
    pipeline_path = os.path.join(model_dir, "vin_decoder_pipeline.pkl")
    if not os.path.exists(pipeline_path):
        pipeline_path = os.path.join(_parent, "chat_cat_short_vin_09", "vin_decoder_pipeline.pkl")
        
    if os.path.exists(pipeline_path):
        try:
            decoder = load_model(pipeline_path)
            res = decoder.predict(chassis_number)
        except Exception as e:
            logger.warning(f"Failed to load pipeline: {e}. Falling back to manual load.")
            from chat_cat_short_vin_09.vin_decoder import VINDecoder
            decoder = VINDecoder(model_dir=model_dir)
            res = decoder.predict(chassis_number)
    else:
        from chat_cat_short_vin_09.vin_decoder import VINDecoder
        decoder = VINDecoder(model_dir=model_dir)
        res = decoder.predict(chassis_number)
        
    res_conf = res.get("confidence_scores", {})
    
    # Read cylinders and no_of_passengers from model prediction (trained models)
    # Fall back to nearest-neighbor lookup if the models haven't been trained yet
    cylinders_val = res.get("cylinders", "UNKNOWN")
    passengers_val = res.get("no_of_passengers", "UNKNOWN")
    
    cylinders_conf = float(res_conf.get("cylinders", 0.0)) if "cylinders" in res_conf else 0.0
    passengers_conf = float(res_conf.get("no_of_passengers", 0.0)) if "no_of_passengers" in res_conf else 0.0
    
    # Neighbor-based fallback when model hasn't been trained for these targets
    if cylinders_val in ("UNKNOWN", None, "") or passengers_val in ("UNKNOWN", None, ""):
        sim_engine = decoder.similarity_engines.get("make")
        if sim_engine and hasattr(sim_engine, "train_df") and sim_engine.train_df is not None:
            neighbors = sim_engine.find_nearest_neighbors(chassis_number, top_n=1)
            if neighbors:
                nb_chassis = neighbors[0][0]
                match_row = sim_engine.train_df[sim_engine.train_df["chassisNumber"] == nb_chassis]
                if not match_row.empty:
                    if cylinders_val in ("UNKNOWN", None, "") and "cylinders" in match_row.columns:
                        c_val = match_row["cylinders"].iloc[0]
                        if pd.notna(c_val) and str(c_val).strip() != "":
                            try:
                                cylinders_val = str(int(float(c_val)))
                            except ValueError:
                                cylinders_val = str(c_val)
                    if passengers_val in ("UNKNOWN", None, "") and "noOfPassengers" in match_row.columns:
                        p_val = match_row["noOfPassengers"].iloc[0]
                        if pd.notna(p_val) and str(p_val).strip() != "":
                            try:
                                passengers_val = str(int(float(p_val)))
                            except ValueError:
                                passengers_val = str(p_val)

    # Helper function to format all predicted attributes as clean strings
    def to_str_val(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "UNKNOWN"
        val_str = str(val).strip()
        if val_str.upper() in ["NAN", "NONE", "UNKNOWN", ""]:
            return "UNKNOWN"
        try:
            float_val = float(val_str)
            if float_val.is_integer():
                return str(int(float_val))
            return str(round(float_val, 2))
        except ValueError:
            pass
        return val_str
    
    # Construct exact requested output dictionary schema
    output = {
        "year": to_str_val(res.get("year")),
        "make": to_str_val(res.get("make")),
        "model": to_str_val(res.get("model")),
        "trim": to_str_val(res.get("trim")),
        "body_type": to_str_val(res.get("body_type")),
        "regional_spec": to_str_val(res.get("regional_spec")),
        "cylinders": to_str_val(cylinders_val),
        "origin": to_str_val(res.get("origin")),
        "no_of_passengers": to_str_val(passengers_val),
        "weight": to_str_val(res.get("weight")),
        "color": to_str_val(res.get("color")),
        "confidence_scores": {
            "make": round(float(res_conf.get("make", 0.0)), 2),
            "model": round(float(res_conf.get("model", 0.0)), 2),
            "year": round(float(res_conf.get("year", 0.0)), 2)
        }
    }
    
    return output

def explain_prediction(chassis_number: str, target: str = "make", model_dir: str = "chat_cat_short_vin_09/models") -> Dict[str, Any]:
    """
    Computes explainability insights for a given prediction target.
    """
    from difflib import SequenceMatcher
    import shap
    
    normalized = normalize_chassis(chassis_number)
    if not normalized:
        raise ValueError("Invalid chassis number provided.")
        
    from chat_cat_short_vin_09.vin_decoder import VINDecoder
    decoder = VINDecoder(model_dir=model_dir)
    
    # 1. Closest prefixes
    sim_engine = decoder.similarity_engines.get("make")
    closest_prefixes = []
    if sim_engine and hasattr(sim_engine, "train_chassis") and sim_engine.train_chassis:
        train_prefixes = sorted(list(set(c[:4] for c in sim_engine.train_chassis)))
        input_prefix = normalized[:4].upper()
        matches = []
        for train_pref in train_prefixes:
            score = SequenceMatcher(None, input_prefix, train_pref).ratio()
            matches.append({"prefix": train_pref, "similarity": round(score, 4)})
        matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        closest_prefixes = matches[:5]
        
    # 2. Local feature attributions
    X_in = extract_features(pd.Series([normalized]), prefix_stats=decoder.prefix_stats)
    
    predictions = {}
    res = decoder.predict(normalized)
    predictions["make"] = res["make"]
    predictions["model"] = res["model"]
    predictions["year"] = res["year"]
    
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
            if len(shap_vals.shape) == 3:
                class_shap = shap_vals[0, :, pred_idx]
            elif len(shap_vals.shape) == 2:
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
            
    for item in local_explanations:
        feat_name = item["feature"]
        item["raw_value"] = str(X_all[feat_name].iloc[0])
        
    return {
        "closest_prefixes": closest_prefixes,
        "feature_attributions": local_explanations[:10]
    }

def main():
    parser = argparse.ArgumentParser(description="9-Character Short Chassis Model Inference Tool")
    parser.add_argument(
        "--chassis",
        type=str,
        required=True,
        help="9-character short chassis number string to decode."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="chat_cat_short_vin_09/models",
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
    logging.basicConfig(level=logging.WARNING)
    main()
