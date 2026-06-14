import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODEL_DIR, REPORTS_DIR, ATTENTION_DIR,
    CLASSIFICATION_TARGETS, REGRESSION_TARGETS
)
from src.tokenizer import VINTokenizer
from src.preprocess import normalize_vin, validate_vin, VINDataPipeline
from src.model import VINTransformerEncoder

def load_inference_pipeline():
    """Loads preprocessing pipeline, Transformer model, and CatBoost models from disk."""
    pipeline_path = os.path.join(MODEL_DIR, "data_pipeline.pkl")
    transformer_path = os.path.join(MODEL_DIR, "transformer_best.pt")
    catboost_path = os.path.join(MODEL_DIR, "catboost.pkl")
    
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}. Run training first.")
        
    pipeline = VINDataPipeline.load(pipeline_path)
    
    # Instantiate Transformer
    vocab_size = pipeline.tokenizer.vocab_size
    target_classes_dict = {col: encoder.num_classes() for col, encoder in pipeline.encoders.items()}
    
    model = VINTransformerEncoder(
        vocab_size=vocab_size,
        target_classes_dict=target_classes_dict,
        regression_targets_list=REGRESSION_TARGETS
    )
    
    if os.path.exists(transformer_path):
        model.load_state_dict(torch.load(transformer_path, map_location=torch.device("cpu")))
    model.eval()
    
    # Load CatBoost
    catboost_models = None
    if os.path.exists(catboost_path):
        with open(catboost_path, "rb") as f:
            catboost_models = pickle.load(f)
            
    # Load signal warnings to flag weak predictions
    weak_targets = []
    warning_path = os.path.join(REPORTS_DIR, "target_signal_analysis_report.md")
    if os.path.exists(warning_path):
        with open(warning_path, "r") as f:
            content = f.read()
        # Find targets listed as WEAK
        matches = re.findall(r"`(\w+)`\s*\|\s*\w+\s*\|\s*[\d,]+\s*\|\s*[\d.]+\%\s*\|\s*[\d.]+\s*\|\s*Position\s*\d+\s*\|\s*\*\*WEAK\*\*", content)
        weak_targets = matches
        
    return pipeline, model, catboost_models, weak_targets


def generate_attention_map(vin, attn_weights, save_filename=None):
    """Generates and saves a heatmap of attention weights over the 17 character positions."""
    os.makedirs(ATTENTION_DIR, exist_ok=True)
    
    # Convert weights to numpy array
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
        
    # Make sure weights are 1D vector of length 17
    weights = np.squeeze(attn_weights)
    
    # Format labels (e.g. 'J (Pos 1)')
    labels = [f"{vin[i]}\n(Pos {i+1})" for i in range(17)]
    
    plt.figure(figsize=(12, 3))
    sns.heatmap(
        [weights],
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=False,
        cbar=False
    )
    plt.title(f"Transformer Position Attention Weights for VIN: {vin}")
    plt.tight_layout()
    
    if save_filename:
        save_path = os.path.join(ATTENTION_DIR, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
    
    # Return figure
    fig = plt.gcf()
    return fig


def predict_single(vin, pipeline, model, catboost_models=None, weak_targets=None, ensemble_weight=0.7):
    """Predicts all attributes for a single VIN string."""
    if weak_targets is None:
        weak_targets = []
        
    cleaned_vin = normalize_vin(vin)
    if not validate_vin(cleaned_vin):
        return {"error": "Invalid VIN string. Must be 17 characters and exclude I, O, Q."}
        
    # Tokenize
    tokens = pipeline.tokenizer.encode(cleaned_vin)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    
    # Model forward pass
    with torch.no_grad():
        outputs = model(tokens_tensor)
        
    attn_weights = outputs["attention_weights"].squeeze(0).cpu().numpy()
    
    results = {}
    
    # Tabular input for CatBoost
    cat_df = pd.DataFrame([list(cleaned_vin)], columns=[f"pos_{i}" for i in range(17)]).astype(str)
    
    # 1. Process Classification targets
    for col in CLASSIFICATION_TARGETS:
        # Transformer probabilities
        trans_logits = outputs[col].squeeze(0)
        trans_probs = F.softmax(trans_logits, dim=-1).cpu().numpy()
        
        # Ensembling if CatBoost is available
        if catboost_models is not None and col in catboost_models:
            cat_probs = catboost_models[col].predict_proba(cat_df)[0]
            # Blend
            probs = (ensemble_weight * trans_probs) + ((1.0 - ensemble_weight) * cat_probs)
        else:
            probs = trans_probs
            
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx] * 100)
        
        # Mapping to string prediction
        pred_label = pipeline.encoders[col].inverse_transform(int(pred_idx))
        
        # Apply confidence threshold
        if confidence < 80.0:
            pred_label = "UNKNOWN"
            
        results[col] = {
            "prediction": pred_label,
            "confidence": round(confidence, 1)
        }
        
        # Check signal strength warnings
        if col in weak_targets:
            results[col]["warning"] = "Low correlation signal detected inside VIN."

    # 2. Process Regression targets
    for col in REGRESSION_TARGETS:
        trans_pred = outputs[col].item()
        
        # Ensemble with CatBoost
        if catboost_models is not None and col in catboost_models:
            cat_pred = catboost_models[col].predict(cat_df)[0]
            pred_val = (ensemble_weight * trans_pred) + ((1.0 - ensemble_weight) * cat_pred)
        else:
            pred_val = trans_pred
            
        # Inverse transform regression scaling
        pred_unscaled = float(pipeline.scalers[col].inverse_transform(pred_val))
        
        # Post-process for physical sanity
        if col == "noOfPassengers":
            pred_unscaled = int(max(1, round(pred_unscaled)))
        else: # weightInKg
            pred_unscaled = float(max(100.0, round(pred_unscaled, 1)))
            
        results[col] = {
            "prediction": pred_unscaled
        }
        
        if col in weak_targets:
            results[col]["warning"] = "Low correlation signal detected inside VIN."
            
    # Include attention map figure/weights
    results["attention_weights"] = attn_weights.tolist()
    results["cleaned_vin"] = cleaned_vin
    
    return results


def predict_batch(df_input, pipeline, model, catboost_models=None, weak_targets=None, ensemble_weight=0.7):
    """Processes a batch of VINs from a DataFrame, appending all predictions."""
    if "chassisNumber" not in df_input.columns:
        raise KeyError("Uploaded CSV must contain 'chassisNumber' column.")
        
    df_output = df_input.copy()
    
    # Initialize output prediction lists
    predictions = {col: [] for col in CLASSIFICATION_TARGETS + REGRESSION_TARGETS}
    
    # Process row by row for simpler ensembling and sanity logic
    for idx, row in df_input.iterrows():
        vin = row["chassisNumber"]
        pred_res = predict_single(vin, pipeline, model, catboost_models, weak_targets, ensemble_weight)
        
        if "error" in pred_res:
            for col in CLASSIFICATION_TARGETS + REGRESSION_TARGETS:
                predictions[col].append("INVALID_VIN" if col in CLASSIFICATION_TARGETS else np.nan)
        else:
            for col in CLASSIFICATION_TARGETS + REGRESSION_TARGETS:
                predictions[col].append(pred_res[col]["prediction"])
                
    # Add predicted columns to output dataframe
    for col in CLASSIFICATION_TARGETS + REGRESSION_TARGETS:
        df_output[f"pred_{col}"] = predictions[col]
        
    return df_output
