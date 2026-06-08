import os
import gc
from typing import Tuple, Dict, Any, List
import argparse
import logging
import pickle
import numpy as np
import pandas as pd

from feature_engineering import prepare_data
from model_utils import save_fallback_model
from evaluate import evaluate_predictions

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def split_and_align_classes(
    X: pd.DataFrame, 
    y_series: pd.Series, 
    test_size: float = 0.20
) -> Tuple[pd.Index, pd.Index]:
    """Helper to perform a stratified split with rare class safety and class alignment."""
    class_counts = y_series.value_counts()
    
    # If there's only 1 class or not enough classes, do a standard split without stratification
    if len(class_counts) <= 1:
        from sklearn.model_selection import train_test_split
        idx_train, idx_val = train_test_split(X.index, test_size=test_size, random_state=42)
        return idx_train, idx_val
        
    majority_class = class_counts.index[0]
    
    # 1. Stratification safety: group rare classes with only 1 member under the majority class
    rare_classes = class_counts[class_counts < 2].index.tolist()
    y_stratify = y_series.copy()
    if len(rare_classes) > 0:
        y_stratify = y_stratify.replace(rare_classes, majority_class)
        
    # Double check if any class in y_stratify still has less than 2 members
    class_counts_strat = y_stratify.value_counts()
    if (class_counts_strat < 2).any():
        from sklearn.model_selection import train_test_split
        idx_train, idx_val = train_test_split(X.index, test_size=test_size, random_state=42)
        return idx_train, idx_val
        
    # 2. Perform split on indexes
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(
        X.index, test_size=test_size, random_state=42,
        stratify=y_stratify
    )
    
    # 3. Class alignment: Ensure every class in validation exists in training
    idx_train_list = list(idx_train)
    idx_val_list = list(idx_val)
    
    train_classes = set(y_series.loc[idx_train_list])
    val_classes = set(y_series.loc[idx_val_list])
    missing_in_train = val_classes - train_classes
    
    if missing_in_train:
        val_rows = y_series.loc[idx_val_list]
        move_mask = val_rows.isin(missing_in_train)
        rows_to_move = val_rows[move_mask].index.tolist()
        
        idx_train_list.extend(rows_to_move)
        moved_set = set(rows_to_move)
        idx_val_list = [idx for idx in idx_val_list if idx not in moved_set]
        
    return pd.Index(idx_train_list), pd.Index(idx_val_list)

def predict_from_target_model(row: pd.Series, target_model: Dict[Any, Any], feature_levels: List[List[str]]) -> Tuple[str, float]:
    """Predicts a target class and confidence score for a single row using backoff hierarchy."""
    for level_idx, features in enumerate(feature_levels):
        key = tuple(row[f] for f in features)
        if len(key) == 1:
            key = key[0]
            
        lookup_dict = target_model[level_idx]
        if key in lookup_dict:
            return lookup_dict[key]  # returns (class_str, probability)
            
    # Return global default if not found at any level
    return target_model["_default"]

def main():
    parser = argparse.ArgumentParser(description="UAE VIN Intelligence System - Fallback Lookup Training Pipeline")
    parser.add_argument("--data_path", type=str, default="../Data/data_methaq(2.0).csv",
                        help="Path to the raw CSV or Parquet dataset")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Output directory for saved models and encoders")
    parser.add_argument("--downsample", type=int, default=-1,
                        help="Downsample training set to this size (negative for no downsampling)")
    parser.add_argument("--test_size", type=float, default=0.20,
                        help="Validation set fraction")
    
    args = parser.parse_args()
    
    # Check data path
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at: {args.data_path}")
        return
        
    logger.info("Starting VIN Intelligence System fallback pipeline training...")
    
    # 1. Feature engineering and cleaning
    X, y, encoders = prepare_data(args.data_path, args.model_dir)
    
    targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "YEAR", "CYLINDERS", "ORIGIN", "NO_OF_PASSENGERS", "WEIGHT", "REGIONAL_SPEC"]
    fallback_model = {}
    
    for target in targets:
        logger.info(f"\n{'='*60}\nBuilding Fallback Model for Target: {target}\n{'='*60}")
        
        # Get target series
        y_target = y[f"{target}_enc"]
        le = encoders[target]
        num_classes = len(le.classes_)
        
        # Hierarchy feature levels
        feature_levels = BACKOFF_HIERARCHY[target]
        logger.info(f"Target: {target} | Backoff hierarchy: {feature_levels}")
        
        # Initial train/validation split
        idx_train, idx_val = split_and_align_classes(X, y_target, args.test_size)
        
        # Downsampling if explicitly requested
        if args.downsample > 0 and len(idx_train) > args.downsample:
            logger.info(f"Downsampling training set from {len(idx_train):,} to {args.downsample:,}...")
            np.random.seed(42)
            idx_train_sampled = np.random.choice(idx_train, size=args.downsample, replace=False)
            
            # Re-align classes on the downsampled subset to make sure no validation label is missing
            idx_train_sampled_list = list(idx_train_sampled)
            idx_val_list = list(idx_val)
            
            train_classes = set(y_target.loc[idx_train_sampled_list])
            val_classes = set(y_target.loc[idx_val_list])
            missing = val_classes - train_classes
            
            if missing:
                val_rows = y_target.loc[idx_val_list]
                move_mask = val_rows.isin(missing)
                rows_to_move = val_rows[move_mask].index.tolist()
                
                idx_train_sampled_list.extend(rows_to_move)
                moved_set = set(rows_to_move)
                idx_val_list = [idx for idx in idx_val_list if idx not in moved_set]
                
            idx_train = pd.Index(idx_train_sampled_list)
            idx_val = pd.Index(idx_val_list)
            
        logger.info(f"Final training split size: {len(idx_train):,}")
        logger.info(f"Final validation split size: {len(idx_val):,}")
        
        # Slice training data
        X_train = X.loc[idx_train]
        y_train = y_target.loc[idx_train].values
        
        target_model = {}
        
        # Calculate global default
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if len(class_counts) > 0:
            best_idx = np.argmax(class_counts)
            global_default_enc = unique_classes[best_idx]
            global_prob = float(class_counts[best_idx] / len(y_train))
            global_default = le.inverse_transform([global_default_enc])[0]
        else:
            global_default = "UNKNOWN"
            global_prob = 1.0
            
        target_model["_default"] = (global_default, global_prob)
        
        # Build pandas dataframes for group count calculations
        train_df = X_train.copy()
        train_df["target_enc"] = y_train
        
        # Compile frequency dictionary for each hierarchy level
        for level_idx, features in enumerate(feature_levels):
            groupby_cols = features.copy()
            counts_df = train_df.groupby(groupby_cols + ["target_enc"]).size().reset_index(name="count")
            
            # Find class with highest count for each feature combination
            best_df = counts_df.sort_values(by=groupby_cols + ["count"], ascending=False)
            best_df = best_df.drop_duplicates(subset=groupby_cols)
            
            # Merge with total counts to compute probability/confidence
            total_counts = train_df.groupby(groupby_cols).size().reset_index(name="total_count")
            best_df = pd.merge(best_df, total_counts, on=groupby_cols)
            best_df["prob"] = best_df["count"] / best_df["total_count"]
            
            # Decode all best_df target_enc at once for extreme speedup
            best_df["class_str"] = le.inverse_transform(best_df["target_enc"].astype(int))
            
            # Convert to dictionary with string class keys
            lookup_dict = {}
            for _, row in best_df.iterrows():
                key = tuple(row[f] for f in groupby_cols)
                if len(key) == 1:
                    key = key[0]
                
                lookup_dict[key] = (row["class_str"], float(row["prob"]))
                
            target_model[level_idx] = lookup_dict
            
        fallback_model[target] = target_model
        
        # Evaluate model on the validation split
        logger.info(f"Evaluating fallback model on validation split (size={len(idx_val):,})...")
        X_val = X.loc[idx_val]
        y_val_enc = y_target.loc[idx_val].values
        y_val_true = le.inverse_transform(y_val_enc)
        
        # Convert X_val to records dict for fast iteration
        all_features_used = list(set(f for lvl in feature_levels for f in lvl))
        val_records = X_val[all_features_used].to_dict(orient="records")
        
        val_preds = []
        for val_row in val_records:
            pred_val, _ = predict_from_target_model(val_row, target_model, feature_levels)
            val_preds.append(pred_val)
            
        evaluate_predictions(y_val_true, val_preds, target)
        
        # Clear memory
        gc.collect()
        
    # Save the fallback model dict
    save_fallback_model(fallback_model, args.model_dir)
    logger.info("Fallback training pipeline completed successfully!")

if __name__ == "__main__":
    main()
