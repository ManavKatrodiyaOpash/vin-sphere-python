import os
import gc
from typing import Tuple
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import prepare_data
from model_utils import create_model, save_catboost_model
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

# Map target name to its specified feature subset
TARGET_FEATURE_MAPPING = {
    "MAKE": ["WMI"],
    "MODEL": ["WMI", "VDS", "YEAR_CODE"],
    "TRIM": ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],
    "BODY_TYPE": ["WMI", "VDS"],
    "ENGINE": ["WMI", "VDS", "YEAR_CODE"]
}

def split_and_align_classes(
    X: pd.DataFrame, 
    y_series: pd.Series, 
    test_size: float = 0.20
) -> Tuple[pd.Index, pd.Index]:
    """Helper to perform a stratified split with rare class safety and class alignment."""
    # 1. Stratification safety: group classes with only 1 member under a single bin
    class_counts = y_series.value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    y_stratify = y_series.copy()
    if len(rare_classes) > 0:
        y_stratify = y_stratify.replace(rare_classes, -1)
        
    # 2. Perform split on indexes
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

def main():
    parser = argparse.ArgumentParser(description="UAE VIN Intelligence System - CatBoost Training Pipeline")
    parser.add_argument("--data_path", type=str, default="../Data/data_methaq(2.0).csv",
                        help="Path to the raw CSV or Parquet dataset")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Output directory for saved models and encoders")
    parser.add_argument("--downsample", type=int, default=100000,
                        help="Downsample training set to this size to prevent OOM errors on 16GB RAM machines")
    parser.add_argument("--test_size", type=float, default=0.20,
                        help="Validation set fraction")
    
    args = parser.parse_args()
    
    # Check data path
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at: {args.data_path}")
        return
        
    logger.info("Starting VIN Intelligence System pipeline training...")
    
    # 1. Feature engineering and cleaning
    X, y, encoders = prepare_data(args.data_path, args.model_dir)
    
    # Save the standard column mappings and details in a json configuration file later
    targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "ENGINE"]
    
    for target in targets:
        logger.info(f"\n{'='*60}\nTraining Pipeline for Target: {target}\n{'='*60}")
        
        # Get target series
        y_target = y[f"{target}_enc"]
        num_classes = len(encoders[target].classes_)
        
        # Feature columns selection
        features = TARGET_FEATURE_MAPPING[target]
        logger.info(f"Target: {target} | Features selected: {features}")
        
        # Initial train/validation split
        idx_train, idx_val = split_and_align_classes(X, y_target, args.test_size)
        
        # 2. Downsampling training indices if necessary (to prevent OOM/bad allocation)
        if args.downsample > 0 and len(idx_train) > args.downsample:
            logger.info(f"Downsampling training set from {len(idx_train):,} to {args.downsample:,} for RAM safety...")
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
        
        # Slice datasets
        X_train, y_train = X.loc[idx_train, features], y_target.loc[idx_train].values
        X_val, y_val = X.loc[idx_val, features], y_target.loc[idx_val].values
        if num_classes <= 1:
            logger.info(f"Target '{target}' has only {num_classes} unique class. Skipping CatBoost model training (will use constant prediction).")
            continue
            
        # 3. Create model with customized parameters
        model = create_model(target, num_classes)
        
        # 4. Train model with early stopping
        logger.info(f"Fitting CatBoostClassifier for target '{target}'...")
        try:
            model.fit(
                X_train, y_train,
                cat_features=features,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
            
            # 5. Evaluate model
            preds = model.predict(X_val)
            # Flatten predictions
            preds = preds.ravel()
            evaluate_predictions(y_val, preds, target)
            
            # 6. Save model to registry
            save_catboost_model(model, args.model_dir, target)
            
        except Exception as e:
            logger.error(f"Failed to train target '{target}': {e}", exc_info=True)
            
        # Proactively clear RAM before the next iteration
        del model
        gc.collect()
        
    logger.info("CatBoost training pipeline completed successfully!")

if __name__ == "__main__":
    main()
