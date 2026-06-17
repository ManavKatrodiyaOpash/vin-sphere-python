import os
import argparse
import logging
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor

from feature_engineering import load_and_preprocess_data, extract_features
from model_utils import save_model
from evaluate import evaluate_classification, evaluate_regression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Map target JSON keys to CSV column names and target types
TARGET_CONFIGS = {
    "make": {"col": "make", "type": "classification"},
    "model": {"col": "model", "type": "classification"},
    "trim": {"col": "trim", "type": "classification"},
    "body_type": {"col": "bodyType", "type": "classification"},
    "year": {"col": "year", "type": "regression"},  # Year predicted via regression
    "color": {"col": "color", "type": "classification"},
    "weight": {"col": "weightInKg", "type": "regression"},  # Weight predicted via regression
    "regional specs": {"col": "regionalSpec", "type": "classification"},
    "origin": {"col": "origin", "type": "classification"}
}

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    target_type: str,
    cat_features: list
) -> Any:
    """
    Trains a CatBoost model based on the target type (Classifier or Regressor).
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        target_name: Name of the target variable.
        target_type: Either 'classification' or 'regression'.
        cat_features: List of categorical feature names.
        
    Returns:
        The trained CatBoost model.
    """
    logger.info(f"Training CatBoost {target_type} model for '{target_name}'...")
    
    if target_type == "classification":
        # CatBoostClassifier for categorical targets
        model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.08,
            depth=6,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50
        )
    else:
        # CatBoostRegressor for numeric targets (year, weight)
        model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.08,
            depth=6,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50
        )
        
    # Fit model with categorical features specified
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_train, y_train),  # CatBoost uses this for early stopping
        verbose=100
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Japanese Import 10-Character Short Chassis Model Training Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/final_clean_10.csv",
        help="Path to the cleaned 10-length chassis CSV file."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="chat_cat_short_vin/models",
        help="Directory to save the trained models."
    )
    
    args = parser.parse_args()
    
    # Check data path
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset file does not exist at path: {args.data_path}")
        return
        
    # 1. Load and Preprocess Dataset
    try:
        df = load_and_preprocess_data(args.data_path)
    except Exception as e:
        logger.error(f"Dataset preprocessing failed: {e}")
        return
        
    # 2. Extract input features for the entire dataset
    # We do this once to avoid redundant computations
    X_all = extract_features(df["chassisNumber"])
    
    # Standardize types of features for CatBoost
    cat_features = [col for col in X_all.columns if col != "serial_number"]
    for col in cat_features:
        X_all[col] = X_all[col].astype(str)
        
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 3. Train models for each target
    for target_key, config in TARGET_CONFIGS.items():
        col_name = config["col"]
        target_type = config["type"]
        
        logger.info(f"Starting model building for: {target_key} (CSV column: {col_name})")
        
        # Rule 12: Ignore rows with missing target values
        valid_indices = df[col_name].dropna().index
        if len(valid_indices) == 0:
            logger.warning(f"No valid rows found for target '{target_key}'. Skipping.")
            continue
            
        X_target = X_all.loc[valid_indices]
        y_target = df.loc[valid_indices, col_name]
        
        # 4. Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # 5. Train Model
        try:
            model = train_model(X_train, y_train, target_key, target_type, cat_features)
        except Exception as e:
            logger.error(f"Failed to train model for '{target_key}': {e}")
            continue
            
        # 6. Evaluate Model
        logger.info(f"Evaluating model for '{target_key}' on test set (size: {len(X_test)})...")
        y_pred = model.predict(X_test)
        
        # Reshape predicted arrays if necessary
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
            
        if target_type == "classification":
            evaluate_classification(y_test, y_pred, target_key)
        else:
            evaluate_regression(y_test, y_pred, target_key)
            
        # 7. Save model using joblib
        # Normalize target name for filename
        safe_target_name = target_key.replace(" ", "_")
        model_filename = f"{safe_target_name}_model.joblib"
        model_path = os.path.join(args.model_dir, model_filename)
        
        try:
            save_model(model, model_path)
        except Exception as e:
            logger.error(f"Could not save model for '{target_key}': {e}")
            
    logger.info("Short chassis training pipeline completed successfully!")

if __name__ == "__main__":
    main()
