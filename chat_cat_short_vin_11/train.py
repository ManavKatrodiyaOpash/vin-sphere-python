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
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_name: str,
    target_type: str,
    cat_features: list,
    task_type: str = "GPU",
    use_early_stopping: bool = True
) -> Any:
    """
    Trains a CatBoost model based on the target type (Classifier or Regressor).
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_val: Validation feature DataFrame for early stopping.
        y_val: Validation target Series for early stopping.
        target_name: Name of the target variable.
        target_type: Either 'classification' or 'regression'.
        cat_features: List of categorical feature names.
        task_type: Device to train on ('CPU' or 'GPU').
        use_early_stopping: Whether to use early stopping on the validation set.
        
    Returns:
        The trained CatBoost model.
    """
    # Determine iterations and learning rate dynamically based on target cardinality to speed up CPU training
    if target_type == "classification":
        num_classes = len(y_train.unique())
        if num_classes > 50:
            iterations = 150
            learning_rate = 0.15
        elif num_classes > 20:
            iterations = 250
            learning_rate = 0.10
        else:
            iterations = 500
            learning_rate = 0.08
    else:
        iterations = 800
        learning_rate = 0.05

    logger.info(f"Training CatBoost {target_type} model for '{target_name}' ({len(y_train.unique()) if target_type == 'classification' else 1} classes) on {task_type} with iterations={iterations}, lr={learning_rate}...")
    early_stopping = 50 if use_early_stopping else None
    
    try:
        if target_type == "classification":
            # CatBoostClassifier for categorical targets
            model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=6,
                random_seed=42,
                verbose=100,
                early_stopping_rounds=early_stopping,
                max_ctr_complexity=1,
                thread_count=4,
                boosting_type="Plain",
                task_type=task_type
            )
        else:
            # CatBoostRegressor for numeric targets (year, weight)
            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=6,
                random_seed=42,
                verbose=100,
                early_stopping_rounds=early_stopping,
                max_ctr_complexity=1,
                thread_count=4,
                boosting_type="Plain",
                task_type=task_type
            )
            
        # Fit model
        if use_early_stopping:
            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                verbose=100
            )
        else:
            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                verbose=100
            )
        return model
    except Exception as e:
        if task_type == "GPU":
            logger.warning(f"Failed to train on GPU for '{target_name}' due to error: {e}. Falling back to CPU...")
            if target_type == "classification":
                model = CatBoostClassifier(
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=6,
                    random_seed=42,
                    verbose=100,
                    early_stopping_rounds=early_stopping,
                    max_ctr_complexity=1,
                    thread_count=4,
                    boosting_type="Plain",
                    task_type="CPU"
                )
            else:
                model = CatBoostRegressor(
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=6,
                    random_seed=42,
                    verbose=100,
                    early_stopping_rounds=early_stopping,
                    max_ctr_complexity=1,
                    thread_count=4,
                    boosting_type="Plain",
                    task_type="CPU"
                )
            if use_early_stopping:
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    verbose=100
                )
            else:
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features,
                    verbose=100
                )
            return model
        else:
            raise e

def main():
    parser = argparse.ArgumentParser(description="Japanese Import 11-Character Short Chassis Model Training Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/final_clean_11.csv",
        help="Path to the cleaned 11-length chassis CSV file."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="chat_cat_short_vin_11/models",
        help="Directory to save the trained models."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU"],
        help="Device to train on ('CPU' or 'GPU'). Defaults to GPU and falls back to CPU if unavailable."
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
    
    # Standardize types of features for CatBoost (excluding numeric features)
    numeric_features = ["serial_number", "first_digit_idx", "last_letter_idx", "num_letters", "num_digits"]
    cat_features = [col for col in X_all.columns if col not in numeric_features]
    
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
        
        # Split train into train_fit and val for early stopping (proper validation split)
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # Class alignment for classification targets
        use_early_stopping = True
        if target_type == "classification":
            train_classes = set(y_train_fit)
            val_classes = set(y_val)
            missing_in_train = val_classes - train_classes
            if missing_in_train:
                # Find indices in validation set that have these missing classes
                val_indices_to_move = y_val[y_val.isin(missing_in_train)].index
                X_train_fit = pd.concat([X_train_fit, X_val.loc[val_indices_to_move]])
                y_train_fit = pd.concat([y_train_fit, y_val.loc[val_indices_to_move]])
                X_val = X_val.drop(val_indices_to_move)
                y_val = y_val.drop(val_indices_to_move)
            
            if len(y_val) == 0:
                X_val = X_train_fit
                y_val = y_train_fit
                use_early_stopping = False
        
        # 5. Train Model
        try:
            model = train_model(
                X_train_fit, y_train_fit,
                X_val, y_val,
                target_key, target_type,
                cat_features, task_type=args.device,
                use_early_stopping=use_early_stopping
            )
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
            
        # 7. Save model using pickle
        # Normalize target name for filename
        safe_target_name = target_key.replace(" ", "_")
        model_filename = f"{safe_target_name}_model.pkl"
        model_path = os.path.join(args.model_dir, model_filename)
        
        try:
            save_model(model, model_path)
        except Exception as e:
            logger.error(f"Could not save model for '{target_key}': {e}")
            
    logger.info("Short chassis training pipeline completed successfully!")

if __name__ == "__main__":
    main()
