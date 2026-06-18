import os
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

# Import classifiers and regressors
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from feature_engineering import load_and_preprocess_data, extract_features
from model_utils import RobustLabelEncoder, save_model, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Column mappings: Internal target name to CSV column name
COLUMN_MAPPING = {
    "make": "make",
    "model": "model",
    "year": "year",
    "trim": "trim",
    "body_type": "bodyType",
    "origin": "origin",
    "regional_specs": "regionalSpec",
    "color": "color",
    "weight": "weightInKg"
}

# Targets and their ML task type
TARGET_CONFIGS = {
    "make": "classification",
    "model": "classification",
    "year": "classification",          # Year as classification to yield confidence scores
    "trim": "classification",
    "body_type": "classification",
    "origin": "classification",
    "regional_specs": "classification",
    "color": "classification",
    "weight": "regression"              # Weight as regression
}

# The order in which hierarchical targets depend on each other
HIERARCHICAL_ORDER = ["make", "model", "year", "trim", "body_type", "origin", "regional_specs", "color", "weight"]

def instantiate_candidate_model(algo_name: str, target_type: str, num_classes: int = 2) -> Any:
    """
    Instantiates candidate classifiers or regressors with parameter tuning
    based on the class cardinality to optimize training time.
    """
    # Scale down estimators for high cardinality classes to speed up CPU training
    if num_classes > 100:
        n_est = 30
        cat_iter = 50
        max_depth = 5
    elif num_classes > 50:
        n_est = 50
        cat_iter = 80
        max_depth = 5
    else:
        n_est = 100
        cat_iter = 150
        max_depth = 6

    if target_type == "classification":
        if algo_name == "CatBoost":
            return CatBoostClassifier(
                iterations=cat_iter,
                learning_rate=0.1,
                depth=max_depth,
                random_seed=42,
                verbose=0,
                thread_count=4,
                allow_writing_files=False
            )
        elif algo_name == "LightGBM":
            return LGBMClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        elif algo_name == "XGBoost":
            return XGBClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss"
            )
        elif algo_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_depth + 4, # Slightly deeper for RF
                random_state=42,
                n_jobs=-1
            )
    else:
        # Regression
        if algo_name == "CatBoost":
            return CatBoostRegressor(
                iterations=150,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=0,
                thread_count=4,
                allow_writing_files=False
            )
        elif algo_name == "LightGBM":
            return LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        elif algo_name == "XGBoost":
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                eval_metric="rmse"
            )
        elif algo_name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
    raise ValueError(f"Unknown algorithm {algo_name} or type {target_type}")

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
    
    args = parser.parse_args()
    
    # 1. Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset file does not exist at: {args.data_path}")
        return
        
    df = load_and_preprocess_data(args.data_path)
    logger.info(f"Loaded dataset with {len(df)} rows.")
    
    # Clean the categorical columns
    for target_name, csv_col in COLUMN_MAPPING.items():
        if csv_col in df.columns:
            df[csv_col] = df[csv_col].astype(str).str.upper().str.strip()
            # Standardize missing labels
            df[csv_col] = df[csv_col].replace(["NAN", "NONE", "", "NAT", "UNDEFINED"], np.nan)
            
    # Extract chassis features once to avoid redundant computations
    X_chassis = extract_features(df["chassisNumber"])
    
    # Prefix-based disjoint splitting
    df["prefix5"] = df["chassisNumber"].astype(str).str[:5]
    unique_prefixes = list(df["prefix5"].unique())
    logger.info(f"Total unique 5-character prefixes: {len(unique_prefixes)}")
    
    train_prefixes, test_prefixes = train_test_split(unique_prefixes, test_size=0.2, random_state=42)
    train_prefixes = set(train_prefixes)
    test_prefixes = set(test_prefixes)
    logger.info(f"Split: {len(train_prefixes)} training prefixes, {len(test_prefixes)} testing prefixes.")
    
    # Save training prefixes and split metadata
    os.makedirs(args.model_dir, exist_ok=True)
    metadata = {
        "train_prefixes": list(train_prefixes),
        "test_prefixes": list(test_prefixes),
        "best_models": {}
    }
    
    # Store test predictions to check joint accuracy
    test_predictions_dict = {}
    
    # Fit target models sequentially following the hierarchical dependency order
    for target_name in HIERARCHICAL_ORDER:
        csv_col = COLUMN_MAPPING[target_name]
        target_type = TARGET_CONFIGS[target_name]
        
        logger.info(f"\n==================================================")
        logger.info(f" TRAINING TARGET: {target_name.upper()} ({target_type})")
        logger.info(f"==================================================")
        
        # Prepare valid rows for this target
        valid_mask = df[csv_col].notna()
        target_df = df[valid_mask].copy()
        target_X_chassis = X_chassis.loc[target_df.index].copy()
        
        if len(target_df) == 0:
            logger.warning(f"No valid rows found for target '{target_name}'. Skipping.")
            continue
            
        # Feature selection: append hierarchical dependencies if applicable
        X_features_raw = target_X_chassis.copy()
        if target_name == "model":
            X_features_raw["make"] = target_df["make"]
        elif target_name == "year":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
        elif target_name == "trim":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
            X_features_raw["year"] = target_df["year"].astype(str)
            
        # Train/Test prefix split masks
        tr_mask = target_df["prefix5"].isin(train_prefixes)
        te_mask = target_df["prefix5"].isin(test_prefixes)
        
        X_train_raw = X_features_raw[tr_mask].copy()
        X_test_raw = X_features_raw[te_mask].copy()
        y_train_raw = target_df.loc[tr_mask, csv_col].copy()
        y_test_raw = target_df.loc[te_mask, csv_col].copy()
        
        logger.info(f"Target '{target_name}' splits: train size = {len(X_train_raw)}, test size = {len(X_test_raw)}")
        if len(X_train_raw) == 0 or len(X_test_raw) == 0:
            logger.warning(f"Insufficient split sizes. Skipping.")
            continue
            
        # Encode categorical features
        cat_cols = [col for col in X_train_raw.columns if X_train_raw[col].dtype == object or isinstance(X_train_raw[col].iloc[0], str)]
        
        fe_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        for col in cat_cols:
            X_train_raw[col] = X_train_raw[col].astype(str)
            X_test_raw[col] = X_test_raw[col].astype(str)
            
        X_train_encoded = X_train_raw.copy()
        X_test_encoded = X_test_raw.copy()
        
        if len(cat_cols) > 0:
            X_train_encoded[cat_cols] = fe_encoder.fit_transform(X_train_raw[cat_cols])
            X_test_encoded[cat_cols] = fe_encoder.transform(X_test_raw[cat_cols])
            
        # Encode targets
        if target_type == "classification":
            target_encoder = RobustLabelEncoder()
            y_train_encoded = target_encoder.fit_transform(y_train_raw)
            y_test_encoded = target_encoder.transform(y_test_raw)
            num_classes = len(y_train_raw.unique())
        else:
            target_encoder = None
            y_train_encoded = y_train_raw.astype(float).values
            y_test_encoded = y_test_raw.astype(float).values
            num_classes = 1
            
        # Candidate models list
        candidate_algos = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
        cv_scores = {algo: [] for algo in candidate_algos}
        
        # 3-Fold GroupKFold Cross Validation on train prefixes
        gkf = GroupKFold(n_splits=3)
        groups = target_df.loc[tr_mask, "prefix5"].values
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_encoded, y_train_raw, groups=groups)):
            X_tr, X_val = X_train_encoded.iloc[tr_idx], X_train_encoded.iloc[val_idx]
            
            # Local encoding for classification to satisfy XGBoost contiguous label requirements
            if target_type == "classification":
                y_tr_raw = y_train_raw.iloc[tr_idx]
                y_val_raw = y_train_raw.iloc[val_idx]
                
                local_encoder = RobustLabelEncoder()
                y_tr = local_encoder.fit_transform(y_tr_raw)
                y_val = local_encoder.transform(y_val_raw)
                
                # Filter out classes in validation fold that are not in training fold
                fallback_idx = local_encoder.class_to_idx[local_encoder.fallback_value]
                val_keep_mask = (y_val != fallback_idx)
                
                if val_keep_mask.sum() > 0:
                    X_val_f = X_val[val_keep_mask]
                    y_val_f = y_val[val_keep_mask]
                else:
                    X_val_f = X_val
                    y_val_f = y_val
            else:
                y_tr = y_train_encoded[tr_idx]
                y_val_f = y_train_encoded[val_idx]
                X_val_f = X_val
                
            for algo in candidate_algos:
                try:
                    model = instantiate_candidate_model(algo, target_type, num_classes=num_classes)
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val_f)
                    
                    if target_type == "classification":
                        score = accuracy_score(y_val_f, preds)
                    else:
                        score = mean_absolute_error(y_val_f, preds)
                        
                    cv_scores[algo].append(score)
                except Exception as e:
                    logger.warning(f"Fold {fold} - Algorithm {algo} failed: {e}")
                    if target_type == "classification":
                        cv_scores[algo].append(0.0)
                    else:
                        cv_scores[algo].append(999999.0)
                        
        # Select best model based on CV performance
        best_algo = None
        if target_type == "classification":
            best_score = -1.0
            for algo, scores in cv_scores.items():
                avg_score = np.mean(scores)
                logger.info(f"Model {algo} CV Mean Accuracy: {avg_score:.4f}")
                if avg_score > best_score:
                    best_score = avg_score
                    best_algo = algo
            logger.info(f"--> Selected BEST Classifier: {best_algo} with CV Accuracy: {best_score:.4f}")
        else:
            best_score = 999999.0
            for algo, scores in cv_scores.items():
                avg_score = np.mean(scores)
                logger.info(f"Model {algo} CV Mean MAE: {avg_score:.4f}")
                if avg_score < best_score:
                    best_score = avg_score
                    best_algo = algo
            logger.info(f"--> Selected BEST Regressor: {best_algo} with CV MAE: {best_score:.4f}")
            
        # Fit selected best model on the entire training set
        best_model = instantiate_candidate_model(best_algo, target_type, num_classes=num_classes)
        best_model.fit(X_train_encoded, y_train_encoded)
        
        # Evaluate on the holdout test set (standalone)
        test_preds = best_model.predict(X_test_encoded)
        if target_type == "classification":
            test_acc = accuracy_score(y_test_encoded, test_preds)
            logger.info(f"Holdout Test set Accuracy (True features): {test_acc:.4f}")
            # Map predictions back to raw labels for joint/hierarchical checks
            raw_test_preds = target_encoder.inverse_transform(test_preds)
        else:
            test_mae = mean_absolute_error(y_test_encoded, test_preds)
            logger.info(f"Holdout Test set MAE (True features): {test_mae:.4f}")
            raw_test_preds = test_preds
            
        # Store predictions for sequential simulation
        test_predictions_dict[target_name] = pd.Series(raw_test_preds, index=y_test_raw.index)
        
        # Save model and encoders using pickle (.pkl)
        save_model(best_model, os.path.join(args.model_dir, f"{target_name}_model.pkl"))
        save_model(fe_encoder, os.path.join(args.model_dir, f"{target_name}_fe_encoder.pkl"))
        if target_encoder is not None:
            save_model(target_encoder, os.path.join(args.model_dir, f"{target_name}_label_encoder.pkl"))
            
        metadata["best_models"][target_name] = {
            "algorithm": best_algo,
            "cv_score": float(best_score),
            "test_score": float(test_acc) if target_type == "classification" else float(test_mae)
        }
        
    # ----------------------------------------------------
    # Evaluate Joint Hierarchical Chain on Holdout Test Set
    # ----------------------------------------------------
    logger.info(f"\n==================================================")
    logger.info(f" EVALUATING JOINT HIERARCHICAL CHAIN ON TEST SET")
    logger.info(f"==================================================")
    
    # We simulate prediction on the holdout test set *sequentially*
    # utilizing predicted values for Make, Model, and Year in features.
    hier_test_df = df[df["prefix5"].isin(test_prefixes)].copy()
    hier_X_chassis = X_chassis[df["prefix5"].isin(test_prefixes)].copy()
    
    # Sequential outputs
    pred_makes = []
    pred_models = []
    pred_years = []
    
    # 1. Make prediction
    make_fe_enc = load_model(os.path.join(args.model_dir, "make_fe_encoder.pkl"))
    make_model = load_model(os.path.join(args.model_dir, "make_model.pkl"))
    make_lbl = load_model(os.path.join(args.model_dir, "make_label_encoder.pkl"))
    
    X_make = hier_X_chassis.copy()
    X_make_enc = X_make.copy()
    cat_cols_make = [col for col in X_make.columns if X_make[col].dtype == object or isinstance(X_make[col].iloc[0], str)]
    for col in cat_cols_make:
        X_make[col] = X_make[col].astype(str)
        X_make_enc[col] = X_make_enc[col].astype(str)
    if len(cat_cols_make) > 0:
        X_make_enc[cat_cols_make] = make_fe_enc.transform(X_make[cat_cols_make])
    make_preds = make_lbl.inverse_transform(make_model.predict(X_make_enc))
    pred_makes = list(make_preds)
    
    # 2. Model prediction
    model_fe_enc = load_model(os.path.join(args.model_dir, "model_fe_encoder.pkl"))
    model_model = load_model(os.path.join(args.model_dir, "model_model.pkl"))
    model_lbl = load_model(os.path.join(args.model_dir, "model_label_encoder.pkl"))
    
    X_model = hier_X_chassis.copy()
    X_model["make"] = pred_makes
    X_model_enc = X_model.copy()
    cat_cols_model = [col for col in X_model.columns if X_model[col].dtype == object or isinstance(X_model[col].iloc[0], str)]
    for col in cat_cols_model:
        X_model[col] = X_model[col].astype(str)
        X_model_enc[col] = X_model_enc[col].astype(str)
    if len(cat_cols_model) > 0:
        X_model_enc[cat_cols_model] = model_fe_enc.transform(X_model[cat_cols_model])
    model_preds = model_lbl.inverse_transform(model_model.predict(X_model_enc))
    pred_models = list(model_preds)
    
    # 3. Year prediction
    year_fe_enc = load_model(os.path.join(args.model_dir, "year_fe_encoder.pkl"))
    year_model = load_model(os.path.join(args.model_dir, "year_model.pkl"))
    year_lbl = load_model(os.path.join(args.model_dir, "year_label_encoder.pkl"))
    
    X_year = hier_X_chassis.copy()
    X_year["make"] = pred_makes
    X_year["model"] = pred_models
    X_year_enc = X_year.copy()
    cat_cols_year = [col for col in X_year.columns if X_year[col].dtype == object or isinstance(X_year[col].iloc[0], str)]
    for col in cat_cols_year:
        X_year[col] = X_year[col].astype(str)
        X_year_enc[col] = X_year_enc[col].astype(str)
    if len(cat_cols_year) > 0:
        X_year_enc[cat_cols_year] = year_fe_enc.transform(X_year[cat_cols_year])
    year_preds = year_lbl.inverse_transform(year_model.predict(X_year_enc))
    pred_years = [str(y) for y in year_preds]
    
    # 4. Trim prediction
    trim_fe_enc = load_model(os.path.join(args.model_dir, "trim_fe_encoder.pkl"))
    trim_model = load_model(os.path.join(args.model_dir, "trim_model.pkl"))
    trim_lbl = load_model(os.path.join(args.model_dir, "trim_label_encoder.pkl"))
    
    X_trim = hier_X_chassis.copy()
    X_trim["make"] = pred_makes
    X_trim["model"] = pred_models
    X_trim["year"] = pred_years
    X_trim_enc = X_trim.copy()
    cat_cols_trim = [col for col in X_trim.columns if X_trim[col].dtype == object or isinstance(X_trim[col].iloc[0], str)]
    for col in cat_cols_trim:
        X_trim[col] = X_trim[col].astype(str)
        X_trim_enc[col] = X_trim_enc[col].astype(str)
    if len(cat_cols_trim) > 0:
        X_trim_enc[cat_cols_trim] = trim_fe_enc.transform(X_trim[cat_cols_trim])
    trim_preds = trim_lbl.inverse_transform(trim_model.predict(X_trim_enc))
    
    # Compute accuracy metrics on the joint predicted chain vs True values
    true_makes = [str(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "UNKNOWN" for x in hier_test_df["make"]]
    true_models = [str(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "UNKNOWN" for x in hier_test_df["model"]]
    true_years = [str(int(float(x))) if (x is not None and not (isinstance(x, float) and np.isnan(x)) and str(x) != "nan") else "UNKNOWN" for x in hier_test_df["year"]]
    true_trims = [str(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "UNKNOWN" for x in hier_test_df["trim"]]
    
    make_preds = [str(x) for x in make_preds]
    model_preds = [str(x) for x in model_preds]
    year_preds_cleaned = [str(int(float(y))) if y not in ['UNKNOWN', '0', 'nan'] else y for y in year_preds]
    trim_preds = [str(x) for x in trim_preds]
    
    logger.info(f"Joint Hierarchical Accuracy on Holdout Test Set:")
    
    # Make accuracy
    make_acc = accuracy_score(true_makes, make_preds)
    logger.info(f" - Make Accuracy: {make_acc:.4f}")
    
    # Model accuracy
    model_acc = accuracy_score(true_models, model_preds)
    logger.info(f" - Model Accuracy (Hierarchical): {model_acc:.4f}")
    
    # Year accuracy
    year_acc = accuracy_score(true_years, year_preds_cleaned)
    logger.info(f" - Year Accuracy (Hierarchical): {year_acc:.4f}")
    
    # Trim accuracy
    trim_acc = accuracy_score(true_trims, trim_preds)
    logger.info(f" - Trim Accuracy (Hierarchical): {trim_acc:.4f}")
    
    # Save metadata
    save_model(metadata, os.path.join(args.model_dir, "metadata.pkl"))
    logger.info("\nTraining pipeline successfully completed! Metadata and models saved.")

if __name__ == "__main__":
    main()
