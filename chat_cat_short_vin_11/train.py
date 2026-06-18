import sys
from pathlib import Path
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))
_self = Path(__file__).resolve().parent
if str(_self) not in sys.path:
    sys.path.insert(0, str(_self))

import os
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

# Import classifiers and regressors
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)

from chat_cat_short_vin_11.feature_engineering import load_and_preprocess_data, extract_features
from chat_cat_short_vin_11.model_utils import RobustLabelEncoder, save_model, load_model
from chat_cat_short_vin_11.prefix_similarity import PrefixSimilarityEngine

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
    "cylinders": "cylinders",
    "no_of_passengers": "noOfPassengers",
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
    "cylinders": "classification",      # Number of cylinders (e.g., 4, 6, 8)
    "no_of_passengers": "classification",  # Number of passengers (e.g., 5, 7, 8)
    "color": "classification",
    "weight": "regression"              # Weight as regression
}

# The order in which hierarchical targets depend on each other
HIERARCHICAL_ORDER = ["make", "model", "year", "trim", "body_type", "origin", "regional_specs", "cylinders", "no_of_passengers", "color", "weight"]

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
        cat_iter = 120
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
                max_depth=max_depth + 4,
                random_state=42,
                n_jobs=-1
            )
        elif algo_name == "ExtraTrees":
            return ExtraTreesClassifier(
                n_estimators=n_est,
                max_depth=max_depth + 4,
                random_state=42,
                n_jobs=-1
            )
    else:
        # Regression
        if algo_name == "CatBoost":
            return CatBoostRegressor(
                iterations=100,
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
        elif algo_name == "ExtraTrees":
            return ExtraTreesRegressor(
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
            
    # Extract baseline chassis features
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
    
    best_estimators = {}
    ordinal_encoders = {}
    target_encoders = {}
    similarity_engines = {}
    
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
        elif target_name == "body_type":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
        elif target_name == "origin":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
        elif target_name == "regional_specs":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
        elif target_name == "cylinders":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
            X_features_raw["body_type"] = target_df["bodyType"]
        elif target_name == "no_of_passengers":
            X_features_raw["make"] = target_df["make"]
            X_features_raw["model"] = target_df["model"]
            X_features_raw["body_type"] = target_df["bodyType"]
            
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
            
        # 1. Fit similarity engine on the training split only to prevent leakage
        sim_engine = PrefixSimilarityEngine()
        df_train_subset = target_df[tr_mask].copy()
        sim_engine.fit(df_train_subset)
        similarity_engines[target_name] = sim_engine
        
        # Transform features
        X_train_sim_feats = sim_engine.transform(df_train_subset["chassisNumber"])
        X_train_all = pd.concat([X_train_raw.reset_index(drop=True), X_train_sim_feats.reset_index(drop=True)], axis=1)
        
        df_test_subset = target_df[te_mask].copy()
        X_test_sim_feats = sim_engine.transform(df_test_subset["chassisNumber"])
        X_test_all = pd.concat([X_test_raw.reset_index(drop=True), X_test_sim_feats.reset_index(drop=True)], axis=1)
        
        # 2. Encode categorical features
        cat_cols = [col for col in X_train_all.columns if X_train_all[col].dtype == object or isinstance(X_train_all[col].iloc[0], str)]
        
        fe_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        for col in cat_cols:
            X_train_all[col] = X_train_all[col].astype(str)
            X_test_all[col] = X_test_all[col].astype(str)
            
        X_train_encoded = X_train_all.copy()
        X_test_encoded = X_test_all.copy()
        
        if len(cat_cols) > 0:
            X_train_encoded[cat_cols] = fe_encoder.fit_transform(X_train_all[cat_cols])
            X_test_encoded[cat_cols] = fe_encoder.transform(X_test_all[cat_cols])
            
        ordinal_encoders[target_name] = fe_encoder
        
        # 3. Encode targets
        if target_type == "classification":
            target_encoder = RobustLabelEncoder()
            y_train_encoded = target_encoder.fit_transform(y_train_raw)
            y_test_encoded = target_encoder.transform(y_test_raw)
            num_classes = len(y_train_raw.unique())
            target_encoders[target_name] = target_encoder
        else:
            target_encoder = None
            y_train_encoded = y_train_raw.astype(float).values
            y_test_encoded = y_test_raw.astype(float).values
            num_classes = 1
            
        # Candidate models list
        candidate_algos = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "ExtraTrees"]
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
        best_estimators[target_name] = best_model
        
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
        
    # Create the unified metadata dict for prefix statistics
    prefix_stats_unified = {}
    for length in range(2, 8):
        prefix_stats_unified[length] = similarity_engines["make"].prefix_stats[length]
        
    # Save statistics and metadata using joblib
    save_model(prefix_stats_unified, os.path.join(args.model_dir, "prefix_statistics.pkl"))
    save_model(metadata, os.path.join(args.model_dir, "feature_metadata.pkl"))
    
    # Save encoders.pkl mapping
    encoders_mapping = {
        "ordinal_encoders": ordinal_encoders,
        "target_encoders": target_encoders,
        "similarity_engines": similarity_engines
    }
    save_model(encoders_mapping, os.path.join(args.model_dir, "encoders.pkl"))
    
    # ----------------------------------------------------
    # Evaluate Joint Hierarchical Chain on Holdout Test Set
    # ----------------------------------------------------
    logger.info(f"\n==================================================")
    logger.info(f" EVALUATING JOINT HIERARCHICAL CHAIN ON TEST SET")
    logger.info(f"==================================================")
    
    # We simulate prediction on the holdout test set *sequentially*
    # utilizing predicted values for Make, Model, and Year in features.
    hier_test_df = df[df["prefix5"].isin(test_prefixes)].copy()
    
    # Sequential outputs
    pred_makes = []
    pred_models = []
    pred_years = []
    pred_trims = []
    pred_body_types = []
    pred_origins = []
    pred_regional_specs = []
    pred_cylinders = []
    pred_passengers = []
    
    # Decode one-by-one simulating deployment flow
    for idx, row in hier_test_df.iterrows():
        chassis = row["chassisNumber"]
        
        # 1. Make
        X_in = extract_features(pd.Series([chassis]))
        sim_eng_make = similarity_engines["make"]
        X_sim_make = sim_eng_make.transform(pd.Series([chassis]))
        X_all_make = pd.concat([X_in.reset_index(drop=True), X_sim_make.reset_index(drop=True)], axis=1)
        X_enc_make = X_all_make.copy()
        cat_cols_make = [col for col in X_all_make.columns if X_all_make[col].dtype == object or isinstance(X_all_make[col].iloc[0], str)]
        for col in cat_cols_make:
            X_enc_make[col] = X_enc_make[col].astype(str)
        X_enc_make[cat_cols_make] = ordinal_encoders["make"].transform(X_enc_make[cat_cols_make])
        make_pred_idx = best_estimators["make"].predict(X_enc_make)[0]
        if isinstance(make_pred_idx, (np.ndarray, list)):
            make_pred_idx = make_pred_idx[0]
        make_val = target_encoders["make"].inverse_transform(make_pred_idx)
        pred_makes.append(make_val)
        
        # 2. Model
        X_in_model = X_in.copy()
        X_in_model["make"] = make_val
        sim_eng_model = similarity_engines["model"]
        X_sim_model = sim_eng_model.transform(pd.Series([chassis]))
        X_all_model = pd.concat([X_in_model.reset_index(drop=True), X_sim_model.reset_index(drop=True)], axis=1)
        X_enc_model = X_all_model.copy()
        cat_cols_model = [col for col in X_all_model.columns if X_all_model[col].dtype == object or isinstance(X_all_model[col].iloc[0], str)]
        for col in cat_cols_model:
            X_enc_model[col] = X_enc_model[col].astype(str)
        X_enc_model[cat_cols_model] = ordinal_encoders["model"].transform(X_enc_model[cat_cols_model])
        model_pred_idx = best_estimators["model"].predict(X_enc_model)[0]
        if isinstance(model_pred_idx, (np.ndarray, list)):
            model_pred_idx = model_pred_idx[0]
        model_val = target_encoders["model"].inverse_transform(model_pred_idx)
        pred_models.append(model_val)
        
        # 3. Year
        X_in_year = X_in.copy()
        X_in_year["make"] = make_val
        X_in_year["model"] = model_val
        sim_eng_year = similarity_engines["year"]
        X_sim_year = sim_eng_year.transform(pd.Series([chassis]))
        X_all_year = pd.concat([X_in_year.reset_index(drop=True), X_sim_year.reset_index(drop=True)], axis=1)
        X_enc_year = X_all_year.copy()
        cat_cols_year = [col for col in X_all_year.columns if X_all_year[col].dtype == object or isinstance(X_all_year[col].iloc[0], str)]
        for col in cat_cols_year:
            X_enc_year[col] = X_enc_year[col].astype(str)
        X_enc_year[cat_cols_year] = ordinal_encoders["year"].transform(X_enc_year[cat_cols_year])
        year_pred_idx = best_estimators["year"].predict(X_enc_year)[0]
        if isinstance(year_pred_idx, (np.ndarray, list)):
            year_pred_idx = year_pred_idx[0]
        year_val = target_encoders["year"].inverse_transform(year_pred_idx)
        pred_years.append(year_val)
        
        # 4. Trim
        X_in_trim = X_in.copy()
        X_in_trim["make"] = make_val
        X_in_trim["model"] = model_val
        X_in_trim["year"] = str(year_val)
        sim_eng_trim = similarity_engines["trim"]
        X_sim_trim = sim_eng_trim.transform(pd.Series([chassis]))
        X_all_trim = pd.concat([X_in_trim.reset_index(drop=True), X_sim_trim.reset_index(drop=True)], axis=1)
        X_enc_trim = X_all_trim.copy()
        cat_cols_trim = [col for col in X_all_trim.columns if X_all_trim[col].dtype == object or isinstance(X_all_trim[col].iloc[0], str)]
        for col in cat_cols_trim:
            X_enc_trim[col] = X_enc_trim[col].astype(str)
        X_enc_trim[cat_cols_trim] = ordinal_encoders["trim"].transform(X_enc_trim[cat_cols_trim])
        trim_pred_idx = best_estimators["trim"].predict(X_enc_trim)[0]
        if isinstance(trim_pred_idx, (np.ndarray, list)):
            trim_pred_idx = trim_pred_idx[0]
        trim_val = target_encoders["trim"].inverse_transform(trim_pred_idx)
        pred_trims.append(trim_val)
        
        # 5. Body Type
        X_in_bt = X_in.copy()
        X_in_bt["make"] = make_val
        X_in_bt["model"] = model_val
        sim_eng_bt = similarity_engines["body_type"]
        X_sim_bt = sim_eng_bt.transform(pd.Series([chassis]))
        X_all_bt = pd.concat([X_in_bt.reset_index(drop=True), X_sim_bt.reset_index(drop=True)], axis=1)
        X_enc_bt = X_all_bt.copy()
        cat_cols_bt = [col for col in X_all_bt.columns if X_all_bt[col].dtype == object or isinstance(X_all_bt[col].iloc[0], str)]
        for col in cat_cols_bt:
            X_enc_bt[col] = X_enc_bt[col].astype(str)
        X_enc_bt[cat_cols_bt] = ordinal_encoders["body_type"].transform(X_enc_bt[cat_cols_bt])
        bt_pred_idx = best_estimators["body_type"].predict(X_enc_bt)[0]
        if isinstance(bt_pred_idx, (np.ndarray, list)):
            bt_pred_idx = bt_pred_idx[0]
        bt_val = target_encoders["body_type"].inverse_transform(bt_pred_idx)
        pred_body_types.append(bt_val)
        
        # 6. Origin
        X_in_ori = X_in.copy()
        X_in_ori["make"] = make_val
        X_in_ori["model"] = model_val
        sim_eng_ori = similarity_engines["origin"]
        X_sim_ori = sim_eng_ori.transform(pd.Series([chassis]))
        X_all_ori = pd.concat([X_in_ori.reset_index(drop=True), X_sim_ori.reset_index(drop=True)], axis=1)
        X_enc_ori = X_all_ori.copy()
        cat_cols_ori = [col for col in X_all_ori.columns if X_all_ori[col].dtype == object or isinstance(X_all_ori[col].iloc[0], str)]
        for col in cat_cols_ori:
            X_enc_ori[col] = X_enc_ori[col].astype(str)
        X_enc_ori[cat_cols_ori] = ordinal_encoders["origin"].transform(X_enc_ori[cat_cols_ori])
        ori_pred_idx = best_estimators["origin"].predict(X_enc_ori)[0]
        if isinstance(ori_pred_idx, (np.ndarray, list)):
            ori_pred_idx = ori_pred_idx[0]
        ori_val = target_encoders["origin"].inverse_transform(ori_pred_idx)
        pred_origins.append(ori_val)
        
        # 7. Regional Specs
        X_in_rs = X_in.copy()
        X_in_rs["make"] = make_val
        X_in_rs["model"] = model_val
        sim_eng_rs = similarity_engines["regional_specs"]
        X_sim_rs = sim_eng_rs.transform(pd.Series([chassis]))
        X_all_rs = pd.concat([X_in_rs.reset_index(drop=True), X_sim_rs.reset_index(drop=True)], axis=1)
        X_enc_rs = X_all_rs.copy()
        cat_cols_rs = [col for col in X_all_rs.columns if X_all_rs[col].dtype == object or isinstance(X_all_rs[col].iloc[0], str)]
        for col in cat_cols_rs:
            X_enc_rs[col] = X_enc_rs[col].astype(str)
        X_enc_rs[cat_cols_rs] = ordinal_encoders["regional_specs"].transform(X_enc_rs[cat_cols_rs])
        rs_pred_idx = best_estimators["regional_specs"].predict(X_enc_rs)[0]
        if isinstance(rs_pred_idx, (np.ndarray, list)):
            rs_pred_idx = rs_pred_idx[0]
        rs_val = target_encoders["regional_specs"].inverse_transform(rs_pred_idx)
        pred_regional_specs.append(rs_val)
        
        # 8. Cylinders
        if "cylinders" in best_estimators:
            X_in_cyl = X_in.copy()
            X_in_cyl["make"] = make_val
            X_in_cyl["model"] = model_val
            X_in_cyl["body_type"] = bt_val
            sim_eng_cyl = similarity_engines["cylinders"]
            X_sim_cyl = sim_eng_cyl.transform(pd.Series([chassis]))
            X_all_cyl = pd.concat([X_in_cyl.reset_index(drop=True), X_sim_cyl.reset_index(drop=True)], axis=1)
            X_enc_cyl = X_all_cyl.copy()
            cat_cols_cyl = [col for col in X_all_cyl.columns if X_all_cyl[col].dtype == object or isinstance(X_all_cyl[col].iloc[0], str)]
            for col in cat_cols_cyl:
                X_enc_cyl[col] = X_enc_cyl[col].astype(str)
            X_enc_cyl[cat_cols_cyl] = ordinal_encoders["cylinders"].transform(X_enc_cyl[cat_cols_cyl])
            cyl_pred_idx = best_estimators["cylinders"].predict(X_enc_cyl)[0]
            if isinstance(cyl_pred_idx, (np.ndarray, list)):
                cyl_pred_idx = cyl_pred_idx[0]
            cyl_val = target_encoders["cylinders"].inverse_transform(cyl_pred_idx)
            pred_cylinders.append(cyl_val)
            
        # 9. No of Passengers
        if "no_of_passengers" in best_estimators:
            X_in_pax = X_in.copy()
            X_in_pax["make"] = make_val
            X_in_pax["model"] = model_val
            X_in_pax["body_type"] = bt_val
            sim_eng_pax = similarity_engines["no_of_passengers"]
            X_sim_pax = sim_eng_pax.transform(pd.Series([chassis]))
            X_all_pax = pd.concat([X_in_pax.reset_index(drop=True), X_sim_pax.reset_index(drop=True)], axis=1)
            X_enc_pax = X_all_pax.copy()
            cat_cols_pax = [col for col in X_all_pax.columns if X_all_pax[col].dtype == object or isinstance(X_all_pax[col].iloc[0], str)]
            for col in cat_cols_pax:
                X_enc_pax[col] = X_enc_pax[col].astype(str)
            X_enc_pax[cat_cols_pax] = ordinal_encoders["no_of_passengers"].transform(X_enc_pax[cat_cols_pax])
            pax_pred_idx = best_estimators["no_of_passengers"].predict(X_enc_pax)[0]
            if isinstance(pax_pred_idx, (np.ndarray, list)):
                pax_pred_idx = pax_pred_idx[0]
            pax_val = target_encoders["no_of_passengers"].inverse_transform(pax_pred_idx)
            pred_passengers.append(pax_val)
            
    # Compute accuracy metrics on the joint predicted chain vs True values
    true_makes = [str(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "UNKNOWN" for x in hier_test_df["make"]]
    true_models = [str(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "UNKNOWN" for x in hier_test_df["model"]]
    true_years = [str(int(float(x))) if (x is not None and not (isinstance(x, float) and np.isnan(x)) and str(x) != "nan") else "UNKNOWN" for x in hier_test_df["year"]]
    true_trims = [str(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "UNKNOWN" for x in hier_test_df["trim"]]
    true_body_types = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in hier_test_df["bodyType"]]
    true_origins = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in hier_test_df["origin"]]
    true_regional_specs = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in hier_test_df["regionalSpec"]]
    true_cylinders = [str(int(float(x))) if pd.notna(x) and str(x).strip() not in ["", "nan"] else "UNKNOWN" for x in hier_test_df["cylinders"]]
    true_passengers = [str(int(float(x))) if pd.notna(x) and str(x).strip() not in ["", "nan"] else "UNKNOWN" for x in hier_test_df["noOfPassengers"]]
    
    make_preds = [str(x) for x in pred_makes]
    model_preds = [str(x) for x in pred_models]
    year_preds_cleaned = [str(int(float(y))) if y not in ['UNKNOWN', '0', 'nan'] else y for y in pred_years]
    trim_preds = [str(x) for x in pred_trims]
    
    logger.info(f"Joint Hierarchical Accuracy on Holdout Test Set:")
    logger.info(f" - Make Accuracy: {accuracy_score(true_makes, make_preds):.4f}")
    logger.info(f" - Model Accuracy (Hierarchical): {accuracy_score(true_models, model_preds):.4f}")
    logger.info(f" - Year Accuracy (Hierarchical): {accuracy_score(true_years, year_preds_cleaned):.4f}")
    logger.info(f" - Trim Accuracy (Hierarchical): {accuracy_score(true_trims, trim_preds):.4f}")
    logger.info(f" - Body Type Accuracy: {accuracy_score(true_body_types, pred_body_types):.4f}")
    logger.info(f" - Origin Accuracy: {accuracy_score(true_origins, pred_origins):.4f}")
    logger.info(f" - Regional Specs Accuracy: {accuracy_score(true_regional_specs, pred_regional_specs):.4f}")
    if pred_cylinders:
        logger.info(f" - Cylinders Accuracy: {accuracy_score(true_cylinders, pred_cylinders):.4f}")
    if pred_passengers:
        logger.info(f" - No of Passengers Accuracy: {accuracy_score(true_passengers, pred_passengers):.4f}")
        
    # Instantiate and save the unified pipeline
    from chat_cat_short_vin_11.vin_decoder import VINDecoder
    decoder_pipeline = VINDecoder(model_dir=args.model_dir)
    save_model(decoder_pipeline, os.path.join(args.model_dir, "vin_decoder_pipeline.pkl"))
    save_model(decoder_pipeline, "chat_cat_short_vin_11/vin_decoder_pipeline.pkl")
    logger.info("Saved unified vin_decoder_pipeline.pkl successfully!")

if __name__ == "__main__":
    main()
