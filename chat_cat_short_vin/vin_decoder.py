import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from feature_engineering import normalize_chassis, extract_features

class VINDecoder:
    def __init__(self, model_dir="chat_cat_short_vin/models"):
        self.model_dir = model_dir
        self.models = {}
        self.ordinal_encoders = {}
        self.target_encoders = {}
        self.similarity_engines = {}
        self.prefix_stats = {}
        
        # Try to load attributes if the files exist
        prefix_stats_path = os.path.join(model_dir, "prefix_statistics.pkl")
        encoders_path = os.path.join(model_dir, "encoders.pkl")
        
        if os.path.exists(prefix_stats_path):
            self.prefix_stats = joblib.load(prefix_stats_path)
            
        if os.path.exists(encoders_path):
            encoders_map = joblib.load(encoders_path)
            self.ordinal_encoders = encoders_map.get("ordinal_encoders", {})
            self.target_encoders = encoders_map.get("target_encoders", {})
            self.similarity_engines = encoders_map.get("similarity_engines", {})
            
            targets = ["make", "model", "trim", "body_type", "year", "origin", "regional_specs", "color", "weight"]
            for t in targets:
                safe_name = t.replace(" ", "_")
                model_path = os.path.join(model_dir, f"{safe_name}_model.pkl")
                if os.path.exists(model_path):
                    self.models[t] = joblib.load(model_path)

    def predict(self, chassis_number: str) -> dict:
        normalized = normalize_chassis(chassis_number)
        if not normalized or len(normalized) != 10:
            raise ValueError("Chassis number must be exactly 10 characters.")
            
        # 1. Base Feature Extraction
        X_in = extract_features(pd.Series([normalized]))
        
        # Predict targets sequentially following the hierarchical chain
        predictions = {}
        confidences = {}
        
        # 1. Make
        sim_eng_make = self.similarity_engines["make"]
        X_sim_make = sim_eng_make.transform(pd.Series([normalized]))
        X_all_make = pd.concat([X_in.reset_index(drop=True), X_sim_make.reset_index(drop=True)], axis=1)
        X_enc_make = X_all_make.copy()
        cat_cols_make = [col for col in X_all_make.columns if X_all_make[col].dtype == object or isinstance(X_all_make[col].iloc[0], str)]
        for col in cat_cols_make:
            X_enc_make[col] = X_enc_make[col].astype(str)
        X_enc_make[cat_cols_make] = self.ordinal_encoders["make"].transform(X_enc_make[cat_cols_make])
        
        make_pred_idx = self.models["make"].predict(X_enc_make)[0]
        if isinstance(make_pred_idx, (np.ndarray, list)):
            make_pred_idx = make_pred_idx[0]
        make_val = self.target_encoders["make"].inverse_transform(make_pred_idx)
        predictions["make"] = str(make_val)
        
        if hasattr(self.models["make"], "predict_proba"):
            confidences["make"] = float(np.max(self.models["make"].predict_proba(X_enc_make)[0]))
        else:
            confidences["make"] = 1.0
            
        # 2. Model
        X_in_model = X_in.copy()
        X_in_model["make"] = predictions["make"]
        sim_eng_model = self.similarity_engines["model"]
        X_sim_model = sim_eng_model.transform(pd.Series([normalized]))
        X_all_model = pd.concat([X_in_model.reset_index(drop=True), X_sim_model.reset_index(drop=True)], axis=1)
        X_enc_model = X_all_model.copy()
        cat_cols_model = [col for col in X_all_model.columns if X_all_model[col].dtype == object or isinstance(X_all_model[col].iloc[0], str)]
        for col in cat_cols_model:
            X_enc_model[col] = X_enc_model[col].astype(str)
        X_enc_model[cat_cols_model] = self.ordinal_encoders["model"].transform(X_enc_model[cat_cols_model])
        
        model_pred_idx = self.models["model"].predict(X_enc_model)[0]
        if isinstance(model_pred_idx, (np.ndarray, list)):
            model_pred_idx = model_pred_idx[0]
        model_val = self.target_encoders["model"].inverse_transform(model_pred_idx)
        predictions["model"] = str(model_val)
        
        if hasattr(self.models["model"], "predict_proba"):
            confidences["model"] = float(np.max(self.models["model"].predict_proba(X_enc_model)[0]))
        else:
            confidences["model"] = 1.0
            
        # 3. Year
        X_in_year = X_in.copy()
        X_in_year["make"] = predictions["make"]
        X_in_year["model"] = predictions["model"]
        sim_eng_year = self.similarity_engines["year"]
        X_sim_year = sim_eng_year.transform(pd.Series([normalized]))
        X_all_year = pd.concat([X_in_year.reset_index(drop=True), X_sim_year.reset_index(drop=True)], axis=1)
        X_enc_year = X_all_year.copy()
        cat_cols_year = [col for col in X_all_year.columns if X_all_year[col].dtype == object or isinstance(X_all_year[col].iloc[0], str)]
        for col in cat_cols_year:
            X_enc_year[col] = X_enc_year[col].astype(str)
        X_enc_year[cat_cols_year] = self.ordinal_encoders["year"].transform(X_enc_year[cat_cols_year])
        
        year_pred_idx = self.models["year"].predict(X_enc_year)[0]
        if isinstance(year_pred_idx, (np.ndarray, list)):
            year_pred_idx = year_pred_idx[0]
        year_val = self.target_encoders["year"].inverse_transform(year_pred_idx)
        try:
            predictions["year"] = int(float(year_val))
        except ValueError:
            predictions["year"] = str(year_val)
            
        if hasattr(self.models["year"], "predict_proba"):
            confidences["year"] = float(np.max(self.models["year"].predict_proba(X_enc_year)[0]))
        else:
            confidences["year"] = 1.0
            
        # 4. Trim
        X_in_trim = X_in.copy()
        X_in_trim["make"] = predictions["make"]
        X_in_trim["model"] = predictions["model"]
        X_in_trim["year"] = str(predictions["year"])
        sim_eng_trim = self.similarity_engines["trim"]
        X_sim_trim = sim_eng_trim.transform(pd.Series([normalized]))
        X_all_trim = pd.concat([X_in_trim.reset_index(drop=True), X_sim_trim.reset_index(drop=True)], axis=1)
        X_enc_trim = X_all_trim.copy()
        cat_cols_trim = [col for col in X_all_trim.columns if X_all_trim[col].dtype == object or isinstance(X_all_trim[col].iloc[0], str)]
        for col in cat_cols_trim:
            X_enc_trim[col] = X_enc_trim[col].astype(str)
        X_enc_trim[cat_cols_trim] = self.ordinal_encoders["trim"].transform(X_enc_trim[cat_cols_trim])
        
        trim_pred_idx = self.models["trim"].predict(X_enc_trim)[0]
        if isinstance(trim_pred_idx, (np.ndarray, list)):
            trim_pred_idx = trim_pred_idx[0]
        trim_val = self.target_encoders["trim"].inverse_transform(trim_pred_idx)
        predictions["trim"] = str(trim_val)
        
        if hasattr(self.models["trim"], "predict_proba"):
            confidences["trim"] = float(np.max(self.models["trim"].predict_proba(X_enc_trim)[0]))
        else:
            confidences["trim"] = 1.0
            
        # 5. Body Type
        X_in_bt = X_in.copy()
        X_in_bt["make"] = predictions["make"]
        X_in_bt["model"] = predictions["model"]
        sim_eng_bt = self.similarity_engines["body_type"]
        X_sim_bt = sim_eng_bt.transform(pd.Series([normalized]))
        X_all_bt = pd.concat([X_in_bt.reset_index(drop=True), X_sim_bt.reset_index(drop=True)], axis=1)
        X_enc_bt = X_all_bt.copy()
        cat_cols_bt = [col for col in X_all_bt.columns if X_all_bt[col].dtype == object or isinstance(X_all_bt[col].iloc[0], str)]
        for col in cat_cols_bt:
            X_enc_bt[col] = X_enc_bt[col].astype(str)
        X_enc_bt[cat_cols_bt] = self.ordinal_encoders["body_type"].transform(X_enc_bt[cat_cols_bt])
        
        bt_pred_idx = self.models["body_type"].predict(X_enc_bt)[0]
        if isinstance(bt_pred_idx, (np.ndarray, list)):
            bt_pred_idx = bt_pred_idx[0]
        bt_val = self.target_encoders["body_type"].inverse_transform(bt_pred_idx)
        predictions["body_type"] = str(bt_val)
        
        if hasattr(self.models["body_type"], "predict_proba"):
            confidences["body_type"] = float(np.max(self.models["body_type"].predict_proba(X_enc_bt)[0]))
        else:
            confidences["body_type"] = 1.0
            
        # 6. Origin
        X_in_ori = X_in.copy()
        X_in_ori["make"] = predictions["make"]
        X_in_ori["model"] = predictions["model"]
        sim_eng_ori = self.similarity_engines["origin"]
        X_sim_ori = sim_eng_ori.transform(pd.Series([normalized]))
        X_all_ori = pd.concat([X_in_ori.reset_index(drop=True), X_sim_ori.reset_index(drop=True)], axis=1)
        X_enc_ori = X_all_ori.copy()
        cat_cols_ori = [col for col in X_all_ori.columns if X_all_ori[col].dtype == object or isinstance(X_all_ori[col].iloc[0], str)]
        for col in cat_cols_ori:
            X_enc_ori[col] = X_enc_ori[col].astype(str)
        X_enc_ori[cat_cols_ori] = self.ordinal_encoders["origin"].transform(X_enc_ori[cat_cols_ori])
        
        ori_pred_idx = self.models["origin"].predict(X_enc_ori)[0]
        if isinstance(ori_pred_idx, (np.ndarray, list)):
            ori_pred_idx = ori_pred_idx[0]
        ori_val = self.target_encoders["origin"].inverse_transform(ori_pred_idx)
        predictions["origin"] = str(ori_val)
        
        if hasattr(self.models["origin"], "predict_proba"):
            confidences["origin"] = float(np.max(self.models["origin"].predict_proba(X_enc_ori)[0]))
        else:
            confidences["origin"] = 1.0
            
        # 7. Regional Specs
        X_in_rs = X_in.copy()
        X_in_rs["make"] = predictions["make"]
        X_in_rs["model"] = predictions["model"]
        sim_eng_rs = self.similarity_engines["regional_specs"]
        X_sim_rs = sim_eng_rs.transform(pd.Series([normalized]))
        X_all_rs = pd.concat([X_in_rs.reset_index(drop=True), X_sim_rs.reset_index(drop=True)], axis=1)
        X_enc_rs = X_all_rs.copy()
        cat_cols_rs = [col for col in X_all_rs.columns if X_all_rs[col].dtype == object or isinstance(X_all_rs[col].iloc[0], str)]
        for col in cat_cols_rs:
            X_enc_rs[col] = X_enc_rs[col].astype(str)
        X_enc_rs[cat_cols_rs] = self.ordinal_encoders["regional_specs"].transform(X_enc_rs[cat_cols_rs])
        
        rs_pred_idx = self.models["regional_specs"].predict(X_enc_rs)[0]
        if isinstance(rs_pred_idx, (np.ndarray, list)):
            rs_pred_idx = rs_pred_idx[0]
        rs_val = self.target_encoders["regional_specs"].inverse_transform(rs_pred_idx)
        predictions["regional_specs"] = str(rs_val)
        
        if hasattr(self.models["regional_specs"], "predict_proba"):
            confidences["regional_specs"] = float(np.max(self.models["regional_specs"].predict_proba(X_enc_rs)[0]))
        else:
            confidences["regional_specs"] = 1.0
            
        # Color & Weight
        for target in ["color", "weight"]:
            if target in self.models:
                sim_eng = self.similarity_engines[target]
                X_sim = sim_eng.transform(pd.Series([normalized]))
                X_all = pd.concat([X_in.reset_index(drop=True), X_sim.reset_index(drop=True)], axis=1)
                X_enc = X_all.copy()
                cat_cols = [col for col in X_all.columns if X_all[col].dtype == object or isinstance(X_all[col].iloc[0], str)]
                for col in cat_cols:
                    X_enc[col] = X_enc[col].astype(str)
                X_enc[cat_cols] = self.ordinal_encoders[target].transform(X_enc[cat_cols])
                
                pred_val = self.models[target].predict(X_enc)[0]
                if isinstance(pred_val, (np.ndarray, list)):
                    pred_val = pred_val[0]
                    
                if target == "weight":
                    predictions["weight"] = float(np.round(pred_val, 2))
                    confidences["weight"] = 1.0
                else:
                    predictions["color"] = str(self.target_encoders["color"].inverse_transform(pred_val))
                    if hasattr(self.models["color"], "predict_proba"):
                        confidences["color"] = float(np.max(self.models["color"].predict_proba(X_enc)[0]))
                    else:
                        confidences["color"] = 1.0
            else:
                if target == "weight":
                    predictions["weight"] = 0.0
                    confidences["weight"] = 0.0
                else:
                    predictions["color"] = "UNKNOWN"
                    confidences["color"] = 0.0

        output = {
            "make": predictions["make"],
            "model": predictions["model"],
            "trim": predictions["trim"],
            "year": predictions["year"],
            "body_type": predictions["body_type"],
            "origin": predictions["origin"],
            "regional_specs": predictions["regional_specs"],
            "color": predictions["color"],
            "weight": predictions["weight"],
            "confidence": {
                "make": round(confidences["make"], 4),
                "model": round(confidences["model"], 4),
                "year": round(confidences["year"], 4)
            }
        }
        return output
