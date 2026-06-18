import sys
import os
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Set up headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve project path and append to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
chat_cat_short_vin_09_path = project_root / "chat_cat_short_vin_09"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(chat_cat_short_vin_09_path) not in sys.path:
    sys.path.append(str(chat_cat_short_vin_09_path))

from chat_cat_short_vin_09.predict import predict_vehicle as predict_09, explain_prediction as explain_09
from chat_cat_short_vin.predict import predict_vehicle as predict_10, explain_prediction as explain_10
from chat_cat_short_vin_11.predict import predict_vehicle as predict_11, explain_prediction as explain_11

# Set up page configurations
st.set_page_config(
    page_title="Short Chassis Decoder",
    page_icon="",
    layout="centered",
)

st.title("Short VIN Decoder (ML Model)")
# User input form
chassis_input = st.text_input(
    "Enter a 9, 10, or 11 Character Short Chassis Number:",
    placeholder="Enter your vin here."
).strip().upper()

# Normalize
normalized = re.sub(r"[^A-Z0-9]", "", chassis_input)
length = len(normalized)

model_to_use = None
if length > 0:
    if length == 9:
        st.info("Detected: 9-Character Chassis Model")
        model_to_use = 9
    elif length == 10:
        st.info("Detected: 10-Character Chassis Model")
        model_to_use = 10
    elif length == 11:
        st.info("Detected: 11-Character Chassis Model")
        model_to_use = 11
    else:
        st.error(f"Invalid length: {length} characters. Must be 9, 10, or 11.")

if st.button("Decode Chassis", use_container_width=True):
    if length == 0:
        st.error("Please enter a chassis number first.")
    elif model_to_use is None:
        st.error("Chassis must be exactly 9, 10, or 11 characters.")
    else:
        st.session_state["last_decoded_chassis"] = normalized
        st.session_state["model_used"] = model_to_use
        
        # Verify model directory exists
        if model_to_use == 9:
            model_dir = project_root / "chat_cat_short_vin_09" / "models"
        elif model_to_use == 10:
            model_dir = project_root / "chat_cat_short_vin" / "models"
        else:
            model_dir = project_root / "chat_cat_short_vin_11" / "models"
            
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            st.warning(f"{model_to_use}-Character models are currently training or not found.")
            st.stop()

# Persistent rendering block
if "last_decoded_chassis" in st.session_state:
    normalized = st.session_state["last_decoded_chassis"]
    model_to_use = st.session_state["model_used"]
    
    if model_to_use == 9:
        model_dir = str(project_root / "chat_cat_short_vin_09" / "models")
        predict_fn = predict_09
        explain_fn = explain_09
    elif model_to_use == 10:
        model_dir = str(project_root / "chat_cat_short_vin" / "models")
        predict_fn = predict_10
        explain_fn = explain_10
    else:
        model_dir = str(project_root / "chat_cat_short_vin_11" / "models")
        predict_fn = predict_11
        explain_fn = explain_11
        
    with st.spinner("Decoding vehicle attributes and calculating confidence scores..."):
        try:
            # 1. Run predictions
            result = predict_fn(normalized, model_dir=model_dir)
            
            # Render predicted attributes in a beautiful formatted table
            st.subheader("Predicted Vehicle Attributes")
            
            table_rows = []
            display_attrs = [
                ("Make", "make"),
                ("Model", "model"),
                ("Model Year", "year"),
                ("Trim", "trim"),
                ("Body Type", "body_type"),
                ("Cylinders", "cylinders"),
                ("Origin", "origin"),
                ("No of Passengers", "no_of_passengers"),
                ("Regional Spec", "regional_spec"),
                ("Color", "color"),
                ("Weight", "weight")
            ]
            
            conf_dict = result.get("confidence_scores", result.get("attribute_confidences", {}))
            for label, key in display_attrs:
                val = result.get(key, "UNKNOWN")
                conf = conf_dict.get(key, 0.0)
                    
                # Format confidence as plain percentage string
                conf_str = f"{conf * 100:.2f}%" if conf > 0.0 else "N/A"
                table_rows.append({
                    "Attribute": label,
                    "Predicted Value": str(val),
                    "Confidence": conf_str
                })
                
            df_display = pd.DataFrame(table_rows)
            st.table(df_display)
            
            # Renders raw response
            with st.subheader("View Raw Model Response (JSON)"):
                st.json(result)
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
