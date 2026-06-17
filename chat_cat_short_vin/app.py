import sys
import os
import re
from pathlib import Path
import streamlit as st
import pandas as pd

# Set up page configurations
st.set_page_config(
    page_title="Short Chassis Decoder",
    page_icon="🚗",
    layout="centered",
)

# Resolve project path and append to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from chat_cat_short_vin.predict import predict_vehicle as predict_10
from chat_cat_short_vin_11.predict import predict_vehicle as predict_11

st.title("Short Chassis Decoder (ML Model)")

chassis_input = st.text_input(
    "Enter a 10 or 11 character Chassis/VIN number to decode:",
    placeholder="e.g. FJ750077443 or GDH20100412"
).strip().upper()

# Normalize
normalized = re.sub(r"[^A-Z0-9]", "", chassis_input)
length = len(normalized)

model_to_use = None
if length > 0:
    if length == 10:
        st.info("Detected: 10-Character Chassis Model")
        model_to_use = 10
    elif length == 11:
        st.info("Detected: 11-Character Chassis Model")
        model_to_use = 11
    else:
        st.error(f"Invalid length: {length} characters. Must be 10 or 11.")

if st.button("Decode", use_container_width=True):
    if length == 0:
        st.error("Please enter a chassis number first.")
    elif model_to_use is None:
        st.error("Chassis must be exactly 10 or 11 characters.")
    else:
        with st.spinner("Decoding..."):
            try:
                # Perform decoding based on detected model
                if model_to_use == 10:
                    model_dir = project_root / "chat_cat_short_vin" / "models"
                    # Check if models exist
                    if not os.path.exists(model_dir) or not os.listdir(model_dir):
                        st.warning("10-Character models are currently training or not found in models directory.")
                        st.stop()
                    result = predict_10(normalized, model_dir=str(model_dir))
                else:
                    model_dir = project_root / "chat_cat_short_vin_11" / "models"
                    if not os.path.exists(model_dir) or not os.listdir(model_dir):
                        st.warning("11-Character models not found in models directory.")
                        st.stop()
                    result = predict_11(normalized, model_dir=str(model_dir))
                
                # Format prediction results into a table matching main app.py style
                conf_dict = result.get("attribute_confidences", {})
                table_data = []
                for attr, key in [
                    ("Make", "make"),
                    ("Model", "model"),
                    ("Trim", "trim"),
                    ("Body Type", "body_type"),
                    ("Model Year", "year"),
                    ("Weight (KG)", "weight"),
                    ("Regional Specs", "regional specs"),
                    ("Origin", "origin"),
                    ("Color", "color"),
                ]:
                    val = result.get(key)
                    conf = conf_dict.get(key, 0.0)
                    conf_str = f"{conf * 100:.2f}%"
                    
                    table_data.append({
                        "Attribute": attr,
                        "Predicted Value": val,
                        "Confidence": conf_str
                    })
                
                df_results = pd.DataFrame(table_data)
                
                st.subheader("Predicted Vehicle Attributes")
                st.table(df_results)
                
                st.subheader("Raw JSON Response")
                st.json(result)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
