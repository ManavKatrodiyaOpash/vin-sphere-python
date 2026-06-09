import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd

# Configure page metadata
st.set_page_config(
    page_title="VIN Decoder",
    page_icon="",
    layout="centered"
)

# Add project root and chat_cat to Python path to import chat_cat
project_root = Path(__file__).resolve().parent.parent
chat_cat_path = project_root / "chat_cat"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(chat_cat_path) not in sys.path:
    sys.path.append(str(chat_cat_path))

from chat_cat.predict import decode_vin

# Define fallback model directory path
MODEL_DIR = str(project_root / "chat_cat" / "models")

st.title("VIN Decoder (ML Model)")

#st.write("Decode vehicle attributes using the fallback hierarchical lookup model.")

# User VIN input panel
vin_input = st.text_input(
    "Enter a 17-character vehicle VIN to decode:",
    placeholder="e.g. JTFHX02POF0099797",
    max_chars=17
).strip().upper()

# Verification safety check
if st.button("Decode", use_container_width=True):
    if len(vin_input) != 17:
        st.error("Invalid input: A standard vehicle VIN must be exactly 17 characters.")
    else:
        with st.spinner("Decoding..."):
            try:
                # Perform prediction using fallback model
                result = decode_vin(vin_input, model_dir=MODEL_DIR)
                
                # Format prediction results into a table
                conf_dict = result.get("attribute_confidences", {})
                
                table_data = [
                    {"Attribute": "Make", "Predicted Value": result.get("make"), "Confidence": f"{conf_dict.get('make', 0.0)*100:.2f}%"},
                    {"Attribute": "Model", "Predicted Value": result.get("model"), "Confidence": f"{conf_dict.get('model', 0.0)*100:.2f}%"},
                    {"Attribute": "Trim", "Predicted Value": result.get("trim"), "Confidence": f"{conf_dict.get('trim', 0.0)*100:.2f}%"},
                    {"Attribute": "Body Type", "Predicted Value": result.get("body_type"), "Confidence": f"{conf_dict.get('body_type', 0.0)*100:.2f}%"},
                    {"Attribute": "Model Year", "Predicted Value": result.get("year"), "Confidence": f"{conf_dict.get('year', 0.0)*100:.2f}%"},
                    {"Attribute": "Cylinders", "Predicted Value": result.get("cylinders"), "Confidence": f"{conf_dict.get('cylinders', 0.0)*100:.2f}%"},
                    {"Attribute": "Origin", "Predicted Value": result.get("origin"), "Confidence": f"{conf_dict.get('origin', 0.0)*100:.2f}%"},
                    {"Attribute": "Number of Passengers", "Predicted Value": result.get("no_of_passengers"), "Confidence": f"{conf_dict.get('no_of_passengers', 0.0)*100:.2f}%"},
                    {"Attribute": "Weight (KG)", "Predicted Value": result.get("weight"), "Confidence": f"{conf_dict.get('weight', 0.0)*100:.2f}%"},
                    {"Attribute": "Regional Specs", "Predicted Value": result.get("regional_spec"), "Confidence": f"{conf_dict.get('regional_spec', 0.0)*100:.2f}%"},
                ]
                
                df_results = pd.DataFrame(table_data)
                
                # Display overall confidence score
                overall_conf = result.get("confidence", 0.0) * 100
                st.subheader(f"Overall Confidence: {overall_conf:.2f}%")
                
                # Display Results Table
                st.table(df_results)
                
                # Display Raw JSON Output
                st.subheader("Raw JSON Response")
                st.json(result)
                
            except Exception as e:
                st.error(f"Error executing prediction fallback model: {e}")
