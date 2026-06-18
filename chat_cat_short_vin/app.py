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
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from chat_cat_short_vin.predict import predict_vehicle as predict_10, explain_prediction as explain_10
from chat_cat_short_vin_11.predict import predict_vehicle as predict_11, explain_prediction as explain_11

# Set up page configurations
st.set_page_config(
    page_title="Short Chassis Decoder",
    page_icon="🚗",
    layout="centered",
)

# Custom premium styling via CSS injection
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .attribute-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    .confidence-badge {
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🚗 Short Chassis Decoder (10 & 11 Chars)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Production-grade model trained to generalize to unseen vehicle chassis prefixes.</p>", unsafe_allow_html=True)

# User input form
chassis_input = st.text_input(
    "Enter a 10 or 11 Character Short Chassis Number:",
    placeholder="e.g. GDH20100412 (11-char) or AE10012345 (10-char)"
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

if st.button("Decode Chassis", use_container_width=True):
    if length == 0:
        st.error("Please enter a chassis number first.")
    elif model_to_use is None:
        st.error("Chassis must be exactly 10 or 11 characters.")
    else:
        # Save choice in session state to persist it for interactive widgets
        st.session_state["last_decoded_chassis"] = normalized
        st.session_state["model_used"] = model_to_use
        
        # Verify model directory exists
        if model_to_use == 10:
            model_dir = project_root / "chat_cat_short_vin" / "models"
            if not os.path.exists(model_dir) or not os.listdir(model_dir):
                st.warning("10-Character models are currently training or not found in models directory.")
                st.stop()
        else:
            model_dir = project_root / "chat_cat_short_vin_11" / "models"
            if not os.path.exists(model_dir) or not os.listdir(model_dir):
                st.warning("11-Character models are currently training or not found in models directory.")
                st.stop()

# Persistent rendering block
if "last_decoded_chassis" in st.session_state:
    normalized = st.session_state["last_decoded_chassis"]
    model_to_use = st.session_state["model_used"]
    
    if model_to_use == 10:
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
            st.subheader("📋 Decoded Vehicle Information")
            
            table_rows = []
            display_attrs = [
                ("Make", "make"),
                ("Model", "model"),
                ("Model Year", "year"),
                ("Trim", "trim"),
                ("Body Type", "body_type"),
                ("Country of Origin", "origin"),
                ("Regional Specs", "regional_specs"),
                ("Color", "color"),
                ("Weight (KG)", "weight")
            ]
            
            for label, key in display_attrs:
                # Key normalization between formats
                if model_to_use == 10:
                    lookup_key = "regional specs" if key == "regional_specs" else key
                    val = result.get(lookup_key, "UNKNOWN")
                    conf_dict = result.get("attribute_confidences", {})
                    conf = conf_dict.get(lookup_key, 0.0)
                else:
                    val = result.get(key, "UNKNOWN")
                    conf = result.get(f"{key}_confidence", 0.0)
                    
                # Highlight colors depending on confidence score
                if conf >= 0.85:
                    badge_color = "rgba(46, 204, 113, 0.15)"
                    text_color = "#2ecc71"
                elif conf >= 0.60:
                    badge_color = "rgba(241, 196, 15, 0.15)"
                    text_color = "#f1c40f"
                else:
                    badge_color = "rgba(231, 76, 60, 0.15)"
                    text_color = "#e74c3c"
                    
                conf_html = f"<span class='confidence-badge' style='background-color: {badge_color}; color: {text_color};'>{conf * 100:.2f}%</span>"
                table_rows.append({
                    "Attribute": label,
                    "Decoded Value": str(val),
                    "Confidence Score": conf_html
                })
                
            df_display = pd.DataFrame(table_rows)
            st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # 2. Explainability Insights Section
            st.markdown("---")
            st.subheader("💡 Model Explainability Insights")
            
            explain_target = st.selectbox(
                "Select an attribute to inspect model feature attribution:",
                options=["make", "model", "year", "trim", "body_type", "origin", "regional_specs"]
            )
            
            exp_res = explain_fn(normalized, target=explain_target, model_dir=model_dir)
            
            # Render closest training prefixes
            st.markdown("##### 🔍 Top Closest Prefixes in Training Set")
            st.markdown("These are the most similar chassis structures seen during model training:")
            
            closest_pref = exp_res.get("closest_prefixes", [])
            if closest_pref:
                cols = st.columns(len(closest_pref))
                for i, match in enumerate(closest_pref):
                    with cols[i]:
                        st.metric(
                            label=f"Match #{i+1}",
                            value=match["prefix"],
                            delta=f"{match['similarity'] * 100:.1f}% Match",
                            delta_color="normal"
                        )
            else:
                st.info("No close prefixes found in training set.")
                    
            # Render local feature attribution chart
            st.markdown(f"##### 📊 Feature Attribution (SHAP Impact) for target: `{explain_target.upper()}`")
            attributions = exp_res.get("feature_attributions", [])
            
            if not attributions:
                st.info("No feature attributions available.")
            else:
                df_attr = pd.DataFrame(attributions)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#1e293b')
                
                if "shap_value" in df_attr.columns:
                    df_attr = df_attr.sort_values(by="shap_value", key=abs, ascending=True)
                    colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in df_attr["shap_value"]]
                    bars = ax.barh(df_attr["feature"], df_attr["shap_value"], color=colors, edgecolor='#334155')
                    ax.set_title(f"SHAP Values (Positive = Favors Prediction, Negative = Opposes)", color='#f8fafc', fontsize=11)
                    ax.set_xlabel("SHAP Impact Score", color='#94a3b8')
                else:
                    # Fallback to feature importances
                    df_attr = df_attr.sort_values(by="importance", ascending=True)
                    bars = ax.barh(df_attr["feature"], df_attr["importance"], color="#3b82f6", edgecolor='#334155')
                    ax.set_title(f"Model Feature Importances", color='#f8fafc', fontsize=11)
                    ax.set_xlabel("Relative Importance Score", color='#94a3b8')
                    
                ax.tick_params(colors='#f8fafc')
                ax.spines['bottom'].set_color('#334155')
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.spines['left'].set_color('#334155')
                ax.grid(axis='x', linestyle='--', alpha=0.1)
                
                # Add raw value labels
                for bar, (_, row) in zip(bars, df_attr.iterrows()):
                    width = bar.get_width()
                    raw_val = row.get("raw_value", "")
                    x_pos = width + 0.01 if width >= 0 else width - 0.1
                    ha = 'left' if width >= 0 else 'right'
                    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f" '{raw_val}'", 
                            va='center', ha=ha, color='#38bdf8', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            # Renders raw response
            with st.expander("📝 View Raw Model Response (JSON)"):
                st.json(result)
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
