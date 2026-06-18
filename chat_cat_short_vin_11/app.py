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

# Setup Python path to import parent directory and local package
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from chat_cat_short_vin_11.predict import predict_vehicle, explain_prediction

# Page config
st.set_page_config(
    page_title="11-Character Short Chassis Decoder",
    page_icon="🚗",
    layout="centered"
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

st.markdown("<h1 class='main-header'>🚗 11-Character Short Chassis Decoder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Production-grade model trained to generalize to unseen vehicle chassis prefixes.</p>", unsafe_allow_html=True)

# User input form
chassis_input = st.text_input(
    "Enter an 11-Character Short Chassis Number:",
    placeholder="e.g. EL500004138 or FJ750077443",
    max_chars=11
).strip().upper()

# Normalize
normalized_chassis = re.sub(r"[^A-Z0-9]", "", chassis_input)

if st.button("Decode Chassis", use_container_width=True):
    if len(normalized_chassis) != 11:
        st.error(f"Invalid length: Input chassis is {len(normalized_chassis)} characters. Must be exactly 11 characters.")
    else:
        model_dir = str(project_root / "chat_cat_short_vin_11" / "models")
        
        # Check if models exist
        if not os.path.exists(os.path.join(model_dir, "metadata.pkl")):
            st.warning("Models are currently training or not found in the models directory. Please run train.py first.")
            st.stop()
            
        with st.spinner("Decoding vehicle attributes and calculating confidence scores..."):
            try:
                # 1. Run predictions
                result = predict_vehicle(normalized_chassis, model_dir=model_dir)
                
                # Render predicted attributes in a beautiful formatted table
                st.subheader("📋 Decoded Vehicle Information")
                
                table_rows = []
                # Map attributes to friendly display labels
                attr_conf = result.get("attribute_confidences", {})
                display_attrs = [
                    ("Make", "make"),
                    ("Model", "model"),
                    ("Model Year", "year"),
                    ("Trim", "trim"),
                    ("Body Type", "body_type"),
                    ("Cylinders", "cylinders"),
                    ("No. of Passengers", "no_of_passengers"),
                    ("Country of Origin", "origin"),
                    ("Regional Specs", "regional_spec"),
                    ("Color", "color"),
                    ("Weight (KG)", "weight"),
                ]
                
                for label, val_key in display_attrs:
                    val = result.get(val_key, "UNKNOWN")
                    conf = attr_conf.get(val_key, result.get(f"{val_key}_confidence", 0.0))
                    
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
                
                # Interactive target selector for explainability
                explain_target = st.selectbox(
                    "Select an attribute to inspect model feature attribution:",
                    options=["make", "model", "year", "trim", "body_type", "origin", "regional_specs"]
                )
                
                exp_res = explain_prediction(normalized_chassis, target=explain_target, model_dir=model_dir)
                
                # Render closest training prefixes
                st.markdown("##### 🔍 Top Closest Prefixes in Training Set")
                st.markdown("These are the most similar chassis structures seen during model training:")
                
                cols = st.columns(5)
                for i, match in enumerate(exp_res["closest_prefixes"]):
                    with cols[i]:
                        st.metric(
                            label=f"Match #{i+1}",
                            value=match["prefix"],
                            delta=f"{match['similarity'] * 100:.1f}% Match",
                            delta_color="normal"
                        )
                        
                # Render local feature attribution chart
                st.markdown(f"##### 📊 Feature Attribution (SHAP Impact) for target: `{explain_target.upper()}`")
                attributions = exp_res["feature_attributions"]
                
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
                        # Fallback
                        df_attr = df_attr.sort_values(by="importance", ascending=True)
                        bars = ax.barh(df_attr["feature"], df_attr["importance"], color="#3b82f6", edgecolor='#334155')
                        ax.set_title(f"Random Forest Feature Importances", color='#f8fafc', fontsize=11)
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
                        raw_val = row["raw_value"]
                        # Place text at the end of the bar
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
