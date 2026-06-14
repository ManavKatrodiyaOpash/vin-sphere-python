import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    MODEL_DIR, REPORTS_DIR,
    CLASSIFICATION_TARGETS, REGRESSION_TARGETS
)
from src.predict import (
    load_inference_pipeline, predict_single, predict_batch, generate_attention_map
)

# Set page config
st.set_page_config(
    page_title="Automotive VIN Attribute Prediction Engine",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #17b978;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5f7fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #fff8e1;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ffb300;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("🚗 VIN Engine")
page = st.sidebar.selectbox(
    "Select Navigation Page",
    ["Single VIN Prediction", "Batch Prediction", "Model Analytics"]
)

# Load Pipeline & Models
@st.cache_resource
def get_inference_resources():
    try:
        pipeline, model, catboost_models, weak_targets = load_inference_pipeline()
        return pipeline, model, catboost_models, weak_targets, None
    except Exception as e:
        return None, None, None, None, str(e)

pipeline, model, cat_models, weak_targets, error_msg = get_inference_resources()

if error_msg:
    st.error(f"Error loading models: {error_msg}")
    st.warning("Please verify that the training script has run successfully and that the files under `models/` have been generated.")
    st.stop()


# ----------------------------------------------------
# PAGE 1: SINGLE VIN PREDICTION
# ----------------------------------------------------
if page == "Single VIN Prediction":
    st.markdown('<div class="main-title">Single Chassis Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter a 17-character chassis number (VIN) to predict vehicle attributes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Chassis Number Input")
        vin_input = st.text_input("VIN String", "JTDBR32E220045678", max_chars=17)
        predict_button = st.button("Run Prediction Engine", type="primary")
        
        # Display warnings if color_raw is weak
        if "color_raw" in weak_targets:
            st.markdown("""
            <div class="warning-card">
                <strong>⚠️ Signal Notice:</strong> The dataset analysis shows that <code>color_raw</code> has a very weak signal inside the chassis number. Predictions for color may not be reliable.
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        if predict_button or vin_input:
            with st.spinner("Decoding Chassis Sequence..."):
                results = predict_single(vin_input, pipeline, model, cat_models, weak_targets)
                
            if "error" in results:
                st.error(results["error"])
            else:
                st.subheader("Predictions Output")
                
                # Format predictions as requested
                output_json = {}
                for col in CLASSIFICATION_TARGETS:
                    output_json[col] = {
                        "prediction": results[col]["prediction"],
                        "confidence": results[col]["confidence"]
                    }
                    
                for col in REGRESSION_TARGETS:
                    output_json[col] = {
                        "prediction": results[col]["prediction"]
                    }
                    
                st.json(output_json)
                
                # Display individual warning logs
                warning_cols = [col for col in results if "warning" in results[col]]
                if warning_cols:
                    st.info(f"Target attributes with weak signal warnings: {', '.join(warning_cols)}")

    # Visualizations section
    if (predict_button or vin_input) and "error" not in results:
        st.write("---")
        st.subheader("Transformer self-attention heatmap")
        st.write("The heatmap below shows which positions in the chassis number had the most influence on the model's prediction vector.")
        
        fig = generate_attention_map(results["cleaned_vin"], results["attention_weights"])
        st.pyplot(fig)


# ----------------------------------------------------
# PAGE 2: BATCH PREDICTION
# ----------------------------------------------------
elif page == "Batch Prediction":
    st.markdown('<div class="main-title">Batch Chassis Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a CSV file containing chassis numbers for batch attribute inference</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Show input preview
            st.subheader("Input Preview")
            st.dataframe(df_input.head(5))
            
            if "chassisNumber" not in df_input.columns:
                st.error("Uploaded CSV file must contain a 'chassisNumber' column.")
            else:
                if st.button("Process Batch Predictions", type="primary"):
                    with st.spinner("Processing batch predictions (this can take a moment)..."):
                        df_output = predict_batch(df_input, pipeline, model, cat_models, weak_targets)
                        
                    st.success("Batch processing complete!")
                    st.subheader("Predictions Preview")
                    st.dataframe(df_output.head(10))
                    
                    # Convert to CSV for download
                    csv = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predicted CSV File",
                        data=csv,
                        file_name="chassis_predictions_output.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")


# ----------------------------------------------------
# PAGE 3: MODEL ANALYTICS
# ----------------------------------------------------
elif page == "Model Analytics":
    st.markdown('<div class="main-title">Model Diagnostics & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dataset statistics, position-wise Cramer\'s V importance, and model validation details</div>', unsafe_allow_html=True)
    
    # 1. Load Cramer's V Target Signal report
    st.subheader("Position-wise Target Signal Strength")
    report_file = os.path.join(REPORTS_DIR, "target_signal_analysis_report.md")
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.info("No target signal analysis report found. Run model training first to generate dataset stats.")
        
    st.write("---")
    
    # 2. Test metrics report
    st.subheader("Validation & Evaluation Metrics")
    metrics_file = os.path.join(REPORTS_DIR, "model_evaluation_metrics.md")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_content = f.read()
        st.markdown(metrics_content)
    else:
        st.info("Evaluation metrics report not found. Run model training first.")
