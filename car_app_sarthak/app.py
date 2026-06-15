import sys
import os
import re
from difflib import SequenceMatcher
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

# Define fallback model directory path and lookup path
MODEL_DIR = str(project_root / "chat_cat" / "models")
LOOKUP_DATA_PATH = project_root / "lookup_data.csv"

@st.cache_data
def load_price_lookup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"make", "model", "year", "trim", "price"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in lookup CSV: {', '.join(sorted(missing_columns))}")

    df = df.copy()
    df["make_key"] = df["make"].apply(normalize_lookup_text)
    df["model_key"] = df["model"].apply(normalize_lookup_text)
    df["trim_key"] = df["trim"].apply(normalize_lookup_text)
    df["year_key"] = df["year"].apply(normalize_lookup_year)
    if "chassisNumber" in df.columns:
        df["vin_key"] = df["chassisNumber"].apply(normalize_lookup_text)
    else:
        df["vin_key"] = ""
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def normalize_lookup_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def normalize_lookup_year(value) -> str:
    year = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(year):
        return ""
    return str(int(year))


def valid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["price"].notna() & (df["price"] > 0)]


def select_price(df: pd.DataFrame):
    df = valid_price_rows(df)
    if df.empty:
        return None
    return float(df["price"].median())


def get_lookup_price(result: dict, vin: str):
    if not LOOKUP_DATA_PATH.exists():
        return None
        
    lookup_df = load_price_lookup(str(LOOKUP_DATA_PATH))

    vin_key = normalize_lookup_text(vin)
    if vin_key:
        vin_matches = lookup_df[lookup_df["vin_key"] == vin_key]
        price = select_price(vin_matches)
        if price is not None:
            return price

    base_matches = lookup_df[
        (lookup_df["make_key"] == normalize_lookup_text(result.get("make")))
        & (lookup_df["model_key"] == normalize_lookup_text(result.get("model")))
        & (lookup_df["year_key"] == normalize_lookup_year(result.get("year")))
    ]

    if base_matches.empty:
        return None

    trim_key = normalize_lookup_text(result.get("trim"))
    if trim_key:
        exact_trim_matches = base_matches[base_matches["trim_key"] == trim_key]
        price = select_price(exact_trim_matches)
        if price is not None:
            return price

        contains_trim_matches = base_matches[
            base_matches["trim_key"].apply(
                lambda csv_trim: bool(csv_trim)
                and (trim_key in csv_trim or csv_trim in trim_key)
            )
        ]
        price = select_price(contains_trim_matches)
        if price is not None:
            return price

        fuzzy_matches = base_matches.copy()
        fuzzy_matches["trim_score"] = fuzzy_matches["trim_key"].apply(
            lambda csv_trim: SequenceMatcher(None, trim_key, csv_trim).ratio()
            if csv_trim
            else 0.0
        )
        fuzzy_matches = fuzzy_matches[fuzzy_matches["trim_score"] >= 0.82]
        if not fuzzy_matches.empty:
            best_score = fuzzy_matches["trim_score"].max()
            price = select_price(fuzzy_matches[fuzzy_matches["trim_score"] == best_score])
            if price is not None:
                return price

    valid_base_matches = valid_price_rows(base_matches)
    if valid_base_matches["trim_key"].nunique() == 1:
        return select_price(valid_base_matches)

    return None

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
                price = get_lookup_price(result, vin_input)
                price_display = "Not found" if price is None else f"{price:,.0f} AED"
                result["lookup_price"] = None if price is None else float(price)
                
                # Valuation calculations (Retail & Trade, with +/- 10% min/max)
                if price is not None:
                    retail_avg = round(price, -2)
                    retail_min = round(retail_avg * 0.9, -2)
                    retail_max = round(retail_avg * 1.1, -2)
                    
                    trade_avg = round(retail_avg * 0.92, -2)
                    trade_min = round(trade_avg * 0.9, -2)
                    trade_max = round(trade_avg * 1.1, -2)
                    
                    valuation_data = {
                        "retail_price": {
                            "average": int(retail_avg),
                            "minimum": int(retail_min),
                            "maximum": int(retail_max)
                        },
                        "trade_price": {
                            "average": int(trade_avg),
                            "minimum": int(trade_min),
                            "maximum": int(trade_max)
                        }
                    }
                else:
                    valuation_data = None
                
                result["valuation"] = valuation_data
                
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
                    {"Attribute": "Price", "Predicted Value": price_display, "Confidence": "Lookup"},
                ]
                
                df_results = pd.DataFrame(table_data)
                
                # Display overall confidence score
                overall_conf = result.get("confidence", 0.0) * 100
                st.subheader(f"Overall Confidence: {overall_conf:.2f}%")
                
                # Display Results Table
                st.table(df_results)
                
                # Display Our Price Valuation Dashboard if available
                if valuation_data:
                    st.subheader("Market Valuation (From Lookup)")
                    valuation_df = pd.DataFrame([
                        {
                            "Price Metric": "Minimum Price (-10%)",
                            "Retail Price (AED)": f"{valuation_data['retail_price']['minimum']:,.0f}",
                            "Trade Price (AED)": f"{valuation_data['trade_price']['minimum']:,.0f}"
                        },
                        {
                            "Price Metric": "Average Price",
                            "Retail Price (AED)": f"{valuation_data['retail_price']['average']:,.0f}",
                            "Trade Price (AED)": f"{valuation_data['trade_price']['average']:,.0f}"
                        },
                        {
                            "Price Metric": "Maximum Price (+10%)",
                            "Retail Price (AED)": f"{valuation_data['retail_price']['maximum']:,.0f}",
                            "Trade Price (AED)": f"{valuation_data['trade_price']['maximum']:,.0f}"
                        }
                    ])
                    st.table(valuation_df)
                
                # Display Raw JSON Output
                st.subheader("Raw JSON Response")
                st.json(result)
                
            except Exception as e:
                st.error(f"Error executing prediction fallback model: {e}")
