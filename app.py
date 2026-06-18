import sys
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="VIN Decoder",
    page_icon="",
    layout="centered",
)

# Add project root and chat_cat to Python path to import chat_cat
project_root = Path(__file__).resolve().parent
chat_cat_path = project_root / "chat_cat"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(chat_cat_path) not in sys.path:
    sys.path.append(str(chat_cat_path))

from chat_cat.predict import decode_vin

def decode_any_vin(vin_str):
    length = len(vin_str)
    if length == 17:
        return decode_vin(vin_str, model_dir=MODEL_DIR)
    elif length == 11:
        from chat_cat_short_vin_11.predict import predict_vehicle as predict_11
        import numpy as np
        model_dir_11 = str(project_root / "chat_cat_short_vin_11" / "models")
        res_11 = predict_11(vin_str, model_dir=model_dir_11)
        
        # res_11 already returns the standardized schema from predict_vehicle
        mapped_res = {
            "make": res_11.get("make"),
            "model": res_11.get("model"),
            "trim": res_11.get("trim"),
            "body_type": res_11.get("body_type"),
            "year": res_11.get("year"),
            "cylinders": res_11.get("cylinders", "UNKNOWN"),
            "origin": res_11.get("origin"),
            "no_of_passengers": res_11.get("no_of_passengers", "UNKNOWN"),
            "weight": res_11.get("weight"),
            "color": res_11.get("color", "UNKNOWN"),
            "regional_spec": res_11.get("regional_spec", res_11.get("regional_specs", "UNKNOWN")),
            "attribute_confidences": res_11.get("attribute_confidences", {}),
            "confidence": res_11.get("confidence", 0.0)
        }
        return mapped_res
    else:
        raise ValueError(f"Invalid VIN/Chassis length: {length}. Must be 11 or 17 characters.")

# Define fallback model and lookup paths
MODEL_DIR = str(project_root / "chat_cat" / "models")
LOOKUP_DATA_PATH = project_root / "lookup_data.csv"
DEPRECIATION_CSV_PATHS = {
    "audi_dep.csv": project_root / "Valuation" / "data" / "audi_dep.csv",
    "byd_dep.csv": project_root / "Valuation" / "data" / "byd_dep.csv",
    "chevelorate_dep.csv": project_root / "Valuation" / "data" / "chevelorate_dep.csv",
    "drivearabia_toyota_prices_newest.csv": project_root / "Valuation" / "data" / "drivearabia_toyota_prices_newest.csv",
    "drivearabia_toyota_prices.csv": project_root / "Valuation" / "data" / "drivearabia_toyota_prices.csv",
    "ford_dep.csv": project_root / "Valuation" / "data" / "ford_dep.csv",
    "gmc_dep.csv": project_root / "Valuation" / "data" / "gmc_dep.csv",
    "honda_dep.csv": project_root / "Valuation" / "data" / "honda_dep.csv",
    "hyundai_dep.csv": project_root / "Valuation" / "data" / "hyundai_dep.csv",
    "infiniti_dep.csv": project_root / "Valuation" / "data" / "infiniti_dep.csv",
    "jeep_dep.csv": project_root / "Valuation" / "data" / "jeep_dep.csv",
    "kia_dep.csv": project_root / "Valuation" / "data" / "kia_dep.csv",
    "lexus_dep.csv": project_root / "Valuation" / "data" / "lexus_dep.csv",
    "lincoln_dep.csv": project_root / "Valuation" / "data" / "lincoln_dep.csv",
    "mazda_dep.csv": project_root / "Valuation" / "data" / "mazda_dep.csv",
    "mini_dep.csv": project_root / "Valuation" / "data" / "mini_dep.csv",
    "mitsubishi_dep.csv": project_root / "Valuation" / "data" / "mitsubishi_dep.csv",
    "porsche_dep.csv": project_root / "Valuation" / "data" / "porsche_dep.csv",
    "rox_dep.csv": project_root / "Valuation" / "data" / "rox_dep.csv",
    "suzuki_dep.csv": project_root / "Valuation" / "data" / "suzuki_dep.csv",
    "tesla_dep.csv": project_root / "Valuation" / "data" / "tesla_dep.csv",
    "toyota_dep.csv": project_root / "Valuation" / "data" / "toyota_dep.csv",
    "volkswagen_dep.csv": project_root / "Valuation" / "data" / "volkswagen_dep.csv"
}

st.title("VIN Decoder (ML Model)")

# User VIN input panel
vin_input = st.text_input(
    "Enter a 17-character vehicle VIN or 11-character chassis number to decode:",
    placeholder="e.g. JTFHX02POF0099797 or EL500004138",
    max_chars=17,
).strip().upper()


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



@st.cache_resource
def load_ml_price_model(model_path: str):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


def get_ml_predicted_price(result: dict) -> float:
    model_path = project_root / "Valuation_ml" / "price_model.cbm"
    if not model_path.exists():
        return None
    try:
        model = load_ml_price_model(str(model_path))
    except Exception:
        return None

    make = str(result.get("make", "")).strip().upper()
    model_name = str(result.get("model", "")).strip().upper()
    trim = str(result.get("trim", "")).strip()
    if not trim or trim.upper() in ["NAN", "NONE", "UNKNOWN", ""]:
        trim = "UNKNOWN"
    else:
        trim = trim.upper()

    year_val = pd.to_numeric(result.get("year"), errors="coerce")
    if pd.isna(year_val):
        return None

    ref_year = 2026
    age = max(ref_year - int(year_val), 0)

    input_df = pd.DataFrame([{
        "make": make,
        "model": model_name,
        "trim": trim,
        "age": age
    }])

    try:
        import numpy as np
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)
        return float(pred_price)
    except Exception:
        return None


def calculate_valuation(price: float) -> dict:
    if price is None or pd.isna(price):
        return None
    try:
        retail_avg = round(price, -2)
        retail_min = round(retail_avg * 0.9, -2)
        retail_max = round(retail_avg * 1.1, -2)
        
        trade_avg = round(retail_avg * 0.92, -2)
        trade_min = round(trade_avg * 0.9, -2)
        trade_max = round(trade_avg * 1.1, -2)
        
        return {
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
    except Exception:
        return None


@st.cache_data
def load_all_depreciation_lookups(paths_dict: dict) -> dict:
    dfs = {}
    for name, path in paths_dict.items():
        if Path(path).exists():
            try:
                dfs[name] = pd.read_csv(path)
                continue
            except Exception:
                pass
        
        # Fallback to GitHub raw URL for Vercel deployment (try current dev branch first)
        github_url = f"https://raw.githubusercontent.com/sarthak-opash/vin-sphere-python/dev-sarthak/Valuation/data/{name}"
        try:
            dfs[name] = pd.read_csv(github_url)
        except Exception:
            # Secondary fallback: try the main branch in case of merge
            main_url = f"https://raw.githubusercontent.com/sarthak-opash/vin-sphere-python/main/Valuation/data/{name}"
            try:
                dfs[name] = pd.read_csv(main_url)
            except Exception:
                pass
    return dfs




def find_matches_in_df(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    make = str(result.get("make", "")).strip().lower()
    model = str(result.get("model", "")).strip().lower().replace(" ", "-")
    if make:
        if not model.startswith(f"{make}-"):
            model_slug = f"{make}-{model}"
        else:
            model_slug = model
    else:
        model_slug = model
        
    year = result.get("year")
    try:
        year_val = int(pd.to_numeric(year))
    except Exception:
        return pd.DataFrame()
        
    matches = df[(df["model_slug"] == model_slug) & (df["year"] == year_val)]
    if matches.empty:
        matches = df[df["model_slug"].str.contains(model, case=False, na=False) & (df["year"] == year_val)]
        if matches.empty:
            return pd.DataFrame()
            
    trim = str(result.get("trim", "")).strip().upper()
    
    def normalize_trim(val):
        if pd.isna(val):
            return ""
        return re.sub(r"[^A-Z0-9]", "", str(val).upper())
        
    trim_key = normalize_trim(trim)
    
    best_match = None
    if trim_key:
        exact_matches = matches[matches["trim_name"].apply(normalize_trim) == trim_key]
        if not exact_matches.empty:
            best_match = exact_matches
        else:
            contains_matches = matches[matches["trim_name"].apply(lambda x: trim_key in normalize_trim(x) or normalize_trim(x) in trim_key)]
            if not contains_matches.empty:
                best_match = contains_matches
            else:
                matches_copy = matches.copy()
                matches_copy["score"] = matches_copy["trim_name"].apply(
                    lambda x: SequenceMatcher(None, trim_key, normalize_trim(x)).ratio() if x else 0.0
                )
                fuzzy_matches = matches_copy[matches_copy["score"] >= 0.82]
                if not fuzzy_matches.empty:
                    best_score = fuzzy_matches["score"].max()
                    best_match = fuzzy_matches[fuzzy_matches["score"] == best_score]
                    
    if best_match is None or best_match.empty:
        best_match = matches
        
    return best_match


def get_depreciated_values_by_file(result: dict) -> dict:
    dfs_dict = load_all_depreciation_lookups(DEPRECIATION_CSV_PATHS)
    matched_values = {}
    
    for filename, df in dfs_dict.items():
        matched_df = find_matches_in_df(df, result)
        if not matched_df.empty and "depreciated_value" in matched_df.columns:
            valid_vals = matched_df["depreciated_value"].dropna()
            if not valid_vals.empty:
                matched_values[filename] = float(valid_vals.median())
                
    return matched_values


# Verification safety check
if st.button("Decode", use_container_width=True):
    if len(vin_input) not in [11, 17]:
        st.error("Invalid input: Input must be exactly 17 characters for a standard VIN or 11 characters for a short chassis.")
    else:
        with st.spinner("Decoding..."):
            try:
                # Perform prediction using fallback model
                result = decode_any_vin(vin_input)

                # Format prediction results into a table
                conf_dict = result.get("attribute_confidences", {})
                # Display overall confidence score
                overall_conf = result.get("confidence", 0.0) * 100

                price = get_lookup_price(result, vin_input)
                price_display = "Not found" if price is None else f"{price:,.0f}"
                result["lookup_price"] = None if price is None else float(price)

                # Calculate dynamic, data-driven confidence based on price variance for the make/model
                try:
                    lookup_df = load_price_lookup(str(LOOKUP_DATA_PATH))
                    
                    make_key = normalize_lookup_text(result.get("make"))
                    model_key = normalize_lookup_text(result.get("model"))
                    
                    model_matches = lookup_df[
                        (lookup_df["make_key"] == make_key) & 
                        (lookup_df["model_key"] == model_key)
                    ]
                    
                    prices = model_matches["price"].dropna()
                    if len(prices) > 1:
                        std_val = prices.std()
                        mean_val = prices.mean()
                        if mean_val > 0:
                            coef_var = std_val / mean_val
                            model_confidence = max(100.0 - (coef_var * 50.0), 60.0)
                            model_confidence = min(model_confidence, 98.0)
                        else:
                            model_confidence = 94.50
                    else:
                        model_confidence = 94.50
                except Exception:
                    model_confidence = 94.50

                ml_price = get_ml_predicted_price(result)
                if ml_price is None:
                    ml_price_display = "Not found"
                    ml_conf_display = "Not found"
                    ml_predicted_conf = None
                else:
                    ml_price_display = f"{ml_price:,.0f}"
                    ml_joint_conf = (overall_conf / 100.0) * model_confidence
                    ml_conf_display = f"{ml_joint_conf:.2f}%"
                    ml_predicted_conf = float(ml_joint_conf / 100.0)
                result["ml_predicted_price"] = None if ml_price is None else float(ml_price)
                result["ml_predicted_price_confidence"] = ml_predicted_conf

                dep_values = get_depreciated_values_by_file(result)
                result["depreciated_values"] = dep_values

                # Calculate valuations (min/avg/max for retail & trade)
                result["lookup_price_valuation"] = calculate_valuation(price)
                result["ml_price_valuation"] = calculate_valuation(ml_price)
                result["depreciated_price_valuations"] = {
                    filename: calculate_valuation(val) for filename, val in dep_values.items()
                } if dep_values else {}

                dep_rows = []
                if not dep_values:
                    dep_rows.append({"Attribute": "Depreciated Value", "Predicted Value": "Not found", "Confidence": "Scraped Data"})
                else:
                    for filename, val in dep_values.items():
                        display_name = filename.rsplit('.', 1)[0]
                        dep_rows.append({
                            "Attribute": f"Depreciated Value ({display_name})",
                            "Predicted Value": f"{val:,.0f}",
                            "Confidence": "Scraped Data"
                        })

                table_data = [
                    {"Attribute": "Make", "Predicted Value": result.get("make"), "Confidence": f"{conf_dict.get('make', 0.0) * 100:.2f}%"},
                    {"Attribute": "Model", "Predicted Value": result.get("model"), "Confidence": f"{conf_dict.get('model', 0.0) * 100:.2f}%"},
                    {"Attribute": "Trim", "Predicted Value": result.get("trim"), "Confidence": f"{conf_dict.get('trim', 0.0) * 100:.2f}%"},
                    {"Attribute": "Body Type", "Predicted Value": result.get("body_type"), "Confidence": f"{conf_dict.get('body_type', 0.0) * 100:.2f}%"},
                    {"Attribute": "Model Year", "Predicted Value": result.get("year"), "Confidence": f"{conf_dict.get('year', 0.0) * 100:.2f}%"},
                    {"Attribute": "Cylinders", "Predicted Value": result.get("cylinders"), "Confidence": f"{conf_dict.get('cylinders', 0.0) * 100:.2f}%"},
                    {"Attribute": "Origin", "Predicted Value": result.get("origin"), "Confidence": f"{conf_dict.get('origin', 0.0) * 100:.2f}%"},
                    {"Attribute": "Number of Passengers", "Predicted Value": result.get("no_of_passengers"), "Confidence": f"{conf_dict.get('no_of_passengers', 0.0) * 100:.2f}%"},
                    {"Attribute": "Weight (KG)", "Predicted Value": result.get("weight"), "Confidence": f"{conf_dict.get('weight', 0.0) * 100:.2f}%"},
                    {"Attribute": "Regional Specs", "Predicted Value": result.get("regional_spec"), "Confidence": f"{conf_dict.get('regional_spec', 0.0) * 100:.2f}%"},
                    {"Attribute": "Price", "Predicted Value": price_display, "Confidence": "Lookup"},
                    {"Attribute": "Price (ML Model)", "Predicted Value": ml_price_display, "Confidence": ml_conf_display},
                ]
                table_data.extend(dep_rows)

                df_results = pd.DataFrame(table_data)

                st.subheader(f"Overall Confidence: {overall_conf:.2f}%")

                # Display Results Table
                st.table(df_results)

                # Render Valuation Dashboard
                has_any_valuation = (price is not None) or (ml_price is not None) or bool(dep_values)
                if has_any_valuation:
                    st.subheader("Market Valuation Dashboard")
                    tab_titles = []
                    if price is not None:
                        tab_titles.append("Market Lookup")
                    if ml_price is not None:
                        tab_titles.append("ML Model Prediction")
                    if dep_values:
                        tab_titles.append("Depreciation Lookup")
                        
                    tabs = st.tabs(tab_titles)
                    tab_index = 0
                    
                    if price is not None:
                        with tabs[tab_index]:
                            val = result["lookup_price_valuation"]
                            st.write("**Retail Price vs. Trade Price (Lookup)**")
                            st.table(pd.DataFrame([
                                {"Price Metric": "Minimum Price (-10%)", "Retail Price (AED)": f"{val['retail_price']['minimum']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['minimum']:,.0f}"},
                                {"Price Metric": "Average Price", "Retail Price (AED)": f"{val['retail_price']['average']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['average']:,.0f}"},
                                {"Price Metric": "Maximum Price (+10%)", "Retail Price (AED)": f"{val['retail_price']['maximum']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['maximum']:,.0f}"}
                            ]))
                            tab_index += 1
                            
                    if ml_price is not None:
                        with tabs[tab_index]:
                            val = result["ml_price_valuation"]
                            st.write("**Retail Price vs. Trade Price (ML Model)**")
                            st.table(pd.DataFrame([
                                {"Price Metric": "Minimum Price (-10%)", "Retail Price (AED)": f"{val['retail_price']['minimum']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['minimum']:,.0f}"},
                                {"Price Metric": "Average Price", "Retail Price (AED)": f"{val['retail_price']['average']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['average']:,.0f}"},
                                {"Price Metric": "Maximum Price (+10%)", "Retail Price (AED)": f"{val['retail_price']['maximum']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['maximum']:,.0f}"}
                            ]))
                            tab_index += 1
                            
                    if dep_values:
                        with tabs[tab_index]:
                            st.write("**Retail Price vs. Trade Price (Depreciation)**")
                            for filename, dep_val in dep_values.items():
                                val = result["depreciated_price_valuations"][filename]
                                display_name = filename.rsplit('.', 1)[0]
                                st.write(f"**Source File: `{display_name}`**")
                                st.table(pd.DataFrame([
                                    {"Price Metric": "Minimum Price (-10%)", "Retail Price (AED)": f"{val['retail_price']['minimum']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['minimum']:,.0f}"},
                                    {"Price Metric": "Average Price", "Retail Price (AED)": f"{val['retail_price']['average']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['average']:,.0f}"},
                                    {"Price Metric": "Maximum Price (+10%)", "Retail Price (AED)": f"{val['retail_price']['maximum']:,.0f}", "Trade Price (AED)": f"{val['trade_price']['maximum']:,.0f}"}
                                ]))
                            tab_index += 1

                # Display Raw JSON Output
                st.subheader("Raw JSON Response")
                st.json(result)

            except Exception as e:
                st.error(f"Error executing prediction fallback model: {e}")
