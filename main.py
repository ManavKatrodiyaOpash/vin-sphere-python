
# pyrefly: ignore [missing-import]
import streamlit as st
import json
from pathlib import Path

# =====================================================
# IMPORT YOUR EXISTING DECODER
# =====================================================
# Rename your current app.py decoder file if needed.
# Must expose: decode_vin(vin)
from car_app.app import decode_vin

# =====================================================
# TOYOTA VDS DATABASE
# =====================================================

@st.cache_data
def load_toyota_vds():
    json_file = "car_app_dev/toyota_vds_year_patterns_v3.json"
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

toyota_rules = load_toyota_vds()

YEAR_MAP = {
    "V":1997,"W":1998,"X":1999,"Y":2000,
    "1":2001,"2":2002,"3":2003,"4":2004,"5":2005,
    "6":2006,"7":2007,"8":2008,"9":2009,
    "A":2010,"B":2011,"C":2012,"D":2013,"E":2014,
    "F":2015,"G":2016,"H":2017,"J":2018,"K":2019,
    "L":2020,"M":2021,"N":2022,"P":2023,"R":2024,
    "S":2025,"T":2026
}

def decode_toyota_vds(vin):
    vds = vin[3:8]
    year_code = vin[9]

    year = YEAR_MAP.get(year_code)
    if year is None:
        return None

    lookup_key = f"{vds}_{year}"

    if lookup_key not in toyota_rules:
        return None

    rule = toyota_rules[lookup_key]

    return {
        "source": "Toyota VDS Database",
        "make": "Toyota",
        "model": rule.get("model"),
        "body_type": rule.get("bodyType"),
        "cylinders": rule.get("cylinders"),
        "regional_spec": rule.get("regionalSpec"),
        "trim": rule.get("trim"),
        "trim_confidence": rule.get("trim_confidence"),
        "samples": rule.get("samples")
    }

# =====================================================
# MASTER PIPELINE
# =====================================================

def simplify_fallback_result(r):

    result = {
        "manufacturer": r.get("manufacturer", "Unknown"),
        "country": r.get("country", "Unknown"),
        "vehicle_type": r.get("vehicle_type", "Unknown"),

        "body_type": r.get("body_type", "Unknown"),
        "engine": r.get("engine", "Unknown"),
        "drive_type": r.get("drive_type", "Unknown"),
        "number_of_doors": r.get("number_of_doors", "Unknown"),

        "series_line": r.get("series_line", "Unknown"),
        # "model_generation": r.get("model_generation", "Unknown"),
        "model_platform": r.get("model_platform", "Unknown"),

        "model_year": r.get("model_year", "Unknown"),
        "plant": r.get("plant", "Unknown"),

        # These don't exist in app.py
        "trim": r.get("trim", "Unknown"),
        "regional_spec": r.get("regional_spec", "Unknown"),
        "cylinders": r.get("cylinders", "Unknown")
    }

    # Only add safety fields if they actually exist

    if r.get("restraint_system") not in [None, "", "Unknown", "No"]:
        result["restraint_system"] = r["restraint_system"]

    if r.get("number_of_airbags") not in [None, "", "Unknown", "No"]:
        result["number_of_airbags"] = r["number_of_airbags"]

    if r.get("curtain_airbags") not in [None, "", "Unknown", "No"]:
        result["curtain_airbags"] = r["curtain_airbags"]

    if r.get("side_airbags") not in [None, "", "Unknown", "No"]:
        result["side_airbags"] = r["side_airbags"]

    if r.get("driver_knee_airbag") not in [None, "", "Unknown", "No"]:
        result["driver_knee_airbag"] = r["driver_knee_airbag"]

    if r.get("passenger_knee_airbag") not in [None, "", "Unknown", "No"]:
        result["passenger_knee_airbag"] = r["passenger_knee_airbag"]

    return result

def master_decode(vin):

    # Stage 1
    vds_result = decode_toyota_vds(vin)

    if vds_result:
        # st.success(f"Decoded using: {result.get('source')}")
        st.success("Data Found in the Database")
        return vds_result

    # Stage 2 Fallback
    st.warning("No match found in the database, using fallback Model Instead")
    
    result = decode_vin(vin)

    result = simplify_fallback_result(result)

    return result

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="VIN Decoder Pipeline",
    layout="wide"
)

st.title("VIN Decoder Pipeline")

vin = st.text_input(
    "Enter VIN",
    max_chars=17
).strip().upper()

if st.button("Decode"):

    if len(vin) != 17:
        st.error("VIN must contain exactly 17 characters")

    else:
        vds = vin[3:8]

        year_code = vin[9]

        year = YEAR_MAP.get(year_code)
        
        if year is None:

            st.error(
                f"Unknown year code: {year_code}"
            )
        
        else:
            lookup_key = f"{vds}_{year}"

            st.write("---")

            st.write("### Extracted")

            st.write(f"VDS: **{vds}**")
            st.write(f"Year: **{year}**")
            st.write(f"Lookup Key: **{lookup_key}**")

            result = master_decode(vin)

            st.json(result)