import streamlit as st
import json
import pandas as pd
from pathlib import Path

# -------------------------
# Load Rules
# -------------------------

@st.cache_data
def load_rules():
    json_file = Path(__file__).parent / "toyota_vds_year_patterns_v3.json"

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

toyota_rules = load_rules()

# -------------------------
# Toyota Year Decoder
# -------------------------
YEAR_MAP = {
    "V":1997,
    "W":1998,
    "X":1999,
    "Y":2000,
    "1":2001,
    "2":2002,
    "3":2003,
    "4":2004,
    "5":2005,
    "6":2006,
    "7":2007,
    "8":2008,   
    "9":2009,
    "A":2010,
    "B":2011,
    "C":2012,
    "D":2013,
    "E":2014,
    "F":2015,
    "G":2016,
    "H":2017,
    "J":2018,
    "K":2019,
    "L":2020,
    "M":2021,
    "N":2022,
    "P":2023,
    "R":2024,
    "S":2025,
    "T":2026
    }

# -------------------------
# Page
# -------------------------
st.set_page_config(
    page_title="Toyota VIN Decoder",
    layout="wide"
)

st.title("Toyota VIN Decoder")
st.write("Testing VDS + Year rule engine")

# -------------------------
# VIN Input
# -------------------------
vin = st.text_input(
    "Enter VIN",
    max_chars=17
).strip().upper()

# -------------------------
# Decode Button
# -------------------------
if st.button("Decode"):

    if len(vin) != 17:

        st.error(
            "VIN must contain exactly 17 characters"
        )

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

            if lookup_key in toyota_rules:

                rule = toyota_rules[lookup_key]

                st.success("Match Found")

                result = {
                    "Model":
                        rule.get("model"),

                    "Body Type":
                        rule.get("bodyType"),

                    "Cylinders":
                        rule.get("cylinders"),

                    "Regional Spec":
                        rule.get("regionalSpec"),

                    "Trim":
                        rule.get("trim"),

                    "Trim Confidence":
                        rule.get("trim_confidence"),

                    "Samples":
                        rule.get("samples")
                }

                st.json(result)

                if "possible_trims" in rule:

                    st.warning(
                        "Multiple trims observed for this pattern"
                    )

                    st.write(
                        pd.DataFrame(
                            {
                                "Possible Trims":
                                rule["possible_trims"]
                            }
                        )
                    )

            else:

                st.error(
                    "Pattern not found in rule database"
                )