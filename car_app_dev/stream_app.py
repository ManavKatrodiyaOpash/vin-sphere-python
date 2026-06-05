import streamlit as st
import json
import pandas as pd
from pathlib import Path
import re

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
if __name__ == "__main__":
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

                        "make": 
                        "Toyota",
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
                    
                    @st.cache_data
                    def load_toyota_master():

                        json_file = "../car_app/toyota.json"

                        with open(json_file, "r", encoding="utf-8") as f:
                            return json.load(f)

                    toyota_data = load_toyota_master()

                    st.warning(
                        "Pattern not found in VDS database. Using Toyota VIN fallback decoder."
                    )

                    pos4 = vin[3]
                    pos5 = vin[4]
                    pos6 = vin[5]
                    pos7 = vin[6]
                    pos8 = vin[7]
                    pos11 = vin[10]
                    wmi = vin[:3]

                    result = {
                        "Make": "Toyota",
                        "Model": "Unknown",
                        "Trim": "Unknown",
                        "Body Type": "Unknown",
                        "Doors": "Unknown",
                        "Drive Type": "Unknown",
                        "Engine": "Unknown",
                        "Cylinders": "Unknown",
                        "Regional Spec": "Unknown",
                        "Plant": "Unknown",
                        "Manufacturer": "Unknown",
                        "Country": "Unknown"
                    }

                    # ------------------
                    # WMI INFO
                    # ------------------

                    wmi_info = toyota_data.get(
                        "wmi",
                        {}
                    ).get(
                        wmi,
                        {}
                    )

                    result["Manufacturer"] = wmi_info.get(
                        "manufacturer",
                        "Unknown"
                    )

                    result["Country"] = wmi_info.get(
                        "country",
                        "Unknown"
                    )

                    # ------------------
                    # PLANT
                    # ------------------

                    result["Plant"] = toyota_data.get(
                        "position_11_plant",
                        {}
                    ).get(
                        pos11,
                        "Unknown"
                    )

                    # ------------------
                    # ERA SELECTION
                    # ------------------

                    era = (
                        toyota_data["era_2010_present"]
                        if year >= 2010
                        else toyota_data["era_1996_2009"]
                    )

                    # ------------------
                    # BODY TYPE
                    # ------------------

                    body_tables = era.get(
                        "position_4_body_type",
                        {}
                    )

                    for section in body_tables.values():

                        if isinstance(section, dict):

                            if pos4 in section:

                                body = section[pos4]

                                if isinstance(body, dict):

                                    result["Body Type"] = body.get(
                                        "body_type",
                                        "Unknown"
                                    )

                                    result["Doors"] = body.get(
                                        "number_of_doors",
                                        "Unknown"
                                    )

                                    result["Drive Type"] = body.get(
                                        "drive_type",
                                        "Unknown"
                                    )

                                break

                    # ------------------
                    # ENGINE
                    # ------------------

                    engine = era.get(
                        "position_5_engine",
                        {}
                    ).get(
                        pos5
                    )

                    if engine:

                        result["Engine"] = engine

                        cyl_match = re.search(
                            r'([0-9]+)[\\.-]?L.*?(I[0-9]|V[0-9]|H[0-9])',
                            engine
                        )

                        if cyl_match:

                            cyl_text = cyl_match.group(2)

                            digits = re.findall(
                                r'[0-9]+',
                                cyl_text
                            )

                            if digits:

                                result["Cylinders"] = digits[0]

                    # ------------------
                    # MODEL
                    # ------------------

                    model = era.get(
                        "position_8_vehicle_line",
                        {}
                    ).get(
                        pos8
                    )

                    if model:

                        result["Model"] = model

                    # ------------------
                    # REGIONAL SPEC
                    # ------------------

                    result["Regional Spec"] = "Unknown"

                    st.json(result)