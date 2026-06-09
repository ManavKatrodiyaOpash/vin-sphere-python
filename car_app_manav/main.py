import streamlit as st
import json
import os
import re
import pandas as pd
from pathlib import Path

# =====================================================
# INITIAL SETUP & GLOBAL CACHE
# =====================================================

st.set_page_config(
    page_title="VIN Decoder Pro",
    page_icon="🚗",
    layout="wide"
)

@st.cache_data
def load_toyota_patterns():
    """Loads the high-confidence statistical Toyota VDS patterns."""
    json_file = "../car_app_dev/toyota_vds_year_patterns_v3.json"
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)
    return {}

toyota_patterns = load_toyota_patterns()

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.vin-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #f0f0ff;
    margin-bottom: 0.2rem;
}

.vin-subtitle {
    font-size: 0.95rem;
    color: #5a5a7a;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.vin-map-container {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.2rem 0;
}

.vin-map-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #4a4a6a;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.vin-chars {
    display: flex;
    gap: 4px;
    flex-wrap: nowrap;
    overflow-x: auto;
    padding-bottom: 0.5rem;
}

.vin-char {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 38px;
}

.char-box {
    width: 38px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 6px;
    border: 1.5px solid #2a2a3a;
    background: #16161f;
    color: #c8c8e8;
    transition: all 0.2s;
}

.char-box.wmi { border-color: #3a6aff; background: #0d1a3a; color: #6a9aff; }
.char-box.vds { border-color: #ff6a3a; background: #2a1008; color: #ff9a6a; }
.char-box.check { border-color: #6aff9a; background: #0a2015; color: #6aff9a; }
.char-box.year { border-color: #ff6aff; background: #1a0a1a; color: #ff9aff; }
.char-box.plant { border-color: #ffcc3a; background: #1a1500; color: #ffcc3a; }
.char-box.serial { border-color: #3affff; background: #001a1a; color: #3affff; }

.char-pos {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    color: #3a3a5a;
    margin-top: 4px;
}

.vin-legend {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: #5a5a7a;
}

.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
}

.section-card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a4a6a;
    margin-bottom: 0.8rem;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.55rem 0;
    border-bottom: 1px solid #16161f;
    gap: 1rem;
}

.info-row:last-child { border-bottom: none; }

.info-key {
    font-size: 0.82rem;
    color: #5a5a7a;
    font-weight: 400;
    white-space: nowrap;
    min-width: 130px;
}

.info-val {
    font-size: 0.88rem;
    color: #d8d8f0;
    font-weight: 500;
    text-align: right;
    word-break: break-word;
}

.info-val.highlight { color: #6a9aff; }
.info-val.good { color: #6aff9a; }
.info-val.warn { color: #ffcc3a; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    font-family: 'Space Mono', monospace;
}

.badge-blue { background: #0d1a3a; color: #6a9aff; border: 1px solid #3a6aff; }
.badge-orange { background: #2a1008; color: #ff9a6a; border: 1px solid #ff6a3a; }
.badge-green { background: #0a2015; color: #6aff9a; border: 1px solid #3aff7a; }
.badge-purple { background: #1a0a2a; color: #cc9aff; border: 1px solid #8a5aff; }
.badge-yellow { background: #1a1500; color: #ffcc3a; border: 1px solid #cc9000; }

.model-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    background: #16161f;
    border: 1px solid #2a2a4a;
    color: #a8a8d0;
    font-size: 0.8rem;
    margin: 3px;
    font-weight: 500;
}

.check-valid {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
}

.check-valid.valid { background: #0a2015; color: #6aff9a; border: 1px solid #3aff7a; }
.check-valid.invalid { background: #2a0808; color: #ff6a6a; border: 1px solid #ff3a3a; }

.error-box {
    background: #1a0808;
    border: 1px solid #3a1010;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #ff8a8a;
    font-size: 0.85rem;
}

.note-box {
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #6aaa6a;
    font-size: 0.78rem;
    font-style: italic;
    margin-top: 0.5rem;
}

.divider {
    border: none;
    border-top: 1px solid #1a1a2a;
    margin: 1.5rem 0;
}

stTextInput > div > div > input {
    background: #111118 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CORE STRUCTURAL VALIDATION UTILITIES
# =====================================================

VIN_VALUES = {
    'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,
    'J':1,'K':2,'L':3,'M':4,'N':5,       'P':7,'R':9,
           'S':2,'T':3,'U':4,'V':5,'W':6,'X':7,'Y':8,'Z':9,
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9
}

VIN_WEIGHTS = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

YEAR_MAP = {
    "V":1997, "W":1998, "X":1999, "Y":2000, "1":2001, "2":2002, "3":2003, "4":2004,
    "5":2005, "6":2006, "7":2007, "8":2008, "9":2009, "A":2010, "B":2011, "C":2012,
    "D":2013, "E":2014, "F":2015, "G":2016, "H":2017, "J":2018, "K":2019, "L":2020,
    "M":2021, "N":2022, "P":2023, "R":2024, "S":2025, "T":2026
}

def validate_check_digit(vin):
    try:
        total = sum(VIN_VALUES.get(c, 0) * VIN_WEIGHTS[i] for i, c in enumerate(vin))
        remainder = total % 11
        expected = 'X' if remainder == 10 else str(remainder)
        return vin[8] == expected, expected
    except Exception:
        return False, "?"

def validate_vin_chars(vin):
    bad = [c for c in vin if c in ('I', 'O', 'Q')]
    return len(bad) == 0, bad

def is_european_format(vin):
    return len(vin) >= 7 and vin[3:6].upper() == "ZZZ"

def get_manufacturer_file(vin):
    vin = vin.upper()
    if len(vin) < 3:
        return None, None
    wmi = vin[:3]
    for fname, mfr in [
        ("nissan.json",     "Nissan"),
        ("toyota.json",     "Toyota"),
        ("honda.json",      "Honda"),
        ("bmw.json",        "BMW"),
        ("audi.json",       "Audi"),
        ("hyundai.json",    "Hyundai"),
        ("mercedes.json",   "Mercedes-Benz"),
        ("ford.json",       "Ford"),
        ("volkswagen.json", "Volkswagen"),
    ]:
        rules = load_rules(fname)
        if rules and wmi in rules.get("wmi", {}):
            return fname, mfr
    return None, None

def load_rules(filename):
    paths = [filename, os.path.join(os.path.dirname(__file__), filename)]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

# =====================================================
# BRAND DECODER LOGIC MODULES
# =====================================================

def decode_toyota(vin, rules, result):
    """Hybrid Toyota Decoder: Merges pattern parsing with systemic fallbacks."""
    vds_5 = vin[3:8]
    pos10 = vin[9]
    year_from_map = YEAR_MAP.get(pos10)
    
    # Resolve Model Year Base
    if year_from_map:
        result["model_year"] = str(year_from_map)
    else:
        year_code_map = rules.get("model_year_codes", {}) or rules.get("position_10_model_year", {}) or rules.get("year_codes", {})
        fallback_y = year_code_map.get(pos10)
        if fallback_y:
            result["model_year"] = str(fallback_y)
            
    current_year_int = int(result["model_year"]) if result["model_year"] != "Unknown" else 0
    lookup_key = f"{vds_5}_{current_year_int}"
    
    # 1st Layer: VDS Pattern Database Match
    if lookup_key in toyota_patterns:
        pattern_rule = toyota_patterns[lookup_key]
        result["notes"].append("Toyota Match Found in Pattern Engine (VDS+Year Matrix).")
        
        result["series_line"] = pattern_rule.get("model") or result["series_line"]
        result["body_type"] = pattern_rule.get("bodyType") or result["body_type"]
        result["cylinder"] = str(pattern_rule.get("cylinders")) if pattern_rule.get("cylinders") else result["cylinder"]
        result["regional_space"] = pattern_rule.get("regionalSpec") or result["regional_space"]
        result["trim"] = pattern_rule.get("trim") or result["trim"]
        result["color"] = pattern_rule.get("color") or result["color"]
        result["weight"] = str(pattern_rule.get("weight")) if pattern_rule.get("weight") else result["weight"]
        
        # New Field Extractions from Pattern DB (Supporting CamelCase and Snake Case)
        if "doors" in pattern_rule or "number_of_doors" in pattern_rule:
            result["number_of_doors"] = str(pattern_rule.get("number_of_doors") or pattern_rule.get("doors"))
        if "driveType" in pattern_rule or "drive_type" in pattern_rule:
            result["drive_type"] = pattern_rule.get("drive_type") or pattern_rule.get("driveType")
        if "transmission" in pattern_rule or "Transmission" in pattern_rule:
            result["Transmission"] = pattern_rule.get("Transmission") or pattern_rule.get("transmission")
        # if "color" in pattern_rule:
        #     result["color"] = pattern_rule.get("color")
        # if "weight" in pattern_rule:
        #     result["weight"] = pattern_rule.get("weight")
        
        if "possible_trims" in pattern_rule:
            result["possible_trims_list"] = pattern_rule["possible_trims"]
    else:
        result["notes"].append("Pattern not found in VDS database. Running structural rule decoder.")

    # 2nd Layer: Fill missing attributes using original structural logic
    pos4, pos5, pos6, pos7, pos8, pos11 = vin[3], vin[4], vin[5], vin[6], vin[7], vin[10]
    era = rules.get("era_2010_present" if current_year_int >= 2010 else "era_1996_2009", {})
    
    # Enhanced multi-value check ensuring drive_type and doors don't stay Unknown if body_type matches from layout
    if result["body_type"] == "Unknown" or result["drive_type"] == "Unknown" or result["number_of_doors"] == "Unknown":
        p4 = era.get("position_4_body_type", {})
        for cat in p4.keys():
            if isinstance(p4[cat], dict) and pos4 in p4[cat]:
                target = p4[cat][pos4]
                if isinstance(target, dict):
                    if result["body_type"] == "Unknown":
                        result["body_type"] = target.get("body_type", "Unknown")
                    if result["drive_type"] == "Unknown":
                        result["drive_type"] = target.get("drive_type", "Unknown")
                    if result["number_of_doors"] == "Unknown":
                        result["number_of_doors"] = target.get("number_of_doors", "Unknown")
                else:
                    if result["body_type"] == "Unknown":
                        result["body_type"] = target
                break

    if result["engine"] == "Unknown" or result["engine"] == "Not Available":
        eng = era.get("position_5_engine", {}).get(pos5)
        if eng:
            result["engine"] = eng
            if result["cylinder"] == "Unknown":
                cyl_match = re.search(r'([0-9]+)[\\.-]?L.*?(I[0-9]|V[0-9]|H[0-9])', eng)
                if cyl_match:
                    cyl_text = cyl_match.group(2)
                    digits = re.findall(r'[0-9]+', cyl_text)
                    if digits:
                        result["cylinder"] = digits[0]

    if current_year_int >= 2010:
        if result["restraint_system"] == "Unknown":
            rs = era.get("position_6_restraint", {}).get(pos6)
            if isinstance(rs, dict):
                result["restraint_system"] = rs.get("restraint_system", "Unknown")
                result["number_of_airbags"] = rs.get("number_of_airbags")
                result["curtain_airbags"] = rs.get("curtain_airbags")
                if rs.get("driver_knee_airbag"): result["driver_knee_airbag"] = "Yes"
                if rs.get("passenger_knee_airbag"): result["passenger_knee_airbag"] = "Yes"
                if rs.get("side_airbags"): result["side_airbags"] = "Yes"
            elif rs:
                result["restraint_system"] = rs
                
        if result["series_line"] == "Unknown":
            p7 = era.get("position_7_series", {})
            for grp in p7.values():
                if isinstance(grp, dict) and pos7 in grp:
                    result["series_line"] = grp[pos7]
                    break
    else:
        if result["series_line"] == "Unknown":
            ser = era.get("position_6_series", {})
            for grp in ser.values():
                if isinstance(grp, dict) and pos6 in grp:
                    result["series_line"] = grp[pos6]
                    break
                    
        if result["restraint_system"] == "Unknown":
            rs = era.get("position_7_restraint_passenger", {}).get(pos7)
            if isinstance(rs, dict):
                result["restraint_system"] = rs.get("restraint_system", "Unknown")
                result["number_of_airbags"] = rs.get("number_of_airbags")
                result["curtain_airbags"] = rs.get("curtain_airbags")
                if rs.get("side_airbags"): result["side_airbags"] = "Yes"
                if rs.get("driver_knee_airbag"): result["driver_knee_airbag"] = "Yes"
                if rs.get("passenger_knee_airbag"): result["passenger_knee_airbag"] = "Yes"
            elif rs:
                result["restraint_system"] = rs

    if result["series_line"] == "Unknown":
        vl = era.get("position_8_vehicle_line", {}).get(pos8)
        if vl: result["series_line"] = vl
    result["model_platform"] = era.get("position_8_vehicle_line", {}).get(pos8, "Unknown")

    plant = rules.get("position_11_plant", {}).get(pos11)
    if plant: result["plant"] = plant

    return result

def decode_nissan(vin, rules, result):
    pos4, pos5, pos6, pos7, pos8, pos10, pos11 = vin[3], vin[4], vin[5], vin[6], vin[7], vin[9], vin[10]
    year_map = rules.get("position_10_model_year", {})
    year_val = year_map.get(pos10)
    if year_val: result["model_year"] = str(year_val)
    model_year_int = int(year_val) if year_val else 0

    vtype = result.get("vehicle_type", "").lower()
    is_truck_wmi = any(t in vtype for t in ("truck", "multi-purpose", "mpv", "van", "bus", "standard"))
    is_infiniti = "infiniti" in vtype

    p4_block = rules.get("position_4_engine_line", {})
    era_key = "era_2010_present" if model_year_int >= 2010 else "era_1997_2009"
    era_p4 = p4_block.get(era_key, {})
    p4_sub = "trucks_mpv" if is_truck_wmi else "passenger_cars"
    engine_val = era_p4.get(p4_sub, {}).get(pos4)
    if not engine_val:
        engine_val = era_p4.get("passenger_cars" if is_truck_wmi else "trucks_mpv", {}).get(pos4)
    if engine_val:
        if isinstance(engine_val, dict):
            result["series_line"] = engine_val.get("series_line", "Unknown")
            result["engine"] = engine_val.get("engine", "Unknown")
        else:
            result["engine"] = engine_val

    p5_val = rules.get("position_5_platform_line", {}).get(pos5)
    if p5_val: result["model_platform"] = p5_val

    p6_block = rules.get("position_6_generation_code", {})
    if is_infiniti: p6_pref = ["infiniti_luxury", "nissan_passenger_ev", "nissan_truck_suv_mpv"]
    elif is_truck_wmi: p6_pref = ["nissan_truck_suv_mpv", "nissan_passenger_ev", "infiniti_luxury"]
    else: p6_pref = ["nissan_passenger_ev", "infiniti_luxury", "nissan_truck_suv_mpv"]
    
    p6_val = None
    for sub_name in p6_pref:
        sub = p6_block.get(sub_name, {})
        if pos6 in sub:
            p6_val = sub[pos6]
            break
    if p6_val: result["model_generation"] = p6_val

    p7_block = rules.get("position_7", {})
    if model_year_int >= 2020:
        p7_sub = "era_2020_present_mpv_suv_trims" if is_truck_wmi else "era_2020_present_passenger_trims"
        p7_val = p7_block.get(p7_sub, {}).get(pos7) or p7_block.get("era_2020_present_passenger_trims" if is_truck_wmi else "era_2020_present_mpv_suv_trims", {}).get(pos7)
        if isinstance(p7_val, dict): result["trim"] = p7_val.get("trim", "Unknown")
        result["notes"].append("Nissan MY2020+: pos7=Trim Level.")
    else:
        p7_sub = "era_1997_2019_truck_cabs" if is_truck_wmi else "era_1997_2019_body_styles"
        p7_val = p7_block.get(p7_sub, {}).get(pos7) or p7_block.get("era_1997_2019_body_styles" if is_truck_wmi else "era_1997_2019_truck_cabs", {}).get(pos7)
        if isinstance(p7_val, dict):
            result["body_type"] = p7_val.get("body_type", "Unknown")
            result["number_of_doors"] = p7_val.get("number_of_doors", "Unknown")
            if p7_val.get("bed_type"): result["bed_type"] = p7_val["bed_type"]

    p8_block = rules.get("position_8_restraints", {})
    p8_dict = p8_block.get("mpv_truck" if is_truck_wmi else "passenger_cars", {})
    p8_val = p8_dict.get(pos8)
    if not p8_val and model_year_int >= 2023: p8_val = p8_dict.get(pos8 + "_2023")
    if not p8_val:
        other_dict = p8_block.get("passenger_cars" if is_truck_wmi else "mpv_truck", {})
        p8_val = other_dict.get(pos8) or (other_dict.get(pos8 + "_2023") if model_year_int >= 2023 else None)
        
    if isinstance(p8_val, dict):
        result["restraint_system"] = p8_val.get("restraint_system", "Unknown")
        result["number_of_airbags"] = p8_val.get("number_of_airbags")
        if p8_val.get("front_airbags"): result["front_airbags"] = "Yes"
        if p8_val.get("side_airbags"): result["side_airbags"] = "Yes"
        if p8_val.get("curtain_airbags"): result["curtain_airbags"] = "Yes"
        if p8_val.get("knee_airbags"): result["driver_knee_airbag"] = "Yes"
        if p8_val.get("rear_side_airbags"): result["rear_airbags"] = "Yes"
        if p8_val.get("front_center_airbag"): result["front_center_airbag"] = "Yes"
    elif isinstance(p8_val, str):
        result["restraint_system"] = p8_val

    plant = rules.get("position_11_plants", {}).get(pos11)
    if plant: result["plant"] = plant
    return result

def decode_bmw(vin, rules, result):
    p4 = rules.get("position_4_model_series", {}).get(vin[3])
    if p4: result["series_line"] = p4
    p5 = rules.get("position_5_body_subvariant", {}).get(vin[4])
    if p5: result["body_type"] = p5
    p6 = rules.get("position_6_engine_family", {}).get(vin[5])
    if p6: result["engine"] = p6
    p7 = rules.get("position_7_drivetrain_variant", {}).get(vin[6])
    if p7: result["restraint_system"] = p7
    p8 = rules.get("position_8_market_steering", {}).get(vin[7])
    if p8: result["model_platform"] = p8
    result["notes"].append("BMW: pos7=drivetrain/variant (RWD/xDrive/eDrive), pos8=market/steering. GCC spec NOT in VIN.")
    return result

def decode_audi(vin, rules, result):
    if is_european_format(vin):
        result["model_generation"] = "European Format — pos4-6: ZZZ filler (no data)"
        euro = rules.get("euro_format", {})
        p7 = euro.get("position_7_model_line", {}).get(vin[6])
        if p7: result["series_line"] = p7
        p8 = euro.get("position_8_engine_family", {}).get(vin[7])
        if p8: result["engine"] = p8
        result["notes"].append("Audi European-format VIN: pos4-6=ZZZ. Engine family from pos8.")
    else:
        us = rules.get("us_format", {})
        p4 = us.get("position_4_model_series", {}).get(vin[3])
        if p4: result["series_line"] = p4
        p5 = us.get("position_5_engine", {}).get(vin[4])
        if p5: result["engine"] = p5
        p6 = us.get("position_6_restraint", {}).get(vin[5])
        if p6: result["restraint_system"] = p6
        p7 = us.get("position_7_body_style", {}).get(vin[6])
        if p7: result["body_type"] = p7
        p8 = us.get("position_8_drive_transmission", {}).get(vin[7])
        if p8: result["model_platform"] = p8
        result["notes"].append("Audi US-format VIN: full NHTSA pos4-8 encoding.")
    return result

def decode_hyundai(vin, rules, result):
    pos4, pos5, pos6, pos7, pos8, pos9, pos10 = vin[3], vin[4], vin[5], vin[6], vin[7], vin[8], vin[9]
    p4 = rules.get("position_4_model_line", {}).get(pos4)
    if p4: result["series_line"] = p4

    model_year_int = int(result["model_year"]) if result["model_year"] != "Unknown" else 0

    if model_year_int >= 2001:
        p5 = rules.get("position_5_post2001_trim", {}).get(pos5)
        if isinstance(p5, dict): result["model_generation"] = result["trim"] = p5.get("trim", "Unknown")
        elif p5: result["model_generation"] = result["trim"] = p5
        
        p6 = rules.get("position_6_post2001_body", {}).get(pos6)
        if isinstance(p6, dict):
            if p6.get("body_type"): result["body_type"] = p6["body_type"]
            if p6.get("number_of_doors"): result["number_of_doors"] = p6["number_of_doors"]
            if p6.get("drive_type"): result["drive_type"] = p6["drive_type"]
            if p6.get("cab_type"): result["cab_type"] = p6["cab_type"]
            if p6.get("series_line"): result["series_line"] = p6["series_line"]
        result["notes"].append("Hyundai MY2001+: pos5=trim level, pos6=body type.")
    else:
        p5 = rules.get("position_5_pre2001_body", {}).get(pos5)
        if isinstance(p5, dict): result["body_type"] = p5.get("body_type")
        p6 = rules.get("position_6_pre2001_trim", {}).get(pos6)
        if p6: result["model_generation"] = result["trim"] = p6
        result["notes"].append("Hyundai pre-2001: pos5=body style, pos6=trim level.")

    restraint_groups = rules.get("position_7_restraint", {})
    p7 = None
    for grp in restraint_groups.values():
        if isinstance(grp, dict) and pos7 in grp:
            p7 = grp[pos7]
            break
    if isinstance(p7, dict):
        result["restraint_system"] = p7.get("restraint_system", "Unknown")
        result["number_of_airbags"] = p7.get("number_of_airbags", "Unknown")
        result["curtain_airbags"] = p7.get("curtain_airbags", "Unknown")
        if p7.get("front_airbags"): result["front_airbags"] = "Yes"
        if p7.get("side_airbags"): result["side_airbags"] = "Yes"
        if p7.get("rear_airbags"): result["rear_airbags"] = "Yes"
    elif p7:
        result["restraint_system"] = p7

    p8 = rules.get("position_8_engine", {}).get(pos8)
    if p8: result["engine"] = p8
    
    if result["country"] == "India":
        p9 = rules.get("position_9_Transmission", {}).get(pos9)
        result["Transmission"] = p9 if p9 else "Unknown"
    return result

def decode_mercedes(vin, rules, result):
    if is_european_format(vin):
        result["model_generation"] = "European Format — pos4-6: ZZZ filler (no data)"
        euro = rules.get("euro_format", {})
        p7 = euro.get("position_7_body_variant", {}).get(vin[6])
        if p7: result["body_type"] = p7
        p8 = euro.get("position_8_drivetrain", {}).get(vin[7])
        if p8: result["model_platform"] = p8
        result["notes"].append("Mercedes European VIN: pos4-6=ZZZ filler.")
    else:
        us = rules.get("us_format", {})
        p4 = us.get("position_4_model_class", {}).get(vin[3])
        if p4: result["series_line"] = p4
        p5 = us.get("position_5_engine_family", {}).get(vin[4])
        if p5: result["engine"] = p5
        p6 = us.get("position_6_restraint", {}).get(vin[5])
        if p6: result["restraint_system"] = p6
        p7 = us.get("position_7_body_variant", {}).get(vin[6])
        if p7: result["body_type"] = p7
        p8 = us.get("position_8_drivetrain", {}).get(vin[7])
        if p8: result["model_platform"] = p8
    return result

def decode_ford(vin, rules, result):
    p4 = rules.get("position_4_model_line", {}).get(vin[3])
    if p4: result["series_line"] = p4
    p5 = rules.get("position_5_engine", {}).get(vin[4])
    if p5: result["engine"] = p5
    p6 = rules.get("position_6_restraint", {}).get(vin[5])
    if p6: result["restraint_system"] = p6
    p7 = rules.get("position_7_body_style", {}).get(vin[6])
    if p7: result["body_type"] = p7
    p8 = rules.get("position_8_trim_drivetrain", {}).get(vin[7])
    if p8: result["model_platform"] = p8
    return result

def decode_volkswagen(vin, rules, result):
    if is_european_format(vin):
        result["model_generation"] = "European Format — pos4-6: ZZZ filler (no data)"
        euro = rules.get("euro_format", {})
        p7 = euro.get("position_7_model_line", {}).get(vin[6])
        if p7: result["series_line"] = p7
        p8 = euro.get("position_8_engine", {}).get(vin[7])
        if p8: result["engine"] = p8
    else:
        us = rules.get("us_format", {})
        p4 = us.get("position_4_model", {}).get(vin[3])
        if p4: result["series_line"] = p4
        p5 = us.get("position_5_engine", {}).get(vin[4])
        if p5: result["engine"] = p5
        p6 = us.get("position_6_restraint", {}).get(vin[5])
        if p6: result["restraint_system"] = p6
        p7 = us.get("position_7_body", {}).get(vin[6])
        if p7: result["body_type"] = p7
        p8 = us.get("position_8_transmission", {}).get(vin[7])
        if p8: result["model_platform"] = p8
    return result

# =====================================================
# CORE PIPELINE DECODER
# =====================================================

def decode_vin(vin):
    vin = vin.upper().strip()

    valid_chars, bad_chars = validate_vin_chars(vin)
    check_ok, expected_check = validate_check_digit(vin)

    filename, mfr_name = get_manufacturer_file(vin)
    rules = load_rules(filename) if filename else None

    result = {
        "vin": vin, "wmi": vin[:3], "vds": vin[3:9], "vis": vin[9:],
        "check_digit": vin[8], "check_digit_valid": check_ok,
        "check_digit_expected": expected_check,
        "valid_chars": valid_chars, "invalid_chars_found": bad_chars,
        "manufacturer": mfr_name or "Unknown",
        "country": "Unknown", "vehicle_type": "Unknown", "wmi_description": "Unknown",
        "body_type": "Unknown", "engine": "Unknown", "trim" : "Unknown",
        "drive_type": "Unknown", "number_of_doors": "Unknown",
        "restraint_system": "Unknown", "number_of_airbags": None,
        "curtain_airbags": None, "driver_knee_airbag": None, "side_airbags": None,
        "passenger_knee_airbag": None, "front_airbags": None, "rear_airbags": None,
        "front_center_airbag": None, "Transmission": "Unknown",
        "model_platform": "Unknown", "series_line": "Unknown", "model_generation": "Unknown",
        "model_year": "Unknown", "plant": "Unknown", "serial_number": vin[11:],
        "pos4": vin[3], "pos5": vin[4], "pos5_6": vin[4:6], "pos6": vin[5],
        "pos7": vin[6], "pos8": vin[7], "pos9": vin[8], "pos10": vin[9], "pos11": vin[10],
        "notes": [],
        "cylinder" : "Unknown", "color" : "Unknown", "weight" : "Unknown", 
        "regional_space" : "Unknown", "no_of_passanger" : "Unknown"
    }

    if not rules:
        result["notes"].append("Unable to identify manufacturer schemas via WMI context lookup.")
        return result

    wmi_info = rules.get("wmi", {}).get(result["wmi"], {})
    result["country"] = wmi_info.get("country", "Unknown")
    result["vehicle_type"] = wmi_info.get("vehicle_type") or wmi_info.get("type", "Unknown")
    result["wmi_description"] = wmi_info.get("manufacturer", mfr_name or "Unknown")

    # Establish Timeline Context
    year_map = rules.get("model_year_codes", {}) or rules.get("position_10_model_year", {}) or rules.get("year_codes", {})
    year_val = year_map.get(result["pos10"])
    if year_val: result["model_year"] = str(year_val)

    # Manufacturer Routing Rules
    if mfr_name == "Toyota":
        result = decode_toyota(vin, rules, result)
    elif mfr_name == "Nissan":
        result = decode_nissan(vin, rules, result)
    elif mfr_name == "BMW":
        result = decode_bmw(vin, rules, result)
    elif mfr_name == "Audi":
        result = decode_audi(vin, rules, result)
    elif mfr_name == "Hyundai":
        result = decode_hyundai(vin, rules, result)
    elif mfr_name == "Mercedes-Benz":
        result = decode_mercedes(vin, rules, result)
    elif mfr_name == "Ford":
        result = decode_ford(vin, rules, result)
    elif mfr_name == "Volkswagen":
        result = decode_volkswagen(vin, rules, result)

    # Post-Process Plant Codes dynamically if missing
    if result["plant"] == "Unknown":
        plant_lookup = rules.get("position_11_plant", {}).get(result["pos11"]) or rules.get("position_11_plants", {}).get(result["pos11"])
        if plant_lookup: result["plant"] = plant_lookup

    return result

# =====================================================
# FORMATTING ENGINE
# =====================================================

def shorten_text(text):
    if not text or text == "Unknown":
        return "Not Available"
    text = str(text)
    if "(" in text: text = text.split("(")[0]
    if "/" in text: text = text.split("/")[0]
    return text.strip()

def clean_value(text):
    if not text or text == "Unknown":
        return "Not Available"
    return text

def vin_map_html(vin, country, manufacturer):
    segments = {
        0: "wmi", 1: "wmi", 2: "wmi",
        3: "vds", 4: "vds", 5: "vds", 6: "vds", 7: "vds",
        8: "check", 9: "year", 10: "plant",
        11: "serial", 12: "serial", 13: "serial", 14: "serial", 15: "serial", 16: "serial",
    }
    labels = {i: f"P{i+1}" for i in range(17)}
    chars_html = "".join([f'<div class="vin-char"><div class="char-box {segments.get(i, "")}">{c}</div><div class="char-pos">{labels[i]}</div></div>' for i, c in enumerate(vin)])
    
    check_digit_text = "Transmission (9)" if country == "India" and manufacturer == "Hyundai" else "Check Digit (9)"

    legend = f"""
    <div class="vin-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#3a6aff"></div>WMI (1-3)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff6a3a"></div>VDS (4-8)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#6aff9a"></div>{check_digit_text}</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff6aff"></div>Model Year (10)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ffcc3a"></div>Plant (11)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3affff"></div>Serial (12-17)</div>
    </div>"""

    return f'<div class="vin-map-container"><div class="vin-map-title">VIN Structure Breakdown</div><div class="vin-chars">{chars_html}</div>{legend}</div>'

def info_row(key, value, style=""):
    return f'<div class="info-row"><span class="info-key">{key}</span><span class="info-val {style}">{value}</span></div>'

def section(title, rows_html):
    return f'<div class="section-card"><div class="section-label">{title}</div>{rows_html}</div>'

# =====================================================
# INTERACTIVE RUNTIME ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    st.markdown('<div class="vin-header">VIN Decoder</div>', unsafe_allow_html=True)
    st.markdown('<div class="vin-subtitle">VEHICLE IDENTIFICATION NUMBER ANALYSIS TOOL</div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        vin_input = st.text_input(
            "VIN Input",
            placeholder="Enter 17-character VIN  e.g.  JN8AS5MV3BW269745",
            max_chars=17,
            label_visibility="collapsed"
        ).strip().upper()

    with col_btn:
        decode_btn = st.button("Decode →", use_container_width=True)

    st.markdown("---")

    if vin_input:
        remaining = 17 - len(vin_input)
        if remaining > 0:
            st.markdown(f'<p style="color:#5a5a7a; font-size:0.8rem; font-family: Space Mono, monospace;">{len(vin_input)}/17 chars — {remaining} more needed</p>', unsafe_allow_html=True)

    if decode_btn or (vin_input and len(vin_input) == 17):
        if len(vin_input) != 17:
            st.markdown('<div class="error-box">⚠ VIN must be exactly 17 characters.</div>', unsafe_allow_html=True)
        else:
            r = decode_vin(vin_input)

            # Normalization Layer
            for k in ["body_type","engine","drive_type","number_of_doors","series_line","model_platform","restraint_system","plant","trim","cylinder","regional_space", "color", "weight"]:
                r[k] = clean_value(shorten_text(r.get(k)))

            # Render HTML Mapping Structure
            st.markdown(vin_map_html(vin_input, r["country"], r["manufacturer"]), unsafe_allow_html=True)

            # Integrity Banners
            check_label = ""
            if r["country"] != "India" or r["manufacturer"] != "Hyundai":
                check_label = ('<span class="check-valid valid">✓ CHECK DIGIT VALID</span>' if r["check_digit_valid"] else f'<span class="check-valid invalid">✗ CHECK DIGIT INVALID (expected {r["check_digit_expected"]})</span>')
                
            char_label = ('<span class="check-valid valid">✓ CHARACTERS VALID</span>' if r["valid_chars"] else f'<span class="check-valid invalid">✗ INVALID CHARS: {", ".join(r["invalid_chars_found"])}</span>')
            st.markdown(f'<div style="display:flex; gap:12px; margin-bottom:1rem;">{check_label}{char_label}</div>', unsafe_allow_html=True)

            # Notes
            for note in r["notes"]:
                st.markdown(f'<div class="note-box">ℹ {note}</div>', unsafe_allow_html=True)

            # Layout Architecture
            c1, c2 = st.columns(2)

            with c1:
                rows = info_row("Manufacturer", r["manufacturer"], "highlight") + \
                       info_row("Country", r["country"]) + \
                       info_row("Vehicle Type", r["vehicle_type"]) + \
                       info_row("WMI", f'<span class="badge badge-blue">{r["wmi"]}</span>') + \
                       info_row("WMI Entity", r["wmi_description"])
                st.markdown(section("World Manufacturer Identifier", rows), unsafe_allow_html=True)

                rows2 = info_row("Model Year", f'<span class="badge badge-purple">{r["model_year"]}</span>') + \
                        info_row("Plant Code", f'<span class="badge badge-yellow">{r["pos11"]}</span>') + \
                        info_row("Plant", r["plant"]) + \
                        info_row("Serial Number", f'<span style="font-family:Space Mono,monospace;color:#3affff">{r["serial_number"]}</span>')
                st.markdown(section("Vehicle Identity Section", rows2), unsafe_allow_html=True)

            with c2:
                rows = info_row("Model", r["series_line"])
                if r["model_platform"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Make", r["model_platform"])
                rows += info_row("Body Type", r["body_type"])
                if r["trim"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Trim", r["trim"])
                if r["Transmission"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Transmission", r["Transmission"])
                if r["drive_type"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Drive Type", r["drive_type"])
                if r["number_of_doors"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Number of Doors", r["number_of_doors"])
                if r.get("bed_type") not in ["Unknown", "Not Available", None, ""]:
                    rows += info_row("Bed Type", r["bed_type"])
                if r["regional_space"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Regional Spec", r["regional_space"])
                if r["cylinder"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Cylinders", r["cylinder"])
                rows += info_row("Color", r["color"])
                rows += info_row("Weight", r["weight"])
                
                st.markdown(section("Vehicle Descriptor Section", rows), unsafe_allow_html=True)


                rows2 = info_row("Engine", r["engine"])
                if r["restraint_system"] not in ["Unknown", "Not Available", None]:
                    rows2 += info_row("Restraint System", r["restraint_system"])
                if r["number_of_airbags"] not in [None, "", "Unknown", "Not Available"]:
                    rows2 += info_row("No of Airbags", r["number_of_airbags"])
                if r.get("front_airbags"): rows2 += info_row("Front Airbags", "Yes")
                if r.get("rear_airbags"): rows2 += info_row("Rear Airbags", "Yes")
                if r.get("curtain_airbags") not in [None, "", "Unknown", "Not Available"]:
                    rows2 += info_row("Curtain Airbags", r["curtain_airbags"])
                if r.get("side_airbags"): rows2 += info_row("Side Airbags", "Yes")
                if r.get("driver_knee_airbag"): rows2 += info_row("Driver Knee Airbag", "Yes")
                if r.get("passenger_knee_airbag"): rows2 += info_row("Passenger Knee Airbag", "Yes")
                if r.get("front_center_airbag"): rows2 += info_row("Front Center Airbag", "Yes")
                    
                st.markdown(section("Powertrain & Safety", rows2), unsafe_allow_html=True)

            # Statistical Ambiguity/Multi-Trim Handler Block
            if "possible_trims_list" in r:
                st.warning("Multiple trim variance distributions observed across identical VDS sequences.")
                st.write(pd.DataFrame({"Possible Trims Identified": r["possible_trims_list"]}))

            # Segment Pills Bar
            st.markdown(f"""
            <div class="section-card" style="margin-top:0.5rem;">
                <div class="section-label">VIN Segments</div>
                <div style="display:flex; gap:1rem; flex-wrap:wrap; align-items:center;">
                    <div>
                        <div style="font-size:0.65rem; color:#3a6aff; font-family:Space Mono,monospace; letter-spacing:0.15em; margin-bottom:4px;">WMI</div>
                        <span style="font-family:Space Mono,monospace; font-size:1.4rem; color:#6a9aff; letter-spacing:4px;">{r["wmi"]}</span>
                    </div>
                    <div style="color:#2a2a4a; font-size:1.2rem;">·</div>
                    <div>
                        <div style="font-size:0.65rem; color:#ff6a3a; font-family:Space Mono,monospace; letter-spacing:0.15em; margin-bottom:4px;">VDS (4-8)</div>
                        <span style="font-family:Space Mono,monospace; font-size:1.4rem; color:#ff9a6a; letter-spacing:4px;">{r["vds"][:5]}</span>
                    </div>
                    <div style="color:#2a2a4a; font-size:1.2rem;">·</div>
                    <div>
                        <div style="font-size:0.65rem; color:#6aff9a; font-family:Space Mono,monospace; letter-spacing:0.15em; margin-bottom:4px;">CHECK</div>
                        <span style="font-family:Space Mono,monospace; font-size:1.4rem; color:#6aff9a; letter-spacing:4px;">{r["vds"][5]}</span>
                    </div>
                    <div style="color:#2a2a4a; font-size:1.2rem;">·</div>
                    <div>
                        <div style="font-size:0.65rem; color:#ff6aff; font-family:Space Mono,monospace; letter-spacing:0.15em; margin-bottom:4px;">YEAR+PLANT</div>
                        <span style="font-family:Space Mono,monospace; font-size:1.4rem; color:#ff9aff; letter-spacing:4px;">{r["vis"][:2]}</span>
                    </div>
                    <div style="color:#2a2a4a; font-size:1.2rem;">·</div>
                    <div>
                        <div style="font-size:0.65rem; color:#3affff; font-family:Space Mono,monospace; letter-spacing:0.15em; margin-bottom:4px;">SERIAL</div>
                        <span style="font-family:Space Mono,monospace; font-size:1.4rem; color:#3affff; letter-spacing:4px;">{r["serial_number"]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Debug Expander
            with st.expander("Raw Decoded Object Context Map"):
                st.json(r)