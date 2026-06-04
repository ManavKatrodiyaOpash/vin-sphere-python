import streamlit as st
import json
import os
import re

st.set_page_config(
    page_title="VIN Decoder Pro",
    page_icon="🚗",
    layout="wide"
)

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
.badge-yellow { background: #1a1500; color: #ffcc3a; border: 1px solid #cc9a00; }

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
# VIN CHECK DIGIT VALIDATOR
# =====================================================

VIN_VALUES = {
    'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,
    'J':1,'K':2,'L':3,'M':4,'N':5,       'P':7,'R':9,
           'S':2,'T':3,'U':4,'V':5,'W':6,'X':7,'Y':8,'Z':9,
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9
}

VIN_WEIGHTS = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def validate_check_digit(vin):
    """Returns True if check digit (pos9) is valid."""
    try:
        total = sum(VIN_VALUES.get(c, 0) * VIN_WEIGHTS[i] for i, c in enumerate(vin))
        remainder = total % 11
        expected = 'X' if remainder == 10 else str(remainder)
        return vin[8] == expected, expected
    except Exception:
        return False, "?"

def validate_vin_chars(vin):
    """VIN cannot contain I, O, Q."""
    bad = [c for c in vin if c in ('I', 'O', 'Q')]
    return len(bad) == 0, bad


# =====================================================
# MANUFACTURER ROUTING
# =====================================================

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


# =====================================================
# LOAD JSON
# =====================================================

def load_rules(filename):
    # Try local dir first, then script dir
    paths = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


# =====================================================
# VW GROUP FORMAT DETECTION (Audi / Mercedes / VW)
# =====================================================

def is_european_format(vin):
    """Returns True if pos4-6 == 'ZZZ' — ISO filler used by VW Group / Mercedes EU-format VINs."""
    return len(vin) >= 7 and vin[3:6].upper() == "ZZZ"


# =====================================================
# BMW DECODER
# =====================================================

def decode_bmw(vin, rules, result):
    pos4 = vin[3]; pos5 = vin[4]; pos6 = vin[5]
    pos7 = vin[6]; pos8 = vin[7]

    p4 = rules.get("position_4_model_series", {}).get(pos4)
    if p4: result["series_line"] = p4

    p5 = rules.get("position_5_body_subvariant", {}).get(pos5)
    if p5: result["body_type"] = p5

    p6 = rules.get("position_6_engine_family", {}).get(pos6)
    if p6: result["engine"] = p6

    p7 = rules.get("position_7_drivetrain_variant", {}).get(pos7)
    if p7: result["restraint_system"] = p7  # drivetrain stored here for display

    p8 = rules.get("position_8_market_steering", {}).get(pos8)
    if p8: result["model_platform"] = p8

    result["notes"].append("BMW: pos7=drivetrain/variant (RWD/xDrive/eDrive), pos8=market/steering. GCC spec NOT in VIN.")
    return result


# =====================================================
# AUDI DECODER
# =====================================================

def decode_audi(vin, rules, result):
    pos7 = vin[6]; pos8 = vin[7]

    if is_european_format(vin):
        result["model_generation"] = "European Format — pos4-6: ZZZ filler (no data)"
        euro = rules.get("euro_format", {})
        p7 = euro.get("position_7_model_line", {}).get(pos7)
        if p7: result["series_line"] = p7
        p8 = euro.get("position_8_engine_family", {}).get(pos8)
        if p8: result["engine"] = p8
        result["notes"].append("Audi European-format VIN: pos4-6=ZZZ (no data). Model from pos7. Engine family from pos8. Full trim requires OEM PR codes.")
    else:
        us = rules.get("us_format", {})
        p4 = us.get("position_4_model_series", {}).get(vin[3])
        if p4: result["series_line"] = p4
        p5 = us.get("position_5_engine", {}).get(vin[4])
        if p5: result["engine"] = p5
        p6 = us.get("position_6_restraint", {}).get(vin[5])
        if p6: result["restraint_system"] = p6
        p7 = us.get("position_7_body_style", {}).get(pos7)
        if p7: result["body_type"] = p7
        p8 = us.get("position_8_drive_transmission", {}).get(pos8)
        if p8: result["model_platform"] = p8
        result["notes"].append("Audi US-format VIN: full NHTSA pos4-8 encoding.")
    return result


# =====================================================
# HYUNDAI DECODER
# =====================================================

def decode_hyundai(vin, rules, result):
    pos4 = vin[3]; pos5 = vin[4]; pos6 = vin[5]
    pos7 = vin[6]; pos8 = vin[7]; pos10 = vin[9]

    p4 = rules.get("position_4_model_line", {}).get(pos4)
    if p4: result["series_line"] = p4

    # Era split: pos5/pos6 swapped at MY2001
    year_map = rules.get("model_year_codes", {})
    year_val = year_map.get(pos10)
    model_year_int = int(year_val) if year_val else 0

    if model_year_int >= 2003:
        p5 = rules.get("position_5_post2001_trim", {}).get(pos5)
        if p5: result["model_generation"] = p5
        p6 = rules.get("position_6_post2001_body", {}).get(pos6)
        if p6: result["body_type"] = p6
        result["notes"].append("Hyundai MY2003+: pos5=trim level, pos6=body type.")
    else:
        p5 = rules.get("position_5_pre2001_body", {}).get(pos5)
        if p5: result["body_type"] = p5
        p6 = rules.get("position_6_pre2001_trim", {}).get(pos6)
        if p6: result["model_generation"] = p6
        result["notes"].append("Hyundai pre-2001: pos5=body style, pos6=trim level.")

    p7 = rules.get("position_7_restraint", {}).get(pos7)
    if p7: result["restraint_system"] = p7

    p8 = rules.get("position_8_engine", {}).get(pos8)
    if p8: result["engine"] = p8

    return result


# =====================================================
# MERCEDES-BENZ DECODER
# =====================================================

def decode_mercedes(vin, rules, result):
    pos7 = vin[6]; pos8 = vin[7]

    if is_european_format(vin):
        result["model_generation"] = "European Format — pos4-6: ZZZ filler (no data)"
        euro = rules.get("euro_format", {})
        p7 = euro.get("position_7_body_variant", {}).get(pos7)
        if p7: result["body_type"] = p7
        p8 = euro.get("position_8_drivetrain", {}).get(pos8)
        if p8: result["model_platform"] = p8
        result["notes"].append("Mercedes European-format VIN (WDD/WDC): pos4-6=ZZZ filler. Body/variant=pos7. Drivetrain/4MATIC=pos8. AMG Line/packages NOT in VIN.")
    else:
        us = rules.get("us_format", {})
        p4 = us.get("position_4_model_class", {}).get(vin[3])
        if p4: result["series_line"] = p4
        p5 = us.get("position_5_engine_family", {}).get(vin[4])
        if p5: result["engine"] = p5
        p6 = us.get("position_6_restraint", {}).get(vin[5])
        if p6: result["restraint_system"] = p6
        p7 = us.get("position_7_body_variant", {}).get(pos7)
        if p7: result["body_type"] = p7
        p8 = us.get("position_8_drivetrain", {}).get(pos8)
        if p8: result["model_platform"] = p8
        result["notes"].append("Mercedes US-format VIN (4JG Vance, AL): full NHTSA pos4-8 encoding.")
    return result


# =====================================================
# FORD DECODER
# =====================================================

def decode_ford(vin, rules, result):
    pos4 = vin[3]; pos5 = vin[4]; pos6 = vin[5]
    pos7 = vin[6]; pos8 = vin[7]

    p4 = rules.get("position_4_model_line", {}).get(pos4)
    if p4: result["series_line"] = p4

    p5 = rules.get("position_5_engine", {}).get(pos5)
    if p5: result["engine"] = p5

    p6 = rules.get("position_6_restraint", {}).get(pos6)
    if p6: result["restraint_system"] = p6

    p7 = rules.get("position_7_body_style", {}).get(pos7)
    if p7: result["body_type"] = p7

    p8 = rules.get("position_8_trim_drivetrain", {}).get(pos8)
    if p8: result["model_platform"] = p8

    result["notes"].append("Ford: strict NHTSA format. pos8=trim+drivetrain combined — precise trim may need OEM DB.")
    return result


# =====================================================
# VOLKSWAGEN DECODER
# =====================================================

def decode_volkswagen(vin, rules, result):
    pos7 = vin[6]; pos8 = vin[7]

    if is_european_format(vin):
        result["model_generation"] = "European Format — pos4-6: ZZZ filler (no data)"
        euro = rules.get("euro_format", {})
        p7 = euro.get("position_7_model_line", {}).get(pos7)
        if p7: result["series_line"] = p7
        p8 = euro.get("position_8_engine", {}).get(pos8)
        if p8: result["engine"] = p8
        result["notes"].append("VW European-format VIN: pos4-6=ZZZ (zero data). Model=pos7. Engine family=pos8. Full trim/options require PR codes on spare-wheel-well sticker.")
    else:
        us = rules.get("us_format", {})
        p4 = us.get("position_4_model", {}).get(vin[3])
        if p4: result["series_line"] = p4
        p5 = us.get("position_5_engine", {}).get(vin[4])
        if p5: result["engine"] = p5
        p6 = us.get("position_6_restraint", {}).get(vin[5])
        if p6: result["restraint_system"] = p6
        p7 = us.get("position_7_body", {}).get(pos7)
        if p7: result["body_type"] = p7
        p8 = us.get("position_8_transmission", {}).get(pos8)
        if p8: result["model_platform"] = p8
        result["notes"].append("VW US/Mexico-format VIN: full NHTSA pos4-8 encoding.")
    return result


# =====================================================
# CORE DECODER — aligned with corrected JSON schema
# =====================================================


def decode_vin(vin):
    vin = vin.upper().strip()

    valid_chars, bad_chars = validate_vin_chars(vin)
    check_ok, expected_check = validate_check_digit(vin)

    filename, mfr_name = get_manufacturer_file(vin)
    rules = load_rules(filename) if filename else None

    wmi = vin[:3]
    vds = vin[3:9]
    vis = vin[9:]

    pos4 = vin[3]
    pos5 = vin[4]
    pos5_6 = vin[4:6]
    pos6 = vin[5]
    pos7 = vin[6]
    pos8 = vin[7]
    pos9 = vin[8]
    pos10 = vin[9]
    pos11 = vin[10]
    serial = vin[11:]

    result = {
        "vin": vin, "wmi": wmi, "vds": vds, "vis": vis,
        "check_digit": pos9, "check_digit_valid": check_ok,
        "check_digit_expected": expected_check,
        "valid_chars": valid_chars, "invalid_chars_found": bad_chars,
        "manufacturer": mfr_name or "Unknown",
        "country": "Unknown", "vehicle_type": "Unknown",
        "wmi_description": "Unknown",
        "body_type": "Unknown", "engine": "Unknown",
        "drive_type": "Unknown", "number_of_doors": "Unknown",
        "restraint_system": "Unknown", "number_of_airbags": None,
        "curtain_airbags": None, "driver_knee_airbag": None, "side_airbags": None,
        "passenger_knee_airbag": None, "model_platform": "Unknown",
        "series_line": "Unknown", "model_generation": "Unknown",
        "model_year": "Unknown", "plant": "Unknown",
        "serial_number": serial,
        "pos4": pos4, "pos5": pos5, "pos5_6": pos5_6,
        "pos6": pos6, "pos7": pos7, "pos8": pos8,
        "pos9": pos9, "pos10": pos10, "pos11": pos11,
        "notes": []
    }

    if not rules:
        result["notes"].append("Unable to identify manufacturer from WMI.")
        return result

    wmi_info = rules.get("wmi", {}).get(wmi, {})
    result["country"] = wmi_info.get("country", "Unknown")
    result["vehicle_type"] = wmi_info.get("vehicle_type", "Unknown")
    result["wmi_description"] = wmi_info.get("manufacturer", mfr_name or "Unknown")

    # YEAR
    year_map = (
        rules.get("model_year_codes", {})
        or rules.get("position_10_model_year", {})
        or rules.get("year_codes", {})
    )

    year_val = year_map.get(pos10)
    if year_val:
        result["model_year"] = str(year_val)

    # TOYOTA ERA-BASED DECODER
    if mfr_name == "Toyota" and year_val:

        era = rules.get(
            "era_2010_present" if int(year_val) >= 2010 else "era_1996_2009",
            {}
        )

        p4 = era.get("position_4_body_type", {})
        for cat in p4.keys():
            if isinstance(p4[cat], dict) and pos4 in p4[cat]:
                target = p4[cat][pos4]
                if isinstance(target, dict):
                    result["body_type"] = target.get("body_type", "Unknown")
                    result["drive_type"] = target.get("drive_type", "Unknown")
                    result["number_of_doors"] = target.get("number_of_doors", "Unknown")
                else:
                    result["body_type"] = target
                break

        eng = era.get("position_5_engine", {}).get(pos5)
        if eng:
            result["engine"] = eng

        if int(year_val) >= 2010:
            rs = era.get("position_6_restraint", {}).get(pos6)
            if rs:
                if isinstance(rs, dict):
                    result["restraint_system"] = rs.get("restraint_system", "Unknown")
                    result["number_of_airbags"] = rs.get("number_of_airbags")
                    result["curtain_airbags"] = rs.get("curtain_airbags")

                    if rs.get("driver_knee_airbag"):
                        result["driver_knee_airbag"] = "Yes"

                    if rs.get("passenger_knee_airbag"):
                        result["passenger_knee_airbag"] = "Yes"
                    
                    if rs.get("side_airbags"):
                        result["side_airbags"] = "Yes"
                else:
                    result["restraint_system"] = rs

            p7 = era.get("position_7_series", {})
            for grp in p7.values():
                if isinstance(grp, dict) and pos7 in grp:
                    result["series_line"] = grp[pos7]
                    break
        else:
            ser = era.get("position_6_series", {})
            for grp in ser.values():
                if isinstance(grp, dict) and pos6 in grp:
                    result["series_line"] = grp[pos6]
                    break

            rs = era.get("position_7_restraint_passenger", {}).get(pos7)
            if rs:
                if isinstance(rs, dict):
                    result["restraint_system"] = rs.get("restraint_system", "Unknown")
                    result["number_of_airbags"] = rs.get("number_of_airbags")
                    result["curtain_airbags"] = rs.get("curtain_airbags")

                    if rs.get("side_airbags"):
                        result["side_airbags"] = "Yes"

                    if rs.get("driver_knee_airbag"):
                        result["driver_knee_airbag"] = "Yes"

                    if rs.get("passenger_knee_airbag"):
                        result["passenger_knee_airbag"] = "Yes"
                else:
                    result["restraint_system"] = rs

        vl = era.get("position_8_vehicle_line", {}).get(pos8)
        if vl:
            result["model_platform"] = vl

        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant:
            result["plant"] = plant

        return result

    # Generic Nissan/Honda logic remains unchanged
    # ── NEW BRAND ROUTING ───────────────────────────
    if mfr_name == "BMW":
        result = decode_bmw(vin, rules, result)
        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant: result["plant"] = plant
        return result

    if mfr_name == "Audi":
        result = decode_audi(vin, rules, result)
        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant: result["plant"] = plant
        return result

    if mfr_name == "Hyundai":
        year_map2 = rules.get("model_year_codes", {})
        yv = year_map2.get(pos10)
        if yv: result["model_year"] = str(yv)
        result = decode_hyundai(vin, rules, result)
        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant: result["plant"] = plant
        return result

    if mfr_name == "Mercedes-Benz":
        result = decode_mercedes(vin, rules, result)
        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant: result["plant"] = plant
        return result

    if mfr_name == "Ford":
        result = decode_ford(vin, rules, result)
        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant: result["plant"] = plant
        return result

    if mfr_name == "Volkswagen":
        result = decode_volkswagen(vin, rules, result)
        plant = rules.get("position_11_plant", {}).get(pos11)
        if plant: result["plant"] = plant
        return result

    # Generic Nissan/Honda fallback (unchanged)
    p4_map = rules.get("position_4", {})
    if p4_map.get(pos4):
        result["series_line"] = p4_map[pos4]

    p56_map = rules.get("position_5_and_6", {})
    if p56_map.get(pos5_6):
        result["model_platform"] = p56_map[pos5_6]

    p7_map = rules.get("position_7", {})
    if p7_map.get(pos7):
        result["body_type"] = p7_map[pos7]

    p8_map = rules.get("position_8", {})
    if p8_map.get(pos8):
        result["restraint_system"] = p8_map[pos8]

    plant = rules.get("position_11_plant", {}).get(pos11)
    if plant:
        result["plant"] = plant

    return result



def shorten_text(text):
    if not text or text == "Unknown":
        return "Not Available"
    text = str(text)

    if "(" in text:
        text = text.split("(")[0]

    if "/" in text:
        text = text.split("/")[0]

    return text.strip()


def clean_value(text):
    if not text or text == "Unknown":
        return "Not Available"
    return text


# =====================================================
# VIN VISUAL MAP HTML
# =====================================================

def vin_map_html(vin):
    segments = {
        0: "wmi", 1: "wmi", 2: "wmi",
        3: "vds", 4: "vds", 5: "vds", 6: "vds", 7: "vds",
        8: "check",
        9: "year",
        10: "plant",
        11: "serial", 12: "serial", 13: "serial",
        14: "serial", 15: "serial", 16: "serial",
    }
    labels = {
        0:"P1",1:"P2",2:"P3",
        3:"P4",4:"P5",5:"P6",6:"P7",7:"P8",
        8:"P9",9:"P10",10:"P11",
        11:"P12",12:"P13",13:"P14",14:"P15",15:"P16",16:"P17"
    }
    chars_html = ""
    for i, c in enumerate(vin):
        cls = segments.get(i, "")
        chars_html += f"""
        <div class="vin-char">
            <div class="char-box {cls}">{c}</div>
            <div class="char-pos">{labels[i]}</div>
        </div>"""

    legend = """
    <div class="vin-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#3a6aff"></div>WMI (1-3)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff6a3a"></div>VDS (4-8)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#6aff9a"></div>Check Digit (9)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff6aff"></div>Model Year (10)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ffcc3a"></div>Plant (11)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3affff"></div>Serial (12-17)</div>
    </div>"""

    return f"""
    <div class="vin-map-container">
        <div class="vin-map-title">VIN Structure Breakdown</div>
        <div class="vin-chars">{chars_html}</div>
        {legend}
    </div>"""


def info_row(key, value, style=""):
    return f"""
    <div class="info-row">
        <span class="info-key">{key}</span>
        <span class="info-val {style}">{value}</span>
    </div>"""


def section(title, rows_html):
    return f"""
    <div class="section-card">
        <div class="section-label">{title}</div>
        {rows_html}
    </div>"""


# =====================================================
# UI
# =====================================================

st.markdown('<div class="vin-header">Toyota VIN Decoder</div>', unsafe_allow_html=True)
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

# ── LIVE CHAR COUNT ──────────────────────────────────
if vin_input:
    remaining = 17 - len(vin_input)
    if remaining > 0:
        st.markdown(f'<p style="color:#5a5a7a; font-size:0.8rem; font-family: Space Mono, monospace;">{len(vin_input)}/17 chars — {remaining} more needed</p>', unsafe_allow_html=True)

if decode_btn or (vin_input and len(vin_input) == 17):

    if len(vin_input) != 17:
        st.markdown('<div class="error-box">⚠ VIN must be exactly 17 characters.</div>', unsafe_allow_html=True)

    else:
        r = decode_vin(vin_input)

        for k in ["body_type","engine","drive_type","number_of_doors","series_line","model_platform","restraint_system","plant"]:
            r[k] = clean_value(shorten_text(r.get(k)))


        # ── VIN MAP ─────────────────────────────────
        st.markdown(vin_map_html(vin_input), unsafe_allow_html=True)

        # ── VALIDATION BANNER ───────────────────────
        check_label = (
            '<span class="check-valid valid">✓ CHECK DIGIT VALID</span>'
            if r["check_digit_valid"] else
            f'<span class="check-valid invalid">✗ CHECK DIGIT INVALID (expected {r["check_digit_expected"]})</span>'
        )
        char_label = (
            '<span class="check-valid valid">✓ CHARACTERS VALID</span>'
            if r["valid_chars"] else
            f'<span class="check-valid invalid">✗ INVALID CHARS: {", ".join(r["invalid_chars_found"])}</span>'
        )
        st.markdown(
            f'<div style="display:flex; gap:12px; margin-bottom:1rem;">{check_label}{char_label}</div>',
            unsafe_allow_html=True
        )

        # ── NOTES / WARNINGS ────────────────────────
        for note in r["notes"]:
            st.markdown(f'<div class="note-box">ℹ {note}</div>', unsafe_allow_html=True)

        # ── LAYOUT: 3 COLUMNS ───────────────────────
        c1, c2, c3 = st.columns(3)

        with c1:
            rows = ""
            rows += info_row("Manufacturer", r["manufacturer"], "highlight")
            rows += info_row("Country", r["country"])
            rows += info_row("Vehicle Type", r["vehicle_type"])
            rows += info_row("WMI", f'<span class="badge badge-blue">{r["wmi"]}</span>')
            rows += info_row("WMI Entity", r["wmi_description"])
            st.markdown(section("World Manufacturer Identifier", rows), unsafe_allow_html=True)

            rows2 = ""
            rows2 += info_row("Model Year", f'<span class="badge badge-purple">{r["model_year"]}</span>')
            rows2 += info_row("Plant Code", f'<span class="badge badge-yellow">{r["pos11"]}</span>')
            rows2 += info_row("Plant", r["plant"])
            rows2 += info_row("Serial Number", f'<span style="font-family:Space Mono,monospace;color:#3affff">{r["serial_number"]}</span>')
            st.markdown(section("Vehicle Identity Section", rows2), unsafe_allow_html=True)

        with c2:
            rows = ""
            rows += info_row("Series / Line", r["series_line"])
            rows += info_row("Body Type", r["body_type"])
            rows += info_row("Drive Type", r["drive_type"])
            rows += info_row("Number of Doors", r["number_of_doors"])
            rows += info_row("Model Platform", r["model_platform"])
            st.markdown(section("Vehicle Descriptor Section", rows), unsafe_allow_html=True)

            rows2 = ""

            rows2 += info_row("Engine", r["engine"])

            if r["restraint_system"] not in ["Unknown", "Not Available", None]:
                rows2 += info_row("Restraint System", r["restraint_system"])

            if r.get("number_of_airbags"):
                rows2 += info_row("Airbags", r["number_of_airbags"])

            if r.get("curtain_airbags"):
                rows2 += info_row("Curtain Airbags", r["curtain_airbags"])
            
            if r.get("side_airbags"):
                rows2 += info_row("Side Airbags", r["side_airbags"])

            if r.get("driver_knee_airbag"):
                rows2 += info_row("Driver Knee Airbag", "Yes")

            if r.get("passenger_knee_airbag"):
                rows2 += info_row("Passenger Knee Airbag", "Yes")
                
            st.markdown(section("Powertrain & Safety", rows2), unsafe_allow_html=True)

        with c3:
            rows = ""
            rows += info_row("Pos 4", f'{r["pos4"]} → {r["series_line"][:40] if r["series_line"] != "Unknown" else "—"}')
            rows += info_row("Pos 5+6", f'{r["pos5_6"]} → {r["model_generation"][:40] if r["model_generation"] != "Unknown" else "—"}')
            rows += info_row("Pos 7", f'{r["pos7"]} → {r["body_type"][:40] if r["body_type"] != "Unknown" else r["restraint_system"][:40] if r["restraint_system"] != "Unknown" else "—"}')
            rows += info_row("Pos 8", f'{r["pos8"]} → {r["restraint_system"][:40] if r["restraint_system"] != "Unknown" else r["model_platform"][:40] if r["model_platform"] != "Unknown" else "—"}')
            rows += info_row("Pos 9 (Check)", f'{r["check_digit"]} {"✓" if r["check_digit_valid"] else "✗"}', "good" if r["check_digit_valid"] else "warn")
            rows += info_row("Pos 10 (Year)", f'{r["pos10"]} → {r["model_year"]}')
            rows += info_row("Pos 11 (Plant)", f'{r["pos11"]} → {r["plant"][:35]}')
            st.markdown(section("Position-by-Position Map", rows), unsafe_allow_html=True)

        # ── SEGMENT PILLS ────────────────────────────
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

        # ── RAW JSON ─────────────────────────────────
        with st.expander("Raw Decoded JSON"):
            st.json(r)