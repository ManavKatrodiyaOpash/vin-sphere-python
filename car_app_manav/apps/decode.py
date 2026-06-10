from .rules import get_manufacturer_file, load_rules, load_brand_patterns
from .vin_check import validate_check_digit, validate_vin_chars, YEAR_MAP
from .styles import is_european_format

# =====================================================
# TOYOTA DECODER
# =====================================================

bmw_patterns = load_brand_patterns("bmw")
# audi_patterns = load_brand_patterns("audi")
chevrolet_patterns = load_brand_patterns("chevrolet")
ford_patterns = load_brand_patterns("ford")
gmc_patterns = load_brand_patterns("gmc")
honda_patterns = load_brand_patterns("honda")
hyundai_patterns = load_brand_patterns("hyundai")
infiniti_patterns = load_brand_patterns("infiniti")
jeep_patterns = load_brand_patterns("jeep")
jetour_patterns = load_brand_patterns("jetour")
kia_patterns = load_brand_patterns("kia")
land_rover_patterns = load_brand_patterns("land_rover")
lexus_patterns = load_brand_patterns("lexus")
mazda_patterns = load_brand_patterns("mazda")
mercedes_benz_patterns = load_brand_patterns("mercedes_benz")
mitsubishi_patterns = load_brand_patterns("mitsubishi")
nissan_patterns = load_brand_patterns("nissan")
suzuki_patterns = load_brand_patterns("suzuki")
tesla_patterns = load_brand_patterns("tesla")
toyota_patterns = load_brand_patterns("toyota")
volkswagen_patterns = load_brand_patterns("volkswagen")

def _apply_pattern(vin, patterns, brand_name, result):
    """Generic pattern engine: lookup VDS+year in brand patterns, inject fields, return hit bool."""
    try:
        current_year_int = int(result.get("model_year", 0))
    except:
        current_year_int = 0

    vds_5 = vin[3:8]
    lookup_key = f"{vds_5}_{current_year_int}"

    if lookup_key not in patterns:
        result["notes"].append(
            f"{brand_name}: pattern not found in VDS database. Running structural rule decoder."
        )
        return False

    p = patterns[lookup_key]
    result["notes"].append(f"{brand_name} match found in Pattern Engine (VDS+Year Matrix).")

    if p.get("model"):
        result["series_line"] = p["model"]
    if p.get("bodyType"):
        result["body_type"] = p["bodyType"]
    if p.get("cylinders"):
        result["cylinder"] = str(p["cylinders"])
    if p.get("regionalSpec"):
        result["regional_space"] = p["regionalSpec"]
    if p.get("trim"):
        result["trim"] = p["trim"]
    if p.get("noOfPassengers"):
        result["no_of_passengers"] = p["noOfPassengers"]
    if p.get("color"):
        result["color"] = p["color"]
    if p.get("weightInKg"):
        result["weight"] = str(p["weightInKg"])
    if p.get("number_of_doors") or p.get("doors"):
        result["number_of_doors"] = str(p.get("number_of_doors") or p.get("doors"))
    if p.get("driveType") or p.get("drive_type"):
        result["drive_type"] = p.get("drive_type") or p.get("driveType")
    if p.get("Transmission") or p.get("transmission"):
        result["Transmission"] = p.get("Transmission") or p.get("transmission")
    if p.get("possible_trims"):
        result["possible_trims_list"] = p["possible_trims"]

    return True


def decode_toyota(vin, rules, result):
    """Hybrid Toyota Decoder: Pattern Engine + Structural Fallback Decoder"""

    import re

    vds_5 = vin[3:8]
    pos4 = vin[3]
    pos5 = vin[4]
    pos6 = vin[5]
    pos7 = vin[6]
    pos8 = vin[7]
    pos10 = vin[9]
    pos11 = vin[10]

    # -------------------------------------------------
    # MODEL YEAR
    # -------------------------------------------------

    year_from_map = YEAR_MAP.get(pos10)

    if year_from_map:
        result["model_year"] = str(year_from_map)
    else:
        year_code_map = (
            rules.get("model_year_codes", {})
            or rules.get("position_10_model_year", {})
            or rules.get("year_codes", {})
        )

        fallback_y = year_code_map.get(pos10)

        if fallback_y:
            result["model_year"] = str(fallback_y)

    try:
        current_year_int = int(result["model_year"])
    except:
        current_year_int = 0

    # -------------------------------------------------
    # TOYOTA PATTERN ENGINE
    # -------------------------------------------------

    lookup_key = f"{vds_5}_{current_year_int}"

    _apply_pattern(vin, toyota_patterns, "Toyota", result)

    # -------------------------------------------------
    # STRUCTURAL FALLBACK DECODER
    # -------------------------------------------------

    era = rules.get(
        "era_2010_present"
        if current_year_int >= 2010
        else "era_1996_2009",
        {}
    )

    # -------------------------------------------------
    # POSITION 4 BODY TYPE
    # -------------------------------------------------

    if (
        result["body_type"] == "Unknown"
        or result["drive_type"] == "Unknown"
        or result["number_of_doors"] == "Unknown"
    ):

        p4 = era.get("position_4_body_type", {})

        for cat in p4.keys():

            if isinstance(p4[cat], dict) and pos4 in p4[cat]:

                target = p4[cat][pos4]

                if isinstance(target, dict):

                    if result["body_type"] == "Unknown":
                        result["body_type"] = target.get(
                            "body_type",
                            "Unknown"
                        )

                    if result["drive_type"] == "Unknown":
                        result["drive_type"] = target.get(
                            "drive_type",
                            "Unknown"
                        )

                    if result["number_of_doors"] == "Unknown":
                        result["number_of_doors"] = target.get(
                            "number_of_doors",
                            "Unknown"
                        )

                else:

                    if result["body_type"] == "Unknown":
                        result["body_type"] = target

                break

    # -------------------------------------------------
    # POSITION 5 ENGINE
    # -------------------------------------------------

    if (
        result["engine"] == "Unknown"
        or result["engine"] == "Not Available"
    ):

        eng = era.get(
            "position_5_engine",
            {}
        ).get(pos5)

        if eng:

            result["engine"] = eng

            if result["cylinder"] == "Unknown":

                cyl_match = re.search(
                    r'(I|V|H)(\d+)',
                    eng,
                    re.IGNORECASE
                )

                if cyl_match:
                    result["cylinder"] = cyl_match.group(2)

    # -------------------------------------------------
    # 2010+
    # -------------------------------------------------

    if current_year_int >= 2010:

        if result["restraint_system"] == "Unknown":

            rs = era.get(
                "position_6_restraint",
                {}
            ).get(pos6)

            if isinstance(rs, dict):

                result["restraint_system"] = rs.get(
                    "restraint_system",
                    "Unknown"
                )

                result["number_of_airbags"] = rs.get(
                    "number_of_airbags"
                )

                result["curtain_airbags"] = rs.get(
                    "curtain_airbags"
                )

                if rs.get("driver_knee_airbag"):
                    result["driver_knee_airbag"] = "Yes"

                if rs.get("passenger_knee_airbag"):
                    result["passenger_knee_airbag"] = "Yes"

                if rs.get("side_airbags"):
                    result["side_airbags"] = "Yes"

                if rs.get("front_airbags"):
                    result["front_airbags"] = "Yes"

                if rs.get("rear_airbags"):
                    result["rear_airbags"] = "Yes"

                if rs.get("front_center_airbag"):
                    result["front_center_airbag"] = "Yes"

            elif rs:

                result["restraint_system"] = rs

        if result["series_line"] == "Unknown":

            p7 = era.get(
                "position_7_series",
                {}
            )

            for grp in p7.values():

                if isinstance(grp, dict) and pos7 in grp:

                    result["series_line"] = grp[pos7]
                    break

    # -------------------------------------------------
    # PRE-2010
    # -------------------------------------------------

    else:

        if result["series_line"] == "Unknown":

            ser = era.get(
                "position_6_series",
                {}
            )

            for grp in ser.values():

                if isinstance(grp, dict) and pos6 in grp:

                    result["series_line"] = grp[pos6]
                    break

        if result["restraint_system"] == "Unknown":

            rs = era.get(
                "position_7_restraint_passenger",
                {}
            ).get(pos7)

            if isinstance(rs, dict):

                result["restraint_system"] = rs.get(
                    "restraint_system",
                    "Unknown"
                )

                result["number_of_airbags"] = rs.get(
                    "number_of_airbags"
                )

                result["curtain_airbags"] = rs.get(
                    "curtain_airbags"
                )

                if rs.get("side_airbags"):
                    result["side_airbags"] = "Yes"

                if rs.get("driver_knee_airbag"):
                    result["driver_knee_airbag"] = "Yes"

                if rs.get("passenger_knee_airbag"):
                    result["passenger_knee_airbag"] = "Yes"

                if rs.get("front_airbags"):
                    result["front_airbags"] = "Yes"

                if rs.get("rear_airbags"):
                    result["rear_airbags"] = "Yes"

                if rs.get("front_center_airbag"):
                    result["front_center_airbag"] = "Yes"

            elif rs:

                result["restraint_system"] = rs

    # -------------------------------------------------
    # POSITION 8 VEHICLE LINE
    # -------------------------------------------------

    if result["series_line"] == "Unknown":

        vl = era.get(
            "position_8_vehicle_line",
            {}
        ).get(pos8)

        if vl:
            result["series_line"] = vl

    result["model_platform"] = era.get(
        "position_8_vehicle_line",
        {}
    ).get(
        pos8,
        result["model_platform"]
    )

    # -------------------------------------------------
    # POSITION 11 PLANT
    # -------------------------------------------------

    plant = rules.get(
        "position_11_plant",
        {}
    ).get(pos11)

    if plant:
        result["plant"] = plant

    return result

# =====================================================
# NISSAN DECODER
# =====================================================

def decode_nissan(vin, rules, result):
    pos4 = vin[3]
    pos5 = vin[4]
    pos6 = vin[5]
    pos7 = vin[6]
    pos8 = vin[7]
    pos10 = vin[9]
    pos11 = vin[10]

    # ── MODEL YEAR ────────────────────────────────────────────────
    year_map = rules.get("position_10_model_year", {})
    year_val = year_map.get(pos10)
    if year_val:
        result["model_year"] = str(year_val)

    _apply_pattern(vin, nissan_patterns, "Nissan", result)

    year_map = rules.get("position_10_model_year", {})
    year_val = year_map.get(pos10)
    if year_val:
        result["model_year"] = str(year_val)
    model_year_int = int(year_val) if year_val else 0

    # is_truck_wmi: covers ALL truck/mpv/van type strings in Nissan JSON
    vtype = result.get("vehicle_type", "").lower()
    is_truck_wmi = any(t in vtype for t in (
        "truck", "multi-purpose", "mpv", "van", "bus", "standard"
    ))

    # Infiniti detection by WMI vehicle_type
    is_infiniti = "infiniti" in vtype

    # ── POSITION 4: ENGINE (era-split, passenger vs truck sub-dict) ────────────
    p4_block = rules.get("position_4_engine_line", {})
    era_key = "era_2010_present" if model_year_int >= 2010 else "era_1997_2009"
    era_p4 = p4_block.get(era_key, {})
    p4_sub = "trucks_mpv" if is_truck_wmi else "passenger_cars"
    engine_val = era_p4.get(p4_sub, {}).get(pos4)
    if not engine_val:
        other_sub = "passenger_cars" if is_truck_wmi else "trucks_mpv"
        engine_val = era_p4.get(other_sub, {}).get(pos4)
    if engine_val:
        if isinstance(engine_val, dict):
            result["series_line"] = engine_val.get("series_line", "Unknown")
            result["engine"] = engine_val.get("engine", "Unknown")
        else:
            result["engine"] = engine_val

    # ── POSITION 5: PLATFORM / MODEL LINE ────────────────────────
    p5_val = rules.get("position_5_platform_line", {}).get(pos5)
    if p5_val:
        result["model_platform"] = p5_val

    # ── POSITION 6: GENERATION CODE ───────────────────────────
    p6_block = rules.get("position_6_generation_code", {})
    if is_infiniti:
        p6_pref = ["infiniti_luxury", "nissan_passenger_ev", "nissan_truck_suv_mpv"]
    elif is_truck_wmi:
        p6_pref = ["nissan_truck_suv_mpv", "nissan_passenger_ev", "infiniti_luxury"]
    else:
        p6_pref = ["nissan_passenger_ev", "infiniti_luxury", "nissan_truck_suv_mpv"]
    p6_val = None
    for sub_name in p6_pref:
        sub = p6_block.get(sub_name, {})
        if pos6 in sub:
            p6_val = sub[pos6]
            break
    if p6_val:
        result["model_generation"] = p6_val
        # result["model_platform"] = p6_val

    # ── POSITION 7: BODY STYLE (pre-2020) or TRIM (2020+) ────────────────────
    p7_block = rules.get("position_7", {})
    if model_year_int >= 2020:
        p7_sub = "era_2020_present_mpv_suv_trims" if is_truck_wmi else "era_2020_present_passenger_trims"
        p7_val = p7_block.get(p7_sub, {}).get(pos7)
        if not p7_val:
            other_p7 = "era_2020_present_passenger_trims" if is_truck_wmi else "era_2020_present_mpv_suv_trims"
            p7_val = p7_block.get(other_p7, {}).get(pos7)
        if isinstance(p7_val, dict):
            result["trim"] = p7_val.get("trim", "Unknown")
        result["notes"].append("Nissan MY2020+: pos7=Trim Level.")
    else:
        p7_sub = "era_1997_2019_truck_cabs" if is_truck_wmi else "era_1997_2019_body_styles"
        p7_val = p7_block.get(p7_sub, {}).get(pos7)
        if not p7_val:
            other_p7 = "era_1997_2019_body_styles" if is_truck_wmi else "era_1997_2019_truck_cabs"
            p7_val = p7_block.get(other_p7, {}).get(pos7)
        if isinstance(p7_val, dict):
            result["body_type"] = p7_val.get("body_type", "Unknown")
            # if p7_val.get("number_of_doors"):
            #     result["number_of_doors"] = str(p7_val["number_of_doors"])
            result["number_of_doors"] = p7_val.get("number_of_doors", "Unknown")
            if p7_val.get("bed_type"):
                result["bed_type"] = p7_val["bed_type"]

    # ── POSITION 8: RESTRAINTS ────────────────────────────────────────────
    p8_block = rules.get("position_8_restraints", {})
    p8_sub = "mpv_truck" if is_truck_wmi else "passenger_cars"
    p8_dict = p8_block.get(p8_sub, {})
    # year-aware lookup: 2023+ may use compound keys like A_2023
    p8_val = p8_dict.get(pos8)
    if not p8_val and model_year_int >= 2023:
        p8_val = p8_dict.get(pos8 + "_2023")
    if not p8_val:
        # fallback to other sub-dict
        other_p8 = "passenger_cars" if is_truck_wmi else "mpv_truck"
        other_dict = p8_block.get(other_p8, {})
        p8_val = other_dict.get(pos8)
        if not p8_val and model_year_int >= 2023:
            p8_val = other_dict.get(pos8 + "_2023")
    if isinstance(p8_val, dict):
        result["restraint_system"] = p8_val.get("restraint_system", "Unknown")
        result["number_of_airbags"] = p8_val.get("number_of_airbags")
        if p8_val.get("front_airbags"):
            result["front_airbags"] = "Yes"
        if p8_val.get("side_airbags"):
            result["side_airbags"] = "Yes"
        if p8_val.get("curtain_airbags"):
            result["curtain_airbags"] = "Yes"
        if p8_val.get("knee_airbags"):
            result["driver_knee_airbag"] = "Yes"
        if p8_val.get("rear_side_airbags"):
            result["rear_airbags"] = "Yes"
        if p8_val.get("front_center_airbag"):
            result["front_center_airbag"] = "Yes"
    elif isinstance(p8_val, str):
        result["restraint_system"] = p8_val

    # ── POSITION 11: PLANT ────────────────────────────────────────────────
    plant = rules.get("position_11_plants", {}).get(pos11)
    if plant:
        result["plant"] = plant

    return result

# =====================================================
# BMW DECODER
# =====================================================

def decode_bmw(vin, rules, result):
    pos4 = vin[3]; pos5 = vin[4]; pos6 = vin[5]
    pos7 = vin[6]; pos8 = vin[7]

    _apply_pattern(vin, bmw_patterns, "BMW", result)

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

    _apply_pattern(vin, audi_patterns, "Audi", result)

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
    pos9 = vin[8]

    _apply_pattern(vin, hyundai_patterns, "Hyundai", result)

    p4 = rules.get("position_4_model_line", {}).get(pos4)
    if p4: result["series_line"] = p4

    # Era split: pos5/pos6 swapped at MY2001
    year_map = rules.get("model_year_codes", {})
    year_val = year_map.get(pos10)
    model_year_int = int(year_val) if year_val else 0

    if model_year_int >= 2001:
        p5 = rules.get("position_5_post2001_trim", {}).get(pos5)
        if p5: 
            if isinstance(p5, dict):
                result["model_generation"] = p5.get("trim", "Unknown")
                result["trim"] = p5.get("trim", "Unknown")
        p6 = rules.get("position_6_post2001_body", {}).get(pos6)

        if isinstance(p6, dict):

            if p6.get("body_type"):
                result["body_type"] = p6["body_type"]

            if p6.get("number_of_doors"):
                result["number_of_doors"] = p6["number_of_doors"]

            if p6.get("drive_type"):
                result["drive_type"] = p6["drive_type"]

            if p6.get("cab_type"):
                result["cab_type"] = p6["cab_type"]

            if p6.get("series_line"):
                result["series_line"] = p6["series_line"]

            if p6.get("raw_description"):
                result["body_style_description"] = p6["raw_description"]
        result["notes"].append("Hyundai MY2003+: pos5=trim level, pos6=body type.")
        
    else:
        p5 = rules.get("position_5_pre2001_body", {}).get(pos5)
        if isinstance(p5, dict):
            result["body_type"] = p5.get("body_type")

            if p5.get("number_of_doors"):
                result["number_of_doors"] = p5["number_of_doors"]

            if p5.get("raw_description"):
                result["body_style_description"] = p5["raw_description"]
                
        p6 = rules.get("position_6_pre2001_trim", {}).get(pos6)
        if p6: 
            result["model_generation"] = p6
            result["trim"] = p6
        result["notes"].append("Hyundai pre-2001: pos5=body style, pos6=trim level.")

    restraint_groups = rules.get("position_7_restraint", {})
    
    p7 = None

    for grp in restraint_groups.values():

        if isinstance(grp, dict) and pos7 in grp:
            p7 = grp[pos7]
            break
    
    if p7: 
        if isinstance(p7, dict):

            result["restraint_system"] = p7.get(
                "restraint_system",
                "Unknown"
            )

            result["number_of_airbags"] = p7.get(
                "number_of_airbags",
                "Unknown"
            )

            result["curtain_airbags"] = p7.get(
                "curtain_airbags",
                "Unknown"   
            )

            if p7.get("front_airbags"):
                result["front_airbags"] = "Yes"

            if p7.get("side_airbags"):
                result["side_airbags"] = "Yes"

            if p7.get("rear_airbags"):
                result["rear_airbags"] = "Yes"

            if p7.get("driver_knee_airbag"):
                result["driver_knee_airbag"] = "Yes"

            if p7.get("passenger_knee_airbag"):
                result["passenger_knee_airbag"] = "Yes"

            if p7.get("front_center_airbag"):
                result["front_center_airbag"] = "Yes"

    p8 = rules.get("position_8_engine", {}).get(pos8)
    if p8: result["engine"] = p8
    
    if result["country"] == "India":
        p9 = rules.get("position_9_Transmission", {}).get(pos9)
        result["Transmission"] = p9 if p9 else "Unknown"

    return result

# =====================================================
# MERCEDES-BENZ DECODER
# =====================================================

def decode_mercedes(vin, rules, result):
    pos7 = vin[6]; pos8 = vin[7]

    _apply_pattern(vin, mercedes_benz_patterns, "Mercedes-Benz", result)

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

    _apply_pattern(vin, ford_patterns, "Ford", result)

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

    _apply_pattern(vin, volkswagen_patterns, "Volkswagen", result)

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
# PATTERN-ONLY DECODERS (no separate structural JSON)
# =====================================================

def decode_lexus(vin, rules, result):
    _apply_pattern(vin, lexus_patterns, "Lexus", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_honda(vin, rules, result):
    _apply_pattern(vin, honda_patterns, "Honda", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_kia(vin, rules, result):
    _apply_pattern(vin, kia_patterns, "Kia", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_mitsubishi(vin, rules, result):
    _apply_pattern(vin, mitsubishi_patterns, "Mitsubishi", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_chevrolet(vin, rules, result):
    _apply_pattern(vin, chevrolet_patterns, "Chevrolet", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_gmc(vin, rules, result):
    _apply_pattern(vin, gmc_patterns, "GMC", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_land_rover(vin, rules, result):
    _apply_pattern(vin, land_rover_patterns, "Land Rover", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_suzuki(vin, rules, result):
    _apply_pattern(vin, suzuki_patterns, "Suzuki", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_jeep(vin, rules, result):
    _apply_pattern(vin, jeep_patterns, "Jeep", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_mazda(vin, rules, result):
    _apply_pattern(vin, mazda_patterns, "Mazda", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_infiniti(vin, rules, result):
    _apply_pattern(vin, infiniti_patterns, "Infiniti", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_tesla(vin, rules, result):
    _apply_pattern(vin, tesla_patterns, "Tesla", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
    return result

def decode_jetour(vin, rules, result):
    _apply_pattern(vin, jetour_patterns, "Jetour", result)
    p11 = rules.get("position_11_plant", {}).get(vin[10]) or rules.get("position_11_plants", {}).get(vin[10])
    if p11: result["plant"] = p11
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
        
        "body_type": "Unknown", "engine": "Unknown", "trim" : "Unknown",
        "drive_type": "Unknown", "number_of_doors": "Unknown",
        
        "restraint_system": "Unknown", "number_of_airbags": None,
        "curtain_airbags": None, "driver_knee_airbag": None, "side_airbags": None,
        "passenger_knee_airbag": None, "front_airbags": None, "rear_airbags": None,
        "front_center_airbag": None, "Transmission": "Unknown",
        
        "model_platform": "Unknown",
        "series_line": "Unknown", "model_generation": "Unknown",
        "model_year": "Unknown", "plant": "Unknown",
        "serial_number": serial,
        "pos4": pos4, "pos5": pos5, "pos5_6": pos5_6,
        "pos6": pos6, "pos7": pos7, "pos8": pos8,
        "pos9": pos9, "pos10": pos10, "pos11": pos11,
        "notes": [],
        
        "cylinder" : "Unknown", "color" : "Unknown", "weight" : "Unknown", 
        "regional_space" : "Unknown", "no_of_passengers" : "Unknown"
    }

    if not rules:
        result["notes"].append("Unable to identify manufacturer from WMI.")
        return result

    wmi_info = rules.get("wmi", {}).get(wmi, {})
    result["country"] = wmi_info.get("country", "Unknown")
    result["vehicle_type"] = wmi_info.get("vehicle_type") or wmi_info.get("type", "Unknown")
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


    # ── BRAND ROUTING ───────────────────────────
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
    elif mfr_name == "Lexus":
        result = decode_lexus(vin, rules, result)
    elif mfr_name == "Honda":
        result = decode_honda(vin, rules, result)
    elif mfr_name == "Kia":
        result = decode_kia(vin, rules, result)
    elif mfr_name == "Mitsubishi":
        result = decode_mitsubishi(vin, rules, result)
    elif mfr_name == "Chevrolet":
        result = decode_chevrolet(vin, rules, result)
    elif mfr_name == "GMC":
        result = decode_gmc(vin, rules, result)
    elif mfr_name == "Land Rover":
        result = decode_land_rover(vin, rules, result)
    elif mfr_name == "Suzuki":
        result = decode_suzuki(vin, rules, result)
    elif mfr_name == "Jeep":
        result = decode_jeep(vin, rules, result)
    elif mfr_name == "Mazda":
        result = decode_mazda(vin, rules, result)
    elif mfr_name == "Infiniti":
        result = decode_infiniti(vin, rules, result)
    elif mfr_name == "Tesla":
        result = decode_tesla(vin, rules, result)
    elif mfr_name == "Jetour":
        result = decode_jetour(vin, rules, result)
    
    if result["plant"] == "Unknown":
        plant_lookup = rules.get("position_11_plant", {}).get(result["pos11"]) or rules.get("position_11_plants", {}).get(result["pos11"])
        if plant_lookup: result["plant"] = plant_lookup

    return result