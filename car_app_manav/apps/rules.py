import os
import json


# =====================================================
# MANUFACTURER ROUTING
# =====================================================

def get_manufacturer_file(vin):
    vin = vin.upper()
    if len(vin) < 3:
        return None, None

    wmi = vin[:3]

    for fname, mfr in [
        ("../JSON/nissan.json",     "Nissan"),
        ("../JSON/toyota.json",     "Toyota"),
        ("../JSON/honda.json",      "Honda"),
        ("../JSON/bmw.json",        "BMW"),
        ("../JSON/audi.json",       "Audi"),
        ("../JSON/hyundai.json",    "Hyundai"),
        ("../JSON/mercedes.json",   "Mercedes-Benz"),
        ("../JSON/ford.json",       "Ford"),
        ("../JSON/volkswagen.json", "Volkswagen"),
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
# LOAD Patterns
# =====================================================

def load_brand_patterns(brand):
    try:
        path = f"../vds_jsons_all_brands/{brand}_vds_patterns.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        path = f"vds_jsons_all_brands/{brand}_vds_patterns.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)