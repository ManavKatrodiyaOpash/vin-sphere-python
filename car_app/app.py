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

    nissan_rules = load_rules("nissan.json")
    toyota_rules = load_rules("toyota.json")

    if nissan_rules and wmi in nissan_rules.get("wmi", {}):
        return "nissan.json", "Nissan"

    if toyota_rules and wmi in toyota_rules.get("wmi", {}):
        return "toyota.json", "Toyota"

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
# CORE DECODER — aligned with corrected JSON schema
# =====================================================

def decode_vin(vin):
    vin = vin.upper().strip()

    # Basic validation
    valid_chars, bad_chars = validate_vin_chars(vin)
    check_ok, expected_check = validate_check_digit(vin)

    filename, mfr_name = get_manufacturer_file(vin)
    rules = load_rules(filename) if filename else None

    wmi = vin[:3]
    vds = vin[3:9]       # pos 4-9
    vis = vin[9:]        # pos 10-17

    pos4 = vin[3]
    pos5 = vin[4]
    pos5_6 = vin[4:6]   # combined for Nissan model+gen
    pos6 = vin[5]
    pos7 = vin[6]
    pos8 = vin[7]
    pos9 = vin[8]        # check digit
    pos10 = vin[9]       # model year
    pos11 = vin[10]      # plant
    serial = vin[11:]    # pos 12-17

    result = {
        # VIN structure
        "vin": vin,
        "wmi": wmi,
        "vds": vds,
        "vis": vis,
        "check_digit": pos9,
        "check_digit_valid": check_ok,
        "check_digit_expected": expected_check,
        "valid_chars": valid_chars,
        "invalid_chars_found": bad_chars,

        # Manufacturer (from rules or fallback)
        "manufacturer": mfr_name or "Unknown",
        "country": "Unknown",
        "vehicle_type": "Unknown",
        "wmi_description": "Unknown",

        # VDS decoded
        "body_type": "Unknown",
        "engine": "Unknown",
        "restraint_system": "Unknown",
        "model_platform": "Unknown",
        "series_line": "Unknown",
        "model_generation": "Unknown",

        # VIS decoded
        "model_year": "Unknown",
        "plant": "Unknown",
        "serial_number": serial,

        # Raw position chars
        "pos4": pos4,
        "pos5": pos5,
        "pos5_6": pos5_6,
        "pos6": pos6,
        "pos7": pos7,
        "pos8": pos8,
        "pos9": pos9,
        "pos10": pos10,
        "pos11": pos11,

        # Notes/warnings
        "notes": [],
    }

    if not rules:
        result["notes"].append("Unable to identify manufacturer from WMI.")
        return result

    # ── WMI ─────────────────────────────────────────
    wmi_data = rules.get("wmi", {})
    wmi_info = wmi_data.get(wmi, {})
    result["country"] = wmi_info.get("country", "Unknown")
    result["vehicle_type"] = wmi_info.get("vehicle_type", "Unknown")
    result["wmi_description"] = wmi_info.get("manufacturer", mfr_name or "Unknown")

    # ── POSITION 4 ──────────────────────────────────
    # Nissan: series/line    Toyota: body type
    p4_map = rules.get("position_4", rules.get("position_4_body_type", {}))
    val = p4_map.get(pos4)
    if val and not val.startswith("_"):
        result["series_line"] = val

    # ── POSITION 5+6 (Nissan combined model+gen) ────
    p56_map = rules.get("position_5_and_6", {})
    if p56_map:
        val56 = p56_map.get(pos5_6)
        if val56:
            result["model_generation"] = val56
        else:
            result["model_generation"] = f"Code {pos5_6} (lookup NHTSA for exact model)"
            result["notes"].append(f"Pos 5+6 combined code '{pos5_6}' not in local table. Use NHTSA for exact model.")

    # ── POSITION 5 (Toyota engine) ──────────────────
    p5_engine = rules.get("position_5_engine_code", {})
    if p5_engine:
        val5 = p5_engine.get(pos5)
        if val5:
            result["engine"] = val5

    # ── POSITION 6 (Toyota series/chassis) ──────────
    p6_series = rules.get("position_6", {})
    if p6_series and not p6_series.get("_note"):
        val6 = p6_series.get(pos6)
        if val6:
            result["model_platform"] = val6

    # ── POSITION 7 ──────────────────────────────────
    # Nissan: body type    Toyota: restraint system
    p7_nissan = rules.get("position_7_north_america", {})
    p7_toyota = rules.get("position_7_restraint_system", {})
    p7_map = p7_nissan or p7_toyota
    val7 = p7_map.get(pos7)
    if val7:
        if p7_nissan:
            result["body_type"] = val7
        else:
            result["restraint_system"] = val7

    # ── POSITION 8 ──────────────────────────────────
    # Nissan: restraint    Toyota: model platform
    p8_nissan = rules.get("position_8_north_america", {})
    p8_toyota = rules.get("position_8_model_platform", {})
    if p8_nissan:
        val8 = p8_nissan.get(pos8)
        if val8:
            result["restraint_system"] = val8
    elif p8_toyota:
        val8 = p8_toyota.get(pos8)
        if val8:
            result["model_platform"] = val8

    # ── POSITION 10 — MODEL YEAR ─────────────────────
    year_map = rules.get("position_10_model_year", rules.get("year_codes", {}))
    year_val = year_map.get(pos10)
    if year_val:
        result["model_year"] = str(year_val)

    # ── POSITION 11 — PLANT ──────────────────────────
    plant_map = rules.get("position_11_plant_north_america",
                rules.get("position_11_plant", {}))
    plant_val = plant_map.get(pos11)
    result["plant"] = plant_val if plant_val else f"Code '{pos11}' (see manufacturer plant list)"

    return result


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

st.markdown('<div class="vin-header">VIN Decoder</div>', unsafe_allow_html=True)
st.markdown('<div class="vin-subtitle">VEHICLE IDENTIFICATION NUMBER ANALYSIS TOOL</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    vin_input = st.text_input(
        "",
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
            rows += info_row("Model / Generation", r["model_generation"])
            rows += info_row("Body Type", r["body_type"])
            rows += info_row("Model Platform", r["model_platform"])
            st.markdown(section("Vehicle Descriptor Section", rows), unsafe_allow_html=True)

            rows2 = ""
            rows2 += info_row("Engine", r["engine"])
            rows2 += info_row("Restraint System", r["restraint_system"])
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