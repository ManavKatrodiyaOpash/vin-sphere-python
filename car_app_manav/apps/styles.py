main_style = """
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
"""

# =====================================================
# VW GROUP FORMAT DETECTION (Audi / Mercedes / VW)
# =====================================================

def is_european_format(vin):
    """Returns True if pos4-6 == 'ZZZ' — ISO filler used by VW Group / Mercedes EU-format VINs."""
    return len(vin) >= 7 and vin[3:6].upper() == "ZZZ"

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

    legend = f"""
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