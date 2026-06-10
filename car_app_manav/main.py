import streamlit as st
import pandas as pd
from apps.styles import (main_style, 
                    shorten_text, 
                    clean_value, 
                    vin_map_html, 
                    info_row, 
                    section
                    )
from apps.decode import decode_vin

# =====================================================
# INITIAL SETUP & GLOBAL CACHE
# =====================================================

st.set_page_config(
    page_title="VIN Decoder Pro",
    page_icon="🚗",
    layout="wide"
)
# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(main_style, unsafe_allow_html=True)

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
            st.markdown(vin_map_html(vin_input), unsafe_allow_html=True)

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
                rows = info_row("Manufacturer", r["manufacturer"], "highlight")
                
                rows += info_row("Country", r["country"])
                
                rows += info_row("Vehicle Type", r["vehicle_type"])
                
                rows += info_row("WMI", f'<span class="badge badge-blue">{r["wmi"]}</span>')
                
                # rows += info_row("WMI Entity", r["wmi_description"])
                
                st.markdown(section("World Manufacturer Identifier", rows), unsafe_allow_html=True)
                

                rows2 = info_row("Model Year", f'<span class="badge badge-purple">{r["model_year"]}</span>')
                
                rows2 += info_row("Plant Code", f'<span class="badge badge-yellow">{r["pos11"]}</span>')
                
                rows2 += info_row("Plant", r["plant"])
                
                rows2 += info_row("Serial Number", f'<span style="font-family:Space Mono,monospace;color:#3affff">{r["serial_number"]}</span>')
                
                st.markdown(section("Vehicle Identity Section", rows2), unsafe_allow_html=True)

            with c2:
                
                rows = info_row("Model", r["series_line"])
                
                # if r["model_platform"] not in ["Unknown", "Not Available", None]:
                #     rows += info_row("Make", r["model_platform"])
                    
                rows += info_row("Body Type", r["body_type"])
                
                if r["trim"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Trim", r["trim"])
                    
                if r["Transmission"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Transmission", r["Transmission"])
                    
                if r["drive_type"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Drive Type", r["drive_type"])
                    
                if r["number_of_doors"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Number of Doors", r["number_of_doors"])
                    
                if r["regional_space"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Regional Spec", r["regional_space"])
                    
                if r["cylinder"] not in ["Unknown", "Not Available", None]:
                    rows += info_row("Cylinders", r["cylinder"])
                
                rows += info_row("No Of Passengers", r["no_of_passengers"])
                    
                rows += info_row("Color", r["color"])
                
                rows += info_row("Weight (Kg)", r["weight"])
                
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
            
            # with c3:
            #     rows = ""
            #     rows += info_row("Pos 4", f'{r["pos4"]} → {r["series_line"][:40] if r["series_line"] != "Unknown" else "—"}')
            #     rows += info_row("Pos 5+6", f'{r["pos5_6"]} → {r["model_generation"][:40] if r["model_generation"] != "Unknown" else "—"}')
            #     rows += info_row("Pos 7", f'{r["pos7"]} → {r["body_type"][:40] if r["body_type"] != "Unknown" else r["restraint_system"][:40] if r["restraint_system"] != "Unknown" else "—"}')
            #     rows += info_row("Pos 8", f'{r["pos8"]} → {r["restraint_system"][:40] if r["restraint_system"] != "Unknown" else r["model_platform"][:40] if r["model_platform"] != "Unknown" else "—"}')
            #     if r["country"] == "India" and r["manufacturer"] == "Hyundai":
            #         rows += info_row("Pos 9", f'{r["pos9"]} → {r["Transmission"]}')
            #     else:
            #         rows += info_row("Pos 9 (Check)", f'{r["check_digit"]} {"✓" if r["check_digit_valid"] else "✗"}', "good" if r["check_digit_valid"] else "warn")
            #     rows += info_row("Pos 10 (Year)", f'{r["pos10"]} → {r["model_year"]}')
            #     rows += info_row("Pos 11 (Plant)", f'{r["pos11"]} → {r["plant"][:35]}')
            #     st.markdown(section("Position-by-Position Map", rows), unsafe_allow_html=True)

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
                allowed_keys = [
                    "manufacturer",
                    "series_line",
                    "model_year",
                    "trim",
                    "body_type",
                    "regional_space",
                    "cylinder",
                    "color",
                    "country",
                    "weight",
                    "no_of_passengers",
                    "vin"
                ]
                filtered_r = {k: r[k] for k in allowed_keys if k in r}
                st.json(filtered_r)