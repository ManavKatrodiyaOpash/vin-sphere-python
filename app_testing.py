import os
import sys
import pandas as pd
from pathlib import Path

# 1. Setup paths to match your main app environment
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 2. Import the logic directly from your app script
try:
    from app_new import decode_any_vin, get_depreciated_values_by_file
except ImportError as e:
    print(f"Error importing functions from app_new.py: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
INPUT_CSV_PATH = project_root / "Data/lookup_data_clean.csv"  # <-- Change to your testing CSV filename
OUTPUT_TXT_PATH = project_root / "missing_depreciation_log.txt"

# Columns expected in your input CSV file
CHASSIS_COL = "chassisNumber"  # <-- Change if your column name is different (e.g., 'vin')
MAKE_COL = "make"
MODEL_COL = "model"
# ---------------------

def run_validation():
    if not INPUT_CSV_PATH.exists():
        print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return

    print(f"Loading dataset from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Check for mandatory columns
    for col in [CHASSIS_COL, MAKE_COL, MODEL_COL]:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in the CSV file.")
            return

    print(f"Starting batch validation for {len(df)} rows. Results will log to {OUTPUT_TXT_PATH}...")
    
    missing_count = 0
    success_count = 0

    # Open the text file to write missing values dynamically
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("=== CARS MISSING DEPRECIATION VALUES ===\n")
        log_file.write(f"{'Make':<20} | {'Model':<30} | {'Chassis/VIN':<20}\n")
        log_file.write("-" * 75 + "\n")

        for index, row in df.iterrows():
            chassis_val = str(row[CHASSIS_COL]).strip().upper()
            csv_make = str(row[MAKE_COL]).strip()
            csv_model = str(row[MODEL_COL]).strip()

            # Skip empty or invalid chassis values early
            if not chassis_val or chassis_val == "NAN" or len(chassis_val) not in [11, 17]:
                continue

            try:
                # Step A: Decode the chassis using your ML models to get predicted metadata
                decoded_result = decode_any_vin(chassis_val)
                
                # Step B: Pass that decoded metadata into your file-matching depreciation logic
                dep_values = get_depreciated_values_by_file(decoded_result)

                # Step C: Verify if any depreciation file matched
                if not dep_values:
                    # Log to file immediately if no file returned a valuation
                    log_file.write(f"{csv_make:<20} | {csv_model:<30} | {chassis_val:<20}\n")
                    missing_count += 1
                else:
                    success_count += 1

            except Exception as e:
                # Fallback logging if an ML prediction or formatting step throws an exception
                log_file.write(f"{csv_make:<20} | {csv_model:<30} | {chassis_val:<20} (Error: {str(e)})\n")
                missing_count += 1

            # Print an on-screen progress indicator every 50 records
            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1}/{len(df)} rows...")

    print("\n=== Validation Completed ===")
    print(f"✅ Vehicles with Depreciation Values: {success_count}")
    print(f"❌ Vehicles missing Depreciation Values: {missing_count}")
    print(f"Logs successfully written to: {OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    run_validation()