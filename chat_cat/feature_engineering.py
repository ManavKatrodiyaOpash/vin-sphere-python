import os
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import LabelEncoder

# Set up logging
logger = logging.getLogger(__name__)

# Column mapping to support both prompt specifications and actual dataset columns (e.g. data_methaq)
COLUMN_MAPPING = {
    "VIN": ["VIN", "chassisNumber", "ChassisNumber", "vin"],
    "MAKE": ["MAKE", "make", "Make"],
    "MODEL": ["MODEL", "model", "Model"],
    "TRIM": ["TRIM", "trim", "Trim"],
    "BODY_TYPE": ["BODY_TYPE", "bodyType", "BodyType", "body_type"],
    "REGIONAL_SPEC": ["regionalSpec", "regional_spec", "regionalSpec"],
    "CYLINDERS": ["CYLINDERS", "cylinders", "Cylinders"],
    "FUEL_TYPE": ["FUEL_TYPE", "fuelType", "fuel_type"],
    "TRANSMISSION": ["TRANSMISSION", "transmission"],
    "YEAR": ["YEAR", "year", "Year"],
    "ORIGIN": ["origin", "Origin", "ORIGIN"],
    "NO_OF_PASSENGERS": ["noOfPassengers", "no_of_passengers"],
    "WEIGHT": ["weightInKg", "weight"],
}

def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map standard expected columns to actual columns present in the DataFrame."""
    resolved = {}
    for standard_col, candidates in COLUMN_MAPPING.items():
        found = False
        for cand in candidates:
            if cand in df.columns:
                resolved[standard_col] = cand
                found = True
                break
        if not found:
            logger.warning(f"Standard column '{standard_col}' could not be matched. Candidates: {candidates}")
    return resolved

def normalize_name(label_type: str, value: Union[str, float, int, None]) -> str:
    """Normalize names (e.g. LAND CRUISER / LANDCRUISER -> Land Cruiser, and numeric values)."""
    if pd.isna(value):
        return "UNKNOWN"
        
    label_type = label_type.upper()
    
    # Handle numeric columns
    if label_type in ["YEAR", "CYLINDERS", "NO_OF_PASSENGERS", "WEIGHT"]:
        try:
            return str(int(float(value)))
        except (ValueError, TypeError):
            return "UNKNOWN"
            
    if not isinstance(value, str):
        return "UNKNOWN"
        
    val = value.strip().upper()
    if not val:
        return "UNKNOWN"
        
    # Specific normalization rules
    if label_type == "MAKE":
        # Mercedes-Benz variations
        if val in ["MERCEDESBENZ", "MERCEDES BENZ", "MERCEDES"]:
            return "Mercedes-Benz"
        # Land Rover variations
        if val in ["LANDROVER", "LAND ROVER"]:
            return "Land Rover"
        return val.title()
        
    elif label_type == "MODEL":
        # Toyota Land Cruiser variations
        if val in ["LANDCRUISER", "LAND CRUISER", "LAND CRISER", "LAND_CRUISER"]:
            return "Land Cruiser"
        # Range Rover variations
        if val in ["RANGEROVER", "RANGE ROVER", "RANGE_ROVER"]:
            return "Range Rover"
        # Patrol variations
        if val == "NISSAN PATROL":
            return "Patrol"
        return val.title()
        
    elif label_type == "TRIM":
        # Standardize trim notation
        val = val.replace(" ", "")
        # Common trim corrections
        if val in ["G-XR", "G.XR"]:
            return "GXR"
        if val in ["V-XR", "V.XR"]:
            return "VXR"
        return val
        
    elif label_type == "BODY_TYPE":
        if val in ["SUV", "S.U.V", "SPORTS UTILITY VEHICLE"]:
            return "SUV"
        if val in ["SEDAN", "SALOON"]:
            return "Sedan"
        if val in ["COUPE", "COUPE'"]:
            return "Coupe"
        return val.title()
        
    elif label_type == "REGIONAL_SPEC":
        if val in ["GCC", "GCC SPEC", "GCC SPECS"]:
            return "GCC"
        return val.title()
        
    elif label_type == "ORIGIN":
        return val.title()
        
    return val

def extract_vin_features(df: pd.DataFrame, vin_col: str) -> pd.DataFrame:
    """Extract character features (POS_1 to POS_17), WMI, VDS, YEAR_CODE, and PLANT_CODE."""
    logger.info("Extracting features from VIN...")
    
    # Ensure all VINs are uppercase and stripped
    v = df[vin_col].astype(str).str.upper().str.strip()
    
    features = pd.DataFrame(index=df.index)
    
    # POS_1 to POS_17 features
    for idx in range(17):
        features[f"POS_{idx+1}"] = v.str[idx]
        
    # High-level components
    features["WMI"] = v.str[0:3]
    features["VDS"] = v.str[3:9]
    features["YEAR_CODE"] = v.str[9]
    features["PLANT_CODE"] = v.str[10]
    
    # Fill any potential nulls with standard placeholder
    features = features.fillna("?")
    
    return features

def prepare_data(data_path: str, save_encoders_dir: str = "models") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """Loads, cleans, engineers features, normalizes target variables, and fits encoders."""
    logger.info(f"Loading raw dataset from {data_path}...")
    
    # Check file type
    if data_path.endswith(".csv"):
        df_raw = pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):
        df_raw = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported dataset file format. Use .csv or .parquet.")
        
    logger.info(f"Loaded raw data of shape: {df_raw.shape}")
    
    # Resolve columns dynamically
    col_map = resolve_columns(df_raw)
    vin_col = col_map["VIN"]
    
    # 1. Clean VINs & remove duplicates
    logger.info("Cleaning VINs and filtering to length 17...")
    df_raw[vin_col] = df_raw[vin_col].astype(str).str.upper().str.strip()
    df_clean = df_raw[df_raw[vin_col].str.len() == 17].copy()
    
    logger.info("Removing duplicate VIN records...")
    df_clean = df_clean.drop_duplicates(subset=[vin_col])
    logger.info(f"Data shape after cleaning and deduplication: {df_clean.shape}")
    
    # 2. Extract VIN features
    X = extract_vin_features(df_clean, vin_col)
    
    # 3. Process targets: Normalize and Label Encode
    logger.info("Normalizing and encoding targets...")
    targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "YEAR", "CYLINDERS", "ORIGIN", "NO_OF_PASSENGERS", "WEIGHT", "REGIONAL_SPEC"]
    y_encoded = pd.DataFrame(index=df_clean.index)
    encoders = {}
    
    os.makedirs(save_encoders_dir, exist_ok=True)
    
    for target in targets:
        actual_col = col_map.get(target)
        if actual_col:
            # Clean and normalize values
            normalized_vals = df_clean[actual_col].apply(lambda x: normalize_name(target, x))
            
            # Fit LabelEncoder
            le = LabelEncoder()
            y_encoded[f"{target}_enc"] = le.fit_transform(normalized_vals)
            encoders[target] = le
            
            # Persist encoder
            encoder_path = os.path.join(save_encoders_dir, f"{target.lower()}_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(le, f)
            logger.info(f"Encoded target '{target}' with {len(le.classes_)} classes. Saved to {encoder_path}")
        else:
            # Fallback if a target column is missing entirely from data (e.g. Engine in some datasets)
            logger.warning(f"Target '{target}' column not found in data! Using 'UNKNOWN' dummy class.")
            le = LabelEncoder()
            y_encoded[f"{target}_enc"] = le.fit_transform(["UNKNOWN"] * len(df_clean))
            encoders[target] = le
            
            encoder_path = os.path.join(save_encoders_dir, f"{target.lower()}_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(le, f)
                
    return X, y_encoded, encoders
