import re
import logging
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def normalize_chassis(chassis: str) -> str:
    """
    Normalizes a chassis number by converting it to uppercase and
    removing all spaces and hyphens.
    """
    if not isinstance(chassis, str):
        logger.warning(f"Expected string for chassis normalization, got {type(chassis)}")
        return ""
    # Convert to uppercase and strip spaces and hyphens
    normalized = chassis.upper().strip().replace(" ", "").replace("-", "")
    return normalized

def extract_features(chassis_series: pd.Series) -> pd.DataFrame:
    """
    Extracts features from a pandas Series of chassis numbers.
    Features extracted:
    - char_1 to char_11: Character positions.
    - prefix_2 to prefix_8: Substring prefixes of lengths 2 to 8.
    - suffix_2 to suffix_4: Substring suffixes of lengths 2 to 4.
    - digit_count: Total digit count.
    - letter_count: Total letter count.
    - first_digit_idx: index of first digit (-1 if none).
    - last_digit_idx: index of last digit (-1 if none).
    - first_letter_idx: index of first letter (-1 if none).
    - last_letter_idx: index of last letter (-1 if none).
    - digit position patterns: char_1_is_digit to char_11_is_digit.
    - letter position patterns: char_1_is_letter to char_11_is_letter.
    - chassis_pattern: String representation mapping letters to L and digits to D (e.g. LLDDDDDDDDD).
    """
    logger.info("Extracting features from 11-character chassis numbers...")
    normalized_series = chassis_series.astype(str).apply(normalize_chassis)
    
    features = {}
    
    # 1. Character features
    for i in range(11):
        features[f"char_{i+1}"] = normalized_series.apply(
            lambda x: x[i] if len(x) > i else "?"
        )
        
    # 2. Prefix features
    for i in range(2, 9):
        features[f"prefix_{i}"] = normalized_series.apply(
            lambda x: x[:i] if len(x) >= i else x.ljust(i, "?")
        )
        
    # 3. Suffix features
    for i in range(2, 5):
        features[f"suffix_{i}"] = normalized_series.apply(
            lambda x: x[-i:] if len(x) >= i else x.rjust(i, "?")
        )
        
    # 4. Numeric features (digit/letter counts and indices)
    features["digit_count"] = normalized_series.apply(
        lambda x: sum(1 for char in x if char.isdigit())
    )
    features["letter_count"] = normalized_series.apply(
        lambda x: sum(1 for char in x if char.isalpha())
    )
    
    # Helper index logic
    features["first_digit_idx"] = normalized_series.apply(
        lambda x: next((idx for idx, c in enumerate(x) if c.isdigit()), -1)
    )
    features["last_digit_idx"] = normalized_series.apply(
        lambda x: next((idx for idx in range(len(x)-1, -1, -1) if x[idx].isdigit()), -1)
    )
    features["first_letter_idx"] = normalized_series.apply(
        lambda x: next((idx for idx, c in enumerate(x) if c.isalpha()), -1)
    )
    features["last_letter_idx"] = normalized_series.apply(
        lambda x: next((idx for idx in range(len(x)-1, -1, -1) if x[idx].isalpha()), -1)
    )
    
    # 5. Position patterns
    for i in range(11):
        features[f"char_{i+1}_is_digit"] = normalized_series.apply(
            lambda x: 1 if len(x) > i and x[i].isdigit() else 0
        )
        features[f"char_{i+1}_is_letter"] = normalized_series.apply(
            lambda x: 1 if len(x) > i and x[i].isalpha() else 0
        )
        
    # 6. Pattern layouts
    features["chassis_pattern"] = normalized_series.apply(
        lambda x: "".join("L" if c.isalpha() else "D" if c.isdigit() else "?" for c in x)
    )
    
    df_features = pd.DataFrame(features)
    logger.info(f"Feature extraction complete. Shape: {df_features.shape}")
    return df_features

def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Loads the short chassis dataset from a CSV file.
    """
    if not data_path:
        raise ValueError("Data path must be provided.")
        
    logger.info(f"Loading dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error loading CSV file from {data_path}: {e}")
        raise
        
    # Standardize column naming if necessary (case variations)
    rename_dict = {}
    for col in df.columns:
        if col.lower() == "chassisnumber":
            rename_dict[col] = "chassisNumber"
            
    if rename_dict:
        df = df.rename(columns=rename_dict)
        
    if "chassisNumber" not in df.columns:
        raise KeyError("Required input column 'chassisNumber' not found in dataset.")
        
    return df
