import re
import logging
import pandas as pd
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)

def normalize_chassis(chassis: str) -> str:
    """
    Normalizes a chassis number by converting it to uppercase and
    removing all spaces and hyphens.
    
    Args:
        chassis: The raw chassis number string.
        
    Returns:
        The normalized uppercase string.
    """
    if not isinstance(chassis, str):
        logger.warning(f"Expected string for chassis normalization, got {type(chassis)}")
        return ""
    # Convert to uppercase and strip spaces and hyphens
    normalized = chassis.upper().strip().replace(" ", "").replace("-", "")
    return normalized

def extract_features(chassis_series: pd.Series) -> pd.DataFrame:
    """
    Extracts features from a pandas Series of 10-character short chassis numbers.
    """
    logger.info("Extracting features from 10-character chassis numbers...")
    normalized_series = chassis_series.astype(str).apply(normalize_chassis)
    
    features = {}
    
    # 1. Character features: char_1 to char_10
    for i in range(10):
        features[f"char_{i+1}"] = normalized_series.apply(
            lambda x: x[i] if len(x) > i else "?"
        )
        
    # 2. Prefix features: prefix_2 to prefix_7
    for i in range(2, 8):
        features[f"prefix_{i}"] = normalized_series.apply(
            lambda x: x[:i] if len(x) >= i else x.ljust(i, "?")
        )
        
    # 3. Suffix features: suffix_2 to suffix_4
    for i in range(2, 5):
        features[f"suffix_{i}"] = normalized_series.apply(
            lambda x: x[-i:] if len(x) >= i else x.rjust(i, "?")
        )
        
    # 4. Pattern layout & length
    features["pattern"] = normalized_series.apply(
        lambda x: "".join("L" if c.isalpha() else "D" if c.isdigit() else "?" for c in x)
    )
    features["pattern_length"] = normalized_series.apply(len)
    
    # 5. Count features
    features["digit_count"] = normalized_series.apply(
        lambda x: sum(1 for char in x if char.isdigit())
    )
    features["letter_count"] = normalized_series.apply(
        lambda x: sum(1 for char in x if char.isalpha())
    )
    # Maintain original count columns for backward compatibility
    features["num_letters"] = features["letter_count"]
    features["num_digits"] = features["digit_count"]
    
    # 6. ASCII Features: ascii_char_1 to ascii_char_10
    for i in range(10):
        features[f"ascii_char_{i+1}"] = normalized_series.apply(
            lambda x: ord(x[i]) if len(x) > i else 0
        )
        
    # 7. N-Gram Features
    features["bigram_1"] = normalized_series.apply(lambda x: x[0:2] if len(x) >= 2 else "??")
    features["bigram_2"] = normalized_series.apply(lambda x: x[1:3] if len(x) >= 3 else "??")
    features["bigram_3"] = normalized_series.apply(lambda x: x[2:4] if len(x) >= 4 else "??")
    features["trigram_1"] = normalized_series.apply(lambda x: x[0:3] if len(x) >= 3 else "???")
    features["trigram_2"] = normalized_series.apply(lambda x: x[1:4] if len(x) >= 4 else "???")
    
    # 8. Position Features
    features["letter_positions"] = normalized_series.apply(
        lambda x: ",".join(str(idx) for idx, char in enumerate(x) if char.isalpha())
    )
    features["digit_positions"] = normalized_series.apply(
        lambda x: ",".join(str(idx) for idx, char in enumerate(x) if char.isdigit())
    )
    
    # 9. Numeric serial section as an integer feature
    def extract_serial_int(val: str) -> int:
        match = re.search(r"(\d+)\D*$", val)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 0
        return 0
    features["serial_number"] = normalized_series.apply(extract_serial_int)
    
    # Letters only sequence
    features["letters_only"] = normalized_series.apply(
        lambda x: "".join(re.findall(r"[A-Z]", x))
    )
    
    # Letter prefix
    def get_letter_prefix(val: str) -> str:
        match = re.match(r"^[A-Z]+", val)
        if match:
            return match.group(0)
        match_any = re.search(r"[A-Z]+", val)
        return match_any.group(0) if match_any else "?"
    features["letter_prefix"] = normalized_series.apply(get_letter_prefix)
    
    # Index of first digit
    def get_first_digit_idx(val: str) -> int:
        for idx, char in enumerate(val):
            if char.isdigit():
                return idx
        return -1
    features["first_digit_idx"] = normalized_series.apply(get_first_digit_idx)
    
    # Index of last letter
    def get_last_letter_idx(val: str) -> int:
        last_idx = -1
        for idx, char in enumerate(val):
            if char.isalpha():
                last_idx = idx
        return last_idx
    features["last_letter_idx"] = normalized_series.apply(get_last_letter_idx)
    
    df_features = pd.DataFrame(features)
    logger.info(f"Feature extraction complete. Shape: {df_features.shape}")
    return df_features

def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Loads the short chassis dataset from a CSV file and normalizes columns.
    
    Args:
        data_path: The filesystem path to the CSV file.
        
    Returns:
        A loaded and preprocessed pandas DataFrame.
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
    # The expected input column is 'chassisNumber'
    rename_dict = {}
    for col in df.columns:
        if col.lower() == "chassisnumber":
            rename_dict[col] = "chassisNumber"
            
    if rename_dict:
        df = df.rename(columns=rename_dict)
        
    if "chassisNumber" not in df.columns:
        raise KeyError("Required input column 'chassisNumber' not found in dataset.")
        
    return df
