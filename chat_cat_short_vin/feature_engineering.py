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
    Extracts features from a pandas Series of chassis numbers.
    Features extracted:
    - pos_0 to pos_9: Individual character positions.
    - prefix_2 to prefix_5: Substring prefixes of lengths 2, 3, 4, and 5.
    - serial_number: The last contiguous numeric sequence converted to an integer.
    - letters_only: All alphabetical characters extracted in order.
    - letter_prefix: First contiguous block of alphabetical characters.
    - first_digit_idx: Zero-indexed position of the first numeric digit.
    - last_letter_idx: Zero-indexed position of the last alphabetical character.
    - num_letters: Total count of letters in the chassis string.
    - num_digits: Total count of digits in the chassis string.
    
    Args:
        chassis_series: Pandas Series containing the chassis numbers.
        
    Returns:
        A pandas DataFrame with engineered features.
    """
    logger.info("Extracting features from chassis numbers...")
    normalized_series = chassis_series.astype(str).apply(normalize_chassis)
    
    features = {}
    
    # 1. Character positions pos_0 to pos_9
    for i in range(10):
        features[f"pos_{i}"] = normalized_series.apply(
            lambda x: x[i] if len(x) > i else "?"
        )
        
    # 2. Prefixes of length 2, 3, 4, and 5
    features["prefix_2"] = normalized_series.apply(
        lambda x: x[:2] if len(x) >= 2 else x.ljust(2, "?")
    )
    features["prefix_3"] = normalized_series.apply(
        lambda x: x[:3] if len(x) >= 3 else x.ljust(3, "?")
    )
    features["prefix_4"] = normalized_series.apply(
        lambda x: x[:4] if len(x) >= 4 else x.ljust(4, "?")
    )
    features["prefix_5"] = normalized_series.apply(
        lambda x: x[:5] if len(x) >= 5 else x.ljust(5, "?")
    )
    
    # 3. Numeric serial section as an integer feature
    def extract_serial_int(val: str) -> int:
        match = re.search(r"(\d+)\D*$", val)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 0
        return 0
        
    features["serial_number"] = normalized_series.apply(extract_serial_int)
    
    # 4. Letters only sequence
    features["letters_only"] = normalized_series.apply(
        lambda x: "".join(re.findall(r"[A-Z]", x))
    )
    
    # 5. Letter prefix
    def get_letter_prefix(val: str) -> str:
        match = re.match(r"^[A-Z]+", val)
        if match:
            return match.group(0)
        match_any = re.search(r"[A-Z]+", val)
        return match_any.group(0) if match_any else "?"
    features["letter_prefix"] = normalized_series.apply(get_letter_prefix)
    
    # 6. Index of first digit
    def get_first_digit_idx(val: str) -> int:
        for idx, char in enumerate(val):
            if char.isdigit():
                return idx
        return -1
    features["first_digit_idx"] = normalized_series.apply(get_first_digit_idx)
    
    # 7. Index of last letter
    def get_last_letter_idx(val: str) -> int:
        last_idx = -1
        for idx, char in enumerate(val):
            if char.isalpha():
                last_idx = idx
        return last_idx
    features["last_letter_idx"] = normalized_series.apply(get_last_letter_idx)
    
    # 8. Number of letters and digits
    features["num_letters"] = normalized_series.apply(
        lambda x: sum(1 for char in x if char.isalpha())
    )
    features["num_digits"] = normalized_series.apply(
        lambda x: sum(1 for char in x if char.isdigit())
    )
    
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
