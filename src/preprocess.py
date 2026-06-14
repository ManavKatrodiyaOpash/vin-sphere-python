import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.config import (
    CLASSIFICATION_TARGETS, REGRESSION_TARGETS,
    RARE_THRESHOLD, DEFAULT_BATCH_SIZE, DATA_PATH
)
from src.tokenizer import VINTokenizer

class RobustLabelEncoder:
    """A label encoder that groups rare classes and handles unseen classes at inference."""
    def __init__(self, rare_threshold=5, default_value="UNKNOWN"):
        self.rare_threshold = rare_threshold
        self.default_value = default_value
        self.classes_ = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.default_idx = 0

    def fit(self, series):
        # Convert to string and handle nulls
        series = series.astype(str).fillna(self.default_value)
        counts = series.value_counts()
        
        # Keep classes with counts >= threshold
        frequent_classes = counts[counts >= self.rare_threshold].index.tolist()
        
        # Ensure default_value is included
        if self.default_value not in frequent_classes:
            frequent_classes.append(self.default_value)
            
        self.classes_ = sorted(frequent_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}
        self.default_idx = self.class_to_idx[self.default_value]
        return self

    def transform(self, series):
        series = series.astype(str).fillna(self.default_value)
        return np.array([self.class_to_idx.get(val, self.default_idx) for val in series], dtype=np.int64)

    def inverse_transform(self, indices):
        if isinstance(indices, (int, np.integer)):
            return self.idx_to_class.get(indices, self.default_value)
        return [self.idx_to_class.get(idx, self.default_value) for idx in indices]

    def num_classes(self):
        return len(self.classes_)


class RegressionScaler:
    """Standard scaler to normalize regression targets."""
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, series):
        # Ignore NaNs during fit
        valid_vals = series.dropna()
        if len(valid_vals) > 0:
            self.mean = float(valid_vals.mean())
            self.std = float(valid_vals.std())
            if self.std == 0:
                self.std = 1.0
        return self

    def transform(self, series):
        vals = series.fillna(self.mean).values
        return (vals - self.mean) / self.std

    def inverse_transform(self, vals):
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        return vals * self.std + self.mean


def normalize_vin(vin: str) -> str:
    """Normalize and clean VIN input string."""
    if not isinstance(vin, str):
        return ""
    vin = vin.upper().strip()
    vin = re.sub(r"[^A-Z0-9]", "", vin)
    return vin


def validate_vin(vin: str) -> bool:
    """Basic structural and standard validation of VIN."""
    if len(vin) != 17:
        return False
    if any(char in vin for char in ["I", "O", "Q"]):
        return False
    return True


class VINDataset(Dataset):
    def __init__(self, tokenized_vins, classification_targets=None, regression_targets=None):
        self.tokenized_vins = torch.tensor(tokenized_vins, dtype=torch.long)
        self.classification_targets = classification_targets
        self.regression_targets = regression_targets

    def __len__(self):
        return len(self.tokenized_vins)

    def __getitem__(self, idx):
        item = {"tokens": self.tokenized_vins[idx]}
        
        if self.classification_targets is not None:
            for col, target_vals in self.classification_targets.items():
                item[col] = torch.tensor(target_vals[idx], dtype=torch.long)
                
        if self.regression_targets is not None:
            for col, target_vals in self.regression_targets.items():
                item[col] = torch.tensor(target_vals[idx], dtype=torch.float)
                
        return item


class VINDataPipeline:
    def __init__(self, rare_threshold=RARE_THRESHOLD):
        self.tokenizer = VINTokenizer()
        self.rare_threshold = rare_threshold
        self.encoders = {}
        self.scalers = {}

    def fit_transform(self, df):
        # 1. Clean VIN
        print("Cleaning and validating VINs...")
        df["chassisNumber"] = df["chassisNumber"].apply(normalize_vin)
        df = df[df["chassisNumber"].apply(validate_vin)].copy()
        
        # Tokenize VINs
        print("Tokenizing VINs...")
        tokenized_vins = np.array([self.tokenizer.encode(vin) for vin in df["chassisNumber"]], dtype=np.int64)
        
        # Fit encoders and scalers
        classification_data = {}
        for col in CLASSIFICATION_TARGETS:
            print(f"Fitting encoder for {col}...")
            self.encoders[col] = RobustLabelEncoder(rare_threshold=self.rare_threshold)
            self.encoders[col].fit(df[col])
            classification_data[col] = self.encoders[col].transform(df[col])
            
        regression_data = {}
        for col in REGRESSION_TARGETS:
            print(f"Fitting scaler for {col}...")
            self.scalers[col] = RegressionScaler()
            self.scalers[col].fit(df[col])
            regression_data[col] = self.scalers[col].transform(df[col])
            
        return tokenized_vins, classification_data, regression_data

    def transform(self, df):
        df = df.copy()
        df["chassisNumber"] = df["chassisNumber"].apply(normalize_vin)
        tokenized_vins = np.array([self.tokenizer.encode(vin) for vin in df["chassisNumber"]], dtype=np.int64)
        
        classification_data = {}
        for col in CLASSIFICATION_TARGETS:
            classification_data[col] = self.encoders[col].transform(df[col])
            
        regression_data = {}
        for col in REGRESSION_TARGETS:
            regression_data[col] = self.scalers[col].transform(df[col])
            
        return tokenized_vins, classification_data, regression_data

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


def prepare_data_loaders(sample_size=None, test_size=0.1, val_size=0.1, batch_size=DEFAULT_BATCH_SIZE, random_state=42):
    """Loads Cleaned.csv, splits into Train/Val/Test, prepares pipelines, and returns DataLoaders."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset Cleaned.csv not found at {DATA_PATH}")

    print(f"Loading full dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if "chassisNumber" not in df.columns:
        chassis_col = [c for c in df.columns if "chassis" in c.lower()]
        if chassis_col:
            df = df.rename(columns={chassis_col[0]: "chassisNumber"})
        else:
            raise KeyError("chassisNumber column not found in dataset.")

    # Clean and validate upfront
    df["chassisNumber"] = df["chassisNumber"].apply(normalize_vin)
    df = df[df["chassisNumber"].apply(validate_vin)].copy()
    print(f"Total valid VIN records: {len(df):,}")

    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        print(f"Sampling dataset to {sample_size:,} records...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    print("Splitting dataset into Train, Val, and Test...")
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), random_state=random_state)
    
    print(f"Train size: {len(train_df):,}, Val size: {len(val_df):,}, Test size: {len(test_df):,}")

    pipeline = VINDataPipeline()
    train_tokens, train_cls, train_reg = pipeline.fit_transform(train_df)
    
    val_tokens, val_cls, val_reg = pipeline.transform(val_df)
    test_tokens, test_cls, test_reg = pipeline.transform(test_df)
    
    train_dataset = VINDataset(train_tokens, train_cls, train_reg)
    val_dataset = VINDataset(val_tokens, val_cls, val_reg)
    test_dataset = VINDataset(test_tokens, test_cls, test_reg)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, pipeline
