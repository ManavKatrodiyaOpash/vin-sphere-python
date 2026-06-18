import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Add project root and local folder to sys.path
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define character vocabulary: uppercase letters A-Z, digits 0-9, and fallback "?"
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
vocab_size = len(VOCAB)

def encode_vin(vin: str) -> list:
    vin = str(vin).upper().strip()[:9].ljust(9, "?")
    return [char_to_idx.get(c, char_to_idx["?"]) for c in vin]

class VINDataset(Dataset):
    def __init__(self, vins: list, labels: list):
        self.vins = torch.tensor([encode_vin(v) for v in vins], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.vins[idx], self.labels[idx]

# Define a Character-level CNN or BiLSTM
class VINCharBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 32, hidden_dim: int = 64):
        super(VINCharBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x) # (batch, 9, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded) # (batch, 9, hidden_dim*2)
        # Global max pool over sequence dimension
        pooled, _ = torch.max(lstm_out, dim=1) # (batch, hidden_dim*2)
        logits = self.fc(pooled)
        return logits

def train_and_eval_dl(train_df, test_df, target_col, epochs=80, batch_size=8):
    # Prepare target labels
    le = LabelEncoder()
    # Combine to fit LabelEncoder to handle unseen test labels
    combined_labels = pd.concat([train_df[target_col], test_df[target_col]]).astype(str)
    le.fit(combined_labels)
    
    train_labels = le.transform(train_df[target_col].astype(str))
    test_labels = le.transform(test_df[target_col].astype(str))
    
    num_classes = len(le.classes_)
    
    train_dataset = VINDataset(train_df['chassisNumber'].tolist(), train_labels)
    test_dataset = VINDataset(test_df['chassisNumber'].tolist(), test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = VINCharBiLSTM(vocab_size=vocab_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def main():
    data_path = os.path.join(_parent, 'Data', 'final_clean_9.csv')
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    df['chassisNumber'] = df['chassisNumber'].astype(str).str.upper().str.strip()
    
    # Align target columns
    column_mapping = {
        "make": "make",
        "model": "model",
        "year": "year",
        "trim": "trim",
        "body_type": "bodyType",
        "origin": "origin",
        "regional_spec": "regionalSpec"
    }
    
    # Fill NAs
    for key, col in column_mapping.items():
        df[col] = df[col].astype(str).str.upper().str.strip().fillna("UNKNOWN")
        df[col] = df[col].replace(["NAN", "NONE", "", "NAT", "UNDEFINED"], "UNKNOWN")

    # Prefix-based disjoint splits (exact same split as train.py)
    df["prefix4"] = df["chassisNumber"].astype(str).str[:4]
    unique_prefixes = list(df["prefix4"].unique())
    
    train_prefixes, test_prefixes = train_test_split(unique_prefixes, test_size=0.2, random_state=42)
    train_prefixes = set(train_prefixes)
    test_prefixes = set(test_prefixes)
    
    train_df = df[df["prefix4"].isin(train_prefixes)].copy()
    test_df = df[df["prefix4"].isin(test_prefixes)].copy()
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # We will load metadata/best_models from ML train.py run if available, to compare.
    ml_best_scores = {}
    metadata_path = os.path.join(_parent, 'chat_cat_short_vin_09', 'models', 'feature_metadata.pkl')
    if os.path.exists(metadata_path):
        try:
            import joblib
            meta = joblib.load(metadata_path)
            best_models = meta.get("best_models", {})
            for key, info in best_models.items():
                ml_best_scores[key] = info.get("test_score", 0.0)
        except Exception as e:
            print(f"Warning: could not load ML metadata: {e}")
            
    print("\nTraining character-level deep learning BiLSTM models...")
    dl_results = {}
    for target_key, csv_col in column_mapping.items():
        # Match naming with ML target names (e.g. 'regional_specs' instead of 'regional_spec')
        ml_key = "regional_specs" if target_key == "regional_spec" else target_key
        
        acc = train_and_eval_dl(train_df, test_df, csv_col)
        dl_results[ml_key] = acc
        print(f"Target: {ml_key.upper()} - Character BiLSTM Holdout Accuracy: {acc:.2%}")
        
    print("\n=======================================================")
    print("      MODEL PERFORMANCE COMPARISON (HOLDOUT ACCURACY)   ")
    print("=======================================================")
    print(f"| {'Target':<18} | {'Tree-Based (CatBoost/LGBM)':<28} | {'Char-level BiLSTM (DL)':<23} |")
    print(f"|{'-'*20}|{'-'*30}|{'-'*25}|")
    for target in ['make', 'model', 'year', 'trim', 'body_type', 'origin', 'regional_specs']:
        ml_score = ml_best_scores.get(target, 0.0)
        dl_score = dl_results.get(target, 0.0)
        print(f"| {target:<18} | {ml_score:<28.2%} | {dl_score:<23.2%} |")
    print("=======================================================")
    
if __name__ == "__main__":
    main()
