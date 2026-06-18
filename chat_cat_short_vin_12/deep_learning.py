import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Add parent directory to path
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Column mapping from internal targets to CSV columns
COLUMN_MAPPING = {
    "make": "make",
    "model": "model",
    "year": "year",
    "trim": "trim",
    "body_type": "bodyType",
    "origin": "origin",
    "regional_specs": "regionalSpec",
    "cylinders": "cylinders",
    "no_of_passengers": "noOfPassengers",
    "color": "color"
}

# Vocabulary for short chassis VINs
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
char_to_idx = {char: idx + 1 for idx, char in enumerate(VOCAB)}
vocab_size = len(VOCAB) + 1  # +1 for padding/unknown (index 0)

def encode_vin(vin_str: str) -> np.ndarray:
    vin_str = str(vin_str).upper().strip().replace(" ", "").replace("-", "")
    encoded = []
    for char in vin_str[:12]:
        encoded.append(char_to_idx.get(char, 0))
    # Pad to 12 if shorter
    while len(encoded) < 12:
        encoded.append(0)
    return np.array(encoded, dtype=np.int64)

class CharacterLSTM(nn.Module):
    def __init__(self, num_classes_dict, vocab_size=vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for target, num_classes in num_classes_dict.items()
        })
        
    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        # Global max pooling over sequence length
        pooled = torch.max(lstm_out, dim=1)[0]
        logits = {target: head(pooled) for target, head in self.heads.items()}
        return logits

def main():
    data_path = os.path.join(_parent, 'Data', 'final_clean_12.csv')
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows.")

    # Drop rows without chassis number
    df = df[df['chassisNumber'].notna()].copy()
    df['chassisNumber'] = df['chassisNumber'].astype(str).str.upper().str.strip()

    # Preprocess targets and build label encoders
    label_encoders = {}
    encoded_targets = {}
    num_classes_dict = {}

    for target, csv_col in COLUMN_MAPPING.items():
        if csv_col in df.columns:
            df[csv_col] = df[csv_col].astype(str).str.upper().str.strip()
            df[csv_col] = df[csv_col].replace(["NAN", "NONE", "", "NAT", "UNDEFINED"], "UNKNOWN")
        else:
            df[csv_col] = "UNKNOWN"

        le = LabelEncoder()
        encoded_targets[target] = le.fit_transform(df[csv_col].fillna("UNKNOWN"))
        label_encoders[target] = le
        num_classes_dict[target] = len(le.classes_)
        print(f"Target {target}: {num_classes_dict[target]} classes")

    # Encode inputs
    vins_encoded = np.stack(df['chassisNumber'].apply(encode_vin).values)

    # Disjoint split by prefix5
    df["prefix5"] = df["chassisNumber"].str[:5]
    unique_prefixes = list(df["prefix5"].unique())
    train_prefixes, test_prefixes = train_test_split(unique_prefixes, test_size=0.2, random_state=42)
    train_prefixes = set(train_prefixes)
    test_prefixes = set(test_prefixes)

    train_mask = df["prefix5"].isin(train_prefixes)
    test_mask = df["prefix5"].isin(test_prefixes)

    X_train = vins_encoded[train_mask]
    X_test = vins_encoded[test_mask]

    y_train_dict = {t: encoded_targets[t][train_mask] for t in COLUMN_MAPPING.keys()}
    y_test_dict = {t: encoded_targets[t][test_mask] for t in COLUMN_MAPPING.keys()}

    # Create PyTorch datasets and dataloaders
    class MultiTaskDataset(torch.utils.data.Dataset):
        def __init__(self, X, y_dict):
            self.X = torch.tensor(X, dtype=torch.long)
            self.y_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in y_dict.items()}
            self.targets = list(y_dict.keys())

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            item = {'x': self.X[idx]}
            for t in self.targets:
                item[t] = self.y_dict[t][idx]
            return item

    train_dataset = MultiTaskDataset(X_train, y_train_dict)
    test_dataset = MultiTaskDataset(X_test, y_test_dict)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Instantiate model
    model = CharacterLSTM(num_classes_dict).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining character-level multi-task BiLSTM model...")
    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            x_batch = batch['x'].to(device)
            optimizer.zero_grad()
            
            logits_dict = model(x_batch)
            loss = 0.0
            for target in COLUMN_MAPPING.keys():
                y_batch = batch[target].to(device)
                loss += criterion(logits_dict[target], y_batch)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            
        epoch_loss = total_loss / len(train_dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

    # Evaluate on holdout test set
    model.eval()
    test_preds = {t: [] for t in COLUMN_MAPPING.keys()}
    test_trues = {t: [] for t in COLUMN_MAPPING.keys()}

    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch['x'].to(device)
            logits_dict = model(x_batch)
            
            for target in COLUMN_MAPPING.keys():
                preds = torch.argmax(logits_dict[target], dim=1).cpu().numpy()
                test_preds[target].extend(preds)
                test_trues[target].extend(batch[target].numpy())

    # Calculate metrics
    dl_accuracies = {}
    print("\nHoldout Test Set Accuracies (Deep Learning model):")
    print("=" * 45)
    for target in COLUMN_MAPPING.keys():
        acc = np.mean(np.array(test_preds[target]) == np.array(test_trues[target]))
        dl_accuracies[target] = acc
        print(f"{target.upper():<20} : {acc:.2%}")

    # Try to compare with CatBoost if tree results exist in metadata
    cb_accuracies = {}
    metadata_path = os.path.join(_parent, 'chat_cat_short_vin_12', 'models', 'feature_metadata.pkl')
    if os.path.exists(metadata_path):
        try:
            tree_meta = joblib.load(metadata_path)
            for target in COLUMN_MAPPING.keys():
                if target in tree_meta.get("best_models", {}):
                    cb_accuracies[target] = tree_meta["best_models"][target].get("test_score", 0.0)
        except Exception as e:
            print(f"Could not load tree model metadata: {e}")

    # Write report comparing deep learning vs Tree-based model
    report_lines = [
        "# Deep Learning vs Tree-Based Model Comparison Report\n",
        "This report compares the performance of a character-level multi-task BiLSTM neural network against our sequential Gradient Boosting tree-based pipeline on unseen VIN prefix holdout sets.\n",
        "## Performance Metrics Comparison\n",
        "| Attribute | Character-level BiLSTM Accuracy | Tree-based Pipeline (Best Model) | Winner |",
        "|---|---|---|---|"
    ]

    for target in COLUMN_MAPPING.keys():
        dl_acc = dl_accuracies[target]
        cb_acc = cb_accuracies.get(target, "N/A")
        
        if cb_acc != "N/A":
            winner = "BiLSTM" if dl_acc > cb_acc else "Tree Model"
            cb_str = f"{cb_acc:.2%}"
        else:
            winner = "BiLSTM (No Tree Model data)"
            cb_str = "N/A"
            
        report_lines.append(f"| {target.upper()} | {dl_acc:.2%} | {cb_str} | {winner} |")

    report_path = os.path.join(_parent, 'chat_cat_short_vin_12', 'models', 'deep_learning_comparison.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved comparison report to {report_path}")

if __name__ == "__main__":
    main()
