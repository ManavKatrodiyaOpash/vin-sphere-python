import os
import re
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import (
    DATA_PATH, MODEL_DIR, REPORTS_DIR,
    CLASSIFICATION_TARGETS, REGRESSION_TARGETS, ALL_TARGETS,
    DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_EPOCHS,
    EARLY_STOPPING_PATIENCE, WEIGHT_DECAY, RARE_THRESHOLD
)
from src.tokenizer import VINTokenizer
from src.preprocess import (
    normalize_vin, validate_vin, prepare_data_loaders,
    VINDataPipeline, VINDataset
)
from src.model import VINTransformerEncoder
from src.evaluate import calculate_classification_metrics, calculate_regression_metrics


def calculate_entropy(series):
    """Calculates Shannon entropy in bits."""
    probs = series.value_counts(normalize=True)
    return -sum(probs * np.log2(probs + 1e-12))


def calculate_cramers_v(x, y):
    """Calculates Cramer's V association score between two categorical variables."""
    contingency_table = pd.crosstab(x, y)
    if contingency_table.size == 0 or min(contingency_table.shape) <= 1:
        return 0.0
    
    try:
        chi2 = stats.chi2_contingency(contingency_table)[0]
    except Exception:
        return 0.0
        
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0
        
    phi2 = chi2 / n
    r, k = contingency_table.shape
    
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    
    denominator = min((k_corr - 1), (r_corr - 1))
    if denominator <= 0:
        return 0.0
        
    return np.sqrt(phi2_corr / denominator)


def analyze_target_signals(df):
    """Analyzes each target for type, missingness, cardinality, and association strength with VIN positions."""
    print("\n=== Running Automatic Target Signal Analysis ===")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Normalization
    df = df.copy()
    df["chassisNumber"] = df["chassisNumber"].apply(normalize_vin)
    df = df[df["chassisNumber"].apply(validate_vin)].copy()
    
    # Split characters
    for i in range(17):
        df[f"pos_{i}"] = df["chassisNumber"].str[i]
        
    # Sample for quick Cramer's V calculation (up to 100k rows)
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    analysis_report = [
        "# Automatic Target Signal Analysis Report\n",
        f"Analyzed a representative sample of **{sample_size:,}** valid records.\n",
        "## Positional Signal Strength Table\n",
        "| Target Column | Type | Cardinality | Missing % | Max Cramer's V | Best Position | Signal Status |",
        "| --- | --- | --- | --- | --- | --- | --- |"
    ]
    
    warnings = []
    weak_targets = []
    
    for col in ALL_TARGETS:
        missing_pct = df[col].isnull().mean() * 100
        nunique = df[col].nunique()
        
        is_regression = col in REGRESSION_TARGETS
        dtype_str = "Regression" if is_regression else "Classification"
        
        # Discretize continuous targets for Cramer's V
        if is_regression:
            target_series = pd.qcut(df_sample[col].fillna(df_sample[col].median()), q=10, labels=False, duplicates="drop")
        else:
            target_series = df_sample[col].astype(str).fillna("UNKNOWN")
            
        max_cramers_v = 0.0
        best_pos = -1
        for i in range(17):
            cv = calculate_cramers_v(df_sample[f"pos_{i}"], target_series)
            if cv > max_cramers_v:
                max_cramers_v = cv
                best_pos = i + 1
                
        signal_status = "STRONG" if max_cramers_v >= 0.10 else "WEAK"
        if signal_status == "WEAK":
            weak_targets.append(col)
            warn_msg = f"Target '{col}' has weak position association (Max Cramer's V = {max_cramers_v:.3f} at Position {best_pos})."
            warnings.append(warn_msg)
            print(f"  [WARNING] {warn_msg}")
        else:
            print(f"  [INFO] Target '{col}' has strong association (Max Cramer's V = {max_cramers_v:.3f} at Position {best_pos}).")
            
        analysis_report.append(
            f"| `{col}` | {dtype_str} | {nunique:,} | {missing_pct:.2f}% | {max_cramers_v:.3f} | Position {best_pos} | **{signal_status}** |"
        )
        
    # Add Warnings section
    analysis_report.append("\n## Signal Warnings")
    if warnings:
        analysis_report.append("The following targets show **weak structural signals** in the VIN. Standard neural networks may struggle to predict them reliably because they might not be systematically encoded inside the chassis number:")
        for w in warnings:
            analysis_report.append(f"- :warning: {w}")
    else:
        analysis_report.append("- No weak target signals detected. All targets show reasonable position correlation.")
        
    # Write report file
    report_path = os.path.join(REPORTS_DIR, "target_signal_analysis_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(analysis_report))
    print(f"Target analysis report saved to: {report_path}")
    
    return weak_targets


class MultiTaskFocalLoss(nn.Module):
    """Focal Loss with support for multi-class, class weights, and label smoothing."""
    def __init__(self, alpha_dict=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha_dict = alpha_dict if alpha_dict is not None else {}

    def forward(self, inputs, targets, target_name):
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)
        
        # Get class weights if available
        alpha = self.alpha_dict.get(target_name, None)
        if alpha is not None:
            alpha = alpha.to(inputs.device)

        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(inputs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Cross entropy loss with smoothed labels
            ce_loss = -torch.sum(true_dist * log_probs, dim=-1)
            
            # Apply class weights
            if alpha is not None:
                class_weights = alpha[targets]
                ce_loss = ce_loss * class_weights
        else:
            ce_loss = F.nll_loss(log_probs, targets, weight=alpha, reduction='none')

        # Focal factor
        probs = torch.exp(-F.nll_loss(log_probs, targets, reduction='none'))
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(train_loader, num_classes_dict):
    """Computes inverse-frequency weights for class balance."""
    print("Computing class weights from training loader labels...")
    counts_dict = {col: torch.zeros(num_classes) for col, num_classes in num_classes_dict.items()}
    
    for batch in train_loader:
        for col in num_classes_dict.keys():
            targets = batch[col]
            for val in targets:
                counts_dict[col][val] += 1
                
    weights_dict = {}
    for col, counts in counts_dict.items():
        total = counts.sum().item()
        num_classes = len(counts)
        counts = torch.clamp(counts, min=1)
        weights = total / (num_classes * counts)
        # Normalize
        weights = weights / weights.mean()
        weights_dict[col] = weights
        
    return weights_dict


def train_epoch(model, loader, optimizer, loss_fn_cls, loss_fn_reg, reg_weight, device):
    model.train()
    epoch_loss = 0.0
    
    for batch in loader:
        optimizer.zero_grad()
        
        tokens = batch["tokens"].to(device)
        
        # Teacher forcing targets with 50% probability during training
        teacher_forcing_targets = None
        if torch.rand(1).item() < 0.5:
            teacher_forcing_targets = {
                col: batch[col].to(device) for col in ["make_grouped", "model_final", "trim_raw"]
            }
            
        outputs = model(tokens, teacher_forcing_targets)
        
        # Compute losses
        loss = 0.0
        # Classification Loss
        for col in CLASSIFICATION_TARGETS:
            logits = outputs[col]
            targets = batch[col].to(device)
            loss += loss_fn_cls(logits, targets, col)
            
        # Regression Loss
        for col in REGRESSION_TARGETS:
            preds = outputs[col]
            targets = batch[col].to(device)
            loss += reg_weight * loss_fn_reg(preds, targets)
            
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)


def validate(model, loader, loss_fn_cls, loss_fn_reg, reg_weight, device):
    model.eval()
    val_loss = 0.0
    all_targets = {col: [] for col in ALL_TARGETS}
    all_preds = {col: [] for col in ALL_TARGETS}
    
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            outputs = model(tokens)
            
            # Compute loss
            loss = 0.0
            for col in CLASSIFICATION_TARGETS:
                logits = outputs[col]
                targets = batch[col].to(device)
                loss += loss_fn_cls(logits, targets, col)
                
            for col in REGRESSION_TARGETS:
                preds = outputs[col]
                targets = batch[col].to(device)
                loss += reg_weight * loss_fn_reg(preds, targets)
                
            val_loss += loss.item()
            
            # Store outputs
            for col in CLASSIFICATION_TARGETS:
                all_targets[col].append(batch[col].numpy())
                all_preds[col].append(outputs[col].cpu().numpy())
                
            for col in REGRESSION_TARGETS:
                all_targets[col].append(batch[col].numpy())
                all_preds[col].append(outputs[col].cpu().numpy())
                
    # Concatenate targets and predictions
    for col in ALL_TARGETS:
        all_targets[col] = np.concatenate(all_targets[col], axis=0)
        all_preds[col] = np.concatenate(all_preds[col], axis=0)
        
    metrics = {"loss": val_loss / len(loader)}
    
    # Compute metrics for classification
    for col in CLASSIFICATION_TARGETS:
        col_metrics = calculate_classification_metrics(all_targets[col], all_preds[col])
        for k, v in col_metrics.items():
            metrics[f"{col}_{k}"] = v
            
    # Compute metrics for regression
    for col in REGRESSION_TARGETS:
        col_metrics = calculate_regression_metrics(all_targets[col], all_preds[col])
        for k, v in col_metrics.items():
            metrics[f"{col}_{k}"] = v
            
    return metrics


def train_catboost_ensemble(train_tokens, train_cls, train_reg, val_tokens, val_cls, val_reg, pipeline):
    """Trains a quick CatBoost model for each target for ensemble validation."""
    print("\n=== Training CatBoost Models for Ensemble ===")
    from catboost import CatBoostClassifier, CatBoostRegressor
    
    catboost_models = {}
    
    # Prep tabular formats
    X_train = pd.DataFrame(train_tokens, columns=[f"pos_{i}" for i in range(17)]).astype(str)
    X_val = pd.DataFrame(val_tokens, columns=[f"pos_{i}" for i in range(17)]).astype(str)
    cat_features = [f"pos_{i}" for i in range(17)]
    
    for col in CLASSIFICATION_TARGETS:
        print(f"Training CatBoost Classifier for {col}...")
        y_train = train_cls[col]
        y_val = val_cls[col]
        num_classes = len(np.unique(y_train))
        
        model = CatBoostClassifier(
            iterations=80,
            depth=5,
            learning_rate=0.15,
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
            random_seed=42,
            verbose=False
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=10,
            verbose=False
        )
        catboost_models[col] = model
        
    for col in REGRESSION_TARGETS:
        print(f"Training CatBoost Regressor for {col}...")
        y_train = train_reg[col]
        y_val = val_reg[col]
        
        model = CatBoostRegressor(
            iterations=80,
            depth=5,
            learning_rate=0.15,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=10,
            verbose=False
        )
        catboost_models[col] = model
        
    # Save CatBoost models dict
    catboost_save_path = os.path.join(MODEL_DIR, "catboost.pkl")
    with open(catboost_save_path, "wb") as f:
        pickle.dump(catboost_models, f)
    print(f"CatBoost ensemble models saved to: {catboost_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Transformer-Based VIN Predictor")
    parser.add_argument("--sample_size", type=int, default=50000, help="Number of rows to sample for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    # Check data file
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Chassis dataset not found at {DATA_PATH}")
        
    # 1. Automatic Dataset Target and Signal analysis
    df = pd.read_csv(DATA_PATH)
    weak_targets = analyze_target_signals(df)
    
    # 2. Prepare DataLoaders
    train_loader, val_loader, test_loader, pipeline = prepare_data_loaders(
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )
    
    # Save Pipeline (encoders, tokenizer, scalers)
    pipeline_save_path = os.path.join(MODEL_DIR, "data_pipeline.pkl")
    pipeline.save(pipeline_save_path)
    print(f"Saved preprocessing pipeline to: {pipeline_save_path}")
    
    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    vocab_size = pipeline.tokenizer.vocab_size
    target_classes_dict = {col: encoder.num_classes() for col, encoder in pipeline.encoders.items()}
    
    class_weights = compute_class_weights(train_loader, target_classes_dict)
    
    model = VINTransformerEncoder(
        vocab_size=vocab_size,
        target_classes_dict=target_classes_dict,
        regression_targets_list=REGRESSION_TARGETS
    ).to(device)
    
    # Loss functions
    loss_fn_cls = MultiTaskFocalLoss(alpha_dict=class_weights, gamma=2.0, label_smoothing=0.1)
    loss_fn_reg = nn.HuberLoss(delta=1.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 4. Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(MODEL_DIR, "transformer_best.pt")
    
    print("\n=== Starting Transformer Model Training ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn_cls, loss_fn_reg,
            reg_weight=0.5, device=device
        )
        
        val_metrics = validate(
            model, val_loader, loss_fn_cls, loss_fn_reg,
            reg_weight=0.5, device=device
        )
        
        scheduler.step()
        
        val_loss = val_metrics["loss"]
        lr_curr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_curr:.6f}")
        
        # Display sample classification accuracy
        make_acc = val_metrics.get("make_grouped_accuracy", 0.0)
        model_acc = val_metrics.get("model_final_accuracy", 0.0)
        print(f"   Validation Acc -> make_grouped: {make_acc*100:.1f}% | model_final: {model_acc*100:.1f}%")
        
        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("   ==> Saved new best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # 5. Load best model and evaluate on Test set
    print("\n=== Evaluating Best Model on Test Dataset ===")
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = validate(model, test_loader, loss_fn_cls, loss_fn_reg, reg_weight=0.5, device=device)
    print(f"Test Set Loss: {test_metrics['loss']:.4f}")
    for col in CLASSIFICATION_TARGETS:
        print(f"  {col:16s} Accuracy: {test_metrics[f'{col}_accuracy']*100:.1f}% | Top-3 Accuracy: {test_metrics[f'{col}_top_3_accuracy']*100:.1f}%")
    for col in REGRESSION_TARGETS:
        print(f"  {col:16s} MAE: {test_metrics[f'{col}_mae']:.2f} | R2: {test_metrics[f'{col}_r2']:.3f}")
        
    # Save final validation metrics as reports metadata
    val_report_path = os.path.join(REPORTS_DIR, "validation_metrics_report.pkl")
    with open(val_report_path, "wb") as f:
        pickle.dump(test_metrics, f)
        
    # Save a text report
    metrics_txt = ["# Model Evaluation Report\n"]
    for k, v in test_metrics.items():
        metrics_txt.append(f"- **{k}**: {v:.4f}")
    with open(os.path.join(REPORTS_DIR, "model_evaluation_metrics.md"), "w") as f:
        f.write("\n".join(metrics_txt))
        
    # 6. Optional: Train CatBoost Ensemble
    # Extract raw lists from training dataset for tabular training
    train_tokens = train_loader.dataset.tokenized_vins.numpy()
    train_cls = {col: train_loader.dataset.classification_targets[col].numpy() for col in CLASSIFICATION_TARGETS}
    train_reg = {col: train_loader.dataset.regression_targets[col].numpy() for col in REGRESSION_TARGETS}
    
    val_tokens = val_loader.dataset.tokenized_vins.numpy()
    val_cls = {col: val_loader.dataset.classification_targets[col].numpy() for col in CLASSIFICATION_TARGETS}
    val_reg = {col: val_loader.dataset.regression_targets[col].numpy() for col in REGRESSION_TARGETS}
    
    train_catboost_ensemble(
        train_tokens, train_cls, train_reg,
        val_tokens, val_cls, val_reg,
        pipeline
    )
    
    print("\nTraining script finished successfully!")


if __name__ == "__main__":
    main()
