"""
Market Value Prediction Model for vin-sphere-python

README SECTION & POSITIONING
-----------------------------
1. What the model does:
   Learns vehicle market value directly from historical retail valuations.

2. What the model captures:
   - Brand (make) effects
   - Model effects
   - Trim effects
   - Vehicle age effects

3. What the model cannot capture:
   - Mileage
   - Condition
   - Accident history
   - Regional market fluctuations
   - Specification differences not encoded in trim

4. Positioning within VIN Sphere:
   This model is a lightweight market-value estimator trained on retail valuation data.
   It should be considered a complementary cross-check to the existing depreciation-based
   valuation workflow and not a replacement for the richer Module 3/4 valuation models
   trained on the full insurance dataset containing additional vehicle attributes.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
from difflib import SequenceMatcher

# Configurations
REF_YEAR = 2026
CAT_FEATURES = ["make", "model", "trim"]

# Resolve paths relative to the script's directory
script_dir = Path(__file__).resolve().parent
SRC_FILE = script_dir / "lookup_data_clean.csv"
MODEL_OUT = script_dir / "price_model.cbm"
METRICS_OUT = script_dir / "holdout_metrics.txt"

# Caches for fast inference
_lookup_db = None
_cov_cache = None


def normalize_lookup_text(value) -> str:
    """Normalizes text for key generation by removing non-alphanumeric chars and capitalizing."""
    if pd.isna(value) or value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().strip())


def main():
    print("=== Step 1: Data Loading & Preprocessing ===")
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"Cleaned source file not found at: {SRC_FILE}")

    # Load pre-cleaned and normalized dataset
    df = pd.read_csv(SRC_FILE, dtype=str)
    initial_rows = len(df)
    print(f"Loaded {initial_rows:,} pre-cleaned rows.")

    # Convert numeric columns
    df["year_n"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_n"] = pd.to_numeric(df["price"], errors="coerce")

    # Coerce and handle missing values
    df = df[df["year_n"].notna() & df["price_n"].notna() & df["make"].notna() & df["model"].notna()].copy()
    
    # Outlier filtering (extreme or placeholder prices)
    cleaned_df = df[(df["price_n"] >= 500) & (df["price_n"] <= 5000000)].copy()
    dropped_outliers = len(df) - len(cleaned_df)
    print(f"Filtered out {dropped_outliers:,} outlier prices (< 500 or > 5,000,000 AED).")
    
    df = cleaned_df

    # Impute missing trims
    df["trim"] = df["trim"].fillna("UNKNOWN")

    # Normalize make/model/trim for matching/training
    df["make"] = df["make"].str.upper().str.strip()
    df["model"] = df["model"].str.upper().str.strip()
    df["trim"] = df["trim"].str.upper().str.strip()

    # Feature Engineering
    df["year_n"] = df["year_n"].astype(int)
    df["age"] = (REF_YEAR - df["year_n"]).clip(lower=0)

    print("\n=== Step 2: Feature Engineering ===")
    X = df[["make", "model", "trim", "age"]]
    y = np.log1p(df["price_n"])
    print(f"Dataset features: make, model, trim (categorical), age (numeric). Target: log1p(price).")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"Train split size: {len(X_train):,}, Test split size: {len(X_test):,}")

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

    print("\n=== Step 3: Model Training ===")
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="MAE",
        early_stopping_rounds=50,
        random_seed=42,
        task_type="CPU",
        verbose=100
    )
    
    model.fit(train_pool, eval_set=test_pool)
    model.save_model(str(MODEL_OUT))
    print(f"Model saved to {MODEL_OUT}")

    print("\n=== Step 4: Model Evaluation ===")
    pred_log = model.predict(X_test)
    pred_price = np.expm1(pred_log)
    actual_price = np.expm1(y_test).values

    # Calculate metrics on original price scale
    mae = np.mean(np.abs(pred_price - actual_price))
    rmse = np.sqrt(np.mean((pred_price - actual_price) ** 2))
    errors = np.abs(pred_price - actual_price) / actual_price
    mape = np.mean(errors) * 100

    print(f"Standalone ML Model Evaluation Metrics (n={len(X_test):,}):")
    print(f"  MAE  : AED {mae:,.2f}")
    print(f"  RMSE : AED {rmse:,.2f}")
    print(f"  MAPE : {mape:.2f}%")

    print("\nStandalone ML Model Accuracy Tolerances:")
    for tolerance in [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]:
        acc = np.mean(errors <= tolerance) * 100
        print(f"  Within {int(tolerance*100):3d}% error : {acc:.2f}%")

    # Evaluate Hybrid Lookup + ML Model on the Holdout set
    # Create the lookup keys on training set
    train_lookup_df = X_train.copy()
    train_lookup_df["price_n"] = np.expm1(y_train)
    train_lookup_df["make_key"] = train_lookup_df["make"].apply(normalize_lookup_text)
    train_lookup_df["model_key"] = train_lookup_df["model"].apply(normalize_lookup_text)
    train_lookup_df["trim_key"] = train_lookup_df["trim"].apply(normalize_lookup_text)
    train_lookup_df["year_key"] = (REF_YEAR - train_lookup_df["age"]).astype(int).astype(str)
    train_lookup_df["vin_key"] = ""  # No vin column in training features

    def evaluate_lookup_price(row):
        make_key = normalize_lookup_text(row["make"])
        model_key = normalize_lookup_text(row["model"])
        trim_key = normalize_lookup_text(row["trim"])
        year_key = str(int(REF_YEAR - row["age"]))
        
        base_matches = train_lookup_df[
            (train_lookup_df["make_key"] == make_key)
            & (train_lookup_df["model_key"] == model_key)
            & (train_lookup_df["year_key"] == year_key)
        ]

        if base_matches.empty:
            return None, None

        if trim_key and trim_key != "UNKNOWN":
            exact_trim_matches = base_matches[base_matches["trim_key"] == trim_key]
            if not exact_trim_matches.empty:
                return float(exact_trim_matches["price_n"].median()), "exact_trim"

            contains_trim_matches = base_matches[
                base_matches["trim_key"].apply(
                    lambda csv_trim: bool(csv_trim)
                    and (trim_key in csv_trim or csv_trim in trim_key)
                )
            ]
            if not contains_trim_matches.empty:
                return float(contains_trim_matches["price_n"].median()), "contains_trim"

            fuzzy_matches = base_matches.copy()
            fuzzy_matches["trim_score"] = fuzzy_matches["trim_key"].apply(
                lambda csv_trim: SequenceMatcher(None, trim_key, csv_trim).ratio()
                if csv_trim
                else 0.0
            )
            fuzzy_matches = fuzzy_matches[fuzzy_matches["trim_score"] >= 0.82]
            if not fuzzy_matches.empty:
                best_score = fuzzy_matches["trim_score"].max()
                best_match_df = fuzzy_matches[fuzzy_matches["trim_score"] == best_score]
                return float(best_match_df["price_n"].median()), "fuzzy_trim"

        if base_matches["trim_key"].nunique() == 1:
            return float(base_matches["price_n"].median()), "single_trim_fallback"

        return None, None

    hybrid_preds = []
    methods_used = []
    
    test_eval_df = X_test.copy()
    test_eval_df["cb_pred"] = pred_price
    
    for idx, row in test_eval_df.iterrows():
        price, method = evaluate_lookup_price(row)
        if price is not None:
            hybrid_preds.append(price)
            methods_used.append(method)
        else:
            hybrid_preds.append(row["cb_pred"])
            methods_used.append("catboost")

    hybrid_preds = np.array(hybrid_preds)
    hybrid_mae = np.mean(np.abs(hybrid_preds - actual_price))
    hybrid_rmse = np.sqrt(np.mean((hybrid_preds - actual_price) ** 2))
    hybrid_errors = np.abs(hybrid_preds - actual_price) / actual_price
    hybrid_mape = np.mean(hybrid_errors) * 100

    print("\n=== Step 5: Hybrid Lookup + ML Model Evaluation ===")
    print(f"Hybrid MAE  : AED {hybrid_mae:,.2f}")
    print(f"Hybrid RMSE : AED {hybrid_rmse:,.2f}")
    print(f"Hybrid MAPE : {hybrid_mape:.2f}%")
    print(f"Lookup hit rate: {np.mean(np.array(methods_used) != 'catboost') * 100:.2f}%")
    
    print("\nHybrid Model Accuracy Tolerances:")
    for tolerance in [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]:
        acc = np.mean(hybrid_errors <= tolerance) * 100
        print(f"  Within {int(tolerance*100):3d}% error : {acc:.2f}%")

    # Feature Importance (Printing top 20 most important features)
    print("\nFeature Importances:")
    fi = model.get_feature_importance(train_pool)
    sorted_features = sorted(zip(X.columns, fi), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(sorted_features[:20], 1):
        print(f"  {rank}. {name:10s} : {score:.2f}%")

    # Save metrics file
    with open(METRICS_OUT, "w") as f:
        f.write(f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nMAPE={mape:.2f}%\n")
        f.write(f"Hybrid_MAE={hybrid_mae:.2f}\nHybrid_MAPE={hybrid_mape:.2f}%\n")
        f.write(f"Accuracy_Within_75_Percent={np.mean(hybrid_errors <= 0.75) * 100:.2f}%\n")
    print(f"Metrics written to {METRICS_OUT}")


def _init_inference_caches():
    """Initializes the database caches to speed up single-row predictions."""
    global _lookup_db, _cov_cache
    if _lookup_db is not None and _cov_cache is not None:
        return

    csv_path = Path(__file__).resolve().parent / "lookup_data_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned source file not found at: {csv_path}")

    # Load and process full dataset
    df = pd.read_csv(csv_path, dtype=str)
    df["year_n"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_n"] = pd.to_numeric(df["price"], errors="coerce")

    # Clean and filter outliers
    df = df[df["year_n"].notna() & df["price_n"].notna() & (df["price_n"] >= 500) & (df["price_n"] <= 5000000) & df["make"].notna() & df["model"].notna()].copy()
    df["year_n"] = df["year_n"].astype(int)
    df["make"] = df["make"].str.upper().str.strip()
    df["model"] = df["model"].str.upper().str.strip()
    df["trim"] = df["trim"].fillna("UNKNOWN").str.upper().str.strip()

    # Pre-calculate normalized keys
    df["make_key"] = df["make"].apply(normalize_lookup_text)
    df["model_key"] = df["model"].apply(normalize_lookup_text)
    df["trim_key"] = df["trim"].apply(normalize_lookup_text)
    df["year_key"] = df["year_n"].astype(str)
    if "chassisNumber" in df.columns:
        df["vin_key"] = df["chassisNumber"].apply(normalize_lookup_text)
    else:
        df["vin_key"] = ""

    _lookup_db = df

    # Build CoV cache for make/model combinations
    cov_dict = {}
    grouped = df.groupby(["make_key", "model_key"])["price_n"]
    for keys, prices in grouped:
        if len(prices) > 1:
            std_val = prices.std()
            mean_val = prices.mean()
            if mean_val > 0:
                coef_var = std_val / mean_val
                model_confidence = max(100.0 - (coef_var * 50.0), 60.0)
                model_confidence = min(model_confidence, 98.0)
            else:
                model_confidence = 94.50
        else:
            model_confidence = 94.50
        cov_dict[keys] = model_confidence
    _cov_cache = cov_dict


def get_hierarchical_price(make: str, model: str, trim: str, year: int, vin: str = "") -> tuple:
    """Attempts to match the input to the cleaned database using a hierarchical lookup strategy."""
    _init_inference_caches()

    make_key = normalize_lookup_text(make)
    model_key = normalize_lookup_text(model)
    trim_key = normalize_lookup_text(trim)
    year_key = str(int(year))

    vin_key = normalize_lookup_text(vin)
    if vin_key:
        vin_matches = _lookup_db[_lookup_db["vin_key"] == vin_key]
        if not vin_matches.empty:
            return float(vin_matches["price_n"].median()), "vin_exact"

    base_matches = _lookup_db[
        (_lookup_db["make_key"] == make_key)
        & (_lookup_db["model_key"] == model_key)
        & (_lookup_db["year_key"] == year_key)
    ]

    if base_matches.empty:
        return None, None

    if trim_key and trim_key != "UNKNOWN":
        exact_trim_matches = base_matches[base_matches["trim_key"] == trim_key]
        if not exact_trim_matches.empty:
            return float(exact_trim_matches["price_n"].median()), "exact_trim"

        contains_trim_matches = base_matches[
            base_matches["trim_key"].apply(
                lambda csv_trim: bool(csv_trim)
                and (trim_key in csv_trim or csv_trim in trim_key)
            )
        ]
        if not contains_trim_matches.empty:
            return float(contains_trim_matches["price_n"].median()), "contains_trim"

        fuzzy_matches = base_matches.copy()
        fuzzy_matches["trim_score"] = fuzzy_matches["trim_key"].apply(
            lambda csv_trim: SequenceMatcher(None, trim_key, csv_trim).ratio()
            if csv_trim
            else 0.0
        )
        fuzzy_matches = fuzzy_matches[fuzzy_matches["trim_score"] >= 0.82]
        if not fuzzy_matches.empty:
            best_score = fuzzy_matches["trim_score"].max()
            best_match_df = fuzzy_matches[fuzzy_matches["trim_score"] == best_score]
            return float(best_match_df["price_n"].median()), "fuzzy_trim"

    if base_matches["trim_key"].nunique() == 1:
        return float(base_matches["price_n"].median()), "single_trim_fallback"

    return None, None


def predict_price(make: str, model: str, trim: str, year: int) -> float:
    """
    Predict price helper function. Uses hierarchical database lookup and falls back to CatBoost.
    """
    price, _ = get_hierarchical_price(make, model, trim, year)
    if price is not None:
        return price

    model_path = Path(__file__).resolve().parent / "price_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")

    cb_model = CatBoostRegressor()
    cb_model.load_model(str(model_path))

    make_val = str(make).strip()
    model_val = str(model).strip()
    trim_val = str(trim).strip() if pd.notna(trim) and trim else "UNKNOWN"
    
    age_val = max(REF_YEAR - int(year), 0)

    # Build inference DataFrame matching columns
    input_df = pd.DataFrame([{
        "make": make_val,
        "model": model_val,
        "trim": trim_val,
        "age": float(age_val)
    }])

    pred_log = cb_model.predict(input_df)[0]
    return float(np.expm1(pred_log))


def predict_price_with_confidence(make: str, model: str, trim: str, year: int) -> dict:
    """
    Predict price and output the dynamic confidence score. Aligned with app.py valuation logic.
    """
    price, method = get_hierarchical_price(make, model, trim, year)
    
    _init_inference_caches()
    make_key = normalize_lookup_text(make)
    model_key = normalize_lookup_text(model)
    model_confidence = _cov_cache.get((make_key, model_key), 94.50)

    if price is None:
        model_path = Path(__file__).resolve().parent / "price_model.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")

        cb_model = CatBoostRegressor()
        cb_model.load_model(str(model_path))

        make_val = str(make).strip()
        model_val = str(model).strip()
        trim_val = str(trim).strip() if pd.notna(trim) and trim else "UNKNOWN"
        age_val = max(REF_YEAR - int(year), 0)

        input_df = pd.DataFrame([{
            "make": make_val,
            "model": model_val,
            "trim": trim_val,
            "age": float(age_val)
        }])

        pred_log = cb_model.predict(input_df)[0]
        price = float(np.expm1(pred_log))
        method = "catboost"

    return {
        "price": price,
        "confidence": round(model_confidence / 100.0, 4),
        "method": method
    }


if __name__ == "__main__":
    main()
