"""
train.py — Model Training Pipeline (All States)

Reads processed_data_all.parquet (all states).
Loads best_params.json if available (from hpo.py).
Registers trained model in model_registry.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import joblib
import gc
import sys
from datetime import datetime

# FIX #4: import shared path constant so train.py and retrain_manager.py
# always look at the same real_time_sales.csv location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.constants import REALTIME_SALES_SUBPATH

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(DATA_DIR, "data", "processed", "processed_data_all.parquet")
PARAMS_FILE = os.path.join(DATA_DIR, "models", "best_params.json")
MODEL_OUTPUT = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")

# state_id added for multi-state support
CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                 'wday', 'month', 'event_name_1', 'event_type_1']
NUM_FEATURES  = ['sell_price', 'lag_7', 'lag_28',
                 'rolling_mean_7', 'rolling_std_7',
                 'rolling_mean_28', 'rolling_std_28']
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET   = 'sales'

DEFAULT_PARAMS = {
    'objective':        'regression',
    'metric':           'rmse',
    'verbosity':        -1,
    'boosting_type':    'gbdt',
    'learning_rate':    0.05,
    'num_leaves':       63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'n_jobs':           -1,
    'seed':             42,
}

def load_params() -> dict:
    """Load best params from HPO if available, else use defaults."""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            best = json.load(f)
        print(f"  Loaded best params from {PARAMS_FILE}")
        params = {**DEFAULT_PARAMS, **best}
    else:
        print("  No best_params.json found — using default parameters.")
        params = DEFAULT_PARAMS
    return params

def load_data():
    # Fallback to CA-only data if all-states file not yet generated
    if os.path.exists(INPUT_FILE):
        print(f"Loading {INPUT_FILE}...")
        df = pd.read_parquet(INPUT_FILE)
    else:
        fallback = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
        print(f"  Warning: all-states data not found, falling back to {fallback}")
        df = pd.read_parquet(fallback)
    
    # FIX #4: use REALTIME_SALES_SUBPATH constant — matches retrain_manager.py exactly
    realtime_file = os.path.join(DATA_DIR, *REALTIME_SALES_SUBPATH)
    if os.path.exists(realtime_file):
        rt_df = pd.read_csv(realtime_file)
        df = pd.concat([df, rt_df], ignore_index=True)
        print(f"  Appended {len(rt_df)} rows from real_time_sales.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def split_data(df):
    train_mask = df['date'] < '2015-01-01'
    test_mask  = (df['date'] >= '2015-01-01') & (df['date'] < '2016-01-01')
    train_df = df[train_mask].dropna(subset=FEATURES + [TARGET])
    test_df  = df[test_mask].dropna(subset=FEATURES + [TARGET])
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df

def train_model(train_df, test_df, params):
    print("Training LightGBM model...")
    train_data = lgb.Dataset(train_df[FEATURES], label=train_df[TARGET],
                              categorical_feature=CAT_FEATURES)
    valid_data = lgb.Dataset(test_df[FEATURES], label=test_df[TARGET],
                              categorical_feature=CAT_FEATURES)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    return model

def evaluate_model(model, test_df) -> dict:
    """Calculate MAE and RMSE on the test set."""
    X_test = test_df[FEATURES].copy()
    for col in CAT_FEATURES:
        if col in X_test.columns and hasattr(X_test[col], 'cat'):
            X_test[col] = X_test[col].cat.codes
    
    preds = np.clip(model.predict(X_test.values), 0, None)
    actuals = test_df[TARGET].values
    mae  = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    print(f"  Test MAE: {mae:.4f} | Test RMSE: {rmse:.4f}")
    return {"mae": mae, "rmse": rmse}

def main():
    params   = load_params()
    df       = load_data()
    train_df, test_df = split_data(df)
    del df; gc.collect()
    
    model   = train_model(train_df, test_df, params)
    metrics = evaluate_model(model, test_df)
    
    # Determine training scope
    training_data = "all_states" if os.path.exists(INPUT_FILE) else "CA_only"
    
    # Register in model registry
    try:
        from src.mlops.model_registry import register_model
        version = register_model(model, metrics, params, training_data=training_data)
        print(f"  Registered as version {version}.")
    except Exception as e:
        print(f"  Could not register model: {e}")
        joblib.dump(model, MODEL_OUTPUT)
        print(f"  Saved directly to {MODEL_OUTPUT}.")
    
    # Always save test sample for UI
    test_sample_path = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")
    test_df.head(10000).to_parquet(test_sample_path)
    print(f"  Saved test sample to {test_sample_path}.")
    print("Training complete!")

if __name__ == "__main__":
    main()
