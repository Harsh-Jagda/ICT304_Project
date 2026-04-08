"""
hpo.py — Hyperparameter Optimization with Optuna

Finds the best LightGBM hyperparameters using Bayesian optimization.
Saves results to best_params.json for use by train.py.

Run AFTER data_prep.py has generated processed_data_all.parquet.
Usage: python hpo.py
"""
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import gc

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(DATA_DIR, "data", "processed", "processed_data_all.parquet")
PARAMS_OUT = os.path.join(DATA_DIR, "models", "best_params.json")

# Must match train.py
CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                 'wday', 'month', 'event_name_1', 'event_type_1']
NUM_FEATURES  = ['sell_price', 'lag_7', 'lag_28',
                 'rolling_mean_7', 'rolling_std_7',
                 'rolling_mean_28', 'rolling_std_28']
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET   = 'sales'

N_TRIALS = 15  # ~30-40 min total on this dataset


def load_and_split():
    """Load data and return train/validation sets."""
    fallback = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
    path = DATA_FILE if os.path.exists(DATA_FILE) else fallback
    print(f"Loading data from {os.path.basename(path)}...")
    
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Use a 3-month validation window within training years to speed up HPO
    # Train: 2011-2013, Val: 2014 (keeps HPO fast)
    train_df = df[df['date'] < '2014-01-01'].dropna(subset=FEATURES + [TARGET])
    val_df   = df[(df['date'] >= '2014-01-01') & (df['date'] < '2015-01-01')].dropna(subset=FEATURES + [TARGET])
    
    # Sample to speed up HPO (use 50% of data)
    train_df = train_df.sample(frac=0.5, random_state=42)
    val_df   = val_df.sample(frac=0.5, random_state=42)
    
    print(f"  HPO Train: {len(train_df):,} | HPO Val: {len(val_df):,}")
    del df; gc.collect()
    return train_df, val_df


def objective(trial, train_df, val_df):
    """Optuna objective: minimize MAE on validation set."""
    params = {
        'objective':        'regression',
        'metric':           'mae',       # optimize MAE directly
        'verbosity':        -1,
        'boosting_type':    'gbdt',
        'n_jobs':           -1,
        'seed':             42,
        # Search space
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 31, 255),
        'min_child_samples':trial.suggest_int('min_child_samples', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':     trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1':        trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2':        trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    train_data = lgb.Dataset(train_df[FEATURES], label=train_df[TARGET],
                              categorical_feature=CAT_FEATURES)
    val_data   = lgb.Dataset(val_df[FEATURES], label=val_df[TARGET],
                              categorical_feature=CAT_FEATURES)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
    )
    
    # Encode categoricals to integer codes before prediction
    X_val = val_df[FEATURES].copy()
    for col in CAT_FEATURES:
        if col in X_val.columns:
            X_val[col] = X_val[col].astype('category').cat.codes
    
    preds   = np.clip(model.predict(X_val.values), 0, None)
    actuals = val_df[TARGET].values
    mae     = float(np.mean(np.abs(preds - actuals)))
    return mae


def main():
    print("="*60)
    print(" HYPERPARAMETER OPTIMIZATION (Optuna) ".center(60, "="))
    print(f" Trials: {N_TRIALS} — this will take ~30-45 minutes ".center(60, "="))
    print("="*60)
    
    train_df, val_df = load_and_split()
    
    study = optuna.create_study(direction='minimize', study_name='lgbm_wms_hpo')
    study.optimize(
        lambda trial: objective(trial, train_df, val_df),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    
    best = study.best_params
    best_mae = study.best_value
    
    print("\n" + "="*60)
    print(f"  Best MAE: {best_mae:.4f}")
    print(f"  Best params: {json.dumps(best, indent=4)}")
    
    with open(PARAMS_OUT, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"\n  Saved to {PARAMS_OUT}")
    print("  ✓ Now run: python train.py")


if __name__ == "__main__":
    main()
