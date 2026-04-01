import pandas as pd
import numpy as np

def process_uploaded_csv(csv_path):
    if csv_path.endswith(".parquet"):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)

    if "sales" not in df.columns:
        return None, "CSV must contain 'sales' column"

    # Basic cleaning
    df["sales"] = df["sales"].fillna(0)

    # --- ADVANCED FEATURE ENGINEERING ---
    # We add 7, 14, and 28 day windows to capture weekly and monthly seasonality
    for window in [7, 14, 28]:
        # Rolling Standard Deviation (Volatility)
        col_std = f"rolling_std_{window}"
        if col_std not in df.columns:
            df[col_std] = df["sales"].rolling(window=window).std().fillna(0)
        
        # Rolling Mean (Trend)
        col_mean = f"rolling_mean_{window}"
        if col_mean not in df.columns:
            df[col_mean] = df["sales"].rolling(window=window).mean().fillna(0)
            
        # Lags (Historical context)
        col_lag = f"lag_{window}"
        if col_lag not in df.columns:
            df[col_lag] = df["sales"].shift(window).fillna(0)

    # Categorical handling (Must match LightGBM training)
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'event_name_1', 'event_type_1']
    for col in categorical_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            df[col] = df[col].astype('category')

    # Fill any remaining NaN in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df, None
