import pandas as pd

def process_uploaded_csv(csv_path):
    if csv_path.endswith(".parquet"):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)
    
    if "sales" not in df.columns:
        return None, "CSV must contain 'sales' column"
    
    # basic cleaning
    df["sales"] = df["sales"].fillna(0)
    
    # ✅ Convert categorical columns to category dtype (MUST match training)
    # These are the exact columns LightGBM expects as categorical
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'event_name_1', 'event_type_1']
    for col in categorical_cols:
        if col in df.columns:
            # Fill NaN before converting to category
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            df[col] = df[col].astype('category')
    
    # create rolling std if missing
    if "rolling_std_7" not in df.columns:
        df["rolling_std_7"] = df["sales"].rolling(7).std().fillna(0)
    
    # simple lag
    if "lag_7" not in df.columns:
        df["lag_7"] = df["sales"].shift(7).fillna(0)
    
    # Fill any remaining NaN in numeric columns with 0
    numeric_cols = df.select_dtypes(include=['float16', 'float32', 'float64', 'int']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df, None
