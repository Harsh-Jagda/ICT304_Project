import pandas as pd
import numpy as np

def process_uploaded_csv(csv_path):
    if csv_path.endswith(".parquet"):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)

    # Force all column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    if "sales" not in df.columns:
        return None, "Data must contain 'sales' column"

    # Fill basic missing values
    df["sales"] = df["sales"].fillna(0)
    df["sell_price"] = df["sell_price"].fillna(df["sell_price"].mean()) if "sell_price" in df.columns else 0.0

    # Grouping by item_id is crucial for correct lag/rolling math
    group_col = 'item_id' if 'item_id' in df.columns else None

    for window in [7, 28]:
        col_lag = f"lag_{window}"
        if col_lag not in df.columns:
            df[col_lag] = df.groupby(group_col)['sales'].transform(lambda x: x.shift(window)).fillna(0) if group_col else df['sales'].shift(window).fillna(0)

        col_mean = f"rolling_mean_{window}"
        if col_mean not in df.columns:
            df[col_mean] = df.groupby(group_col)['sales'].transform(lambda x: x.shift(1).rolling(window).mean()).fillna(0) if group_col else 0

        col_std = f"rolling_std_{window}"
        if col_std not in df.columns:
            df[col_std] = df.groupby(group_col)['sales'].transform(lambda x: x.shift(1).rolling(window).std()).fillna(0) if group_col else 0

    # Generate wday and month for the ML model
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['wday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    else:
        df['wday'], df['month'] = 0, 1

    # Categorical handling: Force everything to string then category
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
    for col in categorical_cols:
        if col in df.columns:
            # strip() removes accidental spaces like "TX " vs "TX"
            df[col] = df[col].astype(str).str.strip().fillna('Unknown').astype('category')
        else:
            df[col] = pd.Categorical(['Unknown'] * len(df))

    return df, None
