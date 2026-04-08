"""
data_prep.py — Data Preparation Pipeline (All States)

Updated to process ALL states (CA, TX, WI) instead of CA only.
Adds state_id as an explicit feature column for the model.
Output: processed_data_all.parquet
"""
import pandas as pd
import numpy as np
import os
import gc

# -------------------------------------------------------------------------
# CONSTANTS & CONFIG
# -------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(DATA_DIR, "data", "processed", "processed_data_all.parquet")

def reduce_mem_usage(df, verbose=True):
    """
    Standard memory reduction function to downcast numeric types.
    Saves the user's laptop from dying.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def load_sales():
    """Load ALL states — no filter."""
    print("Loading full sales data (ALL states: CA, TX, WI)...")
    sales = pd.read_csv(os.path.join(DATA_DIR, "data", "raw", "sales_train_evaluation.csv"))
    print(f"  States found: {sorted(sales['state_id'].unique())}")
    print(f"  Total rows (items): {len(sales)}")
    return sales

def melt_sales(sales):
    print("Melting sales data from wide to long format...")
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_melted = pd.melt(sales, id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')
    del sales
    gc.collect()
    return sales_melted

def merge_calendar_and_prices(df):
    print("Merging calendar and price data...")
    calendar = pd.read_csv(os.path.join(DATA_DIR, "data", "raw", "calendar.csv"))
    df = df.merge(calendar, on='d', how='left')
    del calendar
    
    prices = pd.read_csv(os.path.join(DATA_DIR, "data", "raw", "sell_prices.csv"))
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del prices
    gc.collect()
    return df

def clean_data(df):
    print("Cleaning data (price imputation, pre-launch zeros, outlier capping)...")
    
    # 1. Price imputation
    df['sell_price'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.ffill().bfill())
    
    # 2. Filter pre-launch zeros
    df['non_zero'] = (df['sales'] > 0).astype(int)
    df['cum_sales'] = df.groupby(['id'])['non_zero'].transform('cumsum')
    df = df[df['cum_sales'] > 0].copy()
    df.drop(['non_zero', 'cum_sales'], axis=1, inplace=True)
    
    # 3. Outlier capping at 99th percentile × 2 (per category)
    q99 = df.groupby('cat_id')['sales'].transform(lambda x: x.quantile(0.99))
    df['sales'] = np.where(df['sales'] > q99 * 2, (q99 * 2).astype(int), df['sales'])
    
    return df

def create_features(df):
    print("Generating time-series features (lags & rolling windows)...")
    df.sort_values(['id', 'date'], inplace=True)
    
    # Lags
    for lag in [7, 28]:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].transform(lambda x: x.shift(lag))
        
    # Rolling stats
    for win in [7, 28]:
        df[f'rolling_mean_{win}'] = df.groupby('id')['sales'].transform(
            lambda x: x.shift(1).rolling(win).mean())
        df[f'rolling_std_{win}'] = df.groupby('id')['sales'].transform(
            lambda x: x.shift(1).rolling(win).std())
        
    return df

def load_sales_for_state(state: str):
    """Load sales for a single state — memory-efficient."""
    print(f"  Loading {state}...")
    sales = pd.read_csv(os.path.join(DATA_DIR, "data", "raw", "sales_train_evaluation.csv"))
    sales = sales[sales['state_id'] == state].reset_index(drop=True)
    return sales

def process_state(state: str) -> str:
    """Full pipeline for one state. Returns path to tmp parquet."""
    sales        = load_sales_for_state(state)
    sales_melted = melt_sales(sales)
    df           = merge_calendar_and_prices(sales_melted)
    del sales_melted; gc.collect()
    df           = clean_data(df)
    df           = create_features(df)
    # Ensure date is stored as datetime64, not string
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df           = reduce_mem_usage(df, verbose=False)
    
    tmp_path = os.path.join(DATA_DIR, f"_tmp_{state}.parquet")
    df.to_parquet(tmp_path, index=False)
    print(f"  {state} done: {len(df):,} rows saved to {tmp_path}")
    del df; gc.collect()
    return tmp_path

def main():
    states = ['CA', 'TX', 'WI']
    tmp_paths = []
    
    for state in states:
        print(f"\n=== Processing {state} ===")
        tmp_paths.append(process_state(state))
    
    print("\nCombining all states...")
    combined = pd.concat([pd.read_parquet(p) for p in tmp_paths], ignore_index=True)
    print(f"Final dataset: {len(combined):,} rows | {combined['state_id'].nunique()} states")
    
    print(f"Saving to {OUTPUT_FILE}...")
    combined.to_parquet(OUTPUT_FILE, index=False)
    
    # Clean up tmp files (silently skip if already moved/deleted by OneDrive etc.)
    for p in tmp_paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
