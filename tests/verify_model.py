import pandas as pd
import joblib
import os
import numpy as np

# Path config
DATA_DIR = r"c:\Users\INRUI\OneDrive\Desktop\m5-forecasting-accuracy"
MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
TEST_DATA_PATH = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")

def verify_real_predictions():
    print("--- Testing with Real Model and Real Data ---")
    
    # 1. Load resources
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(TEST_DATA_PATH)
    
    # 2. Select a few items for clarity
    sample_items = df['item_id'].unique()[:3]
    df_sample = df[df['item_id'].isin(sample_items)].copy()
    
    # 3. Preparation (LightGBM needs specific features)
    features = model.feature_name()
    X = df_sample[features].copy()
    
    # Pre-process categories (as done in recommendation.py)
    CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'wday', 'month', 'event_name_1', 'event_type_1']
    for col in CAT_FEATURES:
        if col in X.columns:
            if hasattr(X[col], 'cat'):
                X[col] = X[col].cat.codes
            else:
                X[col] = X[col].astype('category').cat.codes

    # 4. Predict
    print(f"Predicting demand for {len(df_sample)} rows...")
    predictions = model.predict(X.values)
    df_sample['predicted_demand'] = predictions
    
    # 5. Show results for 1 item
    print("\nSample Predictions (Item: " + sample_items[0] + "):")
    item_view = df_sample[df_sample['item_id'] == sample_items[0]].tail(5)
    print(item_view[['date', 'sales', 'predicted_demand', 'sell_price']])
    
    # Summary stats
    print("\nPrediction Summary:")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  Max: {predictions.max():.4f}")
    print(f"  Mean: {predictions.mean():.4f}")

if __name__ == "__main__":
    verify_real_predictions()
