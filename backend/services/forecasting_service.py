from .data_prep_service import process_uploaded_csv
import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wms_lgbm_model.pkl")

def run_forecast(csv_path):
    try:
        print(f"DEBUG: Processing file: {csv_path}")
        
        df, err = process_uploaded_csv(csv_path)
        if err:
            print(f"DEBUG: Data prep error: {err}")
            return None, err
        
        print(f"DEBUG: Loaded {len(df)} rows")
        
        # Use RANDOM sampling instead of tail() to get diverse categories
        if len(df) > 10000:
            print(f"DEBUG: Randomly sampling 10,000 rows from {len(df)} total rows")
            df = df.sample(n=10000, random_state=42).copy()  # Changed from tail() to sample()
        
        # Check category distribution
        if 'cat_id' in df.columns:
            cat_counts = df['cat_id'].value_counts().to_dict()
            print(f"DEBUG: Category distribution: {cat_counts}")
        
        model = joblib.load(MODEL_PATH)
        feature_names = model.feature_name()
        
        # Fill missing features
        missing_features = [f for f in feature_names if f not in df.columns]
        for feat in missing_features:
            if feat in ['item_id', 'dept_id', 'cat_id', 'store_id', 'event_name_1', 'event_type_1']:
                df[feat] = pd.Categorical(['Unknown'] * len(df))
            else:
                df[feat] = 0
        
        # Select and encode features
        X = df[feature_names].copy()
        cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'event_name_1', 'event_type_1']
        for col in cat_cols:
            if col in X.columns:
                if X[col].dtype.name == 'category':
                    X[col] = X[col].cat.codes
                elif X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col]).codes
        
        # Predict
        X_numpy = X.values
        preds = model.predict(X_numpy)
        df["forecast"] = preds
        
        # Calculate metrics
        overall_daily = float(df["forecast"].mean())
        
        # Breakdown by category
        if 'cat_id' in df.columns:
            category_breakdown = df.groupby('cat_id')['forecast'].agg([
                ('avg_per_item', 'mean'),
                ('total_daily', 'sum'),
                ('item_count', 'count')
            ]).round(2).to_dict('index')
        else:
            category_breakdown = {}
        
        result = {
            "daily": round(overall_daily, 2),
            "weekly": round(overall_daily * 7, 2),
            "monthly": round(overall_daily * 30, 2),
            "by_category": category_breakdown,
            "total_items": len(df),
            "high_demand_items": int(len(df[df['forecast'] > 1]))
        }
        
        print(f"DEBUG: Forecast results: {result}")
        return result, None
    
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in run_forecast:")
        print(traceback.format_exc())
        return None, f"Forecast error: {str(e)}"
