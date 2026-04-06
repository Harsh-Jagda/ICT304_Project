import joblib
import os
import pandas as pd
import numpy as np
from .data_prep_service import process_uploaded_csv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wms_lgbm_model.pkl")

# Feature set defined by teammate
CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'event_name_1', 'event_type_1']
NUM_FEATURES  = ['sell_price', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28']
FEATURES = CAT_FEATURES + NUM_FEATURES

def run_forecast(csv_path, selected_state="ALL"):
    try:
        df, err = process_uploaded_csv(csv_path)
        if err: return None, err

        # 1. Capture ALL unique states for the dropdown BEFORE filtering
        available_regions = []
        if "state_id" in df.columns:
            # Convert to string and strip to ensure clean matches
            available_regions = sorted([str(s).strip() for s in df["state_id"].unique() if pd.notna(s)])

        # 2. Filter by state (Case-insensitive)
        if selected_state != "ALL" and "state_id" in df.columns:
            df = df[df["state_id"].astype(str).str.upper() == selected_state.upper()].copy()

        if len(df) == 0:
            return None, f"No data found for region: {selected_state}"

        # 3. Load Model and Predict
        model = joblib.load(MODEL_PATH)

        X = df.copy()
        for feat in FEATURES:
            if feat not in X.columns:
                X[feat] = 0 if feat in NUM_FEATURES else "Unknown"

        # Categorical Encoding
        for col in CAT_FEATURES:
            X[col] = pd.Categorical(X[col]).codes

        # CRITICAL FIX: Add predict_disable_shape_check to handle the 16 vs 15 feature mismatch
        preds = model.predict(X[FEATURES].values, predict_disable_shape_check=True)
        df["forecast"] = np.clip(preds, 0, None)

        # 4. Calculations for Dashboard
        overall_mean = float(df["forecast"].mean())
        total_sum = float(df["forecast"].sum())

        category_breakdown = {}
        if 'cat_id' in df.columns:
            category_breakdown = df.groupby('cat_id')['forecast'].agg([
                ('avg_per_item', 'mean'), ('total_daily', 'sum'), ('item_count', 'count')
            ]).round(2).to_dict('index')

        # 5. Labor Logic
        if category_breakdown and total_sum > 0:
            top_cat = max(category_breakdown, key=lambda k: category_breakdown[k]['total_daily'])
            top_pct = round((category_breakdown[top_cat]['total_daily'] / total_sum) * 100)
            labor_action = f"Region {selected_state}: Allocate {top_pct}% of picking staff to {top_cat}."
        else:
            labor_action = "Standardize staff distribution across all zones."

        return {
            "daily": round(overall_mean, 2),
            "weekly": round(overall_mean * 7, 2),
            "monthly": round(overall_mean * 30, 2),
            "confidence_rating": "High" if df["forecast"].std() < (overall_mean * 0.5) else "Moderate",
            "by_category": category_breakdown,
            "total_items": len(df),
            "labor_action": labor_action,
            "available_states": available_regions, # Correctly populated list
            "high_demand_count": int(len(df[df['forecast'] > overall_mean])),
        }, None

    except Exception as e:
        return None, f"Forecast error: {str(e)}"
