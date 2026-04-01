from .data_prep_service import process_uploaded_csv
import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wms_lgbm_model.pkl")

def run_forecast(csv_path):
    try:
        df, err = process_uploaded_csv(csv_path)
        if err: return None, err

        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42).copy()

        model = joblib.load(MODEL_PATH)
        feature_names = model.feature_name()

        # Dynamic Feature Alignment
        for feat in feature_names:
            if feat not in df.columns:
                if any(x in feat for x in ['id', 'dept', 'cat', 'store', 'event']):
                    df[feat] = pd.Categorical(['Unknown'] * len(df))
                else:
                    df[feat] = 0

        X = df[feature_names].copy()
        for col in X.select_dtypes(include=['category', 'object']).columns:
            X[col] = pd.Categorical(X[col]).codes

        preds = model.predict(X.values)
        df["forecast"] = preds

        # Use the sum for labor math and mean for global stats
        total_forecast_sum = float(df["forecast"].sum())
        overall_daily_mean = float(df["forecast"].mean())

        forecast_volatility = df["forecast"].std()
        confidence = "High" if forecast_volatility < (overall_daily_mean * 0.3) else "Moderate"

        category_breakdown = {}
        if 'cat_id' in df.columns:
            category_breakdown = df.groupby('cat_id')['forecast'].agg([
                ('avg_per_item', 'mean'),
                ('total_daily', 'sum'),
                ('item_count', 'count')
            ]).round(2).to_dict('index')

        # --- FOOLPROOF LABOR INSIGHT ---
        if category_breakdown and total_forecast_sum > 0:
            top_cat = max(category_breakdown, key=lambda k: category_breakdown[k]['total_daily'])
            # Calculate % based on TOTAL SUM of all categories
            top_pct = round((category_breakdown[top_cat]['total_daily'] / total_forecast_sum) * 100)
            labor_action = f"Allocate {top_pct}% of picking staff to the {top_cat} zone today."
        else:
            labor_action = "Maintain balanced staff distribution across all zones."

        return {
            "daily": round(overall_daily_mean, 2),
            "weekly": round(overall_daily_mean * 7, 2),
            "monthly": round(overall_daily_mean * 30, 2),
            "confidence_rating": confidence,
            "by_category": category_breakdown,
            "total_items": len(df),
            "high_demand_count": int(len(df[df['forecast'] > overall_daily_mean])),
            "labor_action": labor_action
        }, None

    except Exception as e:
        return None, f"Forecast error: {str(e)}"
