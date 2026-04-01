import pandas as pd
import numpy as np
import math
import json
import os

def compute_risk(csv_path, category="FOODS", json_filename="lead_times.json"):
    try:
        upload_path = os.path.join(os.path.dirname(__file__), "..", "uploads", json_filename)
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", json_filename)
        lead_path = upload_path if os.path.exists(upload_path) else model_path

        if not os.path.exists(lead_path):
            return None, f"Config {json_filename} not found."

        with open(lead_path, "r") as f:
            lead_times = json.load(f)

        matched_key = next((k for k in lead_times if k in category), None)
        lead = lead_times[matched_key] if matched_key else {"lead_time_days": 3, "service_level_z": 1.645}

        df = pd.read_parquet(csv_path) if csv_path.endswith(".parquet") else pd.read_csv(csv_path)

        category_data = df[df["cat_id"] == category] if "cat_id" in df.columns else df
        if category_data.empty: category_data = df

        # --- FOOLPROOF STATS ---
        avg = float(category_data["sales"].mean()) if "sales" in category_data.columns else 0.0
        # Prevent math error if avg is 0
        safe_avg = max(avg, 0.001) 
        
        std = float(category_data["rolling_std_7"].mean()) if "rolling_std_7" in category_data.columns else 0.5
        if np.isnan(std) or std <= 0: std = 0.1

        lead_time = lead.get("lead_time_days", 3)
        z = lead.get("service_level_z", 1.645)

        # 1. ROP Math: Demand during lead time + Safety Buffer
        safety = z * std * math.sqrt(lead_time)
        rop = (avg * lead_time) + safety

        # 2. Volatility (Coefficient of Variation)
        volatility_index = round(std / safe_avg, 2)
        risk_level = "Low"
        if volatility_index > 0.6: risk_level = "Medium"
        if volatility_index > 1.2: risk_level = "High"

        # 3. Days of Cover (Current ROP / Demand)
        days_of_cover = round(rop / safe_avg, 1)

        # --- ACTION LOGIC ---
        if avg == 0:
            action = " NO DEMAND: Review SKU for decommissioning."
        elif days_of_cover < lead_time:
            action = " CRITICAL: Lead time exceeds stock. Expedite order."
        elif days_of_cover < (lead_time + 5):
            action = "REORDER: Stock levels approaching threshold."
        elif days_of_cover > 60:
            action = " OVERSTOCK: Significant surplus. Pause purchasing."
        else:
            action = " OPTIMAL: Levels aligned with demand."

        return {
            "category": category,
            "rop": round(rop, 2),
            "safety_stock": round(safety, 2),
            "avg_daily_demand": round(avg, 2),
            "demand_volatility": volatility_index,
            "risk_status": risk_level,
            "estimated_days_of_cover": days_of_cover,
            "lead_time": lead_time,
            "recommended_action": action
        }, None

    except Exception as e:
        return None, f"Risk analysis error: {str(e)}"
