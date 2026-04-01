import pandas as pd
import numpy as np
import math
import json
import os

def compute_risk(csv_path, category="FOODS", json_filename="lead_times.json"):
    try:
        print(f"DEBUG RISK: Category: {category} | Config: {json_filename}")

        # ---------------- FIND THE CORRECT JSON ----------------
        # Define the two possible locations
        # 1. User Uploads (check this first for custom configs)
        upload_path = os.path.join(os.path.dirname(__file__), "..", "uploads", json_filename)
        # 2. Permanent Model folder (the 'Special Exception' fallback)
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", json_filename)

        if os.path.exists(upload_path):
            lead_path = upload_path
            print(f"Loading user-uploaded config from: {upload_path}")
        elif os.path.exists(model_path):
            lead_path = model_path
            print(f"Loading demo config from: {model_path}")
        else:
            return None, f"Config {json_filename} not found in model or uploads."

        with open(lead_path, "r") as f:
            lead_times = json.load(f)

        # Match category from JSON
        matched_key = None
        for key in lead_times:
            if key in category:
                matched_key = key
                break

        if matched_key:
            lead = lead_times[matched_key]
        else:
            # Global Fallback
            lead = {"lead_time_days": 3, "service_level_z": 1.645}

        # ---------------- LOAD DATA ----------------
        try:
            if csv_path.endswith(".parquet"):
                df = pd.read_parquet(csv_path, columns=["sales", "cat_id", "rolling_std_7"])
            else:
                df = pd.read_csv(csv_path)
        except Exception as e:
            return None, f"Read Error: {str(e)}"

        # ---------------- CALCULATIONS ----------------
        # Filter to category
        if "cat_id" in df.columns:
            category_data = df[df["cat_id"] == category]
        else:
            category_data = df

        if len(category_data) == 0:
            category_data = df # Fallback to full dataset if category empty

        avg = float(category_data["sales"].mean()) if "sales" in category_data.columns else 0.0
        
        if "rolling_std_7" in category_data.columns:
            std = float(category_data["rolling_std_7"].mean())
        else:
            std = 0.5
            
        if np.isnan(std) or std <= 0: std = 0.5

        lead_time = lead.get("lead_time_days", 3)
        z = lead.get("service_level_z", 1.645)

        # Safety Stock = Z * StdDev * sqrt(Lead Time)
        safety = z * std * math.sqrt(lead_time)
        # ROP = (Avg Demand * Lead Time) + Safety Stock
        rop = (avg * lead_time) + safety

        return {
            "category": category,
            "rop": round(rop, 2),
            "safety": round(safety, 2),
            "lead_time": lead_time,
            "service_level_z": z,
            "avg_daily_demand": round(avg, 2),
            "explanation": f"Source: {json_filename}"
        }, None

    except Exception as e:
        return None, f"Risk analysis error: {str(e)}"
