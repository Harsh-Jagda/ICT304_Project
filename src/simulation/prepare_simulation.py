import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from tqdm import tqdm

# FIX #1 + #2: import canonical ROP formula and shared financial constants
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.constants import DEFAULT_UNIT_PRICE_USD, HOLDING_COST_RATE, DEFAULT_LEAD_TIME_DAYS
from src.business.recommendation import calculate_rop

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
LT_PATH = os.path.join(DATA_DIR, "config", "lead_times.json")
OUT_PATH = os.path.join(DATA_DIR, "data", "outputs", "sim_results.parquet")

CAT_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
            'wday', 'month', 'event_name_1', 'event_type_1']

def predict_batch(model, rows: pd.DataFrame) -> np.ndarray:
    features = model.feature_name()
    X = rows[features].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes
    return np.clip(model.predict(X.values), 0, None)

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    with open(LT_PATH) as f: lead_times = json.load(f)

    # Focus on CA_1 for simulation
    df = df[df['store_id'] == 'CA_1'].copy()
    df['date'] = pd.to_datetime(df['date'])

    # Get last 365 days
    max_date = df['date'].max()
    start_date = max_date - pd.Timedelta(days=365)
    sim_df = df[df['date'] >= start_date].copy().sort_values('date')

    print(f"Pre-computing predictions for {len(sim_df)} rows...")
    sim_df['ai_demand'] = predict_batch(model, sim_df)
    
    items = sim_df['item_id'].unique()
    metrics = []
    item_states_ai = {}
    item_states_base = {}

    print("Initializing states...")
    for item in items:
        # Give them starting stock so they don't immediately stock out
        init_stock = int(np.random.randint(15, 30))
        item_states_ai[item] = {'stock': init_stock, 'on_order': 0, 'deliveries': []}
        item_states_base[item] = {'stock': init_stock, 'on_order': 0, 'deliveries': []}

    # FIX #2: use constants instead of magic numbers
    sell_price = DEFAULT_UNIT_PRICE_USD
    holding_cost_rate = HOLDING_COST_RATE
    
    days = sorted(sim_df['date'].unique())
    
    print("Simulating 365 days...")
    for day_idx, current_date in enumerate(tqdm(days)):
        day_df = sim_df[sim_df['date'] == current_date]
        
        day_stock_ai = 0.0
        day_stock_base = 0.0
        day_stockout_ai = 0
        day_stockout_base = 0
        day_lost_ai = 0.0
        day_lost_base = 0.0
        
        for _, row in day_df.iterrows():
            item = row['item_id']
            actual_sales = row['sales']
            ai_pred = row['ai_demand']
            cat_key = item.split('_')[0] if isinstance(item, str) else ''
            item_lt_cfg = lead_times.get(cat_key, {})
            lt = item_lt_cfg.get('lead_time_days', DEFAULT_LEAD_TIME_DAYS)
            z_score = item_lt_cfg.get('service_level_z', 1.645)
            
            s_ai = item_states_ai[item]
            s_base = item_states_base[item]
            
            # Deliveries
            arr_ai = [d for d in s_ai['deliveries'] if d['day'] <= day_idx]
            for d in arr_ai:
                s_ai['stock'] += d['qty']
                s_ai['on_order'] -= d['qty']
            s_ai['deliveries'] = [d for d in s_ai['deliveries'] if d['day'] > day_idx]

            arr_base = [d for d in s_base['deliveries'] if d['day'] <= day_idx]
            for d in arr_base:
                s_base['stock'] += d['qty']
                s_base['on_order'] -= d['qty']
            s_base['deliveries'] = [d for d in s_base['deliveries'] if d['day'] > day_idx]
            
            # Sales (True Demand approx)
            demand = actual_sales
            if demand == 0 and s_base['stock'] == 0:
                demand = ai_pred
            
            if s_ai['stock'] >= demand:
                s_ai['stock'] -= demand
            else:
                missed = demand - s_ai['stock']
                day_lost_ai += (missed * sell_price)
                if missed > 0: day_stockout_ai += 1
                s_ai['stock'] = 0

            if s_base['stock'] >= demand:
                s_base['stock'] -= demand
            else:
                missed = demand - s_base['stock']
                day_lost_base += (missed * sell_price)
                if missed > 0: day_stockout_base += 1
                s_base['stock'] = 0

            day_stock_ai += s_ai['stock']
            day_stock_base += s_base['stock']
            
            # Reorder Policy
            std_7 = row['rolling_std_7'] if not pd.isnull(row['rolling_std_7']) else 0.5

            # FIX #1: use canonical calculate_rop() — no more inline formula
            # AI policy: uses LightGBM prediction + dynamic service level
            _ss_ai, rop_ai = calculate_rop(ai_pred, lt, std_7, z_score=z_score)
            if s_ai['stock'] + s_ai['on_order'] < rop_ai:
                qty = max(0, int(rop_ai - s_ai['stock'] - s_ai['on_order'] + (ai_pred * lt)))
                if qty > 0:
                    s_ai['on_order'] += qty
                    s_ai['deliveries'].append({'day': day_idx + lt, 'qty': qty})

            # Baseline policy: uses 7-day moving average + z=1.0 (68% service level)
            ma_7 = row['rolling_mean_7'] if not pd.isnull(row['rolling_mean_7']) else 1.0
            _ss_base, rop_base = calculate_rop(ma_7, lt, std_7, z_score=1.0)
            if s_base['stock'] + s_base['on_order'] < rop_base:
                qty = max(0, int(rop_base - s_base['stock'] - s_base['on_order'] + (ma_7 * lt)))
                if qty > 0:
                    s_base['on_order'] += qty
                    s_base['deliveries'].append({'day': day_idx + lt, 'qty': qty})
                
        metrics.append({
            'date': current_date,
            'ai_stockout_items': day_stockout_ai,
            'base_stockout_items': day_stockout_base,
            'ai_lost_revenue': day_lost_ai,
            'base_lost_revenue': day_lost_base,
            'ai_holding_cost': day_stock_ai * holding_cost_rate,
            'base_holding_cost': day_stock_base * holding_cost_rate
        })

    out_df = pd.DataFrame(metrics)
    out_df.to_parquet(OUT_PATH)
    print(f"Simulation saved to {OUT_PATH}")

if __name__ == '__main__':
    main()
