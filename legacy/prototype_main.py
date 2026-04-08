import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import timedelta

# --- CONFIG ---
# Automatically use the directory where this script is located — works on any machine
DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
LEAD_TIME_PATH = os.path.join(DATA_DIR, "config", "lead_times.json")
DATA_PATH = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")

STORE_ID = 'CA_1'

def load_resources():
    print("Loading AI model and resources...")
    model = joblib.load(MODEL_PATH)
    with open(LEAD_TIME_PATH, 'r') as f:
        lead_times = json.load(f)
    
    print("Loading dataset (this may take a moment for the large file)...")
    df = pd.read_parquet(DATA_PATH)
    
    # CRITICAL: Ensure date column is parsed as datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Encode categoricals for model compatibility
    CAT_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'event_name_1', 'event_type_1']
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Pre-calculate department averages for Cold Start fallback
    avg_stats = df.groupby('dept_id', observed=True).agg(
        avg_demand=('sales', 'mean'),
        avg_std=('rolling_std_7', 'mean')
    ).to_dict('index')
    
    return model, lead_times, df, avg_stats

def get_prediction_for_row(model, df_ref, row):
    """Get model prediction for a single row."""
    feature_names = model.feature_name()
    X = {}
    for feat in feature_names:
        val = row[feat]
        # Convert categoricals to integer codes for model
        if hasattr(val, 'categories') or (feat in ['item_id', 'dept_id', 'cat_id', 'store_id', 'event_name_1', 'event_type_1']):
            try:
                X[feat] = df_ref[feat].cat.codes[row.name] if hasattr(df_ref[feat], 'cat') else 0
            except:
                X[feat] = 0
        else:
            X[feat] = val if not pd.isnull(val) else 0
    
    X_df = pd.DataFrame([X])
    pred = np.clip(model.predict(X_df.values)[0], 0, None)
    return pred

def prepare_snapshot(df, model, lead_times, store_id=STORE_ID):
    """Select 20 diverse items (10 FOODS, 5 HOBBIES, 5 HOUSEHOLD) from one store."""
    store_df = df[df['store_id'] == store_id].copy()
    
    # Pick snapshot date: 31 days before the end of data
    max_data_date = store_df['date'].max()
    snapshot_date = max_data_date - timedelta(days=31)
    
    print(f"Snapshot date: {snapshot_date.date()} | Simulation window: next 30 days")
    
    current_data = store_df[store_df['date'] == snapshot_date]
    
    def safe_sample(data, category, n, id_range=None):
        cat_data = data[data['cat_id'] == category]
        if id_range:
            # Filter to only include items with local_id in the specified range
            lo, hi = id_range
            cat_data = cat_data[
                cat_data['item_id'].astype(str).apply(
                    lambda x: lo <= int(x.split('_')[2]) <= hi if len(x.split('_')) >= 3 and x.split('_')[2].isdigit() else False
                )
            ]
        if len(cat_data) == 0:
            # Fallback: any date with this category (with same id_range if needed)
            cat_data = store_df[store_df['cat_id'] == category].drop_duplicates('item_id')
        return cat_data.sample(n=min(n, len(cat_data)), random_state=42)
    
    # FOODS: only IDs 001-100 per requirements
    foods = safe_sample(current_data, 'FOODS', 10, id_range=(1, 100))
    hobbies = safe_sample(current_data, 'HOBBIES', 5,  id_range=(1, 100))
    household = safe_sample(current_data, 'HOUSEHOLD', 5, id_range=(1, 100))
    
    snapshot = pd.concat([foods, hobbies, household]).reset_index(drop=False)
    
    items_state = []
    for display_idx, (_, row) in enumerate(snapshot.iterrows(), start=1):
        parts = str(row['item_id']).split('_')
        category = f"{parts[0]}_{parts[1]}" if len(parts) >= 3 else str(row['item_id'])
        local_id = parts[2] if len(parts) >= 3 else "???"
        
        pred_demand = get_prediction_for_row(model, df, row)
        dept = str(row['dept_id'])
        lt = lead_times.get(dept, {}).get('lead_time_days', 2)
        z = lead_times.get(dept, {}).get('service_level_z', 1.645)
        std_val = row['rolling_std_7'] if not pd.isnull(row['rolling_std_7']) else 0.5
        buffer = z * std_val * np.sqrt(lt)
        rop = (pred_demand * lt) + buffer
        
        current_stock = np.random.randint(2, 20)
        stockout_days = current_stock / (pred_demand + 1e-6)
        restock_date = snapshot_date + timedelta(days=int(stockout_days))
        
        items_state.append({
            'index': display_idx,
            'full_id': str(row['item_id']),
            'category': category,
            'local_id': local_id,
            'dept_id': dept,
            'store_id': store_id,
            'stock': current_stock,
            'demand': pred_demand,
            'lead_time': lt,
            'buffer': buffer,
            'rop': rop,
            'restock_date': restock_date,
            'std_7': std_val,
            'snapshot_date': snapshot_date
        })
    
    return items_state

def print_table(items):
    if not items:
        return
    snapshot_date = items[0]['snapshot_date']
    print("\n" + "="*110)
    print(f" AI WAREHOUSE MANAGEMENT SYSTEM — Store: {STORE_ID}  |  Simulated Date: {snapshot_date.date()} ".center(110, "#"))
    print("="*110)
    print(f"{'#':<3} | {'Category':<10} | {'Sub ID':<7} | {'Stock':<6} | {'Demand/day':<12} | {'Est. Restock':<12} | Status")
    print("-" * 110)
    for item in items:
        status = "⚠ CRITICAL" if item['stock'] < item['rop'] else "✓ OK"
        print(f"{item['index']:<3} | {item['category']:<10} | {item['local_id']:<7} | "
              f"{item['stock']:<6} | {item['demand']:<12.2f} | {item['restock_date'].date()} | {status}")
    print("-" * 110)

def global_simulation(items, df, days):
    """Simulate and compare predicted vs actual sales for all 20 items."""
    print(f"\n" + "-"*90)
    print(f" GLOBAL SIMULATION — {days}-DAY WINDOW ".center(90, "-"))
    print("-" * 90)
    print(f"{'#':<3} | {'Item':<16} | {'Predicted':<12} | {'Actual':<12} | {'Error %':<10}")
    print("-" * 90)
    
    total_pred, total_actual = 0, 0
    
    for item in items:
        pred_sales = item['demand'] * days
        
        start = item['snapshot_date'] + timedelta(days=1)
        end = item['snapshot_date'] + timedelta(days=days)
        
        mask = (
            (df['item_id'] == item['full_id']) &
            (df['store_id'] == item['store_id']) &
            (df['date'] >= start) &
            (df['date'] <= end)
        )
        actual_sales = df.loc[mask, 'sales'].sum()
        
        if actual_sales == 0 and pred_sales == 0:
            error = 0.0
        else:
            error = abs(pred_sales - actual_sales) / (actual_sales + 1.0) * 100
        
        flag = " ⚠" if error > 100 else ""
        print(f"{item['index']:<3} | {item['category']}_{item['local_id']:<12} | "
              f"{pred_sales:<12.1f} | {actual_sales:<12.1f} | {error:.1f}%{flag}")
        
        total_pred += pred_sales
        total_actual += actual_sales
    
    overall_error = abs(total_pred - total_actual) / (total_actual + 1.0) * 100
    print("-" * 90)
    print(f"  TOTAL: Predicted {total_pred:.1f} | Actual {total_actual:.1f} | Agg. Error {overall_error:.1f}%")
    print("  Note: Individual errors may be high for zero-sales days (sparse retail data).")
    input("\nPress Enter to return to menu...")

def adjust_any_item(items, lead_times):
    """Adjust a parameter for a specific item by index."""
    try:
        idx = int(input("\nEnter item index to adjust (1-20): ")) - 1
        if 0 <= idx < len(items):
            item = items[idx]
            print(f"\n  Adjusting: {item['category']}_{item['local_id']}")
            print("  1. Change Stock")
            print("  2. Change Lead Time")
            print("  3. Change Buffer")
            attr = input("  Select parameter [1/2/3]: ").strip()
            
            if attr == '1':
                item['stock'] = int(input(f"  New Stock (current {item['stock']}): "))
            elif attr == '2':
                item['lead_time'] = int(input(f"  New Lead Time in days (current {item['lead_time']}): "))
                # Recalculate buffer & ROP automatically
                z = lead_times.get(item['dept_id'], {}).get('service_level_z', 1.645)
                item['buffer'] = z * item['std_7'] * np.sqrt(item['lead_time'])
            elif attr == '3':
                item['buffer'] = float(input(f"  New Buffer (current {item['buffer']:.1f}): "))
            
            item['rop'] = (item['demand'] * item['lead_time']) + item['buffer']
            stockout_days = item['stock'] / (item['demand'] + 1e-6)
            item['restock_date'] = item['snapshot_date'] + timedelta(days=int(stockout_days))
            print("\n  ✓ Parameters updated and recommendations recalculated.")
        else:
            print("Invalid index.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def handle_new_item(items, avg_stats, lead_times):
    """Cold Start: register a new item using department-level averages."""
    print("\n--- REGISTER NEW ITEM (Cold Start) ---")
    known_depts = [k for k in avg_stats.keys() if avg_stats[k]['avg_demand'] > 0]
    print(f"  Available Departments: {sorted(known_depts)}")
    dept = input("  Enter Department (e.g. FOODS_1): ").strip()
    if dept not in avg_stats:
        print(f"  Error: Department '{dept}' not found. New categories are not allowed.")
        return
    
    # ID validation: must be 100-999 and not already in the list
    existing_ids = {item['local_id'] for item in items}
    while True:
        item_num_str = input("  Enter new item ID (100-999, must not already exist): ").strip()
        if not item_num_str.isdigit():
            print("  Error: ID must be a number.")
            continue
        item_num = int(item_num_str)
        if not (100 <= item_num <= 999):
            print("  Error: ID must be between 100 and 999.")
            continue
        if str(item_num) in existing_ids:
            print(f"  Error: ID '{item_num}' already exists in the current list.")
            continue
        break
    
    # Ask for initial stock
    while True:
        stock_str = input("  Enter initial stock amount: ").strip()
        if stock_str.isdigit() and int(stock_str) >= 0:
            initial_stock = int(stock_str)
            break
        print("  Error: Stock must be a non-negative integer.")
    
    pred_demand = avg_stats[dept]['avg_demand']
    std_7 = avg_stats[dept]['avg_std']
    lt = lead_times.get(dept, {}).get('lead_time_days', 2)
    z = lead_times.get(dept, {}).get('service_level_z', 1.645)
    buffer = z * std_7 * np.sqrt(lt)
    rop = (pred_demand * lt) + buffer
    snapshot_date = items[0]['snapshot_date'] if items else pd.Timestamp.now()
    stockout_days = initial_stock / (pred_demand + 1e-6)
    restock_date = snapshot_date + timedelta(days=int(stockout_days))
    
    parts = dept.split('_')
    category = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else dept
    
    items.append({
        'index': len(items) + 1,
        'full_id': f"{dept}_{item_num}",
        'category': category,
        'local_id': str(item_num),
        'dept_id': dept,
        'store_id': STORE_ID,
        'stock': initial_stock,
        'demand': pred_demand,
        'lead_time': lt,
        'buffer': buffer,
        'rop': rop,
        'restock_date': restock_date,
        'std_7': std_7,
        'snapshot_date': snapshot_date
    })
    print(f"  ✓ Item {dept}_{item_num} added (stock: {initial_stock}, est. demand: {pred_demand:.2f}/day).")

def main_loop(model, lead_times, df, avg_stats):
    items = prepare_snapshot(df, model, lead_times)
    
    while True:
        print_table(items)
        print("\n[MAIN MENU]")
        print("  1. Simulate (Global — Day / Week / Month)")
        print("  2. Adjust Item (by Index)")
        print("  3. Exit")
        
        choice = input("\nSelect action [1-3]: ").strip()
        
        if choice == '3':
            print("System shutdown. Goodbye!")
            break
        elif choice == '1':
            print("\n  Simulation horizon:")
            print("  1. Day (1 day)")
            print("  2. Week (7 days)")
            print("  3. Month (30 days)")
            h = input("  Choose [1/2/3]: ").strip()
            days_map = {'1': 1, '2': 7, '3': 30}
            days = days_map.get(h, 7)
            global_simulation(items, df, days)
        elif choice == '2':
            adjust_any_item(items, lead_times)
        else:
            print("  Please enter 1, 2 or 3.")

if __name__ == "__main__":
    model, lead_times, df, avg_stats = load_resources()
    main_loop(model, lead_times, df, avg_stats)
