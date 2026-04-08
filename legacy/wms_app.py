"""
wms_app.py — AI Warehouse Management System (Full Version)

Full production CLI with:
  - Multi-store selection (CA, TX, WI once retrained)
  - Simulate / Adjust / System Status
  - Fast vectorized predictions and simulation for ALL items
  - Purchase Orders & Pagination

Usage: python wms_app.py
"""
import os
import json
import csv
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
_ALL_DATA   = os.path.join(DATA_DIR, "data", "processed", "processed_data_all.parquet")
_CA_DATA    = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
DATA_PATH   = _ALL_DATA if os.path.exists(_ALL_DATA) else _CA_DATA
MODEL_PATH  = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
LT_PATH     = os.path.join(DATA_DIR, "config", "lead_times.json")
RT_PATH     = os.path.join(DATA_DIR, "real_time_sales.csv")
REGISTRY    = os.path.join(DATA_DIR, "models", "model_registry.json")

CAT_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
            'wday', 'month', 'event_name_1', 'event_type_1']

# ─────────────────────────────────────────────
# BANNER & MODEL INFO
# ─────────────────────────────────────────────

def get_model_info() -> str:
    if os.path.exists(REGISTRY):
        try:
            with open(REGISTRY) as f: reg = json.load(f)
            prod = next((r for r in reversed(reg) if r.get("is_production")), None)
            if prod: return f"Model {prod['version']} | MAE {prod['mae']:.4f} | data: {prod['training_data']}"
        except: pass
    return "Model: wms_lgbm_model (baseline)"

def print_banner(store_id: str):
    w = 110
    print("\n" + "=" * w)
    print(f"  AI WAREHOUSE DSS & SIMULATOR  |  Store: {store_id}".center(w))
    print(f"  {get_model_info()}".center(w))
    print("=" * w)

# ─────────────────────────────────────────────
# LOAD & SNAPSHOT
# ─────────────────────────────────────────────

def load_resources():
    print("Loading AI model and 46M row database...")
    model = joblib.load(MODEL_PATH)
    with open(LT_PATH) as f: lead_times = json.load(f)
    print(f"  Data: {'all states' if os.path.exists(_ALL_DATA) else 'California only'}")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    for col in CAT_COLS:
        if col in df.columns: df[col] = df[col].astype('category')
    return model, lead_times, df

def select_store(df: pd.DataFrame) -> str:
    stores = sorted(df['store_id'].astype(str).unique())
    print("\nAvailable stores:")
    for i, s in enumerate(stores, 1):
        print(f"  {i}. {s}")
    while True:
        raw = input(f"\nSelect store [1-{len(stores)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(stores):
            return stores[int(raw) - 1]

def predict_batch(model, rows: pd.DataFrame) -> np.ndarray:
    features = model.feature_name()
    X = rows[features].copy()
    for col in CAT_COLS:
        if col in X.columns: X[col] = X[col].astype('category').cat.codes
    return np.clip(model.predict(X.values), 0, None)

def build_snapshot(df: pd.DataFrame, model, lead_times: dict, store_id: str) -> dict:
    store_df = df[df['store_id'] == store_id]
    snap_date = store_df['date'].max() - timedelta(days=31)
    
    current = store_df[store_df['date'] == snap_date].drop_duplicates('item_id').copy()
    print(f"  Snapshot: {snap_date.date()} | Inventory: {len(current):,} items")

    preds = predict_batch(model, current)
    items = {}
    for i, (_, row) in enumerate(current.iterrows()):
        demand = float(preds[i])
        dept = str(row['dept_id'])
        lid = str(row['item_id']).split('_')[2] if len(str(row['item_id']).split('_'))>2 else "??"
        
        lt_cfg = lead_times.get(dept, lead_times.get(str(row['cat_id']).split('_')[0], {"lead_time_days": 2, "service_level_z": 1.645}))
        lt, z = lt_cfg['lead_time_days'], lt_cfg['service_level_z']
        std_v = float(row['rolling_std_7']) if not pd.isnull(row['rolling_std_7']) else 0.5
        
        buf = z * std_v * np.sqrt(lt)
        rop = demand * lt + buf
        stock = int(np.random.randint(2, 20))
        
        items[str(row['item_id'])] = {
            'full_id': str(row['item_id']), 'category': str(row['cat_id']), 'local_id': lid,
            'dept_id': dept, 'store_id': store_id, 'stock': stock, 'demand': demand,
            'on_order': 0, 'lead_time': lt, 'buffer': buf, 'rop': rop, 
            'std_7': std_v, 'snap_date': snap_date, 'z': z
        }
    return items

# ─────────────────────────────────────────────
# PAGINATION
# ─────────────────────────────────────────────

def display_inventory_page(items: dict, page: int = 1, chunk: int = 15) -> list:
    sorted_items = sorted(items.values(), key=lambda x: (x['stock']+x['on_order']) - x['rop'])
    total_pages = max(1, (len(sorted_items) + chunk - 1) // chunk)
    page = max(1, min(page, total_pages))
    
    start = (page - 1) * chunk
    view = sorted_items[start:start+chunk]
    
    w = 115
    print("\n" + "=" * w)
    print(f" INVENTORY (Sorted by Criticality) | Page {page}/{total_pages} ".center(w, "#"))
    print("=" * w)
    print(f"  {'#':<3} {'Item ID':<20} {'Stock':>5} {'Ordered':>7} {'Dmd/d':>7} {'ROP':>6}  Status")
    print("-" * w)
    
    # Return the mapped list so users can select by standard index 1-15
    idx_map = []
    for i, it in enumerate(view, 1):
        status = "🔴 SHORTAGE" if (it['stock'] + it['on_order']) < it['rop'] else "🟢 OK"
        print(f"  {i:<3} {it['full_id']:<20} {it['stock']:>5} {it['on_order']:>7} "
              f"{it['demand']:>7.2f} {it['rop']:>6.1f}  {status}")
        idx_map.append(it['full_id'])
    print("-" * w)
    return idx_map, page, total_pages

# ─────────────────────────────────────────────
# SIMULATE (Optimized O(1) DF filtering)
# ─────────────────────────────────────────────

def simulate(items: dict, df: pd.DataFrame, store_id: str):
    print("\n  Simulation horizon:")
    print("  1. Day   (1 day)")
    print("  2. Week  (7 days)")
    print("  3. Month (30 days)")
    h = input("  Choose [1/2/3]: ").strip()
    days = {'1': 1, '2': 7, '3': 30}.get(h, 7)

    w = 95
    print(f"\n{'-'*w}")
    print(f" SIMULATION: {days}-DAY WINDOW (Processing {len(items):,} items) ".center(w, "-"))
    print(f"{'-'*w}")
    
    snap_date = list(items.values())[0]['snap_date']
    start = snap_date + timedelta(days=1)
    end   = snap_date + timedelta(days=days)

    mask = (df['store_id'].astype(str) == store_id) & (df['date'] >= start) & (df['date'] <= end)
    df_window = df[mask]
    
    actual_sales = df_window.groupby('item_id', observed=True)['sales'].sum().to_dict()

    total_pred = 0.0
    total_actual = 0.0

    for item_id, it in items.items():
        pred = it['demand'] * days
        actual = float(actual_sales.get(item_id, 0.0))
        total_pred += pred
        total_actual += actual

    overall = 0.0 if total_actual == 0 else abs(total_pred - total_actual) / (total_actual + 1.0) * 100
    print(f"\n  Simulation Complete!")
    print(f"  TOTAL predicted: {total_pred:,.1f} units")
    print(f"  TOTAL actual:    {total_actual:,.1f} units")
    print(f"  Agg. Error:      {overall:.1f}%")
    input("\n  Press Enter to return to menu...")

# ─────────────────────────────────────────────
# MANAGERIAL ACTIONS
# ─────────────────────────────────────────────

def adjust(items: dict, idx_map: list):
    sel = input("\n  Enter Item # from current page (or type full ID): ").strip()
    item_id = None
    if sel.isdigit() and 1 <= int(sel) <= len(idx_map):
        item_id = idx_map[int(sel)-1]
    elif sel in items:
        item_id = sel
    else:
        print("  Item not found."); return

    it = items[item_id]
    print(f"\n  [Adjusting {item_id}] 1.Stock 2.LeadTime 3.Buffer")
    c = input("  Choice: ").strip()
    try:
        if c == '1': it['stock'] = int(input("  New Stock: "))
        elif c == '2': it['lead_time'] = int(input("  New Lead Time: ")); it['buffer'] = it['z'] * it['std_7'] * np.sqrt(it['lead_time'])
        elif c == '3': it['buffer'] = float(input("  New Buffer: "))
    except ValueError:
        print("  Invalid."); return
    it['rop'] = it['demand'] * it['lead_time'] + it['buffer']
    print("  Updated.")

def log_sale(items: dict, store_id: str, idx_map: list):
    sel = input("\n  Enter Item # to log sale for (or full ID): ").strip()
    item_id = None
    if sel.isdigit() and 1 <= int(sel) <= len(idx_map): item_id = idx_map[int(sel)-1]
    elif sel in items: item_id = sel
    else: print("  Not found."); return

    try: qty = int(input(f"  Units sold today for {item_id}: "))
    except: return

    it = items[item_id]
    row = {'date': datetime.now().strftime('%Y-%m-%d'), 'item_id': item_id, 'store_id': store_id, 
           'state_id': store_id.split('_')[0], 'dept_id': it['dept_id'], 'cat_id': it['category'], 'sales': qty}
    
    import csv
    wh = not os.path.exists(RT_PATH)
    with open(RT_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if wh: w.writeheader()
        w.writerow(row)

    it['stock'] = max(0, it['stock'] - qty)
    print(f"  {qty} sold. Stock is now {it['stock']}.")

def generate_po(items: dict):
    orders = []
    for it in items.values():
        need = int(it['rop'] - it['stock'] - it['on_order'] + (it['demand'] * it['lead_time']))
        if need > 0:
            orders.append(it)
            it['on_order'] += need
    
    if not orders:
        print("\n  ✅ Stock healthy. No Purchase Order needed.")
        return
        
    fn = f"purchase_order_{list(items.values())[0]['store_id']}.csv"
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Item ID', 'Qty Ordered', 'XAI Explanation'])
        for it in orders:
            w.writerow([it['full_id'], int(it['on_order']), f"Stock < ROP ({it['rop']:.1f}). Replenishing for {it['lead_time']}d lead time."])
    
    print(f"\n  📝 Purchase Order generated for {len(orders)} items -> saved to {fn}")
    print(f"  Items marked as 'On Order'.")

def fulfill_po(items: dict):
    count = 0
    for it in items.values():
        if it['on_order'] > 0:
            it['stock'] += it['on_order']
            it['on_order'] = 0
            count += 1
    if count: print(f"\n  🚚 Delivery arrived! Restocked {count} items.")
    else: print("\n  📦 No pending orders to fulfill.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    model, lead_times, df = load_resources()
    store_id = select_store(df)
    items = build_snapshot(df, model, lead_times, store_id)
    page = 1

    while True:
        print_banner(store_id)
        idx_map, page, total_pages = display_inventory_page(items, page)

        print("\n  MENU:")
        print("  [N]ext Page | [P]rev Page | [J]ump to Page")
        print("  1. Fast Simulate Demand vs Actual (Macro)")
        print("  2. Adjust Stock/Params")
        print("  3. Log Sale")
        print("  4. GENERATE Purchase Order (Managerial Action)")
        print("  5. APPROVE Delivery / Restock (Managerial Action)")
        print("  6. Change Store  |  7. Exit")

        c = input("\n  Select: ").strip().upper()

        if c == '7': break
        elif c == 'N': page += 1
        elif c == 'P': page -= 1
        elif c == 'J':
            try: page = int(input("  Page #: "))
            except: pass
        elif c == '1': simulate(items, df, store_id)
        elif c == '2': adjust(items, idx_map)
        elif c == '3': log_sale(items, store_id, idx_map)
        elif c == '4': generate_po(items)
        elif c == '5': fulfill_po(items)
        elif c == '6':
            store_id = select_store(df)
            items = build_snapshot(df, model, lead_times, store_id)
            page = 1
        elif c == 'S':
            pass

if __name__ == "__main__":
    main()
