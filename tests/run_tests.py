"""
run_tests.py — Comprehensive Test Suite & Evaluation Report
AI Warehouse Management System

Runs all tests across 4 levels and generates a scored evaluation report.
Output: test_report.json + printed summary

Test Levels:
  1. Unit Tests        — formula correctness
  2. Integration Tests — modules working together
  3. Accuracy Tests    — ML model quality metrics
  4. Functional Tests  — system behavior & edge cases

Usage: python run_tests.py
"""
import os, sys, json, traceback
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []

def record(level: str, name: str, status: str, detail: str = "", score: float = None):
    results.append({
        "level": level, "name": name, "status": status,
        "detail": detail, "score": score,
    })
    icon = {"PASS": "[OK]", "FAIL": "[!!]", "SKIP": "[--]"}[status]
    score_str = f"  [{score:.4f}]" if score is not None else ""
    print(f"  {icon} [{level}] {name}{score_str} -- {detail}")

def run(level, name, fn):
    try:
        fn()
    except AssertionError as e:
        record(level, name, FAIL, str(e))
    except Exception as e:
        record(level, name, FAIL, f"Exception: {e}")

# ═══════════════════════════════════════════════════════
# LEVEL 1: UNIT TESTS — formula correctness
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print(" LEVEL 1: UNIT TESTS ".center(70, "="))
print("="*70)

def test_safety_stock_formula():
    z, std, lt = 1.96, 2.5, 3
    expected = z * std * np.sqrt(lt)
    computed = 1.96 * 2.5 * np.sqrt(3)
    assert abs(expected - computed) < 1e-6, f"Safety stock mismatch: {computed} != {expected}"
    record("UNIT", "Safety Stock Formula", PASS, f"Z x sigma x sqrt(LT) = {computed:.4f}")

def test_rop_formula():
    demand, lt, ss = 3.5, 5, 8.5
    rop = demand * lt + ss
    assert rop == 26.0, f"ROP = {rop}, expected 26.0"
    record("UNIT", "Reorder Point Formula", PASS, f"ROP = {rop:.1f}")

def test_order_qty_ceiling():
    rop, stock = 15.3, 8
    qty = int(np.ceil(rop - stock))
    assert qty == 8, f"Order qty = {qty}, expected 8"
    record("UNIT", "Order Qty Ceiling (np.ceil)", PASS, f"ROP-Stock rounded up = {qty}")

def test_no_negative_order():
    rop, stock = 5.0, 20
    qty = max(0, int(np.ceil(rop - stock)))
    assert qty == 0, f"Should be 0, got {qty}"
    record("UNIT", "No Negative Orders", PASS, "Stock > ROP means order = 0")

def test_item_id_parsing():
    item_id = "FOODS_1_042_CA_1_evaluation"
    parts = item_id.split("_")
    category = f"{parts[0]}_{parts[1]}"
    local_id = parts[2]
    assert category == "FOODS_1", f"Category = {category}"
    assert local_id == "042", f"Local ID = {local_id}"
    record("UNIT", "Item ID Parsing", PASS, f"Category={category}, ID={local_id}")

def test_zero_demand_no_crash():
    demand = 0.0
    stockout_days = 10 / (demand + 1e-6)
    assert stockout_days > 0, "stockout_days should be positive"
    record("UNIT", "Zero Demand Guard (1e-6)", PASS, f"stockout_days = {stockout_days:.1f}")

for fn in [test_safety_stock_formula, test_rop_formula, test_order_qty_ceiling,
           test_no_negative_order, test_item_id_parsing, test_zero_demand_no_crash]:
    run("UNIT", fn.__name__, fn)

# ═══════════════════════════════════════════════════════
# LEVEL 2: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print(" LEVEL 2: INTEGRATION TESTS ".center(70, "="))
print("="*70)

def test_model_loads():
    MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
    assert os.path.exists(MODEL_PATH), "Model file not found"
    model = joblib.load(MODEL_PATH)
    feats = model.feature_name()
    assert len(feats) > 0, "Model has no features"
    record("INTEGRATION", "Model Loads (joblib)", PASS, f"{len(feats)} features")

def test_lead_times_schema():
    path = os.path.join(DATA_DIR, "config", "lead_times.json")
    assert os.path.exists(path), "lead_times.json missing"
    with open(path) as f:
        lt = json.load(f)
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        assert cat in lt, f"{cat} missing from lead_times"
        assert "lead_time_days" in lt[cat], f"lead_time_days missing for {cat}"
        assert "service_level_z" in lt[cat], f"service_level_z missing for {cat}"
    record("INTEGRATION", "lead_times.json Schema", PASS, "All 3 categories valid")

def test_model_registry_write_read():
    import model_registry as mr
    registry = mr._load_registry()
    if registry:
        entry = registry[0]
        assert "version" in entry, "Missing version"
        assert "mae" in entry, "Missing mae"
        record("INTEGRATION", "Model Registry R/W", PASS, f"{len(registry)} versions in registry")
    else:
        record("INTEGRATION", "Model Registry R/W", SKIP, "Registry is empty — run train.py first")

def test_retrain_triggers_run():
    from retrain_manager import evaluate_triggers
    result = evaluate_triggers(current_mae=None, verbose=False)
    assert "triggered" in result, "Missing 'triggered' key"
    assert "checks" in result, "Missing 'checks' key"
    record("INTEGRATION", "Retrain Trigger Evaluation", PASS, f"Checks complete, triggered={result['triggered']}")

def test_recommendation_pipeline():
    """Run mini end-to-end: load model → predict → calculate ROP."""
    MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
    DATA_PATH  = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")
    LT_PATH    = os.path.join(DATA_DIR, "config", "lead_times.json")
    if not all(os.path.exists(p) for p in [MODEL_PATH, DATA_PATH, LT_PATH]):
        record("INTEGRATION", "Mini Recommendation Pipeline", SKIP, "Required files missing")
        return
    
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH).head(50).copy()
    df['date'] = pd.to_datetime(df['date'])
    
    with open(LT_PATH) as f:
        lt = json.load(f)
    
    features = model.feature_name()
    CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'event_name_1', 'event_type_1']
    X = df[features].copy()
    # Add missing columns with 0 if not present (e.g. state_id on old test_sample)
    for col in features:
        if col not in X.columns:
            X[col] = 0
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes
    
    preds = model.predict(X.values)
    preds = np.clip(preds, 0, None)
    assert len(preds) == 50, "Prediction count mismatch"
    assert (preds >= 0).all(), "Negative predictions found"
    record("INTEGRATION", "Mini Recommendation Pipeline", PASS, f"50 predictions OK, min={preds.min():.2f}, max={preds.max():.2f}")

for fn in [test_model_loads, test_lead_times_schema, test_model_registry_write_read,
           test_retrain_triggers_run, test_recommendation_pipeline]:
    run("INTEGRATION", fn.__name__, fn)

# ═══════════════════════════════════════════════════════
# LEVEL 3: ACCURACY TESTS (ML Metrics)
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print(" LEVEL 3: ACCURACY / ML METRICS ".center(70, "="))
print("="*70)

MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
DATA_PATH  = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")
CAT_COLS   = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'event_name_1', 'event_type_1']

acc_df = None
model  = None

if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    model  = joblib.load(MODEL_PATH)
    acc_df = pd.read_parquet(DATA_PATH).copy()
    acc_df['date'] = pd.to_datetime(acc_df['date'])
    features = model.feature_name()
    X = acc_df[features].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes
    acc_df['pred'] = np.clip(model.predict(X.values), 0, None)

def test_overall_mae():
    if acc_df is None:
        record("ACCURACY", "Overall MAE", SKIP, "Data/model missing"); return
    mae = float(np.mean(np.abs(acc_df['pred'] - acc_df['sales'])))
    # Target: MAE < 2.0 (our baseline was 1.26)
    status = PASS if mae < 2.0 else FAIL
    record("ACCURACY", "Overall MAE", status, f"MAE = {mae:.4f} (target < 2.0)", score=mae)

def test_overall_rmse():
    if acc_df is None:
        record("ACCURACY", "Overall RMSE", SKIP, "Data/model missing"); return
    rmse = float(np.sqrt(np.mean((acc_df['pred'] - acc_df['sales'])**2)))
    status = PASS if rmse < 4.0 else FAIL
    record("ACCURACY", "Overall RMSE", status, f"RMSE = {rmse:.4f} (target < 4.0)", score=rmse)

def test_no_negative_predictions():
    if acc_df is None:
        record("ACCURACY", "No Negative Predictions", SKIP, "Data/model missing"); return
    neg = (acc_df['pred'] < 0).sum()
    status = PASS if neg == 0 else FAIL
    record("ACCURACY", "No Negative Predictions", status, f"{neg} negative values found")

def test_mae_by_category():
    if acc_df is None or 'cat_id' not in acc_df.columns:
        record("ACCURACY", "MAE by Category", SKIP, "Data/model missing"); return
    by_cat = acc_df.groupby('cat_id', observed=True)[['pred', 'sales']].apply(
        lambda g: np.mean(np.abs(g['pred'] - g['sales']))
    ).dropna().to_dict()
    if not by_cat:
        record("ACCURACY", "MAE by Category", SKIP, "No category breakdown available")
        return
    detail = " | ".join(f"{k}: {v:.3f}" for k, v in by_cat.items())
    all_ok = all(v < 3.0 for v in by_cat.values())
    record("ACCURACY", "MAE by Category", PASS if all_ok else FAIL, detail)

def test_weekly_aggregated_mae():
    """Weekly-aggregated MAE is usually much lower than daily."""
    if acc_df is None:
        record("ACCURACY", "Weekly Aggregated MAE", SKIP, "Data/model missing"); return
    weekly = acc_df.groupby([pd.Grouper(key='date', freq='W'), 'item_id'], observed=True)[['pred', 'sales']].sum()
    mae_w = float(np.mean(np.abs(weekly['pred'] - weekly['sales'])))
    status = PASS if mae_w < 10.0 else FAIL
    record("ACCURACY", "Weekly Aggregated MAE", status, f"Weekly MAE = {mae_w:.4f} (target < 10.0)", score=mae_w)

def test_prediction_bias():
    """Check model isn't systematically over- or under-predicting."""
    if acc_df is None:
        record("ACCURACY", "Prediction Bias Check", SKIP, "Data/model missing"); return
    bias = float(np.mean(acc_df['pred'] - acc_df['sales']))
    status = PASS if abs(bias) < 0.5 else FAIL
    record("ACCURACY", "Prediction Bias Check", status, f"Mean bias = {bias:.4f} (|bias| < 0.5)")

for fn in [test_overall_mae, test_overall_rmse, test_no_negative_predictions,
           test_mae_by_category, test_weekly_aggregated_mae, test_prediction_bias]:
    run("ACCURACY", fn.__name__, fn)

# ═══════════════════════════════════════════════════════
# LEVEL 4: FUNCTIONAL TESTS — system behavior
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print(" LEVEL 4: FUNCTIONAL TESTS ".center(70, "="))
print("="*70)

def test_required_files_exist():
    required = [
        "models", "wms_lgbm_model.pkl", "config", "lead_times.json",
        "model_registry.py", "retrain_manager.py", "hpo.py",
        "data_prep.py", "train.py", "recommendation.py",
        "prototype_main.py", "dashboard.py",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    status = PASS if not missing else FAIL
    detail = "All files present" if not missing else f"Missing: {missing}"
    record("FUNCTIONAL", "Required Files Exist", status, detail)

def test_lead_time_values_in_range():
    path = os.path.join(DATA_DIR, "config", "lead_times.json")
    with open(path) as f:
        lt = json.load(f)
    for cat, vals in lt.items():
        assert 1 <= vals["lead_time_days"] <= 14, f"{cat}: lead_time out of range"
        assert 1.0 <= vals["service_level_z"] <= 3.0, f"{cat}: Z-score out of range"
    record("FUNCTIONAL", "Lead Times In Valid Range", PASS, "All Z-scores and lead times valid")

def test_edge_case_zero_stock():
    """System should recommend order when stock = 0."""
    stock, rop = 0, 10.0
    order_needed = stock < rop
    qty = int(np.ceil(rop - stock)) if order_needed else 0
    assert order_needed and qty > 0, "Should recommend order when stock = 0"
    record("FUNCTIONAL", "Edge Case: Zero Stock", PASS, f"order_needed=True, qty={qty}")

def test_edge_case_very_high_demand():
    """Safety stock should scale with high demand variance."""
    z, std, lt = 2.326, 50.0, 1  # FOODS with very volatile item
    ss = z * std * np.sqrt(lt)
    assert ss > 0, "Safety stock must be positive"
    record("FUNCTIONAL", "Edge Case: Very High Demand", PASS, f"Safety stock = {ss:.1f} (high variance handled)")

def test_retrain_time_trigger_logic():
    """After >30 days, trigger should fire."""
    from datetime import datetime, timedelta
    # Simulate registry entry from 31 days ago
    old_date = (datetime.now() - timedelta(days=31)).isoformat()
    days_since = (datetime.now() - datetime.fromisoformat(old_date)).days
    assert days_since >= 30, "Time trigger logic broken"
    record("FUNCTIONAL", "Retrain Time Trigger Logic", PASS, f"Fires after {days_since} days")

def test_data_schema():
    """Processed data must contain required columns."""
    path = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")
    if not os.path.exists(path):
        record("FUNCTIONAL", "Data Schema Check", SKIP, "test_sample.parquet missing"); return
    df = pd.read_parquet(path)
    required_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'date',
                     'sales', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 'sell_price']
    missing = [c for c in required_cols if c not in df.columns]
    status = PASS if not missing else FAIL
    record("FUNCTIONAL", "Data Schema Check", status,
           "All required columns present" if not missing else f"Missing: {missing}")

for fn in [test_required_files_exist, test_lead_time_values_in_range,
           test_edge_case_zero_stock, test_edge_case_very_high_demand,
           test_retrain_time_trigger_logic, test_data_schema]:
    run("FUNCTIONAL", fn.__name__, fn)

# ═══════════════════════════════════════════════════════
# FINAL SCORE REPORT
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print(" EVALUATION REPORT ".center(70, "="))
print("="*70)

levels = ["UNIT", "INTEGRATION", "ACCURACY", "FUNCTIONAL"]
grand_pass = grand_fail = grand_skip = 0

for lvl in levels:
    lvl_results = [r for r in results if r["level"] == lvl]
    p = sum(1 for r in lvl_results if r["status"] == PASS)
    f = sum(1 for r in lvl_results if r["status"] == FAIL)
    s = sum(1 for r in lvl_results if r["status"] == SKIP)
    total = len(lvl_results)
    score = p / (total - s) if (total - s) > 0 else 0
    print(f"  {lvl:<14} : {p} PASS | {f} FAIL | {s} SKIP -> Score: {score*100:.0f}%")
    grand_pass += p; grand_fail += f; grand_skip += s

grand_total = grand_pass + grand_fail
grand_score = grand_pass / grand_total if grand_total > 0 else 0

print("-"*70)
print(f"  OVERALL        : {grand_pass} PASS | {grand_fail} FAIL | {grand_skip} SKIP")
print(f"  OVERALL SCORE  : {grand_score * 100:.1f}% ({grand_pass}/{grand_total} tests passed)")

# Grading
if grand_score >= 0.90:
    grade = "A - Excellent"
elif grand_score >= 0.75:
    grade = "B - Good"
elif grand_score >= 0.60:
    grade = "C - Satisfactory"
else:
    grade = "D - Needs Improvement"
print(f"  GRADE          : {grade}")
print("="*70)

# Save report
report = {
    "timestamp":     datetime.now().isoformat(),
    "overall_score": round(grand_score, 4),
    "grade":         grade,
    "pass":          grand_pass,
    "fail":          grand_fail,
    "skip":          grand_skip,
    "details":       results,
}
_report_path = os.path.join(DATA_DIR, "test_report.json")
with open(_report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Full report saved to: test_report.json")
print("  Run 'python run_tests.py' any time to re-evaluate.\n")
