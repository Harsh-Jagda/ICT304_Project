"""
retrain_manager.py — Retraining Trigger System
Monitors model performance and triggers retraining when needed.

Triggers (from Requirements_Specification MR1):
  1. Time-based    : >30 days since last training
  2. Performance   : MAE has drifted >20% above baseline
  3. Volume        : new data file has grown significantly
  4. Multi-state   : an unknown state_id has appeared in new data
"""
import os
import json
import pandas as pd
import subprocess
import sys
from datetime import datetime, timedelta

# FIX #4: import shared constants — single source of truth
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config.constants import (
    REALTIME_SALES_SUBPATH,
    TIME_TRIGGER_DAYS,
    DRIFT_THRESHOLD_PCT,
    VOLUME_TRIGGER_THRESHOLD,
)
from src.mlops.model_registry import (
    get_production_model_entry,
    get_baseline_mae,
    list_versions,
    register_model,
)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_FILE = os.path.join(DATA_DIR, "models", "model_registry.json")
# FIX #4: was os.path.join(DATA_DIR, "real_time_sales.csv") — wrong directory.
# Now uses the constant that matches train.py exactly: data/raw/real_time_sales.csv
REALTIME_DATA = os.path.join(DATA_DIR, *REALTIME_SALES_SUBPATH)
DRIFT_LOG = os.path.join(DATA_DIR, "drift_log.json")

# ---- Thresholds (imported from config.constants — see above) ----
KNOWN_STATES = {"CA", "TX", "WI"}


# ──────────────────────────────────────────────
# Individual checks (each returns True = trigger)
# ──────────────────────────────────────────────

def check_time_trigger() -> tuple[bool, str]:
    """Has it been >30 days since last training?"""
    prod = get_production_model_entry()
    if prod is None:
        return True, "No model registered — initial training needed."
    
    trained_at = datetime.fromisoformat(prod["date"])
    days_since = (datetime.now() - trained_at).days
    
    if days_since >= TIME_TRIGGER_DAYS:
        return True, f"Time trigger: {days_since} days since last training (threshold: {TIME_TRIGGER_DAYS})."
    return False, f"Time OK: {days_since} days since training."


def check_performance_drift(current_mae: float | None = None) -> tuple[bool, str]:
    """Has MAE grown >20% above baseline?"""
    if current_mae is None:
        return False, "No current MAE provided — skipping drift check."
    
    baseline = get_baseline_mae()
    if baseline is None:
        return False, "No baseline MAE in registry."
    
    drift_pct = (current_mae - baseline) / baseline
    if drift_pct > DRIFT_THRESHOLD_PCT:
        return True, f"Performance drift: MAE {current_mae:.4f} is {drift_pct*100:.1f}% above baseline {baseline:.4f}."
    return False, f"Performance OK: drift is {drift_pct*100:.1f}% (threshold: {DRIFT_THRESHOLD_PCT*100:.0f}%)."


def check_volume_trigger() -> tuple[bool, str]:
    """Has real_time_sales.csv grown significantly?"""
    if not os.path.exists(REALTIME_DATA):
        return False, "No real-time data file found."
    
    rt_df = pd.read_csv(REALTIME_DATA)
    n_new = len(rt_df)
    
    # FIX #2: use imported constant instead of local magic number
    if n_new >= VOLUME_TRIGGER_THRESHOLD:
        return True, f"Volume trigger: {n_new} new records in real_time_sales.csv (threshold: {VOLUME_TRIGGER_THRESHOLD})."
    return False, f"Volume OK: {n_new} records so far."


def check_new_state_trigger() -> tuple[bool, str]:
    """Has an unknown state_id appeared in real-time data?"""
    if not os.path.exists(REALTIME_DATA):
        return False, "No real-time data file — skipping state check."
    
    try:
        rt_df = pd.read_csv(REALTIME_DATA)
        if "state_id" not in rt_df.columns:
            return False, "No state_id column in real-time data."
        
        new_states = set(rt_df["state_id"].unique())
        unknown = new_states - KNOWN_STATES
        
        if unknown:
            return True, f"New state trigger: unknown state(s) {unknown} detected — model must be retrained to cover them."
        return False, f"States OK: all known ({new_states})."
    except Exception as e:
        return False, f"Could not read real-time data: {e}"


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def evaluate_triggers(current_mae: float | None = None, verbose: bool = True) -> dict:
    """
    Run all checks. Returns a summary dict with trigger status.
    """
    checks = {
        "time":        check_time_trigger(),
        "drift":       check_performance_drift(current_mae),
        "volume":      check_volume_trigger(),
        "new_state":   check_new_state_trigger(),
    }
    
    triggered = {k: v for k, v in checks.items() if v[0]}
    
    if verbose:
        print("\n" + "="*60)
        print(" RETRAINING TRIGGER REPORT ".center(60, "="))
        for name, (fired, reason) in checks.items():
            icon = "🔴 TRIGGERED" if fired else "🟢 OK"
            print(f"  [{icon}] {name.upper():12} — {reason}")
        print("="*60)
        
        if triggered:
            print(f"\n  ⚠ {len(triggered)} trigger(s) fired. Retraining recommended.")
        else:
            print("\n  ✓ No triggers fired. Model is healthy.")
    
    return {
        "triggered": len(triggered) > 0,
        "reasons": [v[1] for v in triggered.values()],
        "checks": {k: {"fired": v[0], "reason": v[1]} for k, v in checks.items()},
        "timestamp": datetime.now().isoformat(),
    }


def run_retraining(auto_confirm: bool = False) -> bool:
    """
    Launch the full training pipeline.
    Returns True if retraining completed successfully.
    """
    result = evaluate_triggers(verbose=True)
    
    if not result["triggered"]:
        print("No retraining needed.")
        return False
    
    if not auto_confirm:
        confirm = input("\nStart retraining now? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Retraining cancelled.")
            return False
    
    # FIX #2: correct script paths — root-level data_prep.py/train.py do NOT exist.
    # Actual locations are inside src/ subdirectories.
    DATA_PREP_SCRIPT = os.path.join(DATA_DIR, "src", "data_pipeline", "data_prep.py")
    TRAIN_SCRIPT     = os.path.join(DATA_DIR, "src", "mlops", "train.py")

    for script_path in [DATA_PREP_SCRIPT, TRAIN_SCRIPT]:
        if not os.path.exists(script_path):
            print(f"  ERROR: Script not found: {script_path}")
            return False

    print("\n[1/2] Running data preparation (all states)...")
    ret = subprocess.run(
        [sys.executable, DATA_PREP_SCRIPT],
        capture_output=True, text=True, cwd=DATA_DIR
    )
    if ret.returncode != 0:
        print(f"  data_prep.py failed:\n{ret.stderr}")
        return False
    print("  data_prep.py complete.")

    print("[2/2] Running training...")
    ret = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        capture_output=True, text=True, cwd=DATA_DIR
    )
    if ret.returncode != 0:
        print(f"  train.py failed:\n{ret.stderr}")
        return False
    print("  train.py complete.")
    
    # Log the retrain event
    log_path = os.path.join(DATA_DIR, "retrain_log.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    log.append({"timestamp": datetime.now().isoformat(), "reasons": result["reasons"]})
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    
    print("\n  ✓ Retraining pipeline complete. Check model_registry for new version.")
    return True


def log_drift(current_mae: float):
    """Append a drift measurement to drift_log.json for tracking over time."""
    log = []
    if os.path.exists(DRIFT_LOG):
        with open(DRIFT_LOG) as f:
            log = json.load(f)
    log.append({"timestamp": datetime.now().isoformat(), "mae": current_mae})
    with open(DRIFT_LOG, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    print("Running trigger evaluation...")
    evaluate_triggers()
    list_versions()
