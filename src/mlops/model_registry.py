"""
model_registry.py — Model Version Control
Tracks all trained model versions, their metrics, and parameters.
Ensures production model is always the best-performing one.
"""
import json
import os
import shutil
from datetime import datetime
import joblib

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_FILE = os.path.join(DATA_DIR, "models", "model_registry.json")
MODELS_DIR = os.path.join(DATA_DIR, "models", "model_versions")


def _load_registry() -> list:
    if not os.path.exists(REGISTRY_FILE):
        return []
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


def _save_registry(registry: list):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(model, metrics: dict, params: dict, training_data: str = "all_states") -> str:
    """
    Register a newly trained model.
    Only promotes to production if metrics are better than current best.
    
    Args:
        model: trained LightGBM model object
        metrics: dict with 'mae' and 'rmse'
        params: dict of hyperparameters used
        training_data: description of training data scope
    
    Returns:
        version string (e.g. 'v3')
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    registry = _load_registry()
    version_num = len(registry) + 1
    version = f"v{version_num}"
    
    model_filename = os.path.join("models", f"wms_lgbm_model_{version}.pkl")
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    entry = {
        "version": version,
        "date": datetime.now().isoformat(),
        "mae": round(metrics.get("mae", 9999), 4),
        "rmse": round(metrics.get("rmse", 9999), 4),
        "params": params,
        "training_data": training_data,
        "model_file": model_path,
        "is_production": False,
    }
    
    # Save versioned model file
    joblib.dump(model, model_path)
    
    # Determine if this should be promoted to production
    best = get_best_model_entry()
    if best is None or entry["mae"] < best["mae"]:
        # Promote: copy to main production path
        production_path = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
        shutil.copy2(model_path, production_path)
        
        # Mark all previous as non-production
        for r in registry:
            r["is_production"] = False
        entry["is_production"] = True
        
        print(f"  ✓ Model {version} promoted to PRODUCTION (MAE: {entry['mae']:.4f})")
    else:
        print(f"  ✗ Model {version} NOT promoted — production {best['version']} is better (MAE: {best['mae']:.4f} vs {entry['mae']:.4f})")
    
    registry.append(entry)
    _save_registry(registry)
    
    return version


def get_best_model_entry() -> dict | None:
    """Return the lowest-MAE model entry from the registry."""
    registry = _load_registry()
    if not registry:
        return None
    return min(registry, key=lambda x: x["mae"])


def get_production_model_entry() -> dict | None:
    """Return the current production model entry."""
    registry = _load_registry()
    prod = [r for r in registry if r.get("is_production")]
    if prod:
        return prod[-1]
    return get_best_model_entry()


def rollback(version: str) -> bool:
    """
    Roll back to a previous model version.
    Copies the versioned model file to the production path.
    """
    registry = _load_registry()
    target = next((r for r in registry if r["version"] == version), None)
    
    if target is None:
        print(f"  Error: version '{version}' not found in registry.")
        return False
    
    if not os.path.exists(target["model_file"]):
        print(f"  Error: model file for '{version}' not found at {target['model_file']}")
        return False
    
    production_path = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
    shutil.copy2(target["model_file"], production_path)
    
    # Update is_production flags
    for r in registry:
        r["is_production"] = (r["version"] == version)
    _save_registry(registry)
    
    print(f"  ✓ Rolled back to {version} (MAE: {target['mae']:.4f})")
    return True


def list_versions():
    """Print a summary table of all registered model versions."""
    registry = _load_registry()
    if not registry:
        print("No models registered yet.")
        return
    
    print("\n" + "="*80)
    print(" MODEL REGISTRY ".center(80, "="))
    print(f"{'Version':<8} | {'Date':<20} | {'MAE':<8} | {'RMSE':<8} | {'Data':<15} | {'Status'}")
    print("-"*80)
    for r in registry:
        status = "★ PRODUCTION" if r.get("is_production") else "  archived"
        date_str = r["date"][:16].replace("T", " ")
        print(f"{r['version']:<8} | {date_str:<20} | {r['mae']:<8.4f} | {r['rmse']:<8.4f} | {r['training_data']:<15} | {status}")
    print("="*80)


def get_baseline_mae() -> float | None:
    """Return the MAE of the earliest registered model (our baseline)."""
    registry = _load_registry()
    if not registry:
        return None
    return registry[0]["mae"]


if __name__ == "__main__":
    list_versions()
