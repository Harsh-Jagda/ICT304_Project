"""
pipeline.py — Project Orchestrator
====================================
FIX #8: Single entry point that runs the full pipeline in the correct order,
with dependency checks at each step.

Usage:
    # Full pipeline from scratch:
    python pipeline.py

    # Skip data prep (already done):
    python pipeline.py --skip-data-prep

    # Only check trigger & retrain if needed:
    python pipeline.py --retrain-only
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


# ── Colour helpers ────────────────────────────────────────────────────────────
def _ok(msg):  print(f"  \033[92m✓\033[0m {msg}")
def _err(msg): print(f"  \033[91m✗\033[0m {msg}")
def _step(n, total, msg): print(f"\n\033[1m[{n}/{total}] {msg}\033[0m")


def run(cmd: list, label: str) -> bool:
    """Run a subprocess and print success/failure. Returns True on success."""
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode == 0:
        _ok(f"{label} complete.")
        return True
    else:
        _err(f"{label} FAILED:")
        print(result.stderr[-2000:])   # show last 2000 chars of error
        return False


def check_file(path: Path, description: str) -> bool:
    """Assert a required file exists. Returns True if found."""
    if path.exists():
        _ok(f"Found: {path.relative_to(ROOT)}")
        return True
    else:
        _err(f"Missing: {path.relative_to(ROOT)}  ← {description}")
        return False


def main():
    parser = argparse.ArgumentParser(description="WMS Full Pipeline Orchestrator")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data_prep.py if processed data already exists")
    parser.add_argument("--skip-train",     action="store_true",
                        help="Skip training if model already exists")
    parser.add_argument("--skip-sim",       action="store_true",
                        help="Skip simulation (scorecard data not needed)")
    parser.add_argument("--retrain-only",   action="store_true",
                        help="Run trigger check and retrain if needed, then exit")
    parser.add_argument("--launch",         action="store_true",
                        help="Launch the Streamlit dashboard after pipeline completes")
    args = parser.parse_args()

    total_steps = 4 if not args.retrain_only else 1
    step = 0

    print("\n" + "═"*60)
    print("  WMS Pipeline Orchestrator")
    print("═"*60)

    # ── Retrain-only mode ─────────────────────────────────────────────────────
    if args.retrain_only:
        _step(1, 1, "Checking retraining triggers...")
        run([sys.executable, str(ROOT / "src" / "mlops" / "retrain_manager.py")],
            "Trigger evaluation")
        return

    # ── Step 1: Data Preparation ──────────────────────────────────────────────
    step += 1
    processed = ROOT / "data" / "processed" / "processed_data_all.parquet"
    ca_processed = ROOT / "data" / "processed" / "processed_data_ca.parquet"

    if args.skip_data_prep:
        _step(step, total_steps, "Data prep: SKIPPED (--skip-data-prep)")
        if not check_file(processed, "Run without --skip-data-prep") and \
           not check_file(ca_processed, "Run without --skip-data-prep"):
            _err("No processed data found at all — cannot skip data prep.")
            sys.exit(1)
    else:
        _step(step, total_steps, "Running data preparation pipeline...")
        # Verify M5 raw files first
        required_raw = [
            ROOT / "data" / "raw" / "sales_train_evaluation.csv",
            ROOT / "data" / "raw" / "calendar.csv",
            ROOT / "data" / "raw" / "sell_prices.csv",
        ]
        missing = [f for f in required_raw if not f.exists()]
        if missing:
            _err("Missing M5 raw data files:")
            for f in missing:
                print(f"    {f.relative_to(ROOT)}")
            print("\n  Download from: https://www.kaggle.com/c/m5-forecasting-accuracy/data")
            sys.exit(1)

        ok = run([sys.executable, str(ROOT / "src" / "data_pipeline" / "data_prep.py")],
                 "data_prep.py")
        if not ok:
            sys.exit(1)

    # ── Step 2: Model Training ────────────────────────────────────────────────
    step += 1
    model_path = ROOT / "models" / "wms_lgbm_model.pkl"

    if args.skip_train:
        _step(step, total_steps, "Model training: SKIPPED (--skip-train)")
        if not check_file(model_path, "Run without --skip-train"):
            sys.exit(1)
    else:
        _step(step, total_steps, "Training LightGBM model...")
        ok = run([sys.executable, str(ROOT / "src" / "mlops" / "train.py")],
                 "train.py")
        if not ok:
            sys.exit(1)
        check_file(model_path, "Model file not created despite successful training")

    # ── Step 3: Simulation ────────────────────────────────────────────────────
    step += 1
    sim_path = ROOT / "data" / "outputs" / "sim_results.parquet"

    if args.skip_sim:
        _step(step, total_steps, "Simulation: SKIPPED (--skip-sim)")
    else:
        _step(step, total_steps, "Running 365-day A/B simulation...")
        ok = run([sys.executable, str(ROOT / "src" / "simulation" / "prepare_simulation.py")],
                 "prepare_simulation.py")
        if not ok:
            print("  ⚠ Simulation failed — director scorecard will show warning but app still works.")
        else:
            check_file(sim_path, "Simulation output not saved")

    # ── Step 4: Launch (optional) ─────────────────────────────────────────────
    step += 1
    _step(step, total_steps, "Pipeline complete!")
    print()
    _ok("Data processed  →  data/processed/")
    _ok(f"Model trained   →  {model_path.relative_to(ROOT)}")
    if not args.skip_sim and sim_path.exists():
        _ok(f"Simulation done →  {sim_path.relative_to(ROOT)}")
    print()

    if args.launch:
        print("  Launching dashboard at http://localhost:8501 ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], cwd=str(ROOT))
    else:
        print("  To launch the dashboard, run:")
        print("    streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()
