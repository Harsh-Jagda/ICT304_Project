## User Guide

### Quick Start (Recommended)

> **The model has already been trained and all data is pre-processed.** You do NOT need to download M5 data or run training. The repository ships with everything required to launch the dashboard immediately.

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Launch the dashboard**
```bash
streamlit run dashboard.py
```

The application will open at `http://localhost:8501`. That's it!

---

### Login Credentials

| Username | Password | Role | Access |
|---|---|---|---|
| `manager` | `123` | Manager | Live Procurement Hub |
| `director` | `123` | Director | ROI Strategic Scorecard |
| `admin` | `123` | Admin | Both pages |

---

### What is pre-included in this repository

| File | Size | Description |
|---|---|---|
| `models/wms_lgbm_model.pkl` | ~6.7 MB | Trained LightGBM model (production version) |
| `models/model_registry.json` | ~1 KB | Version metadata and MAE baseline |
| `models/best_params.json` | ~0.3 KB | Optimised hyperparameters |
| `data/processed/processed_data_ca.parquet` | ~142 MB | California store data (used by dashboard) |
| `data/processed/processed_data_all.parquet` | ~347 MB | All-state data (used for full retraining) |
| `data/processed/test_sample.parquet` | ~0.2 MB | Holdout test set for accuracy verification |
| `data/outputs/sim_results.parquet` | varies | 365-day simulation results (Director scorecard) |

> **Note:** The M5 raw CSV files (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`) are **not** included due to their large size (~1 GB total). They are only needed if you want to re-run the full data pipeline from scratch (see Advanced Setup below).

---

### System Requirements

- **Python**: 3.10 or higher
- **RAM**: 4 GB minimum for dashboard use; 16 GB recommended for full retraining
- **OS**: Windows 10/11, macOS 12+, or Ubuntu 20.04+

---

### Running the Test Suite

To verify all 89 automated tests pass:
```bash
python -m pytest tests/ -v --tb=short --html=tests/report.html --self-contained-html
```
The full HTML report is saved to `tests/report.html`.

---

### Advanced Setup (Rebuild Everything from Scratch)

This section is only needed if you want to retrain the model with fresh data, or if you are running the project on a new machine where the pre-processed files are not present.

**Prerequisite: Download the M5 Dataset**
Download from Kaggle: https://www.kaggle.com/c/m5-forecasting-accuracy/data
Place the following files in `data/raw/`:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

**Option A: Use the orchestrator (recommended)**
```bash
python pipeline.py
```
This runs all steps in order with dependency checks, progress reporting, and friendly error messages. Accepts optional flags:
```bash
python pipeline.py --skip-data-prep  # if processed data already exists
python pipeline.py --skip-sim        # skip 365-day simulation
python pipeline.py --launch          # auto-launch dashboard at the end
```

**Option B: Run steps manually**
```bash
# Step 1: Data preprocessing (~20–30 min)
python src/data_pipeline/data_prep.py

# Step 2: Model training (~5–15 min)
python src/mlops/train.py

# Step 3: 365-day simulation for Director scorecard (~5–10 min)
python src/simulation/prepare_simulation.py

# Step 4: Launch dashboard
streamlit run dashboard.py
```

---

### Verifying Model Accuracy

To reproduce the reported metrics (MAE 1.25, RMSE 2.6) without retraining:
```bash
python tests/verify_accuracy.py
```

The `test_sample.parquet` file contains the year-5 holdout set that was excluded from training. Minor variations (±0.05 MAE) may occur due to LightGBM version differences or CPU parallelism affecting floating-point accumulation order. The `seed=42` parameter controls all random operations.

---

### Training and Validation Data Separation

The M5 dataset spans 5 years (approximately 2011–2016):
- **Training set**: All data before 2015-01-01 (years 1–4, approximately 1,913 days)
- **Test set**: 2015-01-01 to 2016-01-01 (year 5, 365 days) — never seen during training

This **temporal split** is critical: a random split would introduce data leakage (future data informing past predictions), artificially inflating accuracy metrics. The `split_data()` function in `src/mlops/train.py` enforces this split programmatically.