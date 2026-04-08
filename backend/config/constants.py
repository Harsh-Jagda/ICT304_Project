"""
constants.py — Global Business & Model Constants
=================================================
Single source of truth for all hardcoded business parameters.
If a value needs to change (e.g. average unit price), change it HERE only.
"""

# ── Financial parameters ──────────────────────────────────────────────────────
# Average retail unit price used for financial calculations.
# NOTE (L-05): In production this should use the actual sell_price per item
# from the M5 dataset. This flat rate is an academic baseline.
DEFAULT_UNIT_PRICE_USD = 3.50

# Daily holding cost rate per unit (fraction of unit cost per day).
# Represents warehouse space, capital cost, shrinkage risk.
HOLDING_COST_RATE = 0.05

# ── Supply Chain parameters ───────────────────────────────────────────────────
# Default lead time (days) used when department is not in lead_times.json.
DEFAULT_LEAD_TIME_DAYS = 2

# Z-score for 95% service level (Safety Stock formula).
DEFAULT_SERVICE_LEVEL_Z = 1.645

# ── MLOps / Retraining thresholds ─────────────────────────────────────────────
# Number of days before time-based retraining is triggered.
TIME_TRIGGER_DAYS = 30

# Maximum allowed MAE degradation above baseline before drift trigger fires.
DRIFT_THRESHOLD_PCT = 0.20   # 20%

# Minimum new records in real_time_sales.csv to trigger volume-based retrain.
VOLUME_TRIGGER_THRESHOLD = 1000

# ── Paths (relative to project root) ─────────────────────────────────────────
# Canonical path of the real-time sales accumulation file.
# Both train.py and retrain_manager.py MUST use this constant.
REALTIME_SALES_SUBPATH = ("data", "raw", "real_time_sales.csv")
