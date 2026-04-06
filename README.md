# Warehouse AI: Predictive Inventory & Regional Risk Management System

## 1. Project Abstract

**Warehouse AI** is a full-stack decision-support system designed to mitigate the "Bullwhip Effect" in modern supply chains. By combining Gradient Boosted Decision Trees (LightGBM) with statistical safety stock modeling, the system provides warehouse managers with high-precision demand forecasts and automated purchasing recommendations. Unlike traditional inventory systems, Warehouse AI operates with **Regional Granularity**, allowing for hyper-local analysis across different states and categories.

---

## 2. System Architecture & "The Why"

The project is built on a modular **Python-Flask architecture** designed for scalability and interpretability.

### 2.1 The Problem: Uncertainty and Volatility

Manual inventory management suffers from two primary flaws:

- **Lagged Response**: Reactive ordering leads to stockouts or overstocking.
- **Aggregation Bias**: Looking at "Total National Sales" hides the fact that Texas might be out of stock while California is overstocked.

### 2.2 The Solution: The Predictive Engine

The system moves the decision point from *"How much did we sell yesterday?"* to *"How much will we sell in the next lead-time window?"* by leveraging time-series forecasting.

---

## 3. The Machine Learning Engine (LightGBM)

The core of the system is a **LightGBM** (Light Gradient Boosting Machine) model. We chose LightGBM over traditional ARIMA or LSTM models because of its ability to handle categorical features (like `state_id`) and its speed in processing large-scale tabular data.

### 3.1 The 16-Feature Training Matrix

The model was trained on a high-dimensional feature set designed to capture both seasonal trends and sudden volatility.

| Feature Name | Type | Description | Why it matters |
|--------------|------|-------------|----------------|
| `sales` | Target | Historical daily units sold | The dependent variable the model predicts. |
| `lag_7` | Numerical | Sales from exactly 7 days ago | Captures weekly cyclicality. |
| `lag_28` | Numerical | Sales from 28 days ago | Captures monthly cyclicality. |
| `rolling_mean_7` | Numerical | Average sales of the last week | Smooths out daily "noise." |
| `rolling_mean_28` | Numerical | Average sales of the last month | Detects long-term demand shifts. |
| `rolling_std_7` | Numerical | Standard deviation of weekly sales | Measures demand volatility/uncertainty. |
| `sell_price` | Numerical | Current unit price | Accounts for price elasticity of demand. |
| `state_id` | Categorical | Geographic region (e.g., TX, CA, WI) | Enables regional-specific forecasting. |
| `item_id` | Categorical | Unique SKU identifier | Learns specific product behavior. |
| `cat_id` | Categorical | Category (e.g., HOBBIES, FOODS) | Groups products with similar demand profiles. |
| `dept_id` | Categorical | Department identifier | Secondary grouping for hierarchy. |
| `store_id` | Categorical | Specific warehouse/retail location | Captures localized traffic patterns. |
| `event_name_1` | Categorical | Holiday/Promotion name | Accounts for spikes (e.g., SuperBowl). |
| `event_type_1` | Categorical | Type of event (Sporting, National) | Categorizes the nature of the demand spike. |
| `day_of_week` | Categorical | Monday-Sunday | Learns weekday vs. weekend patterns. |
| `month` | Categorical | 1 through 12 | Learns annual seasonality. |

### 3.2 Feature Engineering Logic

The inclusion of **Lag Features** (`lag_7`, `lag_28`) transforms a simple regression task into a time-series supervised learning problem. By looking at `rolling_std_7`, the model effectively "sees" how erratic a product is, which directly informs the Risk Service later.

---

## 4. Supply Chain Risk Framework

The **Risk Service** is the bridge between the ML forecast and actual warehouse operations.

### 4.1 Safety Stock Calculation

We use a statistical buffer to protect against demand variability during the lead time. The formula implemented in `risk_service.py` is:

```
SS = Z × σd × √LT
```

**Where:**
- **Z**: The Service Level Z-Score (defined in your JSON config)
- **σd**: The Standard Deviation of Demand (calculated from the rolling ML data)
- **LT**: The Lead Time in days

### 4.2 Reorder Point (ROP) Logic

The **Reorder Point** is the inventory level that triggers a new purchase order. It is calculated as:

```
ROP = (Average Daily Demand × LT) + SS
```

### 4.3 Days of Cover

This metric tells the user exactly how long the current stock will last based on the projected demand.

- **Red Status**: Days of Cover < Lead Time (Emergency: Stock will run out before a new order arrives)
- **Green Status**: Days of Cover > Lead Time (Safe: Adequate buffer exists)

---

## 5. Regional Granularity & Multi-Tenancy

A key feature of this system is the **State-Based Filtering**.

- When a user selects "Texas" on the dashboard, the backend filters the entire CSV/Parquet dataset for `state_id == 'TX'`
- The LightGBM model then performs inference specifically on that subset
- This prevents "Global Averaging" where high sales in one state mask a critical shortage in another

---

## 6. Technical Implementation (Backend)

### 6.1 The Flask API

- **`/forecast`**: Triggers the `forecasting_service.py`. It takes the active dataset, processes the 16 features, and returns a JSON array of predictions.
- **`/risk`**: Triggers the `risk_service.py`. It calculates the ROP and Safety Stock by cross-referencing the ML forecast with the `lead_times.json` configuration.
- **`/get_available_regions`**: Scans the active CSV for unique `state_id` values to populate the UI dropdowns.

### 6.2 Data Preparation

The `data_prep_service.py` handles missing values and data type conversion. It ensures that even if a user uploads a CSV with missing `event_name_1` tags, the model receives a "None" category rather than crashing.

---

## 7. Installation & Usage Instructions

### 7.1 Prerequisites

- Python 3.10+
- Git
- Virtual Environment (recommended)

### 7.2 Cloning the Repository

Open your terminal or MINGW64 and run:

```bash
git clone https://github.com/Harsh-Jagda/ICT304_Project.git
cd ICT304_Project
```

### 7.3 Virtual Environment Setup

Create and activate the environment to isolate dependencies:

```bash
# Windows
python -m venv venv
source venv/Scripts/activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 7.4 Installing Dependencies

Install the required ML and Web libraries:

```bash
pip install -r requirements.txt
```

### 7.5 Running the Application

Launch the Flask development server by running the following
command where app.py is located. i.e. /backend.
Username and Passwords are stored in users.csv

```bash
flask run
```

**Access the dashboard at:** `http://127.0.0.1:5000`

---

## 8. User Manual

### 8.1 For Administrators

- **Data Upload**: Access `/upload` to refresh the system with new sales data. Ensure your CSV includes the `state_id` and `sales` columns.
- **Configuration**: Upload a new `lead_times.json` to change how safety stock is calculated for different categories.
- **Audit Logs**: Monitor system activity through the dashboard log viewer to see who uploaded data and when.

### 8.2 For Analysts

- **Scenario Testing**: Use the "Active Config" toggle to switch between "Standard" and "Emergency" lead times.
- **Regional Drill-down**: Select a specific state from the dropdown before running a Risk Report to see localized purchasing needs.

### 8.3 For Viewers

- **Executive Reports**: View the Risk Analysis cards. Pay close attention to the "Purchasing Action" box, which provides clear instructions like "Place Order Immediately" or "Stock Healthy."

---

*This documentation provides a comprehensive overview of the Warehouse AI system, from theoretical foundations to practical implementation.*

