This project implements an AI-driven system designed to assist warehouses in forecasting item sales and predicting shortages. Using historical sales, calendar events, and 
pricing data from the Walmart M5 dataset, the system predicts future demand and optimizes ordering decisions to minimize stockouts and reduce inventory holding costs.

Objectives: 
- Data management and preprocessing - storing of user uploaded data securely, allowing historical data retrieval. The system should clean null and inconsistent values. 
- Accurately forecast item demand for three time intervals (day, week, month).
- Risk analysis - predicting shortages and possible losses for the aforementioned intervals

Dataset:
The project uses the following files from the M5 dataset from Kaggle:
- sales_train_validation.csv - Historical daily unit sales.
- calendar.csv - Dates, holidays, promotions and event metadata.
- sell_prices.csv - Weekly item prices per store. 
