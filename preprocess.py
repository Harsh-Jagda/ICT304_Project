import pandas as pd
import numpy as np

# load csv files from https://www.kaggle.com/c/m5-forecasting-accuracy/data?hl=ru-RU
sales_validation_subset = pd.read_csv('C:/Users/idary/OneDrive/Desktop/m5subsets/sales_train_validation.csv')
calendar_subset = pd.read_csv('C:/Users/idary/OneDrive/Desktop/m5subsets/calendar.csv')
sell_prices_subset = pd.read_csv('C:/Users/idary/OneDrive/Desktop/m5subsets/sell_prices.csv')


# use melt() function to transform a wide dataframe into a narrow dataset, converting d1, d2 etc. into a single row "d" and the amount of sales
sales_long = sales_validation_subset.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="sales"
)

# basic optimization - reducing memory from int64 and float64 into int16 and float32 
sales_long["sales"] = sales_long["sales"].astype("int16")
calendar_subset["date"] = pd.to_datetime(calendar_subset["date"])
sell_prices_subset["sell_price"] = sell_prices_subset["sell_price"].astype("float32") 

# merging calendar and prices for time series 
df = sales_long.merge(calendar_subset, on="d", how="left")
df = df.merge(sell_prices_subset, on=["store_id", "item_id", "wm_yr_wk"], how="left")

# function to reduce memory usage for newly formed columns
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "int64":
            df[col] = df[col].astype("int16")
        elif col_type == "float64":
            df[col] = df[col].astype("float32")
    return df

df = reduce_mem_usage(df)

# rest of code - creating time-based features, lag and rolling features to demand patterns (not done yet)
