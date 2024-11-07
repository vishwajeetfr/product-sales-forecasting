import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression

def seasonal_naive_forecast(train_df, test_df):
    dates = (test_df.index - pd.DateOffset(years=1) + pd.DateOffset(days=1)).values
    seasonal_naive_sales = train_df[train_df.index.isin(dates)]['sales']
    return seasonal_naive_sales.to_list()

def holt_winters_forecast(train_df, test_df):
    hw_model = ExponentialSmoothing(train_df['sales'], trend=None, seasonal='add', seasonal_periods=7).fit()
    hw_forecast = hw_model.forecast(len(test_df)).tolist()
    return hw_forecast

def sarima_forecast(train_df, test_df):
    sarima_model = SARIMAX(train_df['sales'], order=(6, 1, 0), seasonal_order=(6, 1, 0, 7))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True).tolist()
    return sarima_forecast

def linear_regression_forecast(train_df, test_df):
    reg_df = train_df.copy()
    for i in range(1, 8):
        reg_df[f'lag_{i}'] = reg_df['sales'].shift(i)
    reg_df = reg_df.dropna()

    X_train = reg_df[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
    y_train = reg_df['sales']
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = test_df.copy()
    for i in range(1, 8):
        X_test[f'lag_{i}'] = X_test['sales'].shift(i)
    X_test = X_test.dropna()[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
    
    linear_forecast = model.predict(X_test).tolist()
    return linear_forecast
