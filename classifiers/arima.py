from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def train_test(data, sig_col='Open', p=1, d=1, q=1):    
    # create base DF for forecast values
    forecast_vals = pd.DataFrame({'day':[], sig_col: []})

    # group the data by days
    day_df = data[sig_col].groupby(pd.Grouper(freq='D'))

    print(f'Training ARIMA with order: {(p,d,q)}')
    
    # for each day, fit ARIMA to the observed market data and predict the next price based on the previous day
    for day_data in day_df:
        # get the day and prices
        [_, market_data] = day_data
    
        # no market data on weekends and holidays
        if len(market_data) == 0:
            continue
    
        # adjust the index so ARIMA knows this is minute by minute data
        market_data.index = pd.DatetimeIndex(market_data.index).to_period('min')
    
        # build, fit, forecast arima
        day_model = ARIMA(market_data, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
        day_model_fit = day_model.fit()
        next_forecast = day_model_fit.forecast()
    
        # convert the forecast index to datetime then to day frequency
        next_forecast.index = next_forecast.index.to_timestamp()
        next_forecast.index = pd.DatetimeIndex(next_forecast.index).to_period('D')
    
        # shift the index to the next day because that's what the forecast is predicting
        forecast_vals.loc[len(forecast_vals)] = [next_forecast.index.shift(1)[0], next_forecast.values[0]]
    
    forecast_vals = forecast_vals.set_index('day')

    return forecast_vals


def convert_forecast_to_classification(forecast):
    # compute d_forecast / d_t
    forecast_dot = np.diff(forecast, axis=0)
    # create empty array
    yhat = np.zeros_like(forecast_dot)
    # set instances with positive change in forecast to 1
    yhat[forecast_dot > 0] = 1

    return yhat
