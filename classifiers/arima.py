from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from tqdm import tqdm


def minute_forecast(data_df, sig_col='Open', fit_window=120, p=1, d=1, q=1):
    # instantiate array to hold: time, real value, predicted value
    forecast_data = np.zeros((data_df.shape[0]-fit_window, 3))
    raw_data = data_df[sig_col].values
    epoch_times = data_df['EpochTime'].values
    for i in tqdm(range(fit_window, data_df.shape[0])):
        # fit the model on the current window
        model = ARIMA(raw_data[i-fit_window:i],
                      order=(p, d, q),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit = model.fit()

        # forecast the model
        next_forecast = model_fit.forecast()

        # store the data
        forecast_data[i-fit_window] = [epoch_times[i], raw_data[i], next_forecast[0]]

    # format into df
    forecast_df = pd.DataFrame(forecast_data, columns=['EpochTime', sig_col, f'Forecast {sig_col}'])
    return forecast_df


def day_forecast(data, sig_col='Open', forecast_sig_col = 'Forecast Open', p=1, d=1, q=1):    
    # create base DF for forecast values
    forecast_vals = pd.DataFrame({'Day': [], forecast_sig_col: []})

    # group the data by days
    day_df = data[sig_col].groupby(pd.Grouper(freq='D'))

    print(f'Training ARIMA with order: {(p,d,q)}')
    
    # for each day, fit ARIMA to the observed market data and predict the next price based on the previous day
    for day_data in day_df:
        # get the day and prices
        [_, market_data] = day_data

        day = market_data[market_data.rank() == 1].index
    
        # no market data on weekends and holidays
        if len(market_data) == 0 or len(day) == 0:
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
        forecast_vals.loc[len(forecast_vals)] = [day[0], next_forecast.values[0]]
    
    forecast_vals = forecast_vals.set_index('Day')
    forecast_vals.index = pd.to_datetime(forecast_vals.index)
    forecast_vals.index = forecast_vals.index.strftime('%Y-%m-%d')

    return forecast_vals


def convert_forecast_to_classification(forecast):
    # compute d_forecast / d_t
    forecast_dot = np.diff(forecast, axis=0)
    # create empty array
    yhat = np.zeros_like(forecast_dot)
    # set instances with positive change in forecast to 1
    yhat[forecast_dot > 0] = 1

    return yhat


def index_df_by_date(data_df):
    # properly index the input data to ARIMA
    date_indexed_df = data_df.copy(deep=True)
    date_indexed_df['DTS'] = pd.to_datetime(date_indexed_df['EpochTime'], unit='s')
    date_indexed_df = date_indexed_df.set_index('DTS')
    
    return date_indexed_df


def get_arima_true_values(date_indexed_df, sig_col='Open'):
    # set up a day-wise dataframe to perform evaluation on
    true_values = date_indexed_df[date_indexed_df[sig_col].groupby(pd.Grouper(freq='D')).rank() == 1][1:][sig_col].to_frame()
    true_values.index = pd.to_datetime(true_values.index)
    true_values.index = true_values.index.strftime('%Y-%m-%d')

    return true_values


if __name__ == '__main__':
    # local test
    df = pd.read_csv('../data/qqq_2022.csv')
    arima_data = minute_forecast(df.iloc[-390*5:])
