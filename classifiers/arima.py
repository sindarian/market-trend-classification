from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import plotting

def train_test(data, sig_col='Open'):
    # get the order params
    p = get_ar_param(raw_signal_df, sig_col)
    d = get_difference_param(raw_signal_df, sig_col)
    q = get_ma_param(raw_signal_df, sig_col)
    
    # create base DF for forecast values
    forecast_vals = pd.DataFrame({'day':[], sig_col: []})

    # create a date-timestamp column so we can groupy by days
    data['DTS'] = pd.to_datetime(data['EpochTime'], unit='s')
    data = data.set_index('DTS')
    day_df = data[sig_col].groupby(pd.Grouper(freq='D'))
    
    # for each day, fit ARIMA to the observed market data and predict the next price based on the previous day
    for day_data in day_df:
        # get the day
        day = day_data[0]
        
        # get the day prices
        market_data = day_data[1]
    
        # no market data on weekends and holidays
        if len(market_data) == 0:
            continue
    
        # adjust the index so ARIMA knows this is minute by minute data
        market_data.index = pd.DatetimeIndex(market_data.index).to_period('min')
    
        # build, fit, forecast arima
        day_model = ARIMA(market_data, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
        day_model_fit = day_model.fit()
        next_forecast = day_model_fit.forecast()
    
        # convert the forecast index to datetime then to day frequency
        next_forecast.index = next_forecast.index.to_timestamp()
        next_forecast.index = pd.DatetimeIndex(next_forecast.index).to_period('D')
    
        # shift the index to the next day because that's what the forecast is predicting
        forecast_vals.loc[len(forecast_vals)] = [next_forecast.index.shift(1)[0], next_forecast.values[0]]
    
    forecast_vals = forecast_vals.set_index('day')
    
    plotting.plot_forecast(data, forecast_vals)

    return forecast_vals

def get_difference_param(raw_signal_df, sig_col='Open'):
    # plot the original series, the first order, and second order difference
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(raw_signal_df[[sig_col]])
    ax1.set_title('Original Series')
    ax1.axes.xaxis.set_visible(False)
    
    ax2.plot(raw_signal_df[[sig_col]].diff())
    ax2.set_title('1st Order Differencing')
    ax2.axes.xaxis.set_visible(False)
    
    ax3.plot(raw_signal_df[[sig_col]].diff().diff())
    ax3.set_title('2nd Order Differencing')
    plt.show()

    # because the data appears relatively stabilized after a first order
    # difference, we use that as the d param
    return 1

def get_ar_param(raw_signal_df, sig_col='Open'):
    plot_pacf(raw_signal_df[[sig_col]].diff().dropna())

    # given the spike at 1 with the values significantly dropping off and
    # staying around 0, we use 1 as the initial autoregression param
    return 1

def get_ma_param(raw_signal_df, sig_col='Open'):
    plot_acf(raw_signal_df[[sig_col]].diff().dropna())

    # given the spike at 1 with the values significantly dropping off and
    # staying around 0, we use 1 as the initial moving average param
    return 1