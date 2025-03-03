from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error

def train_test(data, sig_col='Open', train_split=0.8):
    # split the data
    train_size = int(len(data) * train_split)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    p = get_ar_param(raw_signal_df, sig_col)
    d = get_difference_param(raw_signal_df, sig_col)
    q = get_ar_param(raw_signal_df, sig_col)
    
    # Fit ARIMA model
    model = ARIMA(train[[sig_col]], order=(p,d,q))
    model_fit = model.fit()
    
    print('ARIMA Model Summary:')
    print(model_fit.summary())
    
    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    
    # Plot the results with specified colors
    plt.figure(figsize=(14,7))
    plt.plot(train.index, train[sig_col], label='Train', color='#203147')
    plt.plot(test.index, test[sig_col], label='Test', color='#01ef63')
    plt.plot(test.index, forecast, label='Forecast', color='orange')
    plt.title(f'{sig_col} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel(f'{sig_col} Price')
    plt.legend()
    plt.show()
    

    # Calculate RMSE
    forecast_vals = forecast[:len(test)]
    test_vals = test[sig_col][:len(forecast)]
    rmse = np.sqrt(mean_squared_error(test_sig, forecast))
    print(f"ARIMA RMSE: {rmse:.4f}")

    test['Forecast'] = forecast

    return test

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