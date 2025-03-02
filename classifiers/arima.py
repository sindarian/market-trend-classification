from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

def train_test(data, sig_col='Open', train_split=0.8):
    # split the data
    train_size = int(len(data) * train_split)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    # Fit ARIMA model
    # order contains (p,d,q) where:
    #   p = how far back to look to help predict price
    #   d = finds the difference between consecutive timestamps
    #   q = size of the moving average window
    model = ARIMA(train[[sig_col]], order=(5,1,5), enforce_stationarity=False, enforce_invertibility=False)
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