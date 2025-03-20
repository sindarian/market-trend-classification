# Classifying Market Trends
## Summary

This code base contains an analysis of the market price for QQQ over the year 2022. The data was pulled from Yahoo! Finance and contains a minute by minute price point of the stock. The models used to predict whether the price of QQQ is increasing or decreasing include:
* MLP
* LSTM
* CNN
* ARIMA

## Data
The data contains data points for the entire year of 2022 for QQQ with a frequency of 1 minute. The data is describes as follows:

| Date | Epoch Time | Open | High | Low | Close | Volume|
|-------|---------------|--------|-------|------|--------|-----|
| str in the form of YYYY-MM-DD | decimal representing theUnix epoch time | decimal  representing the opening price| decimal  representing the highest price| decimal representing the lowest price| decimal  representing the closing price| decimal  representing the total traded volume|


## Running the code
The processing and fine details of the data smoothing, model compilation, and model training are found in the `classifiers` and `plotting` folders as well as the `nn_driver.py` file. To run the project, navigate to the `analysis_driver.ipynb` notebook and run it with a fresh kernel with the required dependencies.

## Dependencies

The list of dependencies required to run this project are:

```
statsmodels
numpy
pandas
pprint
sklearn
warnings
matplotlib
tqdm
keras
```

Anaconda Environment Version: conda 25.1.1