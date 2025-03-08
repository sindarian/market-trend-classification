import matplotlib.pyplot as plt
import numpy as np


def plot_label_over_signal(signal_df, label_df, signal_column='Open'):
    # set initial variables
    colors = {'0': 'red', '1': 'green'}
    t = np.arange(signal_df.shape[0])
    classes = label_df['Label'].unique().astype(int)

    # create plot and plot raw signal
    plt.figure()
    plt.title(f'{signal_df.Date.values[0]} - {signal_df.Date.values[-1]}')
    plt.plot(t, signal_df[signal_column].values)

    # plot label values over the signal
    for c_idx in classes:
        # filter on class values
        label_df_slice = label_df.loc[label_df['Label'].values == classes[int(c_idx)]]
        # plot
        plot_idx = np.in1d(signal_df.EpochTime.values, label_df_slice.EpochTime.values)
        plt.scatter(t[plot_idx],
                    signal_df[signal_column].values[plot_idx],
                    color=colors[str(classes[int(c_idx)])],
                    alpha=.5)

    return plt


def plot_forecast(raw_signal_df, forecast_df, sig_col='Close'):
    # plot the observed vs forecast values
    plt.figure(figsize=(14,7))
    plt.plot(raw_signal_df.index, raw_signal_df[sig_col], color='green', label='Observed')
    plt.plot(forecast_df.index, forecast_df, color='black', label='Forecast', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Actual vs Predicted Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def plot_feature_space(feature_space_df, label_df):
    # time synch
    feature_space_df = feature_space_df.loc[np.in1d(feature_space_df.EpochTime, label_df.EpochTime)]
    label_df = label_df.loc[np.in1d(label_df.EpochTime, feature_space_df.EpochTime)]

    # get indices of different label types
    grow_idx = (label_df.values[:, 1] == 1)
    decay_idx = (label_df.values[:, 1] == 0)

    # plot
    # t = np.arange(feature_space_df.shape[0])
    for c in range(1, feature_space_df.shape[1], 2):
        plt.figure()
        plt.title(feature_space_df.columns[c])
        plt.scatter(feature_space_df.values[grow_idx, c], feature_space_df.values[grow_idx, c+1], color='green')
        plt.scatter(feature_space_df.values[decay_idx, c], feature_space_df.values[decay_idx, c+1], color='red')
        plt.xlabel('Time')
        plt.ylabel(feature_space_df.columns[c])
        plt.show()
