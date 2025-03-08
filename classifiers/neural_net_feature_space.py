import pandas as pd
import numpy as np


def moving_average(values, period):
    # compute the moving average over the valid interval
    ma = np.convolve(values, np.ones(period) / period, 'valid')
    return ma


def compute_feature_space_df(values_df, ma_periods, data_col='Close', tanh_scaling=True):
    # extract values
    values = values_df[data_col].values
    # list for df column names
    columns = []
    # pre-allocate contents of the DF
    ma_distro_arr = np.zeros((len(values), len(ma_periods)*2))
    # implement a "cutoff" to account for inability to compute moving average on some time spaces
    cutoff = ma_distro_arr.shape[0]
    # period index
    p_idx = 0
    for c_idx in range(0, ma_distro_arr.shape[1], 2):
        # compute moving average for this period
        p = ma_periods[p_idx]
        ma_values = moving_average(values, p)

        # compute change in moving average w/r to time -- velocity
        ma_dot_values = np.diff(ma_values)

        # store moving average feature values
        ma_distro_arr[-len(ma_values):, c_idx] = values[-ma_values.shape[0]:] - ma_values  # difference between spot price and moving average
        columns.append(f'MA_diff_{p}')
        ma_distro_arr[-len(ma_dot_values):, c_idx+1] = ma_dot_values  # time delta of moving average
        columns.append(f'MA_dot_{p}')

        # update cutoff
        cutoff = min(cutoff, len(ma_values))
        p_idx += 1

    if tanh_scaling:
        ma_distro_arr = np.tanh(ma_distro_arr)

    df = pd.DataFrame(ma_distro_arr[-cutoff:], columns=columns)
    df.insert(0, 'EpochTime', values_df.EpochTime.values[-df.shape[0]:])
    return df


def preprocessing_transformer(feature_space_arr):
    # perform tanh on the feature space values
    transformed_feature_space = np.tanh(feature_space_arr)
    return transformed_feature_space


if __name__ == '__main__':

    # read in data
    data_df = pd.read_csv('../data/qqq_2022.csv')
    # set periods
    periods = np.arange(5, 15)

    # compute moving average distro
    ma_distro_df = compute_feature_space_df(data_df, periods)
    print(ma_distro_df.shape)

    # plot stuff
    # import labeller
    # import sys
    # sys.path.append('..')
    # import plotting.plot_shortcuts as ps
    #
    # label_df = labeller.driver(signal_df=data_df, n_components=8, signal_column='Close')
    # ps.plot_feature_space(ma_distro_df.iloc[-390:], label_df.iloc[-390:])
