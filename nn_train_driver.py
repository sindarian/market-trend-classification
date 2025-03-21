import numpy as np
import pandas as pd

import classifiers.neural_nets as nn
import classifiers.neural_net_feature_space as nnfs
import classifiers.labeller as labeller


def train_mlp(train_df, label_onehot, config=None):
    # set the config if needed
    if config is None:
        config = default_training_config()

    # create model
    model = nn.build_mlp_clf(config)

    # get column names of just the features to test
    feature_cols = list(train_df.columns)
    if 'EpochTime' in feature_cols:
        feature_cols.remove('EpochTime')
    if 'Label' in feature_cols:
        feature_cols.remove('Label')

    # fit model
    model.fit(train_df[feature_cols].values, label_onehot, epochs=config['epochs'])

    return model


def train_lstm(train_df, label_onehot, config=None):
    # set the config if needed
    if config is None:
        config = default_training_config()

    # create model
    model = nn.build_lstm_clf(config)

    # get column names of just the features to test
    feature_cols = list(train_df.columns)
    if 'EpochTime' in feature_cols:
        feature_cols.remove('EpochTime')
    if 'Label' in feature_cols:
        feature_cols.remove('Label')

    # fit model
    model.fit(train_df[feature_cols].values, label_onehot, epochs=config['epochs'])

    return model


def train_cnn(train_df, label_onehot, config=None):
    # set the config if needed
    if config is None:
        config = default_training_config()

    # create model
    model = nn.build_cnn_clf(config)

    # get column names of just the features to test
    feature_cols = list(train_df.columns)
    if 'EpochTime' in feature_cols:
        feature_cols.remove('EpochTime')
    if 'Label' in feature_cols:
        feature_cols.remove('Label')

    # fit model
    model.fit(train_df[feature_cols].values, label_onehot, epochs=config['epochs'])

    return model


def default_training_config():
    periods = np.arange(2, 20, 2)
    config = {'n_fft_components': 8,
              'signal_data_column': 'Close',
              'periods': periods,
              'in_shape': len(periods)*2,
              'out_shape': 2,
              'activation': 'softmax',
              'loss': 'categorical_crossentropy',
              'metrics': ['accuracy'],
              'epochs': 3}

    return config


def convert_onehot(arr):
    return np.eye(2)[arr.astype(int)]


def inverse_onehot(onehot_arr):
    onehot_arr = np.round(onehot_arr)
    arr = np.zeros(onehot_arr.shape[0])
    for c_idx in range(1, onehot_arr.shape[1]):
        insert_idx = (onehot_arr[:, c_idx] == 1)
        arr[insert_idx] = c_idx

    return arr


def split_data(data_df, label_df=None, config=None):
    # set the config if needed
    if config is None:
        config = default_training_config()

    # compute label space if needed
    if label_df is None:
        label_df = labeller.driver(data_df,
                                   n_components=config['n_fft_components'],
                                   signal_column=config['signal_data_column'])

    # compute feature space
    feature_space_df = nnfs.compute_feature_space_df(data_df,
                                                     config['periods'],
                                                     data_col=config['signal_data_column'])

    # time align
    label_df = label_df.loc[np.in1d(label_df.EpochTime.values, feature_space_df.EpochTime.values)]

    # merge the two DF so that the labels and feature values appear alongside each other
    feature_space_df = pd.merge(feature_space_df, label_df[['EpochTime', 'Label']], how='left')

    # train/test split
    split_factor = .8
    train_df, test_df = (feature_space_df.iloc[:int(feature_space_df.shape[0] * split_factor)],
                         feature_space_df.iloc[int(feature_space_df.shape[0] * split_factor):])

    # shuffle rows of train df, onehot encode
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # label_onehot = convert_onehot(train_df.Label.values)

    # return train_df, label_onehot, test_df
    return train_df, test_df


if __name__ == '__main__':
    # create training configuration
    train_config = default_training_config()

    # read in the data
    qqq_df = pd.read_csv('./data/qqq_2022.csv')

    # train the model
    mlp_model = train_mlp(qqq_df, config=train_config)

    print('Ding!')
