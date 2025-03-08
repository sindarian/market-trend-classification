import numpy as np
import pandas as pd

import classifiers.neural_nets as nn
import classifiers.neural_net_feature_space as nnfs
import classifiers.labeller as labeller


def train_mlp(data_df, config):
    # compute label space
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
    feature_space_df = pd.merge(feature_space_df, label_df, how='left')

    # train/test split
    split_factor = .8
    train_df, test_df = (feature_space_df.iloc[:int(feature_space_df.shape[0] * split_factor)],
                         feature_space_df.iloc[int(feature_space_df.shape[0] * split_factor):])

    # shuffle rows of train df, onehot encode
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    label_onehot = np.eye(2)[train_df['Label'].values.astype(int)]

    # create model
    model = nn.build_mlp_clf(config)

    # train model
    feature_cols = list(train_df.columns)
    feature_cols.remove('EpochTime')
    feature_cols.remove('Label')
    model.fit(train_df[feature_cols].values, label_onehot, epochs=3)

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
              'metrics': ['accuracy']}

    return config


if __name__ == '__main__':
    # create training configuration
    train_config = default_training_config()

    # read in the data
    qqq_df = pd.read_csv('./data/qqq_2022.csv')

    # train the model
    mlp_model = train_mlp(qqq_df, train_config)

    print('Ding!')
