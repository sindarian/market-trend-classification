import numpy as np
import pandas as pd

import classifiers.neural_nets as nn
import classifiers.neural_net_feature_space as nnfs
import classifiers.labeller as labeller

# from sklearn.model_selection import GridSearchCV
# from scikeras.wrappers import KerasClassifier

from pprint import pprint


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


# def train_mlp_grid_search(train_df, label_onehot, config):
#     # # set the config if needed
#     # if config is None:
#     #     config = default_training_config()
#
#     # # compute label space if needed
#     # if label_df is None:
#     #     label_df = labeller.driver(data_df,
#     #                                n_components=config['n_fft_components'],
#     #                                signal_column=config['signal_data_column'])
#
#     # # compute feature space
#     # feature_space_df = nnfs.compute_feature_space_df(data_df,
#     #                                                  config['periods'],
#     #                                                  data_col=config['signal_data_column'])
#
#     # # time align
#     # label_df = label_df.loc[np.in1d(label_df.EpochTime.values, feature_space_df.EpochTime.values)]
#
#     # # merge the two DF so that the labels and feature values appear alongside each other
#     # feature_space_df = pd.merge(feature_space_df, label_df[['EpochTime', 'Label']], how='left')
#
#     # # train/test split
#     # split_factor = .8
#     # train_df, test_df = (feature_space_df.iloc[:int(feature_space_df.shape[0] * split_factor)],
#     #                      feature_space_df.iloc[int(feature_space_df.shape[0] * split_factor):])
#
#     # # shuffle rows of train df, onehot encode
#     # train_df = train_df.sample(frac=1).reset_index(drop=True)
#     # label_onehot = convert_onehot(train_df.Label.values)
#
#     feature_cols = list(train_df.columns)
#     feature_cols.remove('EpochTime')
#     feature_cols.remove('Label')
#
#     model = KerasClassifier(model=nn.build_mlp_clf_2, verbose=0)
#
#     # define the grid search parameters
#     optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#     param_grid = dict(model__config_in_shape=[config['in_shape']],
#                       model__config_out_shape=[config['out_shape']],
#                       model__final_activ=[config['activation']],
#                       model__config_loss_str=[config['loss']],
#                       model__config_metrics=[config['metrics']],
#                       model__optimizer=optimizers)
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#     grid_result = grid.fit(train_df[feature_cols].values, label_onehot)
#
#     pprint(f'Best MLP found with: {grid_result.best_params_}')
#
#     model = grid_result.best_estimator_
#
#     return model #, test_df


# def train_lstm_grid_search(train_df, label_onehot, config):
#     # # set the config if needed
#     # if config is None:
#     #     config = default_training_config()
#
#     # # compute label space if needed
#     # if label_df is None:
#     #     label_df = labeller.driver(data_df,
#     #                                n_components=config['n_fft_components'],
#     #                                signal_column=config['signal_data_column'])
#
#     # # compute feature space
#     # feature_space_df = nnfs.compute_feature_space_df(data_df,
#     #                                                  config['periods'],
#     #                                                  data_col=config['signal_data_column'])
#
#     # # time align
#     # label_df = label_df.loc[np.in1d(label_df.EpochTime.values, feature_space_df.EpochTime.values)]
#
#     # # merge the two DF so that the labels and feature values appear alongside each other
#     # feature_space_df = pd.merge(feature_space_df, label_df[['EpochTime', 'Label']], how='left')
#
#     # # train/test split
#     # split_factor = .8
#     # train_df, test_df = (feature_space_df.iloc[:int(feature_space_df.shape[0] * split_factor)],
#     #                      feature_space_df.iloc[int(feature_space_df.shape[0] * split_factor):])
#
#     # # shuffle rows of train df, onehot encode
#     # train_df = train_df.sample(frac=1).reset_index(drop=True)
#     # label_onehot = convert_onehot(train_df.Label.values)
#
#     # train model
#     feature_cols = list(train_df.columns)
#     feature_cols.remove('EpochTime')
#     feature_cols.remove('Label')
#
#     model = KerasClassifier(model=nn.build_lstm_clf_2, verbose=0)
#
#     # define the grid search parameters
#     optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#     param_grid = dict(model__config_in_shape=[config['in_shape']],
#                       model__config_out_shape=[config['out_shape']],
#                       model__final_activ=[config['activation']],
#                       model__config_loss_str=[config['loss']],
#                       model__config_metrics=[config['metrics']],
#                       model__optimizer=optimizers)
#     param_grid['epochs'] = [3]
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#     grid_result = grid.fit(train_df[feature_cols].values, label_onehot)
#
#     pprint(f'Best LSTM found with: {grid_result.best_params_}')
#
#     model = grid_result.best_estimator_
#
#     return model#, test_df


# def train_cnn_grid_search(train_df, label_onehot, config):
#     # # set the config if needed
#     # if config is None:
#     #     config = default_training_config()
#
#     # # compute label space if needed
#     # if label_df is None:
#     #     label_df = labeller.driver(data_df,
#     #                                n_components=config['n_fft_components'],
#     #                                signal_column=config['signal_data_column'])
#
#     # # compute feature space
#     # feature_space_df = nnfs.compute_feature_space_df(data_df,
#     #                                                  config['periods'],
#     #                                                  data_col=config['signal_data_column'])
#
#     # # time align
#     # label_df = label_df.loc[np.in1d(label_df.EpochTime.values, feature_space_df.EpochTime.values)]
#
#     # # merge the two DF so that the labels and feature values appear alongside each other
#     # feature_space_df = pd.merge(feature_space_df, label_df[['EpochTime', 'Label']], how='left')
#
#     # # train/test split
#     # split_factor = .8
#     # train_df, test_df = (feature_space_df.iloc[:int(feature_space_df.shape[0] * split_factor)],
#     #                      feature_space_df.iloc[int(feature_space_df.shape[0] * split_factor):])
#
#     # # shuffle rows of train df, onehot encode
#     # train_df = train_df.sample(frac=1).reset_index(drop=True)
#     # label_onehot = convert_onehot(train_df.Label.values)
#
#     # train model
#     feature_cols = list(train_df.columns)
#     feature_cols.remove('EpochTime')
#     feature_cols.remove('Label')
#
#     model = KerasClassifier(model=nn.build_cnn_clf_2, verbose=0)
#
#     # define the grid search parameters
#     optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#     param_grid = dict(model__config_in_shape=[config['in_shape']],
#                       model__config_out_shape=[config['out_shape']],
#                       model__final_activ=[config['activation']],
#                       model__config_loss_str=[config['loss']],
#                       model__config_metrics=[config['metrics']],
#                       model__optimizer=optimizers)
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#     grid_result = grid.fit(train_df[feature_cols].values, label_onehot)
#
#     pprint(f'Best CNN found with: {grid_result.best_params_}')
#
#     model = grid_result.best_estimator_
#
#     return model#, test_df


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
