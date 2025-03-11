from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv1D, Input

def build_mlp_clf(model_config_dict):
    # extract stuff from config
    in_shape = model_config_dict['in_shape']
    out_shape = model_config_dict['out_shape']
    activ = model_config_dict['activation']

    # build model
    model = Sequential(name="MLP")
    model.add(Dense(32, input_dim=in_shape, activation='relu'))
    model.add(Dense(16, input_dim=in_shape, activation='relu'))
    model.add(Dense(out_shape, activation=activ))

    print(model.summary())

    # compile model
    loss_str = model_config_dict['loss']
    metrics = model_config_dict['metrics']

    model.compile(optimizer='adam', loss=loss_str, metrics=metrics)

    return model

def build_mlp_clf_2(config_in_shape, config_out_shape, final_activ, config_loss_str, config_metrics, optimizer):
    # extract stuff from config
    in_shape = config_in_shape
    out_shape = config_out_shape
    activ = final_activ

    # build model
    model = Sequential(name="MLP")
    model.add(Input(shape=(in_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(out_shape, activation=activ))

    # compile model
    loss_str = config_loss_str
    metrics = config_metrics

    model.compile(optimizer=optimizer, loss=loss_str, metrics=metrics)

    return model

def build_lstm_clf(model_config_dict):
    # extract stuff from config
    in_shape = model_config_dict['in_shape']
    out_shape = model_config_dict['out_shape']
    activ = model_config_dict['activation']

    # build model
    model = Sequential(name="LSTM")
    model.add(LSTM(units=50, activation='relu', dropout=.10, input_shape=(in_shape,1), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(out_shape, activation=activ))

    print(model.summary())

    # compile model
    loss_str = model_config_dict['loss']
    metrics = model_config_dict['metrics']

    model.compile(optimizer='adam', loss=loss_str, metrics=metrics)

    return model

def build_lstm_clf_2(config_in_shape, config_out_shape, final_activ, config_loss_str, config_metrics, optimizer):
    # extract stuff from config
    in_shape = config_in_shape
    out_shape = config_out_shape
    activ = final_activ

    # build model
    model = Sequential(name="LSTM")
    model.add(Input(shape=(in_shape,1)))
    model.add(LSTM(units=50, activation='relu', dropout=.10, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(out_shape, activation=activ))

    # compile model
    loss_str = config_loss_str
    metrics = config_metrics

    model.compile(optimizer=optimizer, loss=loss_str, metrics=metrics)

    return model

def build_cnn_clf(model_config_dict):
    # extract stuff from config
    in_shape = model_config_dict['in_shape']
    out_shape = model_config_dict['out_shape']
    activ = model_config_dict['activation']

    # build model
    model = Sequential(name="CNN")
    model.add(Conv1D(filters=100, kernel_size=5, activation='sigmoid', input_shape=(18,1)))
    model.add(Flatten())
    model.add(Dense(out_shape, activation=activ))

    print(model.summary())

    # compile model
    loss_str = model_config_dict['loss']
    metrics = model_config_dict['metrics']

    model.compile(optimizer='adam', loss=loss_str, metrics=metrics)

    return model

def build_cnn_clf_2(config_in_shape, config_out_shape, final_activ, config_loss_str, config_metrics, optimizer):
    # extract stuff from config
    in_shape = config_in_shape
    out_shape = config_out_shape
    activ = final_activ

    # build model
    model = Sequential(name="CNN")
    model.add(Input(shape=(in_shape,1)))
    model.add(Conv1D(filters=100, kernel_size=5, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(out_shape, activation=activ))

    # compile model
    loss_str = config_loss_str
    metrics = config_metrics

    model.compile(optimizer=optimizer, loss=loss_str, metrics=metrics)

    return model


if __name__ == '__main__':

    mlp_config = dict({'in_shape': 10,
                       'out_shape': 2,
                       'activation': 'softmax',
                       'loss': 'categorical_crossentropy',
                       'metrics': ['accuracy']})
    mlp = build_mlp_clf(mlp_config)
    print(mlp.summary())
    print('Ding!')
