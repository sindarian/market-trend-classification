from keras.models import Sequential
from keras.layers import Dense


def build_mlp_clf(model_config_dict):
    # extract stuff from config
    in_shape = model_config_dict['in_shape']
    out_shape = model_config_dict['out_shape']
    activ = model_config_dict['activation']

    # build model
    model = Sequential()
    model.add(Dense(32, input_dim=in_shape, activation='relu'))
    model.add(Dense(16, input_dim=in_shape, activation='relu'))
    model.add(Dense(out_shape, activation=activ))

    # compile model
    loss_str = model_config_dict['loss']
    metrics = model_config_dict['metrics']

    model.compile(optimizer='adam', loss=loss_str, metrics=metrics)

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
