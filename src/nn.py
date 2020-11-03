import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras_radam import RAdam


def nn_model(n_hidden) -> keras.models.Model:
    input = Input(shape=(4,))
    hidden1 = Dense(n_hidden, activation='relu')(input)
    output1 = Dense(1, activation='linear')(hidden1)
    output2 = Dense(1, activation='linear')(hidden1)
    model = Model(inputs=input, outputs=[output1, output2])
    # adam = Adam(lr=params['learningRate'])
    # model.compile(adam, loss="mse")
    model.compile(RAdam(), loss='mse')

    return model
