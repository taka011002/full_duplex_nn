from src.system_model import SystemModel
from src import modules as m
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as optimizers
from keras_radam import RAdam
import numpy as np


class NNModel():
    def __init__(self, n_hidden):
        self.init_model(n_hidden)

    def init_model(self, n_hidden, learning_rate=None):
        model = Sequential()
        model.add(Dense(n_hidden, activation='relu', input_shape=(4,)))
        model.add(Dense(n_hidden, activation='relu', input_shape=(n_hidden,)))
        model.add(Dense(2, activation='linear'))
        # optimizer = Adam(lr=0.001)
        # optimizer = RAdam()
        optimizer = optimizers.SGD(lr=0.001, momentum=0.8)
        model.compile(optimizer, loss="mse")

        self.nn = model

    def learn(self, system_model: SystemModel, training_ratio: float, n_epochs: int, batch_size: int):
        self.system_model = system_model

        # トレーニングデータの生成
        trainingSamples = int(np.floor(system_model.x.size * training_ratio))
        x_train = system_model.x[0:trainingSamples]
        y_train = system_model.y[0:trainingSamples]
        s_train = system_model.s[0:trainingSamples]

        # NNの入力に合うように1つのベクトルにする
        train = np.zeros((x_train.size, 4))
        train[:, 0] = x_train.real
        train[:, 1] = x_train.imag
        train[:, 2] = y_train.real
        train[:, 3] = y_train.imag

        # テストデータの作成
        x_test = system_model.x[trainingSamples:]
        y_test = system_model.y[trainingSamples:]
        s_test = system_model.s[trainingSamples:]

        # NNの入力に合うように1つのベクトルにする
        test = np.zeros((x_test.size, 4))
        test[:, 0] = x_test.real
        test[:, 1] = x_test.imag
        test[:, 2] = y_test.real
        test[:, 3] = y_test.imag

        # 学習
        self.nn_history = self.nn.fit(train, [s_train.real, s_train.imag], epochs=n_epochs,
                                      batch_size=batch_size, verbose=2,
                                      validation_data=(test, [s_test.real, s_test.imag]))

        # 学習したモデルを評価
        self.pred = self.nn.predict(test)

        # 推定した希望信号の取り出し
        self.s_hat = self.pred[0] + 1j * self.pred[1]

        # 推定信号をデータへ復調する
        self.d_s_hat = m.demodulate_qpsk(self.s_hat)
        # 元々の外部信号のデータ
        self.d_s_test = system_model.d_s[2 * trainingSamples:]

        self.error = np.sum(self.d_s_test != self.d_s_hat)
        self.ber = m.check_error(self.d_s_test, self.d_s_hat)