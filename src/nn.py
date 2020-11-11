from src.system_model import SystemModel
from src import modules as m
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as optimizers
from keras_radam import RAdam
import numpy as np


class NNModel():
    def __init__(self, n_hidden, learning_rate=0.001, h_si_len: int = 1, h_s_len: int = 1):
        self.init_model(n_hidden, learning_rate, h_si_len, h_s_len)

    def init_model(self, n_hidden, learning_rate=0.001, h_si_len: int = 1, h_s_len: int = 1):
        input = Input(shape=((2*h_si_len)+(2*h_s_len),))
        x = Dense(n_hidden, activation='relu')(input)
        # x = Dense(n_hidden, activation='relu')(x)
        output1 = Dense(1, activation='linear')(x)
        output2 = Dense(1, activation='linear')(x)
        model = Model(inputs=input, outputs=[output1, output2])
        # optimizer = optimizers.Adam(lr=learning_rate)
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.8)
        # optimizer = RAdam()
        model.compile(optimizer, loss='mse')

        self.nn = model

    def learn(self, system_model: SystemModel, training_ratio: float, n_epochs: int, batch_size: int, h_si_len: int = 1,
              h_s_len: int = 1):
        self.system_model = system_model

        # トレーニングデータの生成
        trainingSamples = int(np.floor(system_model.x.size * training_ratio))

        # チャネル数分つくる
        x = np.reshape(np.array([system_model.x[i:i + h_si_len] for i in range(system_model.x.size - h_si_len + 1)]), (system_model.x.size - h_si_len + 1, h_si_len))

        x_train = x[0:trainingSamples]
        y_train = system_model.y[0:trainingSamples]
        s_train = system_model.s[0:trainingSamples]

        # NNの入力に合うように1つのベクトルにする
        train = np.zeros((x_train.shape[0], (2*h_si_len)+(2*h_s_len)))
        train[:, 0:h_si_len] = x_train.real
        train[:, h_si_len:2*(h_si_len)] = x_train.imag
        train[:, 2*(h_s_len):3*(h_s_len)] = y_train.real
        train[:, 3*(h_s_len):4*(h_s_len)] = y_train.imag

        # テストデータの作成
        x_test = x[trainingSamples:]
        y_test = system_model.y[trainingSamples:]
        s_test = system_model.s[trainingSamples:(trainingSamples+x_test.shape[0])] # 数が合わなくなる時があるのでx_sの大きさを合わせる

        # NNの入力に合うように1つのベクトルにする
        test = np.zeros((x_test.shape[0], (2*h_si_len)+(2*h_s_len)))
        test[:, 0:h_si_len] = x_test.real
        test[:, h_si_len:2*(h_si_len)] = x_test.imag
        test[:, 2*(h_si_len):3*(h_si_len)] = y_test.real
        test[:, 3*(h_si_len):4*(h_si_len)] = y_test.imag

        # 学習
        self.nn_history = self.nn.fit(train, [s_train.real, s_train.imag], epochs=n_epochs,
                                      batch_size=batch_size, verbose=2,
                                      validation_data=(test, [s_test.real, s_test.imag]))

        # 学習したモデルを評価
        self.pred = self.nn.predict(test)

        # 推定した希望信号の取り出し
        self.s_hat = np.squeeze(self.pred[0] + 1j * self.pred[1], axis=1)

        # 推定信号をデータへ復調する
        self.d_s_hat = m.demodulate_qpsk(self.s_hat)
        # 元々の外部信号のデータ
        self.d_s_test = system_model.d_s[2 * trainingSamples:2*(trainingSamples+x_test.shape[0])]

        self.error = np.sum(self.d_s_test != self.d_s_hat)
        self.ber = m.check_error(self.d_s_test, self.d_s_hat)
