from src.ofdm_system_model import OFDMSystemModel
from src import modules as m
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as optimizers
# from keras_radam import RAdam
import numpy as np


def select_optimizers(key: str, learning_rate: float = None, momentum: float = None) -> object:
    supports = {
        "momentum": optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        "adam": optimizers.Adam(lr=learning_rate),
        # "radam": RAdam()
    }
    return supports.get(key, "nothing")


class OFDMNNModel:
    system_model: OFDMSystemModel
    model: keras.models.Model
    history: keras.callbacks.History
    pred: np.ndarray
    d_s_test: np.ndarray
    d_s_test_count: int
    s_hat: np.ndarray
    d_s_hat: np.ndarray
    error: np.ndarray

    def __init__(self, n_hidden: list, optimizer_key: str, learning_rate: float, h_si_len: int = 1, h_s_len: int = 1,
                 receive_antenna: int = 1, momentum: float = None):
        input = Input(shape=(2 + 2,))
        n_hidden = n_hidden.copy()  # popだと破壊的操作になり，元々のn_hiddenが壊れるので仕方なくcopyしている
        x = Dense(n_hidden.pop(0), activation='relu')(input)
        for n in n_hidden:
            x = Dense(n, activation='relu')(x)
        output1 = Dense(1, activation='linear')(x)
        output2 = Dense(1, activation='linear')(x)
        model = Model(inputs=input, outputs=[output1, output2])
        optimizer = select_optimizers(optimizer_key, learning_rate, momentum)
        model.compile(optimizer, loss='mse')

        self.model = model

    def learn(self, train_system_model: OFDMSystemModel, test_system_model: OFDMSystemModel, training_ratio: float, n_epochs: int, batch_size: int, h_si_len: int = 1,
              h_s_len: int = 1, receive_antenna: int = 1, delay: int = 0, standardization: bool = False):
        self.train_system_model = train_system_model
        self.test_system_model = test_system_model

        x_train = train_system_model.x.reshape(-1, 1)
        y_train = train_system_model.y.reshape(-1, 1)
        s_train = train_system_model.s_hs_rx.reshape(-1, 1)

        # 標準化
        if standardization is True:
            hensa = np.sqrt(np.var(y_train))
            y_train = y_train / hensa

        # NNの入力に合うように1つのベクトルにする
        train = np.zeros((x_train.shape[0], (2 * h_si_len) + (2)))
        train[:, 0:1] = x_train.real
        train[:, 1:2] = x_train.imag
        train[:, 2:3] = y_train.real
        train[:, 3:4] = y_train.imag

        # テストデータの作成
        x_1 = np.hstack((np.zeros(h_si_len - 1), test_system_model.x))
        x_test = np.array(
            [x_1[i:i + h_si_len] for i in range(test_system_model.x.size)]
        ).reshape(test_system_model.x.size, h_si_len)
        # x_test = test_system_model.x.reshape(-1, 1)
        y_test = test_system_model.y.reshape(-1, 1)
        s_test = test_system_model.s_hs_rx.reshape(-1, 1)  # 数が合わなくなる時があるのでx_sの大きさを合わせる

        # 標準化
        if standardization is True:
            y_test = y_test / hensa

        # NNの入力に合うように1つのベクトルにする
        test = np.zeros((x_test.shape[0], (2 * h_si_len) + (2)))
        test[:, 0:1] = x_test.real
        test[:, 1:2] = x_test.imag
        test[:, 2:3] = y_test.real
        test[:, 3:4] = y_test.imag

        # 学習
        self.history = self.model.fit(train, [s_train.real, s_train.imag], epochs=n_epochs,
                                      batch_size=batch_size, verbose=0,
                                      validation_data=(test, [s_test.real, s_test.imag]))

        # 学習したモデルを評価
        self.pred = self.model.predict(test)

        # 推定した希望信号の取り出し
        y = np.squeeze(self.pred[0] + 1j * self.pred[1], axis=1)
        # y = s_test

        s_hat = test_system_model.demodulate_ofdm(y)

        # 推定信号をデータへ復調する
        self.d_s_hat = m.demodulate_qpsk(s_hat)
        self.d_s_hat = self.d_s_hat.reshape(self.d_s_hat.size, 1)
        
        # 元々の外部信号のデータ
        self.d_s_test = test_system_model.d_s
        self.error = np.sum(self.d_s_test != self.d_s_hat)
        print(self.error / self.d_s_test.size)

    @staticmethod
    def train_bits(bits: int, training_ratio: float) -> int:
        return int(bits * training_ratio)

    @staticmethod
    def test_bits(bits: int, training_ratio: float, h_si_len: int = 1) -> int:
        train_bits = OFDMNNModel.train_bits(bits, training_ratio)
        test_bits = bits - train_bits
        offset = 2 * (- 2 * h_si_len + 2)
        return test_bits + offset
