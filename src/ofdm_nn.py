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
        "adam": optimizers.Adam(lr=learning_rate)
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
        input = Input(shape=((2 * h_si_len) + (2 * receive_antenna * h_s_len),))
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

    def learn(self, system_model: OFDMSystemModel, training_ratio: float, n_epochs: int, batch_size: int, h_si_len: int = 1,
              h_s_len: int = 1, receive_antenna: int = 1, delay: int = 0, standardization: bool = False):
        self.system_model = system_model

        # トレーニングデータの生成
        training_samples = int(np.floor(system_model.x.size) * training_ratio)

        # チャネル数分つくる
        x = np.reshape(
            np.array([system_model.x[i:i + h_si_len] for i in range(system_model.x.size - 2 * h_si_len + 2)]),
            (system_model.x.size - 2 * h_si_len + 2, h_si_len))
        y = np.reshape(
            np.array([system_model.y[i:i + h_s_len] for i in range(system_model.y.size - 2 * h_s_len + 2)]),
            (system_model.y.size - 2 * h_s_len + 2, (h_s_len * receive_antenna)))

        x_train = x[0:training_samples]
        y_train = y[0:training_samples]
        s_train = system_model.s[0 + delay:training_samples + delay]  # 遅延をとる

        # 標準化
        if standardization is True:
            hensa = np.sqrt(np.var(y_train))
            y_train = y_train / hensa

        # NNの入力に合うように1つのベクトルにする
        train = np.zeros((x_train.shape[0], (2 * h_si_len) + (receive_antenna * 2 * h_si_len)))
        train[:, 0:h_si_len] = x_train.real
        train[:, h_si_len:(2 * h_si_len)] = x_train.imag
        train[:, (2 * h_si_len):(2 * h_si_len) + (receive_antenna * h_s_len)] = y_train.real
        train[:,
        (2 * h_si_len) + (receive_antenna * h_s_len):(2 * h_si_len) + (receive_antenna * 2 * h_s_len)] = y_train.imag

        # テストデータの作成
        x_test = x[training_samples:]
        y_test = y[training_samples:]
        s_test = system_model.s[
                 training_samples + delay:(training_samples + x_test.shape[0] + delay)]  # 数が合わなくなる時があるのでx_sの大きさを合わせる

        # 標準化
        if standardization is True:
            y_test = y_test / hensa

        # NNの入力に合うように1つのベクトルにする
        test = np.zeros((x_test.shape[0], (2 * h_si_len) + (receive_antenna * 2 * h_si_len)))
        test[:, 0:h_si_len] = x_test.real
        test[:, h_si_len:(2 * h_si_len)] = x_test.imag
        test[:, (2 * h_si_len):(2 * h_si_len) + (receive_antenna * h_s_len)] = y_test.real
        test[:,
        (2 * h_si_len) + (receive_antenna * h_s_len):(2 * h_si_len) + (receive_antenna * 2 * h_s_len)] = y_test.imag

        # 学習
        self.history = self.model.fit(train, [s_train.real, s_train.imag], epochs=n_epochs,
                                      batch_size=batch_size, verbose=0,
                                      validation_data=(test, [s_test.real, s_test.imag]))

        # 学習したモデルを評価
        self.pred = self.model.predict(test)

        # 推定した希望信号の取り出し
        s_hat = np.squeeze(self.pred[0] + 1j * self.pred[1], axis=1)

        # 推定信号をデータへ復調する
        self.d_s_hat = m.demodulate_qpsk(s_hat)
        self.d_s_hat = self.d_s_hat.reshape(self.d_s_hat.size, 1)
        
        # 元々の外部信号のデータ
        self.d_s_test = system_model.d_s[
                        2 * (training_samples + delay):2 * (training_samples + x_test.shape[0] + delay)]
        self.error = np.sum(self.d_s_test != self.d_s_hat)

    @staticmethod
    def train_bits(bits: int, training_ratio: float) -> int:
        return int(bits * training_ratio)

    @staticmethod
    def test_bits(bits: int, training_ratio: float, h_si_len: int = 1) -> int:
        train_bits = OFDMNNModel.train_bits(bits, training_ratio)
        test_bits = bits - train_bits
        offset = 2 * (- 2 * h_si_len + 2)
        return test_bits + offset
