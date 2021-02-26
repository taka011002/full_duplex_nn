from src.previous_research.ofdm_system_model import OFDMSystemModel
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as optimizers
import src.previous_research.fulldeplex as fd
# from keras_radam import RAdam
import numpy as np


def select_optimizers(key: str, learning_rate: float = None, momentum: float = None) -> object:
    supports = {
        "momentum": optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        "Adam": optimizers.Adam(lr=learning_rate),
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

    def __init__(self, n_hidden: list, optimizer_key: str, learning_rate: float, h_si_len: int = 1, momentum: float = None):
        input = Input(shape=((2 * h_si_len),))
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

    def learn(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, n_epochs: int, batch_size: int, h_si_len: int = 1):
        self.h_lin = fd.ls_estimation(train_x, train_y, h_si_len)
        yCanc = fd.si_cancellation_linear(train_x, self.h_lin.flatten(order='F')).reshape((-1, 1), order='F')

        y_train = train_y.reshape((-1, 1), order='F') - yCanc
        self.yVar = np.var(y_train)
        y_train = y_train / np.sqrt(self.yVar)

        x_1 = np.vstack((np.zeros((h_si_len - 1, 1)), train_x.reshape((-1, 1), order='F')))
        x_train = np.array(
            [x_1[i:i + h_si_len] for i in range(train_x.size)]
        ).reshape(train_x.size, h_si_len)

        # NNの入力に合うように1つのベクトルにする
        train = np.zeros((x_train.shape[0], (2 * h_si_len)))
        train[:, 0:(h_si_len)] = x_train.real
        train[:, (h_si_len):(2 * h_si_len)] = x_train.imag

        # テストデータの作成
        yCanc = fd.si_cancellation_linear(test_x, self.h_lin.flatten('F')).reshape((-1, 1), order='F')
        y_test = test_y.reshape((-1, 1)) - yCanc
        y_test = y_test / np.sqrt(self.yVar)

        x_1 = np.vstack((np.zeros((h_si_len - 1, 1)), test_x.reshape((-1, 1), order='F')))
        x_test = np.array(
            [x_1[i:i + h_si_len] for i in range(test_x.size)]
        ).reshape(test_x.size, h_si_len)

        test = np.zeros((x_test.shape[0], (2 * h_si_len)))
        test[:, 0:(h_si_len)] = x_test.real
        test[:, (h_si_len):(2 * h_si_len)] = x_test.imag

        # 学習
        self.history = self.model.fit(train, [y_train.real, y_train.imag], epochs=n_epochs,
                                      batch_size=batch_size, verbose=0,
                                      validation_data=(test, [y_test.real, y_test.imag]))

    def cancel(self, x: np.ndarray, h_si_len):
        self.y_canc_lin = fd.si_cancellation_linear(x, self.h_lin.flatten('F')).reshape((1, -1), order='F')

        x_1 = np.vstack((np.zeros((h_si_len - 1, 1)), x.reshape((-1, 1), order='F')))
        nn_x = np.array(
            [x_1[i:i + h_si_len] for i in range(x.size)]
        ).reshape(x.size, h_si_len)

        # NNの入力に合うように1つのベクトルにする
        x_pred = np.zeros((nn_x.shape[0], (2 * h_si_len)))
        x_pred[:, 0:(h_si_len)] = nn_x.real
        x_pred[:, (h_si_len):(2 * h_si_len)] = nn_x.imag

        # 学習したモデルを評価
        self.pred = self.model.predict(x_pred)
        self.y_canc_non_lin = np.squeeze(self.pred[0] + 1j * self.pred[1], axis=1)

        self.y_hat = self.y_canc_lin + (np.sqrt(self.yVar) * self.y_canc_non_lin)