import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as optimizers
import numpy as np
import src.previous_research.fulldeplex as fd
from keras_radam import RAdam



class NNModel:
    model: keras.models.Model
    history: keras.callbacks.History
    pred: np.ndarray
    y_canc_non_lin: np.ndarray

    def __init__(self, chan_len: int, n_hidden: int, learning_rate: float):
        input = Input(shape=(2 * chan_len,))  # 13*2=26個
        hidden1 = Dense(n_hidden, activation='relu')(input)
        output1 = Dense(1, activation='linear')(hidden1)
        output2 = Dense(1, activation='linear')(hidden1)
        model = Model(inputs=input, outputs=[output1, output2])
        # optimizer = optimizers.Adam(lr=learning_rate)
        optimizer = RAdam()
        # optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.8)
        model.compile(optimizer, loss="mse")

        self.model = model

    def learn(self, x: np.ndarray, y: np.ndarray, training_ratio: float, chanLen: int, epochs: int, batch_size: int):
        training_samples = int(np.floor(x.size * training_ratio))
        x_train = x[0:training_samples]
        y_train = y[0:training_samples]
        x_test = x[training_samples:]
        y_test = y[training_samples:]

        # Step 1: Estimate linear cancellation arameters and perform linear cancellation
        self.h_lin = fd.ls_estimation(x_train, y_train, chanLen)
        yCanc = fd.si_cancellation_linear(x_train, self.h_lin)

        # Normalize data for NN
        y_train = y_train - yCanc
        yVar = np.var(y_train)
        y_train = y_train / np.sqrt(yVar)

        # Prepare training data for NN
        x_train_real = np.reshape(np.array([x_train[i:i + chanLen].real for i in range(x_train.size - chanLen)]),
                                  (x_train.size - chanLen, chanLen))
        x_train_imag = np.reshape(np.array([x_train[i:i + chanLen].imag for i in range(x_train.size - chanLen)]),
                                  (x_train.size - chanLen, chanLen))
        x_train = np.zeros((x_train.size - chanLen, 2 * chanLen))
        x_train[:, 0:chanLen] = x_train_real
        x_train[:, chanLen:2 * chanLen] = x_train_imag
        y_train = np.reshape(y_train[chanLen:], (y_train.size - chanLen, 1))

        # Prepare test data for NN
        yCanc = fd.si_cancellation_linear(x_test, self.h_lin)
        y_test = y_test - yCanc
        y_test = y_test / np.sqrt(yVar)

        x_test_real = np.reshape(np.array([x_test[i:i + chanLen].real for i in range(x_test.size - chanLen)]),
                                 (x_test.size - chanLen, chanLen))
        x_test_imag = np.reshape(np.array([x_test[i:i + chanLen].imag for i in range(x_test.size - chanLen)]),
                                 (x_test.size - chanLen, chanLen))
        x_test = np.zeros((x_test.size - chanLen, 2 * chanLen))
        x_test[:, 0:chanLen] = x_test_real
        x_test[:, chanLen:2 * chanLen] = x_test_imag
        y_test = np.reshape(y_test[chanLen:], (y_test.size - chanLen, 1))

        ##### Training #####
        # Step 2: train NN to do non-linear cancellation
        self.history = self.model.fit(x_train, [y_train.real, y_train.imag], epochs=epochs, batch_size=batch_size,
                            verbose=0, validation_data=(x_test, [y_test.real, y_test.imag]))

    def cancel(self, x: np.ndarray, y: np.ndarray, chanLen: int):
        # Prepare test data for NN
        x_real = np.reshape(np.array([x[i:i + chanLen].real for i in range(x.size - chanLen)]),
                                 (x.size - chanLen, chanLen))
        x_imag = np.reshape(np.array([x[i:i + chanLen].imag for i in range(x.size - chanLen)]),
                                 (x.size - chanLen, chanLen))
        x_pred = np.zeros((x.size - chanLen, 2 * chanLen))
        x_pred[:, 0:chanLen] = x_real
        x_pred[:, chanLen:2 * chanLen] = x_imag

        # Normalize data for NN
        yCanc = fd.si_cancellation_linear(x, self.h_lin)
        y_lin_canc = y - yCanc

        self.pred = self.model.predict(x_pred)
        self.y_canc_non_lin = np.squeeze(self.pred[0] + 1j * self.pred[1], axis=1)

        self.cancelled_y = y_lin_canc[0:self.y_canc_non_lin.shape[0]] - self.y_canc_non_lin
