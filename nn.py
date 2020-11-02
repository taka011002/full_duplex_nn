import modules as m
from system_model import SystemModel
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, SimpleRNN, Dropout
from keras.optimizers import Adam
from keras_radam import RAdam
import matplotlib.pyplot as plt
import os
import datetime

# グラフ
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
dt_now = datetime.datetime.now()
dirname = dt_now.strftime("%Y/%m/%d/%H_%M_%S")
os.makedirs('figures/'+dirname, exist_ok=True)

# パラメータ
n = 2 * 10 ** 4  # サンプルのn数

gamma = 0.3
phi = 3.0

PA_IBO_dB = 5
PA_rho = 2

LNA_IBO_dB = 5
LNA_rho = 2

SNR_MAX = 12
SNR_MIN = 0
sur_num = 12

nHidden = 5
nEpochs = 20
learningRate = 0.004
trainingRatio = 0.9  # 全体のデータ数に対するトレーニングデータの割合
batchSize = 32

# データを生成する
snrs_db = np.linspace(SNR_MIN, SNR_MAX, sur_num)
sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

bers = np.zeros(sur_num)
for index, sigma in enumerate(sigmas):
    system_model = SystemModel(n, sigma, gamma, phi, PA_IBO_dB, PA_rho, LNA_IBO_dB, LNA_rho)

    # NNを生成
    input = Input(shape=(4,))
    hidden1 = Dense(nHidden, activation='relu')(input)
    output1 = Dense(1, activation='linear')(hidden1)
    output2 = Dense(1, activation='linear')(hidden1)
    model = Model(inputs=input, outputs=[output1, output2])
    # adam = Adam(lr=params['learningRate'])
    # model.compile(adam, loss="mse")
    model.compile(RAdam(), loss='mse')

    # トレーニングデータの生成
    trainingSamples = int(np.floor(system_model.x.size * trainingRatio))
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
    history = model.fit(train, [s_train.real, s_train.imag], epochs=nEpochs, batch_size=batchSize, verbose=2,
                        validation_data=(test, [s_test.real, s_test.imag]))

    # 学習したモデルを評価
    pred = model.predict(test)

    # 推定した希望信号の取り出し
    s_hat = np.squeeze(pred[0] + 1j * pred[1], axis=1)
    # 推定信号をデータへ復調する
    d_s_hat = m.demodulate_qpsk(s_hat)
    # 元々の外部信号のデータ
    d_s_test = system_model.d_s[2 * trainingSamples:]

    error = np.sum(d_s_test != d_s_hat)
    ber = error / s_test.size

    bers[index] = ber

    # Plot learning curve
    plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], 'bo-')
    plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['val_loss'], 'ro-')
    plt.ylabel('less')
    plt.xlabel('Training Epoch')
    plt.legend(['Training Frame', 'Test Frame'], loc='lower right')
    plt.grid(which='major', alpha=0.25)
    plt.xlim([0, nEpochs + 1])
    plt.xticks(range(1, nEpochs, 2))
    plt.savefig('figures/'+dirname+'/sigma_'+str(sigma)+'_NNconv.pdf')
    plt.show()


# SNR-BERグラフ
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel(r"SNR (dB)")
ax.set_ylabel(r"$BER$")
ax.set_yscale('log')
ax.set_xlim(SNR_MIN, SNR_MAX)
y_min = pow(10, 0)
y_max = pow(10, -6)
ax.set_ylim(y_max, y_min)
ax.grid(linestyle='--')
ax.legend()

ax.scatter(snrs_db, bers, color="blue")

plt.savefig('figures/'+dirname+'/SNR_BER.pdf')
plt.show()

print("end")