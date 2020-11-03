from src import modules as m
from src.system_model import SystemModel
from src import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
    os.makedirs(dirname, exist_ok=True)
    result_txt = open(dirname+'/result.txt', mode='w')
    print('start', file=result_txt)
    print(dt_now.strftime("%Y/%m/%d %H:%M:%S"), file=result_txt)

    # グラフ
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 22
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # パラメータ
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'PA_IBO_dB': 3,
        'PA_rho': 2,

        'LNA_IBO_dB': 3,
        'LNA_rho': 2,

        'SNR_MIN': 0,
        'SNR_MAX': 12,
        'sur_num': 12,

        'nHidden': 5,
        'nEpochs': 20,
        # 'learningRate': 0.004,
        'trainingRatio': 0.8,  # 全体のデータ数に対するトレーニングデータの割合
        'batchSize': 32
    }
    print('params', file=result_txt)
    print(params, file=result_txt)

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['sur_num'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    # h_si = m.channel()
    # h_s = m.channel()
    # print('h_si:{0.real}+{0.imag}i'.format(h_si), file=result_txt)
    # print('h_s:{0.real}+{0.imag}i'.format(h_s), file=result_txt)
    print('random channels', file=result_txt)

    bers = np.zeros(params['sur_num'])
    for index, sigma in enumerate(sigmas):
        system_model = SystemModel(
            params['n'],
            sigma,
            params['gamma'],
            params['phi'],
            params['PA_IBO_dB'],
            params['PA_rho'],
            params['LNA_IBO_dB'],
            params['LNA_rho'],
            # h_si,
            # h_s,
        )

        # NNを生成
        model = nn.nn_model(params['nHidden'])

        # トレーニングデータの生成
        trainingSamples = int(np.floor(system_model.x.size * params['trainingRatio']))
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
        history = model.fit(train, [s_train.real, s_train.imag], epochs=params['nEpochs'], batch_size=params['batchSize'], verbose=2,
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
        plt.figure()
        plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], 'bo-')
        plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['val_loss'], 'ro-')
        plt.ylabel('less')
        plt.xlabel('Training Epoch')
        plt.legend(['Training Frame', 'Test Frame'], loc='lower right')
        plt.grid(which='major', alpha=0.25)
        plt.xlim([0, params['nEpochs'] + 1])
        plt.xticks(range(1, params['nEpochs'], 2))
        plt.savefig(dirname + '/sigma_' + str(snrs_db[index]) + '_NNconv.pdf', bbox_inches='tight')

    # SNR-BERグラフ
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
    y_min = pow(10, 0)
    y_max = pow(10, -6)
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
    ax.grid(linestyle='--')

    ax.scatter(snrs_db, bers, color="blue")

    plt.savefig(dirname + '/SNR_BER.pdf')

    print("end", file=result_txt)
    result_txt.close()
