from src import modules as m
from src.system_model import SystemModel
from src.nn import NNModel
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import pickle
import logging


class Result:
    def __init__(self, params, errors, losss, val_losss):
        self.params = params
        self.errors = errors
        self.losss = losss
        self.val_losss = val_losss


if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/non_cancell_snr_ber_average/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
    os.makedirs(dirname, exist_ok=True)

    formatter = '%(levelname)s : %(asctime)s : %(message)s'
    logging.basicConfig(filename=dirname + '/log.log', level=logging.INFO, format=formatter)
    logging.info('start')
    logging.info(dt_now.strftime("%Y/%m/%d %H:%M:%S"))

    # グラフ
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.size"] = 22
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # seed固定
    # np.random.seed(0)

    # パラメータ
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'rho': 2,
        'IBO_dB': [7],

        'SNR_MIN': 0,
        'SNR_MAX': 20,
        'SNR_NUM': 5,
        'SNR_AVERAGE': 100,

        'nHidden': 5,
        'nEpochs': 20,
        # 'learningRate': 0.004,
        'trainingRatio': 0.8,  # 全体のデータ数に対するトレーニングデータの割合
        'batchSize': 32,
    }
    logging.info('params')
    logging.info(params)

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    errors = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE']))
    losss = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_losss = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

    # 実行時の時間を記録する
    start = time.time()

    for snr_index in range(params['SNR_AVERAGE']):
        # 通信路は毎回生成する
        h_si = m.channel()
        h_s = m.channel()
        logging.info('random channel')
        logging.info('h_si:{0.real}+{0.imag}i'.format(h_si))
        logging.info('h_s:{0.real}+{0.imag}i'.format(h_s))

        for IBO_index, IBO_dB in enumerate(params['IBO_dB']):
            for index, sigma in enumerate(sigmas):
                logging.info("IBO_dB_index:" + str(IBO_index))
                logging.info("SNR_AVERAGE_index:" + str(snr_index))
                logging.info("sigma_index:" + str(index))
                logging.info("time: %d[sec]" % int(time.time() - start))
                system_model = SystemModel(
                    params['n'],
                    sigma,
                    params['gamma'],
                    params['phi'],
                    IBO_dB,
                    params['rho'],
                    IBO_dB,
                    params['rho'],
                    h_si,
                    h_s,
                )

                demodulate = m.demodulate_qpsk(system_model.y.squeeze())

                error = np.sum(system_model.d_s != demodulate)

                errors[IBO_index][index][snr_index] = error

    logging.info("learn_end_time: %d[sec]" % int(time.time() - start))
    # 結果をdumpしておく
    result = Result(params, errors, losss, val_losss)
    with open(dirname + '/snr_ber_average_ibo.pkl', 'wb') as f:
        pickle.dump(result, f)

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

    # train_data = params['n'] - (params['n'] * params['trainingRatio'])
    # n_ave = train_data * params['SNR_AVERAGE']

    n_ave = params['n'] * params['SNR_AVERAGE']

    color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        ax.plot(snrs_db, bers, color=color_list[IBO_index], marker='o', linestyle='--', label="IBO=%d[dB]" % IBO_db)

    ax.legend()
    plt.savefig(dirname + '/SNR_BER.pdf')

    logging.info("end")
