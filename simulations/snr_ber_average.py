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
    dirname = '../results/snr_ber_average/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
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
    np.random.seed(0)

    # パラメータ
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'PA_IBO_dB': 7,
        'PA_rho': 2,

        'LNA_IBO_dB': 7,
        'LNA_rho': 2,

        'SNR_MIN': 0,
        'SNR_MAX': 20,
        'SNR_NUM': 2,
        'SNR_AVERAGE': 1,

        'nHidden': 5,
        'nEpochs': 40,
        # 'learningRate': 0.004,
        'trainingRatio': 0.8,  # 全体のデータ数に対するトレーニングデータの割合
        'batchSize': 32,
    }
    logging.info('params')
    logging.info(params)

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))
    losss = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_losss = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

    # 実行時の時間を記録する
    start = time.time()

    for snr_index in range(params['SNR_AVERAGE']):
        # 通信路は毎回生成する
        h_si = m.channel()
        h_s = m.channel()
        logging.info('random channel')
        logging.info('h_si:{0.real}+{0.imag}i'.format(h_si))
        logging.info('h_s:{0.real}+{0.imag}i'.format(h_s))

        for index, sigma in enumerate(sigmas):
            logging.info("SNR_AVERAGE_index:" + str(snr_index))
            logging.info("sigma_index:" + str(index))
            logging.info("time: %d[sec]" % int(time.time() - start))
            system_model = SystemModel(
                params['n'],
                sigma,
                params['gamma'],
                params['phi'],
                params['PA_IBO_dB'],
                params['PA_rho'],
                params['LNA_IBO_dB'],
                params['LNA_rho'],
                h_si,
                h_s,
            )

            # NNを生成
            model = NNModel(params['nHidden'])
            model.learn(system_model, params['trainingRatio'], params['nEpochs'], params['batchSize'])

            errors[index][snr_index] = model.error
            losss[index][snr_index][:] = model.history.history['loss']
            val_losss[index][snr_index][:] = model.history.history['val_loss']

    logging.info("learn_end_time: %d[sec]" % int(time.time() - start))
    # 結果をdumpしておく
    result = Result(params, errors, losss, val_losss)
    with open(dirname + '/snr_ber_average.pkl', 'wb') as f:
        pickle.dump(result, f)

    # SNR-BERグラフ
    errors_sum = np.sum(errors, axis=1)
    n_ave = params['n'] * params['SNR_AVERAGE']
    bers = errors_sum / n_ave

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

    ax.plot(snrs_db, bers, color="black", marker='o', linestyle='--', )

    plt.savefig(dirname + '/SNR_BER.pdf')

    # # Plot learning curve
    for index, snrs_db in enumerate(snrs_db):
        plt.figure()
        loss_avg = np.mean(losss[index], axis=0).T
        val_loss_avg = np.mean(val_losss[index], axis=0).T

        plt.xticks(np.arange(0, params['nEpochs'] + 1, 5))
        plt.plot(np.arange(1, len(loss_avg) + 1), loss_avg, 'bo-')
        plt.plot(np.arange(1, len(loss_avg) + 1), val_loss_avg, 'ro-')
        plt.ylabel('less')
        plt.yscale('log')
        plt.xlabel('Training Epoch')
        plt.legend(['Training Frame', 'Test Frame'], loc='lower right')
        plt.grid(which='major', alpha=0.25)
        plt.xlim([0, params['nEpochs'] + 1])
        plt.xticks(range(1, params['nEpochs'], 2))
        plt.savefig(dirname + '/snr_db_' + str(snrs_db) + '_NNconv.pdf', bbox_inches='tight')

    logging.info("end")
