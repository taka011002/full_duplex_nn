from simulations.snr_ber_average import Result #pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import logging

if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/snr_ber_average_load/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
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

    pkl = '../results/snr_ber_average/2020/11/05/00_11_56/snr_ber_average.pkl'
    with open(pkl, 'rb') as f:
        result = pickle.load(f)

    logging.info("loaded_pkl: %s" % pkl)

    params = result.params
    logging.info('params')
    logging.info(params)

    errors = result.errors
    losss = result.losss
    val_losss = result.val_losss

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])

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