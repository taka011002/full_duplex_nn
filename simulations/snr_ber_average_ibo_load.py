from simulations.snr_ber_average_ibo import Result #pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import logging

if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/snr_ber_average_ibo_load/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
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

    pkl = '../results/2020/11/04/23_15_48/snr_ber_average_ibo.pkl'
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

    n_ave = params['n'] * params['SNR_AVERAGE']

    errors_sum_7 = np.sum(errors[0], axis=1)
    bers_7 = errors_sum_7 / n_ave
    ax.plot(snrs_db, bers_7, color="blue", marker='o', linestyle='--', label="IBO=7[dB]")

    errors_sum_5 = np.sum(errors[1], axis=1)
    bers_5 = errors_sum_5 / n_ave
    ax.plot(snrs_db, bers_5, color="red", marker='v', linestyle='--', label="IBO=5[dB]")

    errors_sum_3 = np.sum(errors[2], axis=1)
    bers_3 = errors_sum_3 / n_ave
    ax.plot(snrs_db, bers_3, color="green", marker='s', linestyle='--', label="IBO=3[dB]")
    ax.legend()
    plt.savefig(dirname + '/SNR_BER.pdf')

    # # Plot learning curve
    for index, snrs_db in enumerate(snrs_db):
        plt.figure()

        loss_avg_7 = np.mean(losss[0][index], axis=0).T
        val_loss_avg_7 = np.mean(val_losss[0][index], axis=0).T
        plt.plot(np.arange(1, len(loss_avg_7) + 1), loss_avg_7, 'bo-', label='Training Frame (IBO=7[dB])')
        plt.plot(np.arange(1, len(loss_avg_7) + 1), val_loss_avg_7, 'go-', label='Test Frame (IBO=7[dB])')

        loss_avg_5 = np.mean(losss[1][index], axis=0).T
        val_loss_avg_5 = np.mean(val_losss[1][index], axis=0).T
        plt.plot(np.arange(1, len(loss_avg_5) + 1), loss_avg_5, 'ro-', label='Training Frame (IBO=5[dB])')
        plt.plot(np.arange(1, len(loss_avg_5) + 1), val_loss_avg_5, 'co-', label='Test Frame (IBO=5[dB])')

        loss_avg_3 = np.mean(losss[2][index], axis=0).T
        val_loss_avg_3 = np.mean(val_losss[2][index], axis=0).T
        plt.plot(np.arange(1, len(loss_avg_3) + 1), loss_avg_3, 'mo-', label='Training Frame (IBO=3[dB])')
        plt.plot(np.arange(1, len(loss_avg_3) + 1), val_loss_avg_3, 'yo-', label='Test Frame (IBO=3[dB])')

        plt.ylabel('less')
        plt.yscale('log')
        plt.xlabel('Training Epoch')
        plt.legend()
        plt.grid(which='major', alpha=0.25)
        plt.xlim([0, params['nEpochs'] + 1])
        plt.xticks(range(1, params['nEpochs'], 2))
        plt.savefig(dirname + '/snr_db_' + str(snrs_db) + '_NNconv.pdf', bbox_inches='tight')