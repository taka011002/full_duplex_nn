from simulations.snr_ber_average_ibo import Result #pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import logging
from src import modules as m
from src.system_model import SystemModel

if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/non_cancell_snr_ber_average/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
    # dirname = '../results/keep/momentam_double_hidden/merge/'
    os.makedirs(dirname, exist_ok=True)

    formatter = '%(levelname)s : %(asctime)s : %(message)s'
    logging.basicConfig(filename=dirname + '/log.log', level=logging.INFO, format=formatter)
    logging.info('start')
    logging.info(dt_now.strftime("%Y/%m/%d %H:%M:%S"))

    # グラフ
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 22
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    results = []
    pkl_paths = [
        '../results/keep/momentam_batch_nn_5/15_35_45/snr_ber_average_ibo.pkl',
        '../results/keep/momentam_batch_nn_5/15_35_48/snr_ber_average_ibo.pkl',
        # '../results/keep/momentam/04_02_37/snr_ber_average_ibo.pkl',
        # '../results/keep/momentam/04_02_45/snr_ber_average_ibo.pkl',
    ]

    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []
    val_losss_list = []

    for result in results:
        errors_list.append(result.errors)
        val_losss_list.append(result.val_losss)

    errors = np.concatenate(errors_list, 2)
    val_losss = np.concatenate(val_losss_list, 2)

    logging.info('chainload')
    logging.info(len(results))

    params = results[0].params
    logging.info('params')
    logging.info(params)

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    non_cancell_errors = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'] * len(results)))
    for snr_index in range(params['SNR_AVERAGE'] * len(results)):
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

                demodulate = m.demodulate_qpsk(system_model.y)

                error = np.sum(system_model.d_s != demodulate)

                non_cancell_errors[IBO_index][index][snr_index] = error

    # グラフ作成
    # SNR-BERグラフ
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
    y_min = pow(10, 0)
    y_max = pow(10, -4)
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
    ax.grid(linestyle='--')

    # Plot learning curve
    fig_learning = plt.figure(figsize=(8, 6))
    ax_learning = fig_learning.add_subplot(111)
    ax_learning.set_xlabel("Training Epoch")
    ax_learning.set_ylabel(r'$E_n$')
    ax_learning.set_yscale('log')
    ax_learning.grid(which='major', alpha=0.25)
    ax_learning.set_xlim(0, params['nEpochs'] + 1)
    ax_learning.set_xticks(range(1, params['nEpochs']+1, 2))

    train_data = params['n'] - (params['n'] * params['trainingRatio'])
    n_ave = train_data * params['SNR_AVERAGE'] * len(results)

    color_list = ["k", "r", "g", "b", "c", "m", "y", "w"]

    # 7dBのみ表示する
    params['IBO_dB'] = [7]

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(non_cancell_errors[IBO_index], axis=1)
        length = params['n'] * params['SNR_AVERAGE'] * len(results)
        bers = errors_sum / length
        ax.plot(snrs_db, bers, color='k', marker='o', linestyle='--', label="Non cancelled")

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        ax.plot(snrs_db, bers, color='g', marker='o', linestyle='--', label="NN cancelled(hidden node: 5)")

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        val_loss_avg = np.mean(val_losss[IBO_index][-1], axis=0).T
        plt.plot(np.arange(1, len(val_loss_avg) + 1), val_loss_avg, color='g', marker='o', linestyle='--', label='Test Frame (hidden node: 5)')


    pkl_paths = [
        '../results/keep/momentam_batch_nn_15_2/16_19_50/snr_ber_average_ibo.pkl',
        '../results/keep/momentam_batch_nn_15_2/16_19_52/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []
    val_losss_list = []

    for result in results:
        errors_list.append(result.errors)
        val_losss_list.append(result.val_losss)

    errors = np.concatenate(errors_list, 2)
    val_losss = np.concatenate(val_losss_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        ax.plot(snrs_db, bers, color='b', marker='o', linestyle='--', label="NN cancelled(hidden node: 15)")

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        val_loss_avg = np.mean(val_losss[IBO_index][-1], axis=0).T
        plt.plot(np.arange(1, len(val_loss_avg) + 1), val_loss_avg, color='b', marker='o', linestyle='--', label='Test Frame (hidden node: 15)')

    pkl_paths = [
        '../results/keep/momentam_double_hidden_nn_15_full_2/14_47_35/snr_ber_average_ibo.pkl',
        '../results/keep/momentam_double_hidden_nn_15_full_2/14_47_38/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []
    val_losss_list = []

    for result in results:
        errors_list.append(result.errors)
        val_losss_list.append(result.val_losss)

    errors = np.concatenate(errors_list, 2)
    val_losss = np.concatenate(val_losss_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        ax.plot(snrs_db, bers, color='r', marker='o', linestyle='--', label="NN cancelled(hidden node: 15-15)")

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        val_loss_avg = np.mean(val_losss[IBO_index][-1], axis=0).T
        plt.plot(np.arange(1, len(val_loss_avg) + 1), val_loss_avg, color='r', marker='o', linestyle='--', label='Test Frame (hidden node: 15-15)')

    ax.legend(fontsize=16)
    fig.savefig(dirname + '/SNR_BER.pdf')

    ax_learning.legend(fontsize=16)
    fig_learning.savefig(dirname + '/snr_db_25_NNconv.pdf', bbox_inches='tight')
