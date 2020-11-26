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
        '../results/snr_ber_average_ibo/2020/11/26/01_00_42/snr_ber_average_ibo.pkl',
        # '../results/keep/momentam_batch_nn_5/15_35_48/snr_ber_average_ibo.pkl',
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

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 2)

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

                demodulate = m.demodulate_qpsk(system_model.y.squeeze())

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
    y_max = pow(10, -6)
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
    ax.grid(linestyle='--')

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
        ax.plot(snrs_db, bers, color='b', marker='o', linestyle='--', label="NN cancelled(receive antenna:1)")

    ## ちゃんと調整してforループにする
    pkl_paths = [
        '../results/snr_ber_average_ibo/2020/11/25/22_25_06/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        np.place(bers, bers == 0, None)
        ax.plot(snrs_db, bers, color='y', marker='o', linestyle='--', label="NN cancelled(receive antenna:2)")

    pkl_paths = [
        '../results/snr_ber_average_ibo/2020/11/25/22_25_24/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        np.place(bers, bers == 0, None)
        ax.plot(snrs_db, bers, color='m', marker='o', linestyle='--', label="NN cancelled(receive antenna:3)")

    pkl_paths = [
        '../results/snr_ber_average_ibo/2020/11/25/22_25_40/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        np.place(bers, bers == 0, None)
        ax.plot(snrs_db, bers, color='r', marker='o', linestyle='--', label="NN cancelled(receive antenna:4)")


    pkl_paths = [
        '../results/snr_ber_average_ibo/2020/11/25/20_11_41/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        np.place(bers, bers == 0, None)
        ax.plot(snrs_db, bers, color='g', marker='o', linestyle='--', label="NN cancelled(receive antenna:5)")

    pkl_paths = [
        '../results/snr_ber_average_ibo/2020/11/25/20_11_47/snr_ber_average_ibo.pkl',
    ]

    results = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 2)

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        np.place(bers, bers == 0, None)
        ax.plot(snrs_db, bers, color='c', marker='o', linestyle='--', label="NN cancelled(receive antenna:10)")

    ax.legend(fontsize=12)
    plt.savefig(dirname + '/SNR_BER.pdf')