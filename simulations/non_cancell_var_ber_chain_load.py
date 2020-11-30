from simulations.snr_ber_average_ibo import Result #pklを使うのに必要
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
import json
from tqdm import tqdm

if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/var_ber_chain_load/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
    # dirname = '../results/keep/momentam_batch_nn_15_full_2/merge/'
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
        '../results/var_ber/2020/11/19/17_14_32/snr_ber_average_ibo.pkl',
    ]

    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            logging.info("loaded_pkl: %s" % pkl_path)
            result = pickle.load(f)
            results.append(result)

    # 結合させる
    errors_list = []
    losss_list = []
    val_losss_list = []

    for result in results:
        errors_list.append(result.errors)
        losss_list.append(result.losss)
        val_losss_list.append(result.val_losss)

    errors = np.concatenate(errors_list, 1)
    losss = np.concatenate(losss_list, 1)
    val_losss = np.concatenate(val_losss_list, 1)

    logging.info('chainload')
    logging.info(len(results))

    params = results[0].params
    logging.info('params')
    logging.info(params)

    # データを生成する
    snr = params['snr']
    sigma = m.sigmas(snr)  # SNR(dB)を元に雑音電力を導出
    params['var_min'] = -100
    var_si_var_s_db = np.linspace(params['var_min'], params['var_max'], params['var_num'])
    var_si_var_s = m.to_exact_number(var_si_var_s_db)
    var_s = 1
    var_si_list = var_si_var_s * var_s
    non_cancell_error = np.zeros((params['var_num'], params['number_of_trials']))
    for trials_index in tqdm(range(params['number_of_trials'])):
        for var_si_index, var_si in enumerate(var_si_list):
            # 通信路は毎回生成する
            h_si = m.channel(1, params['h_si_len'], var_si)
            h_s = m.channel(1, params['h_s_len'], var_si)
            logging.info('random channel')
            logging.info('h_si:{0.real}+{0.imag}i'.format(h_si))
            logging.info('h_s:{0.real}+{0.imag}i'.format(h_s))

            logging.info("number_of_trials_index:" + str(trials_index))
            logging.info("var_s_index:" + str(var_si_index))
            system_model = SystemModel(
                params['n'],
                sigma,
                params['gamma'],
                params['phi'],
                params['PA_IBO_db'],
                params['PA_rho'],
                1000,
                1000,
                h_si,
                h_s,
                params['h_si_len'],
                params['h_s_len'],
            )

            demodulate = m.demodulate_qpsk(system_model.y.reshape(system_model.y.shape[0]))
            error = np.sum(system_model.d_s != demodulate)
            non_cancell_error[var_si_index][trials_index] = error

    # SNR-BERグラフ
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\sigma_{\rm SI}^2 / \sigma_{\rm s}^2$(dB)")
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    y_min = pow(10, 0)
    y_max = pow(10, -6)
    ax.set_ylim(y_max, y_min)
    # ax.set_xlim(params['var_min'], params['var_max'])
    ax.set_xlim(params['var_min'], params['var_max'])
    ax.grid(linestyle='--')

    train_data = params['n'] - (params['n'] * params['trainingRatio'])
    train_data = train_data - params['h_si_len'] + 1
    n_ave = train_data * params['number_of_trials'] * len(results)

    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_ave
    non_cancell_error_sum = np.sum(non_cancell_error, axis=1)
    non_cancell_error = non_cancell_error_sum / (params['n'] * params['number_of_trials'])
    ax.plot(var_si_var_s_db, non_cancell_error, color='k', marker='o', linestyle='--', label="Non cancelled")
    # ax.plot(var_si_var_s_db, bers, color='r', marker='o', linestyle='--', label="NN cancelled")

    ax.legend()
    plt.savefig(dirname + '/var_BER.pdf', bbox_inches='tight')

    # Plot learning curve
    # for var_s_index, var_s in enumerate(var_si_var_s_db):
    #     plt.figure()
    #
    #     loss_avg = np.mean(losss[var_s_index], axis=0).T
    #     val_loss_avg = np.mean(val_losss[var_s_index], axis=0).T
    #     plt.plot(np.arange(1, len(loss_avg) + 1), loss_avg, color="b", marker='o', linestyle='--',
    #              label='Training Frame')
    #     plt.plot(np.arange(1, len(loss_avg) + 1), val_loss_avg, color="r",
    #              marker='o', linestyle='--', label='Test Frame')
    #
    #     plt.ylabel('less')
    #     plt.yscale('log')
    #     plt.xlabel('Training Epoch')
    #     plt.legend()
    #     plt.grid(which='major', alpha=0.25)
    #     plt.xlim([0, params['nEpochs'] + 1])
    #     plt.xticks(range(1, params['nEpochs'], 2))
    #     plt.savefig(dirname + '/var_si_var_s_db_' + str(var_s) + '_NNconv.pdf', bbox_inches='tight')