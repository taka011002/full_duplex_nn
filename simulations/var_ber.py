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


class Result:
    params: dict
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray
    nn_models: list

    def __init__(self, params, errors, losss, val_losss, nn_models=None):
        self.params = params
        self.errors = errors
        self.losss = losss
        self.val_losss = val_losss
        self.nn_models = nn_models


def init_logger(dirname: str):
    formatter = '%(levelname)s : %(asctime)s : %(message)s'
    logging.basicConfig(filename=dirname + '/log.log', level=logging.INFO, format=formatter)


if __name__ == '__main__':
    # シミュレーション結果の保存先を作成する
    dt_now = datetime.datetime.now()
    dirname = '../results/var_ber/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
    os.makedirs(dirname, exist_ok=True)

    init_logger(dirname)
    logging.info('start')
    logging.info(dt_now.strftime("%Y/%m/%d %H:%M:%S"))

    # グラフ
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.size"] = 22
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # パラメータ
    with open('configs/var_ber.json') as f:
        params = json.load(f)
        logging.info('params')
        logging.info(params)

    # データを生成する
    snr = params['snr']
    sigma = m.sigmas(snr)  # SNR(dB)を元に雑音電力を導出

    var_si_var_s_db = np.linspace(params['var_min'], params['var_max'], params['var_num'])
    var_si_var_s = m.to_exact_number(var_si_var_s_db)
    var_si = 1
    var_s_list = var_si / var_si_var_s

    errors = np.zeros((params['var_num'], params['number_of_trials']))
    losss = np.zeros((params['var_num'], params['number_of_trials'], params['nEpochs']))
    val_losss = np.zeros((params['var_num'], params['number_of_trials'], params['nEpochs']))

    # 実行時の時間を記録する
    start = time.time()

    nn_models = [[None] * params['number_of_trials'] for i in range(params['var_num'])]
    for trials_index in tqdm(range(params['number_of_trials'])):
        for var_s_index, var_s in enumerate(var_s_list):
            # 通信路は毎回生成する
            h_si = []
            h_s = []
            for i in range(params['receive_antenna']):
                h_si.append(m.channel(1, params['h_si_len'], var_si))
                h_s.append(m.channel(1, params['h_s_len'], var_s))
            logging.info('random channel')

            logging.info("number_of_trials_index:" + str(trials_index))
            logging.info("var_s_index:" + str(var_s_index))
            logging.info("time: %d[sec]" % int(time.time() - start))
            system_model = SystemModel(
                params['n'],
                sigma,
                params['gamma'],
                params['phi'],
                params['PA_IBO_db'],
                params['PA_rho'],
                params['LNA_IBO_dB'],
                params['LNA_rho'],
                h_si,
                h_s,
                params['h_si_len'],
                params['h_s_len'],
                params['receive_antenna'],
            )

            # NNを生成
            nn_model = NNModel(
                params['nHidden'],
                params['learningRate'],
                params['h_si_len'],
                params['h_s_len'],
                params['receive_antenna'],

            )

            nn_model.learn(
                system_model,
                params['trainingRatio'],
                params['nEpochs'],
                params['batchSize'],
                params['h_si_len'],
                params['h_s_len'],
                params['receive_antenna'],
            )

            errors[var_s_index][trials_index] = nn_model.error
            losss[var_s_index][trials_index][:] = nn_model.history.history['loss']
            val_losss[var_s_index][trials_index][:] = nn_model.history.history['val_loss']
            # 学習済みモデルはpklできないので削除する．
            del nn_model.model
            del nn_model.history
            nn_models[var_s_index][trials_index] = nn_model

    logging.info("learn_end_time: %d[sec]" % int(time.time() - start))
    # 結果をdumpしておく
    result = Result(params, errors, losss, val_losss, None)
    with open(dirname + '/snr_ber_average_ibo.pkl', 'wb') as f:
        pickle.dump(result, f)

    # パラメータはわかりやすいように別
    with open(dirname + '/params.json', 'w') as f:
        json.dump(params, f, indent=4)

    # var-BERグラフ
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\sigma_{\rm SI}^2 / \sigma_{\rm s}^2$(dB)")
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    y_min = pow(10, 0)
    y_max = pow(10, -6)
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(params['var_min'], params['var_max'])
    ax.grid(linestyle='--')

    train_data = params['n'] - (params['n'] * params['trainingRatio'])
    train_data = train_data - params['h_si_len'] + 1
    n_ave = train_data * params['number_of_trials']

    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_ave
    ax.plot(var_si_var_s_db, bers, color='k', marker='o', linestyle='--')

    ax.legend()
    plt.savefig(dirname + '/var_BER.pdf')

    # Plot learning curve
    for var_s_index, var_s in enumerate(var_si_var_s_db):
        plt.figure()

        loss_avg = np.mean(losss[var_s_index], axis=0).T
        val_loss_avg = np.mean(val_losss[var_s_index], axis=0).T
        plt.plot(np.arange(1, len(loss_avg) + 1), loss_avg, color="b", marker='o', linestyle='--',
                 label='Training Frame')
        plt.plot(np.arange(1, len(loss_avg) + 1), val_loss_avg, color="r",
                 marker='o', linestyle='--', label='Test Frame')

        plt.ylabel('less')
        plt.yscale('log')
        plt.xlabel('Training Epoch')
        plt.legend()
        plt.grid(which='major', alpha=0.25)
        plt.xlim([0, params['nEpochs'] + 1])
        plt.xticks(range(1, params['nEpochs'], 2))
        plt.savefig(dirname + '/var_si_var_s_db_' + str(var_s) + '_NNconv.pdf', bbox_inches='tight')

    logging.info("end")
