from src import modules as m
from simulations.common import slack
from simulations.common import settings
from src.system_model import SystemModel
from src.nn import NNModel
import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    SIMULATIONS_NAME = 'snr_ber_average_ibo'
    dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    settings.init_simulation_output(dirname)

    # seed固定
    # np.random.seed(0)

    # パラメータ
    with open('configs/snr_ber_average_ibo.json') as f:
        params = json.load(f)
        logging.info(params)

    # パラメータはわかりやすいように別
    with open(dirname + '/params.json', 'w') as f:
        json.dump(params, f, indent=4)

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    errors = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE']))
    losss = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_losss = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

    # 実行時の時間を記録する
    start = time.time()

    # nn_models = [[[None] * params['SNR_AVERAGE'] for i in range(params['SNR_NUM'])] for j in
    #              range(len(params['IBO_dB']))]
    for trials_index in tqdm(range(params['SNR_AVERAGE'])):
        # 通信路は毎回生成する
        h_si = []
        h_s = []
        for i in range(params['receive_antenna']):
            h_si.append(m.channel(1, params['h_si_len']))
            h_s.append(m.channel(1, params['h_s_len']))

        for IBO_index, IBO_dB in enumerate(params['IBO_dB']):
            for sigma_index, sigma in enumerate(sigmas):
                logging.info("IBO_dB_index:" + str(IBO_index))
                logging.info("SNR_AVERAGE_index:" + str(trials_index))
                logging.info("sigma_index:" + str(sigma_index))
                logging.info("time: %d[sec]" % int(time.time() - start))
                system_model = SystemModel(
                    params['n'],
                    sigma,
                    params['gamma'],
                    params['phi'],
                    IBO_dB,
                    params['rho'],
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

                errors[IBO_index][sigma_index][trials_index] = nn_model.error
                losss[IBO_index][sigma_index][trials_index][:] = nn_model.history.history['loss']
                val_losss[IBO_index][sigma_index][trials_index][:] = nn_model.history.history['val_loss']
                # 学習済みモデルはpklできないので削除する．
                # del nn_model.model
                # del nn_model.history
                # nn_models[IBO_index][sigma_index][trials_index] = nn_model

    logging.info("learn_end_time: %d[sec]" % int(time.time() - start))
    # 結果をdumpしておく
    # result = Result(params, errors, losss, val_losss, nn_models)
    # with open(dirname + '/snr_ber_average_ibo.pkl', 'wb') as f:
    #     pickle.dump(result, f)

    result = Result(params, errors, losss, val_losss, None)
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

    train_data = params['n'] - (params['n'] * params['trainingRatio'])
    train_data = train_data - params['h_si_len'] + 1
    n_ave = train_data * params['SNR_AVERAGE']

    color_list = settings.plt_color_list()
    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_ave
        ax.plot(snrs_db, bers, color=color_list[IBO_index], marker='o', linestyle='--', label="IBO=%d[dB]" % IBO_db)

    ax.legend()
    plt.savefig(dirname + '/SNR_BER.pdf')

    output_png = dirname + '/SNR_BER.png'
    plt.savefig(output_png)

    # Plot learning curve
    for sigma_index, snr_db in enumerate(snrs_db):
        plt.figure()

        for IBO_index, IBO_db in enumerate(params['IBO_dB']):
            loss_avg = np.mean(losss[IBO_index][sigma_index], axis=0).T
            val_loss_avg = np.mean(val_losss[IBO_index][sigma_index], axis=0).T
            plt.plot(np.arange(1, len(loss_avg) + 1), loss_avg, color=color_list[IBO_index], marker='o', linestyle='--',
                     label='Training Frame (IBO=%d[dB])' % IBO_db)
            plt.plot(np.arange(1, len(loss_avg) + 1), val_loss_avg, color=color_list[IBO_index + len(params['IBO_dB'])],
                     marker='o', linestyle='--', label='Test Frame (IBO=%d[dB])' % IBO_db)

        plt.ylabel('less')
        plt.yscale('log')
        plt.xlabel('Training Epoch')
        plt.legend()
        plt.grid(which='major', alpha=0.25)
        plt.xlim([0, params['nEpochs'] + 1])
        plt.xticks(range(1, params['nEpochs'], 2))
        plt.savefig(dirname + '/snr_db_' + str(snr_db) + '_NNconv.pdf', bbox_inches='tight')

    slack.upload_file(output_png, "end:" + dirname + "\n" + json.dumps(params, indent=4))
    logging.info("end")
