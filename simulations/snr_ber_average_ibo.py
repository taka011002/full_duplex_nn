from src import modules as m
from simulations.common import settings
from simulations.common import graph
from src.system_model import SystemModel
from src.nn import NNModel
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
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

    params, output_dir = settings.init_simulation(SIMULATIONS_NAME)

    # データを生成する
    snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出
    sigmas = sigmas * np.sqrt(params['receive_antenna'])

    errors = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE']))
    losss = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_losss = np.zeros((len(params['IBO_dB']), params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

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
                    params['optimizer'],
                    params['learningRate'],
                    params['h_si_len'],
                    params['h_s_len'],
                    params['receive_antenna'],
                    params['momentum']
                )
                nn_model.learn(
                    system_model,
                    params['trainingRatio'],
                    params['nEpochs'],
                    params['batchSize'],
                    params['h_si_len'],
                    params['h_s_len'],
                    params['receive_antenna'],
                    params['delay'],
                    params['standardization']
                )

                errors[IBO_index][sigma_index][trials_index] = nn_model.error
                losss[IBO_index][sigma_index][trials_index][:] = nn_model.history.history['loss']
                val_losss[IBO_index][sigma_index][trials_index][:] = nn_model.history.history['val_loss']

                # 学習済みモデルはpklできないので削除する．
                # del nn_model.model
                # del nn_model.history
                # nn_models[IBO_index][sigma_index][trials_index] = nn_model

    result = Result(params, errors, losss, val_losss, None)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])

    n_sum = params["test_bits"] * params['SNR_AVERAGE']

    color_list = graph.plt_color_list()
    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        errors_sum = np.sum(errors[IBO_index], axis=1)
        bers = errors_sum / n_sum
        ber_ax.plot(snrs_db, bers, color=color_list[IBO_index], marker='o', linestyle='--', label="IBO=%d[dB]" % IBO_db)

    ber_ax.legend()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    for sigma_index, snr_db in enumerate(snrs_db):
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params['nEpochs'])

        for IBO_index, IBO_db in enumerate(params['IBO_dB']):
            loss_avg = np.mean(losss[IBO_index][sigma_index], axis=0).T
            val_loss_avg = np.mean(val_losss[IBO_index][sigma_index], axis=0).T
            epoch = np.arange(1, len(loss_avg) + 1)
            learn_ax.plot(epoch, loss_avg, color=color_list[IBO_index], marker='o',
                          linestyle='--',
                          label='Training Frame (IBO=%d[dB])' % IBO_db)
            learn_ax.plot(epoch, val_loss_avg,
                          color=color_list[IBO_index + len(params['IBO_dB'])],
                          marker='o', linestyle='--', label='Test Frame (IBO=%d[dB])' % IBO_db)

        learn_ax.legend()
        plt.savefig(output_dir + '/snr_db_' + str(snr_db) + '_NNconv.pdf', bbox_inches='tight')

    settings.finish_simulation(params, output_dir, output_png_path)
