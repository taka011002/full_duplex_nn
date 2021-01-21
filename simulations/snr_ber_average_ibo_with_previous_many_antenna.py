from src import modules as m
from simulations.common import settings
from simulations.common import graph
from src.system_model import SystemModel
from src.nn import NNModel
from src.previous_research.nn import NNModel as PreviousNNModel
from src.previous_research.system_model import SystemModel as PreviousSystemModel
import src.previous_research.fulldeplex as fd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from tqdm import tqdm
import dataclasses


@dataclasses.dataclass
class Result:
    params: dict
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray
    error_array: np.ndarray
    previous_losss: np.ndarray
    previous_val_losss: np.ndarray
    lin_error_array: np.ndarray
    non_cancell_error_array: np.ndarray


def proposal(params: dict, sigma, h_si, h_s, receive_antenna_count) -> NNModel:
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
        params['h_si_len'],
        params['h_s_len'],
        receive_antenna_count,
        params['TX_IQI'],
        params['PA'],
        params['LNA'],
        params['RX_IQI']
    )

    # NNを生成
    nn_model = NNModel(
        params['nHidden'],
        params['optimizer'],
        params['learningRate'],
        params['h_si_len'],
        params['h_s_len'],
        receive_antenna_count,
        params['momentum']
    )

    nn_model.learn(
        system_model,
        params['trainingRatio'],
        params['nEpochs'],
        params['batchSize'],
        params['h_si_len'],
        params['h_s_len'],
        receive_antenna_count,
        params['delay'],
        params['standardization']
    )

    return nn_model


def non_cancel_simulation(params: dict, sigma, h_si, h_s) -> np.ndarray:
    system_model = PreviousSystemModel(
        sigma,
        params['gamma'],
        params['phi'],
        params['PA_IBO_dB'],
        params['PA_rho'],
        params['LNA_IBO_dB'],
        params['LNA_rho'],
        h_si,
        params['h_si_len'],
        h_s,
        params['h_s_len'],
        params['TX_IQI'],
        params['PA'],
        params['LNA'],
        params['RX_IQI']
    )

    system_model.set_lna_a_sat(
        params['lin_n'],
        params['LNA_IBO_dB'],
    )

    system_model.transceive_si_s(
        params['lin_n'],
    )

    s_hat = system_model.y * h_s.conj() / (np.abs(h_s) ** 2)

    d_hat = m.demodulate_qpsk(s_hat)
    d_hat_len = d_hat.shape[0]
    error = np.sum(system_model.d_s[0:d_hat_len] != d_hat)

    return error


def lin_cancel_simulation(params: dict, sigma, h_si, h_s) -> np.ndarray:
    system_model = PreviousSystemModel(
        sigma,
        params['gamma'],
        params['phi'],
        params['PA_IBO_dB'],
        params['PA_rho'],
        params['LNA_IBO_dB'],
        params['LNA_rho'],
        h_si,
        params['h_si_len'],
        h_s,
        params['h_s_len'],
        params['TX_IQI'],
        params['PA'],
        params['LNA'],
        params['RX_IQI']
    )

    system_model.set_lna_a_sat(
        params['lin_n'],
        params['LNA_IBO_dB'],
    )

    system_model.transceive_si(
        params['lin_n']
    )

    h_lin = fd.ls_estimation(system_model.x[0:int(params['lin_n'] / 2)], system_model.y, params['h_si_len'])

    system_model.transceive_si_s(
        params['lin_n'],
    )

    yCanc = fd.si_cancellation_linear(system_model.x[0:int(params['lin_n'] / 2)], h_lin)
    cancelled_y = system_model.y - yCanc
    s_hat = cancelled_y * h_s.conj() / (np.abs(h_s) ** 2)

    d_hat = m.demodulate_qpsk(s_hat)
    d_hat_len = d_hat.shape[0]
    error = np.sum(system_model.d_s[0:d_hat_len] != d_hat)

    return error


def previous_non_lin(params: dict, sigma, h_si, h_s) -> PreviousNNModel:
    training_samples = int(np.floor(params['n'] * params['trainingRatio']))
    test_n = params['n'] - training_samples

    system_model = PreviousSystemModel(
        sigma,
        params['gamma'],
        params['phi'],
        params['PA_IBO_dB'],
        params['PA_rho'],
        params['LNA_IBO_dB'],
        params['LNA_rho'],
        h_si,
        params['h_si_len'],
        h_s,
        params['h_s_len'],
        params['TX_IQI'],
        params['PA'],
        params['LNA'],
        params['RX_IQI']
    )

    system_model.set_lna_a_sat(
        test_n,
        params['LNA_IBO_dB'],
    )

    system_model.transceive_si(
        params['n']
    )

    previous_nn_model = PreviousNNModel(
        params['h_si_len'],
        params['p_nHidden'],
        params['p_learningRate']
    )

    previous_nn_model.learn(
        system_model.x[0:int(params['n'] / 2)],
        system_model.y,
        params['p_trainingRatio'],
        params['h_si_len'],
        params['p_nEpochs'],
        params['p_batchSize']
    )

    system_model.transceive_si_s(
        test_n,
    )

    previous_nn_model.cancel(
        system_model.x[0:int(test_n / 2)],
        system_model.y,
        params['h_si_len'],
    )

    s_hat = previous_nn_model.cancelled_y * h_s.conj() / (np.abs(h_s) ** 2)

    d_hat = m.demodulate_qpsk(s_hat)
    d_hat_len = d_hat.shape[0]
    error = np.sum(system_model.d_s[0:d_hat_len] != d_hat)
    previous_nn_model.error = error

    return previous_nn_model


if __name__ == '__main__':
    SIMULATIONS_NAME = 'snr_ber_average_ibo_with_previous_many_antenna'

    params, output_dir = settings.init_simulation(SIMULATIONS_NAME)

    # データを生成する
    snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出
    sigmas = sigmas * np.sqrt(params['receive_antenna'])

    errors = np.zeros(
        (params['receive_antenna_max'], params['SNR_NUM'], params['SNR_AVERAGE']))
    losss = np.zeros((params['receive_antenna_max'], params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_losss = np.zeros((params['receive_antenna_max'], params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

    previous_errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))
    previous_losss = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    previous_val_losss = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

    error_array = np.zeros((len(snrs_db), params['SNR_AVERAGE']))
    lin_error_array = np.zeros((len(snrs_db), params['SNR_AVERAGE']))
    non_cancell_error_array = np.zeros((len(snrs_db), params['SNR_AVERAGE']))

    for trials_index in tqdm(range(params['SNR_AVERAGE'])):
        # 通信路は毎回生成する
        h_si = []
        h_s = []
        for i in range(params['receive_antenna_max']):
            h_si.append(m.channel(1, params['h_si_len']))
            h_s.append(m.channel(1, params['h_s_len']))

        for sigma_index, sigma in enumerate(sigmas):
            logging.info("SNR_AVERAGE_index:" + str(trials_index))
            logging.info("sigma_index:" + str(sigma_index))

            for i, receive_antenna_count in enumerate(range(params['receive_antenna_min'], params['receive_antenna_max']+1)):
                proposal_sigma = sigma * receive_antenna_count
                nn_model = proposal(params, proposal_sigma, h_si[:i+1], h_s[:i+1], receive_antenna_count)
                errors[i][sigma_index][trials_index] = nn_model.error
                losss[i][sigma_index][trials_index][:] = nn_model.history.history['loss']
                val_losss[i][sigma_index][trials_index][:] = nn_model.history.history['val_loss']

            non_cancell_error_array[sigma_index][trials_index] = non_cancel_simulation(params, sigma, h_si[0], h_s[0])
            lin_error_array[sigma_index][trials_index] = lin_cancel_simulation(params, sigma, h_si[0], h_s[0])

            previous_nn_model = previous_non_lin(params, sigma, h_si[0], h_s[0])
            error_array[sigma_index][trials_index] = previous_nn_model.error
            previous_losss[sigma_index][trials_index][:] = previous_nn_model.history.history['loss']
            previous_val_losss[sigma_index][trials_index][:] = previous_nn_model.history.history['val_loss']

    result = Result(params, errors, losss, val_losss, error_array, previous_losss, previous_val_losss, lin_error_array,
                    non_cancell_error_array)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
    n_sum = params['lin_n'] * params['SNR_AVERAGE']
    errors_sum = np.sum(non_cancell_error_array, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="Ww/o canceller", ms=12)

    # n_sum = params['lin_n'] * params['SNR_AVERAGE']
    # errors_sum = np.sum(lin_error_array, axis=1)
    # bers = errors_sum / n_sum
    # ber_ax.plot(snrs_db, bers, color="g", marker='o', linestyle='--', label="Previous(Linear)")

    n_sum = params["test_bits"] * params['SNR_AVERAGE']
    errors_sum = np.sum(error_array, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(snrs_db, bers, color="m", marker='o', linestyle='--', label="Conventional", ms=12)

    color_list = graph.plt_color_list()
    for i, receive_antenna_count in enumerate(range(params['receive_antenna_min'], params['receive_antenna_max'] + 1)):
        n_sum = params["test_bits"] * params['SNR_AVERAGE']
        errors_sum = np.sum(errors[i], axis=1)
        bers = errors_sum / n_sum
        ber_ax.plot(snrs_db, bers, color=color_list[i],
                    marker='o', linestyle='--', label="Proposed(Antenna: %d)" % (i+1), ms=12)

    ber_ax.legend(fontsize=24, loc=3)
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    # for sigma_index, snr_db in enumerate(snrs_db):
    #     learn_fig, learn_ax = graph.new_learning_curve_canvas(params['nEpochs'])
    #     loss_avg = np.mean(losss[sigma_index], axis=0).T
    #     val_loss_avg = np.mean(val_losss[sigma_index], axis=0).T
    #     epoch = np.arange(1, len(loss_avg) + 1)
    #
    #     learn_ax.plot(epoch, loss_avg, color="k", marker='o', linestyle='--', label='Training Frame')
    #     learn_ax.plot(epoch, val_loss_avg, color="r", marker='o', linestyle='--', label='Test Frame')
    #     learn_ax.legend()
    #     plt.savefig(output_dir + '/snr_db_' + str(snr_db) + '_NNconv.pdf', bbox_inches='tight')

    settings.finish_simulation(params, output_dir, output_png_path)
