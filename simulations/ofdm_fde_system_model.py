from src import modules as m
import numpy as np
from scipy.linalg import dft
from simulations.common import graph
from simulations.common import settings
import matplotlib.pyplot as plt
from src.ofdm_system_model import OFDMSystemModel
from src.ofdm_nn import OFDMNNModel
from tqdm import tqdm
import dataclasses
import pickle


@dataclasses.dataclass
class Result:
    params: dict
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray


def proposal(params: dict, sigma, h_si, h_s) -> OFDMNNModel:
    training_blocks = int(params['block'] * params['trainingRatio'])
    test_blocks = params['block'] - training_blocks

    train_system_model = OFDMSystemModel(
        training_blocks,
        params['subcarrier'],
        params['CP'],
        sigma,
        params['gamma'],
        params['phi'],
        params['PA_IBO'],
        params['PA_rho'],
        params['LNA_IBO'],
        params['LNA_rho'],
        h_si,
        h_s,
        params['h_si_len'],
        params['h_s_len'],
        params['receive_antenna'],
        params['TX_IQI'],
        params['PA'],
        params['LNA'],
        params['RX_IQI']
    )

    test_system_model = OFDMSystemModel(
        test_blocks,
        params['subcarrier'],
        params['CP'],
        sigma,
        params['gamma'],
        params['phi'],
        params['PA_IBO'],
        params['PA_rho'],
        params['LNA_IBO'],
        params['LNA_rho'],
        h_si,
        h_s,
        params['h_si_len'],
        params['h_s_len'],
        params['receive_antenna'],
        params['TX_IQI'],
        params['PA'],
        params['LNA'],
        params['RX_IQI']
    )

    nn_model = OFDMNNModel(
        params['nHidden'],
        params['optimizer'],
        params['learningRate'],
        params['h_si_len'],
        params['h_s_len'],
        params['receive_antenna'],
        params['momentum']
    )

    nn_model.learn(
        train_system_model,
        test_system_model,
        params['trainingRatio'],
        params['nEpochs'],
        params['batchSize'],
        params['h_si_len'],
        params['h_s_len'],
        params['receive_antenna'],
        params['delay'],
        params['standardization']
    )

    return nn_model


if __name__ == '__main__':
    SIMULATIONS_NAME = 'ofdm_fde_system_model'

    params, output_dir = settings.init_simulation(SIMULATIONS_NAME, ofdm=True)

    F = dft(params['subcarrier'], "sqrtn")
    FH = F.conj().T

    snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)
    sigmas = sigmas * np.sqrt(params['receive_antenna'])
    errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))
    losss = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_losss = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))

    for trials_index in tqdm(range(params['SNR_AVERAGE'])):
        h_si = []
        h_s = []

        for i in range(params['receive_antenna']):
            h_si.append(m.exponential_decay_channel(1, params['h_si_len']))
            h_s.append(m.exponential_decay_channel(1, params['h_s_len']))

        for sigma_index, sigma in enumerate(sigmas):
            nn_model = proposal(params, sigma, h_si, h_s)

            errors[sigma_index][trials_index] = nn_model.error
            losss[sigma_index][trials_index][:] = nn_model.history.history['loss']
            val_losss[sigma_index][trials_index][:] = nn_model.history.history['val_loss']

    result = Result(params, errors, losss, val_losss)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
    n_sum = params['test_bits'] * params['SNR_AVERAGE']
    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="OFDM")

    plt.tight_layout()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    # slack通知用
    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    for sigma_index, snr_db in enumerate(snrs_db):
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params['nEpochs'])
        loss_avg = np.mean(losss[sigma_index], axis=0).T
        val_loss_avg = np.mean(val_losss[sigma_index], axis=0).T
        epoch = np.arange(1, len(loss_avg) + 1)

        learn_ax.plot(epoch, loss_avg, color="k", marker='o', linestyle='--', label='Training Frame')
        learn_ax.plot(epoch, val_loss_avg, color="r", marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.savefig(output_dir + '/snr_db_' + str(snr_db) + '_NNconv.pdf', bbox_inches='tight')

    settings.finish_simulation(params, output_dir, output_png_path)
