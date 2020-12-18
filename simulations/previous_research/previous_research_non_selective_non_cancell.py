from src import modules as m
from simulations.common import settings
from src.previous_research.nn import NNModel as PreviousNNModel
from src.previous_research.system_model import SystemModel as PreviousSystemModel
import src.previous_research.fulldeplex as fd
import matplotlib.pyplot as plt
from simulations.common import graph
import numpy as np
from tqdm import tqdm
import logging
import pickle


class Result:
    params: dict
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray

    def __init__(self, params, errors, losss, val_losss):
        self.params = params
        self.errors = errors
        self.losss = losss
        self.val_losss = val_losss


if __name__ == '__main__':
    SIMULATIONS_NAME = 'previous_research_non_selective_non_cancell'

    params, output_dir = settings.init_simulation(SIMULATIONS_NAME)

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    error_array = np.zeros((len(snrs_db), params['SNR_AVERAGE']))

    for trials_index in tqdm(range(params['SNR_AVERAGE'])):
        h_si = m.channel(1, params['h_si_len'])
        h_s = m.channel(1, params['h_s_len'])

        for sigma_index, sigma in enumerate(sigmas):
            logging.info("SNR_AVERAGE_index:" + str(trials_index))
            logging.info("sigma_index:" + str(sigma_index))

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
            )

            system_model.set_lna_a_sat(
                test_n,
                params['LNA_IBO_dB'],
            )

            system_model.transceive_si_s(
                test_n,
            )

            s_hat = system_model.y * h_s.conj() / (np.abs(h_s)**2)

            d_hat = m.demodulate_qpsk(s_hat)
            d_hat_len = d_hat.shape[0]
            error = np.sum(system_model.d_s[0:d_hat_len] != d_hat)

            error_array[sigma_index][trials_index] = error

    result = Result(params, error_array, None, None)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
    n_sum = d_hat_len * params['SNR_AVERAGE']

    errors_sum = np.sum(error_array, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="Without canceller")
    ber_ax.legend()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    # settings.finish_simulation(params, output_dir, output_png_path)