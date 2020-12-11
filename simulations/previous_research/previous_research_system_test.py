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
    SIMULATIONS_NAME = 'previous_research_nn_test'

    output_dir = '../results/previous_research_nn_test'
    params = {
        "n": 10000,
        "gamma": 0.3,
        "phi": 3.0,

        "PA_rho": 2,
        "PA_IBO_dB": 7,

        "LNA_rho": 2,
        "LNA_IBO_dB": 7,

        "SNR_MIN": 0,
        "SNR_MAX": 25,
        "SNR_NUM": 6,
        "SNR_AVERAGE": 1,

        "nHidden": 10,
        "nEpochs": 20,
        "learningRate": 0.001,
        "trainingRatio": 0.8,
        "batchSize": 32,

        "h_si_len": 1,
        "h_s_len": 1,
    }

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    h_si = m.channel(1, params['h_si_len'])
    h_s = m.channel(1, params['h_s_len'])
    error_array = np.zeros((len(snrs_db)))
    for sigma_index, sigma in enumerate(sigmas):
        # h_si = np.array([0.01+ 0.01j]).reshape((1, 1))
        # h_s = np.array([0 + 1j]).reshape((1, 1))

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
        system_model.transceive_s(
            params['n']
        )

        s_hat = system_model.y * h_s.conj() / (np.abs(h_s)**2)
        d_s_hat = m.demodulate_qpsk(s_hat)
        n_sum = d_s_hat.shape[0]
        error = np.sum(system_model.d_s[0:n_sum] != d_s_hat)
        error_array[sigma_index] = error

    # errors_sum = np.sum(error_array, axis=1)
    errors_sum = error_array
    bers = errors_sum / n_sum

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
    ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="BER")
    ber_ax.legend()
    plt.show()