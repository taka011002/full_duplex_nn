from src import modules as m
import numpy as np
from scipy.linalg import dft
from simulations.common import graph
from simulations.common import settings
import matplotlib.pyplot as plt
from src.ofdm_system_model import OFDMSystemModel
from src.ofdm_nn import OFDMNNModel


graph.init_graph()
dirname = "../results/ofdm/test"
settings.init_output(dirname)

params = {
    "block": 1000,
    "subcarrier": 32,
    "CP": 8,
    "h_si_len": 1,
    "h_s_len": 1,
    "SNR_MIN": 0,
    "SNR_MAX": 30,
    "SNR_NUM": 6,
    "SNR_AVERAGE": 1,
    "equalizer": "ZF",
    "gamma": 0.3,
    "phi": 3.0,
    "PA_IBO": 7,
    "PA_rho": 2,
    "LNA_IBO": 7,
    "LNA_rho": 2,
    "receive_antenna": 1,
    "TX_IQI": True,
    "PA": True,
    "LNA": True,
    "RX_IQI": True,

    "nHidden": [15],
    "nEpochs": 20,
    "optimizer": "momentum",
    "learningRate": 0.001,
    "momentum": 0.8,
    "trainingRatio": 0.8,
    "batchSize": 32,
    "delay": 0,
    "standardization": False,
}

settings.dump_params(params, dirname)

F = dft(params['subcarrier'], "sqrtn")
FH = F.conj().T

snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
sigmas = m.sigmas(snrs_db)
errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))

for trials_index in range(params['SNR_AVERAGE']):
    h_si = m.channel(1, params['h_si_len'])
    h_s = m.channel(1, params['h_s_len'])

    for sigma_index, sigma in enumerate(sigmas):
        system_model = OFDMSystemModel(
            params['block'],
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




# ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
# n_sum = params['subcarrier'] * 2 * params['block'] * params['SNR_AVERAGE']
# errors_sum = np.sum(errors, axis=1)
# bers = errors_sum / n_sum
# ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="OFDM")
#
# plt.tight_layout()
# plt.savefig(dirname + '/SNR_BER.pdf', bbox_inches='tight')
# plt.show()
