from src import modules as m
import numpy as np
from src import ofdm
from scipy.linalg import dft
from simulations.common import graph
from simulations.common import settings
import matplotlib.pyplot as plt

graph.init_graph()
dirname = "../results/ofdm/test"
settings.init_output(dirname)

params = {
    "block": 1000,
    "subcarrier": 5,
    "CP": 4,
    "chanel_len": 4,
    "SNR_MIN": 0,
    "SNR_MAX": 25,
    "SNR_NUM": 6,
    "SNR_AVERAGE": 100,
    "equalizer": "ZF",
}

settings.dump_params(params, dirname)

ofdm_zero = np.hstack((np.zeros((params["subcarrier"], params["CP"])), np.eye(params["subcarrier"])))

F = dft(params['subcarrier'], "sqrtn")
FH = F.conj().T

snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
sigmas = m.sigmas(snrs_db)
errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))
q_errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))

for trials_index in range(params['SNR_AVERAGE']):
    h_si = m.channel(1, params['chanel_len'])
    H = ofdm.toeplitz_channel(h_si.T, params['chanel_len'], params['subcarrier'], params['CP'])
    Hc = ofdm.circulant_channel(h_si.T, params['chanel_len'], params['subcarrier'])

    D = F @ Hc @ FH
    D_1 = np.linalg.inv(D)

    for sigma_index, sigma in enumerate(sigmas):
        d = np.random.choice([0, 1], (params['subcarrier'] * 2 * params['block'], 1))
        s_n = m.modulate_qpsk(d)
        s = s_n.reshape(params['subcarrier'], params['block'])
        x = np.matmul(FH, s)
        x_cp = ofdm.add_cp(x, params['CP'])

        noise = m.awgn((params['subcarrier'] + params['CP'], params['block']), sigma)
        r = np.matmul(H[:, (params['chanel_len'] - 1):], x_cp) + noise
        r_s = r.flatten()

        y_p = r_s.reshape((params['subcarrier'] + params['CP'], params['block']))
        y_remove_cp = np.matmul(ofdm_zero, y_p)
        y = np.matmul(F, y_remove_cp)

        s_hat = np.matmul(D_1, y)
        s_n_hat = s_hat.reshape(params['subcarrier'] * params['block'])
        d_hat = m.demodulate_qpsk(s_n_hat).reshape((params['subcarrier'] * 2 * params['block'], 1))
        error = np.sum(d != d_hat)

        errors[sigma_index][trials_index] = error

        ### QPSK
        # q_r = (h_si * s_n) + m.awgn((params['subcarrier'] * params['block'], 1), sigma)
        # q_r = q_r * h_si.conj() / (np.abs(h_si) ** 2)
        # q_d_hat = m.demodulate_qpsk(q_r.squeeze()).reshape((params['subcarrier'] * 2 * params['block'], 1))
        # q_error = np.sum(d != q_d_hat)
        #
        # q_errors[sigma_index][trials_index] = q_error

ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'], -4, 0)
n_sum = params['subcarrier'] * 2 * params['block'] * params['SNR_AVERAGE']
errors_sum = np.sum(errors, axis=1)
bers = errors_sum / n_sum
ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="OFDM(QPSK)")

errors_sum = np.sum(q_errors, axis=1)
bers = errors_sum / n_sum
ber_ax.plot(snrs_db, bers, color="b", marker='x', linestyle=':', label="QPSK")
ber_ax.legend()

plt.tight_layout()
# plt.savefig(dirname + '/SNR_BER.pdf', bbox_inches='tight')
plt.show()
