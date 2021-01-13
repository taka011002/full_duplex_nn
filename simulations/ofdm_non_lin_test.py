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
    "subcarrier": 32,
    "CP": 8,
    "chanel_len": 0,
    "SNR_MIN": 0,
    "SNR_MAX": 30,
    "SNR_NUM": 6,
    "SNR_AVERAGE": 100,
    "equalizer": "ZF",
    "gamma": 0.3,
    "phi": 3.0
}

settings.dump_params(params, dirname)


F = dft(params['subcarrier'], "sqrtn")
FH = F.conj().T

snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
sigmas = m.sigmas(snrs_db)
errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))

for trials_index in range(params['SNR_AVERAGE']):
    h_si = m.channel(1, params['chanel_len'] + 1)
    Hc = ofdm.Hc(h_si.T, params['chanel_len'], params['subcarrier'])
    Hc_mmse = Hc + np.eye(params['subcarrier'])

    D = F @ Hc @ FH
    D_1 = np.linalg.inv(D)

    for sigma_index, sigma in enumerate(sigmas):
        d = np.random.choice([0, 1], (params['subcarrier'] * 2 * params['block'], 1))
        s_n = m.modulate_qpsk(d)
        s = s_n.reshape(params['subcarrier'], params['block'])
        x = np.matmul(FH, s)

        ####　ここで送信機不整合を作る
        x = m.iq_imbalance(x, params['gamma'], params['phi'])
        x = x # PA

        #### ここまで

        noise = m.awgn((params['subcarrier'], params['block']), sigma)
        r = np.matmul(Hc, x) + noise

        ### ここで受信器不整合を作る
        r = r # LNA
        r = m.iq_imbalance(r, params['gamma'], params['phi'])

        ### ここまで

        y = np.matmul(F, r)

        s_hat = np.matmul(D_1, y)
        s_n_hat = s_hat.reshape(params['subcarrier'] * params['block'])
        d_hat = m.demodulate_qpsk(s_n_hat).reshape((params['subcarrier'] * 2 * params['block'], 1))

        error = np.sum(d != d_hat)
        errors[sigma_index][trials_index] = error

ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
n_sum = params['subcarrier'] * 2 * params['block'] * params['SNR_AVERAGE']
errors_sum = np.sum(errors, axis=1)
bers = errors_sum / n_sum
ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="OFDM")

plt.tight_layout()
plt.savefig(dirname + '/SNR_BER.pdf', bbox_inches='tight')
plt.show()
