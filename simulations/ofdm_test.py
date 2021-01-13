from src import modules as m
import numpy as np
from src import ofdm
from scipy.linalg import dft
from simulations.common import graph
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fft import ifft, fft

# 送信信号
# n = 100
block = 1000
N = 32 # サブキャリア数
P = 32//4 # CP長
M = 0 # 通信路長

params = {
    "SNR_MIN": 0,
    "SNR_MAX": 30,
    "SNR_NUM": 6,
    "SNR_AVERAGE": 1000
}

# _F = np.exp((-1j*2*np.pi/N)*(np.mat(np.arange(0,N,1)).T*np.array(np.arange(0,N,1))))/np.sqrt(N)
F = dft(N, "sqrtn")
FH = F.conj().T

snrs_db = m.snr_db(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
sigmas = m.sigmas(snrs_db)
# sigmas = sigmas / np.sqrt(N)
# sigmas = sigmas / np.sqrt(2)
errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))
q_errors = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE']))

for trials_index in range(params['SNR_AVERAGE']):
    h_si = m.channel(1, M + 1)
    # h_si = np.array([1 + 1j, 1 + 1j]).reshape((1, 2))
    Hc = ofdm.Hc(h_si.T, M, N)
    Hc_mmse = Hc + np.eye(N)
    # Hc_mmse = Hc.conj() / (np.abs(Hc) + np.eye(5))

    D = F @ Hc @ FH
    D_1 = np.linalg.inv(D)

    for sigma_index, sigma in enumerate(sigmas):
        d = np.random.choice([0, 1], (N * 2 * block, 1))
        s_n = m.modulate_qpsk(d)
        s = s_n.reshape(N, block)
        x = np.matmul(FH, s)

        # _sigma = sigma / np.sqrt(N)
        noise = m.awgn((N, block), sigma)
        # x = x + noise
        r = np.matmul(Hc, x) + noise
        # r = np.matmul(Hc, x) + noise
        # r = r + noise

        # y = np.matmul(D, s) + noise
        y = np.matmul(F, r)

        s_hat = np.matmul(D_1, y)
        s_n_hat = s_hat.reshape(N * block)
        d_hat = m.demodulate_qpsk(s_n_hat).reshape((N * 2 * block, 1))
        error = np.sum(d != d_hat)

        errors[sigma_index][trials_index] = error

        ### QPSK
        q_r = (h_si * s_n) + m.awgn((N * block, 1), sigma)
        q_r = q_r / h_si
        q_d_hat = m.demodulate_qpsk(q_r.squeeze()).reshape((N * 2 * block, 1))
        q_error = np.sum(d != q_d_hat)

        q_errors[sigma_index][trials_index] = q_error



ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
n_sum = N * 2 * block * params['SNR_AVERAGE']
errors_sum = np.sum(errors, axis=1)
bers = errors_sum / n_sum
ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='-', label="OFDM")

errors_sum = np.sum(q_errors, axis=1)
bers = errors_sum / n_sum
ber_ax.plot(snrs_db, bers, color="b", marker='o', linestyle='--', label="QPSK")
ber_ax.legend()

plt.show()