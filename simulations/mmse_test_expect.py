from src import modules as m
import matplotlib.pyplot as plt
from simulations.common import graph
import numpy as np
from scipy.linalg import toeplitz
from tqdm import tqdm


def mmse(channel: np.ndarray, noise_var: float) -> np.ndarray:
    H_H = channel.conj().T
    L_w = channel.shape[0]
    h_1 = channel[:, 0].reshape(L_w, 1)
    w = np.linalg.inv(channel @ H_H + noise_var * np.eye(L_w)) @ h_1
    return w


def toeplitz_chanel(h: np.ndarray, h_len: int, L_w: int) -> np.ndarray:
    H_row = np.zeros((h_len + L_w, 1), dtype=complex)
    H_row[:h_s_len] = h
    H_col = np.zeros((L_w + 1, 1), dtype=complex)
    H_col[0] = h[0]
    H = toeplitz(H_col, r=H_row)
    return H


if __name__ == '__main__':
    SIMULATIONS_NAME = 'mmse_test'

    n = 20000
    ave = 100
    h_s_len = 2
    L_h = h_s_len - 1
    L_w = L_h
    snr_min = 0
    snr_max = 25
    snr_dots = 6

    snrs_db = np.linspace(snr_min, snr_max, snr_dots)
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    error_array = np.zeros((len(snrs_db), ave))
    for trials_index in tqdm(range(ave)):
        # h_s = m.channel(1, h_s_len)
        h_s = np.array([1 +1j, 1 + 1j]).reshape((1, 2))

        for sigma_index, sigma in enumerate(sigmas):
            noise_var = sigma ** 2

            d_s = np.random.choice([0, 1], n)
            s = m.modulate_qpsk(d_s)

            H = toeplitz_chanel(h_s.T, h_s_len, L_w)

            size = s.shape[0]
            chanels_s = np.array(
                [s[i:i + size - L_h - L_w] for i in range(L_h + L_w + 1)])  # [s_k, s_k-1, s_k-2]のように通信路の数に合わせる
            y_s = np.matmul(H, chanels_s)

            r = y_s + m.awgn(y_s.shape, sigma)

            W = mmse(H, noise_var)
            z = np.matmul(W.conj().T, r)

            d_hat = m.demodulate_qpsk(z)
            error = np.sum(d_s[0:d_hat.shape[0]] != d_hat)

            error_array[sigma_index][trials_index] = error
            # print(error / d_hat.shape[0])

    ber_fig, ber_ax = graph.new_snr_ber_canvas(snr_min, snr_max)
    n_sum = (n - 2 * h_s_len) * ave

    errors_sum = np.sum(error_array, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="MMSE")

    ber_ax.legend()
    plt.show()
