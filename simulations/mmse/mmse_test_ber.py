from src import modules as m
from src.previous_research.nn import NNModel as PreviousNNModel
from src.previous_research.system_model import SystemModel as PreviousSystemModel
import matplotlib.pyplot as plt
from simulations.common import graph
import numpy as np
from scipy.linalg import toeplitz
from tqdm import tqdm


def mmse(H, noise_var):
    H_H = H.conj().T
    L_w = H.shape[0]
    W = np.dot(np.linalg.inv(np.dot(H, H_H) + noise_var * np.eye(L_w)), H[:, 0])
    return W


def toeplitz_h(h, h_len, L_w):
    H_row = np.zeros((h_len + L_w, 1), dtype=complex)
    H_row[:h_len] = h
    H_col = np.zeros((L_w + 1, 1), dtype=complex)
    H_col[0] = h[0]
    H = toeplitz(H_col, r=H_row)
    return H


if __name__ == '__main__':
    SIMULATIONS_NAME = 'mmse_test'

    n = 1000000
    h_s_len = 2
    L_h = h_s_len - 1
    L_w = h_s_len - 1

    noise_var = 0.05
    sigma = np.sqrt(noise_var)

    h_s = np.random.normal(loc=0, scale=1, size=(1, h_s_len))
    # h_s = np.array([1, 0.5, 0]).reshape(1, h_s_len)
    print(h_s)

    s = np.random.choice([-1, 1], n)

    chanels_s = np.array([s[i:i + L_h + L_w] for i in range(s.size - h_s_len + 1)])  # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
    chanels_s = h_s * chanels_s
    y_s = np.sum(chanels_s, axis=1)

    n = np.random.normal(loc=0, scale=sigma, size=y_s.shape)
    r = y_s + n

    H = toeplitz_h(h_s.T, h_s_len, L_w)

    W = mmse(H, noise_var)
    WH = W.reshape(L_w + 1, 1).conj().T

    r_vec = np.array([r[i:i + h_s_len] for i in range(r.shape[0] - h_s_len + 1)])
    z = WH * r_vec
    z = np.sum(z, axis=1)

    d_hat = np.where(z > 0, 1, -1)
    error = np.sum(s[0:d_hat.shape[0]] != d_hat)
    ber = error / d_hat.shape[0]

    print(ber)

    if ber == 0:
        print("success")