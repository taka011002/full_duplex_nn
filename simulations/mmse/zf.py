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


def zf(channel: np.ndarray) -> np.ndarray:
    H_H = channel.conj().T
    L_w = channel.shape[0]
    h_1 = channel[:, 0].reshape(L_w, 1)
    w = np.linalg.inv(channel @ H_H) @ h_1
    return w


def toeplitz_chanel(h: np.ndarray, h_len: int, L_w: int) -> np.ndarray:
    H_row = np.zeros((h_len + L_w, 1))
    H_row[:h_s_len] = h
    H_col = np.zeros((L_w + 1, 1))
    H_col[0] = h[0]
    H = toeplitz(H_col, r=H_row)
    return H


def channel(len: int):
    h = np.random.normal(loc=0, scale=1, size=(1, len)) + 1j * np.random.normal(loc=0, scale=1, size=(1, len))
    h = h / np.sqrt(2 * len)
    return h


if __name__ == '__main__':
    SIMULATIONS_NAME = 'mmse_test'

    h_s_len = 2
    L_h = h_s_len - 1
    L_w = 10

    # h_s = channel(h_s_len)
    h_s = np.array([1, 1]).reshape(1, 2)
    H = toeplitz_chanel(h_s.T, h_s_len, L_w)
    print(H)

    zf_W = zf(H)
    print("zf")
    print(zf_W.conj().T)
    print(zf_W.conj().T @ H)

    mmse_w = mmse(H, 0)
    print("mmse")
    print(mmse_w.conj().T)
    print(mmse_w.conj().T @ H)
