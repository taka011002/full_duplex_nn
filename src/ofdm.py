import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import circulant

def IDFT(data):
    """
    dataに対して逆離散フーリエ変換(IDFT)を行う．

    :param data:
    :return:
    """
    return np.fft.ifft(data)


def DFT(data):
    return np.fft.fft(data)


def add_cp(data, cp_len):
    cp = data[-cp_len:]
    return np.vstack((cp, data))


def remove_cp(data, cp_len):
    return data[cp_len:]


def toeplitz_channel(h: np.ndarray, h_len: int, subcarrier, cp) -> np.ndarray:
    rev_h = np.flipud(h)
    H_row = np.zeros(((h_len - 1) + subcarrier + cp, 1), dtype=complex)
    H_row[:h_len] = rev_h
    H_col = np.zeros((subcarrier + cp, 1), dtype=complex)
    H_col[0] = rev_h[0]
    H = toeplitz(H_col, r=H_row)
    return H

def circulant_channel(h: np.ndarray, h_len: int, subcarrier) -> np.ndarray:
    H_col = np.zeros((subcarrier, 1), dtype=complex)
    H_col[:h_len] = h
    H = circulant(H_col)
    return H