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


def remove_cp(data, cp_len, data_len):
    return data[cp_len:(cp_len + data_len)]


def toeplitz_channel(h: np.ndarray, h_len: int, subcarrier, cp) -> np.ndarray:
    H_row = np.zeros((h_len + subcarrier + cp, 1), dtype=complex)
    H_row[:h_len+1] = h
    H_col = np.zeros((subcarrier + cp, 1), dtype=complex)
    H_col[0] = h[0]
    H = toeplitz(H_col, r=H_row)
    return H

def Hc(h: np.ndarray, h_len: int, subcarrier) -> np.ndarray:
    # H_row = np.zeros((subcarrier, 1), dtype=complex)
    # H_row[subcarrier-(h_len):] = np.flipud(h)[:h_len]
    # H_row[0] = h[0]
    H_col = np.zeros((subcarrier, 1), dtype=complex)
    H_col[:h_len+1] = h
    # H = toeplitz(H_col, H_row)
    H = circulant(H_col)
    return H