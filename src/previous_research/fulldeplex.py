import numpy as np
from scipy.linalg import toeplitz
import scipy.io


# Estimates parameters for linear cancellation
def ls_estimation(x: np.ndarray, y: np.ndarray, chan_length: int) -> np.ndarray:
    # Construct LS problem
    A = np.reshape([np.flip(x[i + 1:i + chan_length + 1], axis=0) for i in range(x.size - chan_length)],
                   (x.size - chan_length, chan_length))

    # Solve LS problem
    h = np.linalg.lstsq(A, y[chan_length:])[0]

    # Output estimated channels
    return h


def si_cancellation_linear(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Calculate the cancellation signal
    xcan = np.convolve(x, h, mode='full')
    xcan = xcan[0:x.size]

    # Output
    return xcan


def mmse(H, noise_var):
    H_H = H.conj().T
    L_w = H.shape[0]
    h_1 = H[:, 0].reshape(L_w, 1)
    W = np.linalg.inv(H @ H_H + noise_var * np.eye(L_w)) @ h_1
    return W


def toeplitz_h(h, h_len, L_w):
    H_row = np.zeros((h_len + L_w, 1), dtype=complex)
    H_row[:h_len] = h
    H_col = np.zeros((L_w + 1, 1), dtype=complex)
    H_col[0] = h[0]
    H = toeplitz(H_col, r=H_row)
    return H


# Loads testbed data from file
def loadData(fileName, params):
    # Get parameters
    dataOffset = params['dataOffset']
    chanLen = params['hSILen']
    offset = np.maximum(dataOffset - int(np.ceil(chanLen / 2)), 1)

    # Load the file
    matFile = scipy.io.loadmat(fileName)

    # Prepare data
    x = np.squeeze(matFile['txSamples'], axis=1)[:-offset]
    y = np.squeeze(matFile['analogResidual'], axis=1)[offset:]
    y = y - np.mean(y)
    noise = np.squeeze(matFile['noiseSamples'], axis=1)
    noisePower = np.squeeze(matFile['noisePower'], axis=1)

    # Return
    return x, y, noise, noisePower
