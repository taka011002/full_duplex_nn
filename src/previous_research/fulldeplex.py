import numpy as np


# Estimates parameters for linear cancellation
def ls_estimation(x: np.ndarray, y: np.ndarray, chan_length: int) -> np.ndarray:
    # Construct LS problem
    A = np.reshape([np.flip(x[i + 1:i + chan_length + 1], axis=0) for i in range(x.size - chan_length)],
                   (x.size - chan_length, chan_length))

    # Solve LS problem
    h = np.linalg.lstsq(A, y[chan_length:])[0]

    # Output estimated channels
    return h

def si_cancellation_linear(x: np.ndarray, h: np.ndarray)->np.ndarray:
    # Calculate the cancellation signal
    xcan = np.convolve(x, h, mode='full')
    xcan = xcan[0:x.size]

    # Output
    return xcan