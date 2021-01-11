import numpy as np


def IDFT(data):
    """
    dataに対して逆離散フーリエ変換(IDFT)を行う．

    :param data:
    :return:
    """
    return np.fft.ifft(data)
