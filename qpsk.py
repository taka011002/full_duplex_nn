import numpy as np
import modules as m


class QPSK:
    """
    QPSKモデル
    Attributes
    ----------
    d: ndarray
       送信シンボル
    """

    def __init__(self, bit_num):
        """
        :param bit_num: int
            送信ビット数
        """
        self.bits = np.random.choice([0, 1], bit_num)
        self.d = m.modulate_qpsk(self.bits)

    def bers(self, snrs):
        """
        SNRのndarrayを元にBERを求める。

        :param snrs: ndarray
            SNR(db)
        :return: ndarray
        """
        return [self.ber(sigma) for sigma in m.sigmas(snrs)]

    def ber(self, sigma):
        """
        sigmaを元にBERを求める。

        :param sigma: float
        :return: float
        """

        r = self.d + m.awgn(self.d.size, sigma)
        ber = self.check_error(r)
        return ber

    def check_error(self, y):
        """
        受信信号yと送信シンボルdとのBERを求める。

        :param y: ndarray
        :return: float
        """
        error = np.sum(self.bits != m.demodulate_qpsk(y))
        ber = error / self.bits.size
        return ber
